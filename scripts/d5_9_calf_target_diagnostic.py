"""D5.9 follow-up -- log the PROCESSED FL calf position target (the exact
`u` the original, currently-disabled position actuator would receive),
purely to enable an OFFLINE, counterfactual reconstruction of what force
that actuator would have produced if it had stayed active -- the
original actuator is NEVER re-enabled, and nothing computed here is fed
back into control.

WHY THIS IS NEEDED (see D5.9's diagnostic report): `final_raw` (the
12-dim composed action passed to `self._joint_pos_action.process_actions`)
already contains the frozen policy's raw calf action term-by-term, but
`process_actions` applies `scale`+`offset` (GO2_ACTION_SCALE and
`default_joint_pos`) BEFORE the result becomes the actuator's `ctrl`
target. D5.9 confirmed this target (`u`) is not equal to the raw policy
action and is not currently logged anywhere -- this script closes that
one gap.

HOW THIS STAYS READ-ONLY / DIAGNOSTIC-ONLY:
  - `act._joint_pos_action._processed_actions` is a tensor ALREADY
    computed every step by the existing, unmodified `JointPositionAction.
    process_actions` call inside `ResidualRecoveryAction.process_actions`
    (see actions.py, `self._joint_pos_action.process_actions(final_raw)`).
    This script only READS that tensor at the FL-calf's local index
    (confirmed via the existing assertion in actions.py that
    `self._joint_pos_action.target_names == WALK_JOINT_ORDER`, so
    `FL_calf_joint` is index 2) -- no new computation, no write, no
    feedback into `final_raw`/`delta_a`/the auxiliary actuator commands.
  - The original calf position actuator's gain/bias are NEVER modified
    or re-enabled here -- `_apply_hybrid_actuators` runs exactly as in
    every prior D5.x script, unmodified. This script adds ONE extra
    logged column and nothing else.
  - Identical build/rollout pattern to d5_6_base_pose_diagnostic.py
    (same `build_ablation_env`, same deterministic zero-residual-action
    rollout, same seeds/config knobs). Output goes to a SEPARATE
    directory (`logs/d5_9_diagnostics/seed{N}/`) so no existing D5.5/
    D5.6/D5.9(report) outputs are touched.

Offline force reconstruction (done AFTER this script, in a separate
analysis pass, NOT during the simulation):
    F_hyp = clip(stiffness*(u - calf_pos) - damping*calf_vel,
                 -effort_limit, +effort_limit)
  with GO2_ACTUATOR_CALF's stiffness=40.0, damping=2.0, effort_limit=45,
  confirmed against MuJoCo's mjGAIN_FIXED/mjBIAS_AFFINE force law
  (gainprm[0]=stiffness, biasprm[1]=-stiffness, biasprm[2]=-damping ->
  F = stiffness*u - stiffness*q - damping*qdot, clipped to forcerange)
  in `mjlab.utils.spec.create_position_actuator`.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_3_extended_recovery_window import build_ablation_env  # noqa: E402


@dataclass
class D59Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 200
  max_recovery_steps_override: int = 48
  ablate_hip_wire: bool = False
  out_dir: str = ""


def run(cfg: D59Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_ablation_env(
    d3_cfg, device, cfg.ablate_hip_wire, cfg.max_recovery_steps_override
  )
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  # D5.9: validate the local action index for FL_calf_joint within the
  # existing JointPositionAction term BEFORE trusting it for logging.
  assert list(act._joint_pos_action.target_names)[2] == "FL_calf_joint", (
    f"Unexpected local index for FL_calf_joint: "
    f"{list(act._joint_pos_action.target_names)}"
  )
  calf_local_action_idx = 2

  import mujoco

  mj_model = env.sim.mj_model

  def geom_name(gid: int) -> str:
    if gid < 0:
      return "<none>"
    n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom#{gid}"

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  state = {
    "enter_line": None,
    "exit_line": None,
    "stage1_first_step": None,
    "stage2_first_step": None,
  }

  def do_step(step: int, action: torch.Tensor) -> dict:
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    prev_phase = int(act.phase[0])
    env.step(action)
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      state["exit_line"] = (
        f"EXIT RECOVERY step={step} elapsed={int(act.recovery_elapsed[0])} "
        f"timeout={bool(act.recovery_failed_timeout[0])}"
      )

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())
    fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    fl_ground_found = bool((fl_ground.data.found[0].reshape(-1) > 0).any())
    fr_wire = env.scene.sensors["fr_wire_contact_diag"]
    fr_wire_found = bool((fr_wire.data.found[0].reshape(-1) > 0).any())

    contact = env.sim.data.contact
    nacon = int(env.sim.data.nacon[0])
    hip_wire_contact_now = False
    calf_wire_contact_now = False
    for ci in range(nacon):
      if int(contact.worldid[ci]) != 0:
        continue
      n0, n1 = geom_name(int(contact.geom[ci][0])), geom_name(int(contact.geom[ci][1]))
      pair = (n0, n1)
      if any("FL_hip_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        hip_wire_contact_now = True
      if any("FL_calf2_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        calf_wire_contact_now = True

    calf_forward_dist = float(act.calf_forward_dist[0])
    calf_height_above_wire = float(act.calf_height_above_wire[0])
    calf_side_far = bool(act.calf_side_far[0])
    foot_forward_dist = float(act.foot_forward_dist[0])
    calf_clearance_achieved = bool(act._calf_clearance_achieved[0])
    foot_crossed_achieved = bool(act._foot_crossed_achieved[0])
    crossing_achieved = bool(act._crossing_achieved[0])
    release_counter = int(act._release_counter[0])
    recovery_failed_timeout_flag = bool(act._recovery_failed_timeout[0])
    released_sustained = crossing_achieved and (
      release_counter >= act.cfg.release_required_steps
    )
    genuine_recovery_complete = crossing_achieved and released_sustained

    # --- D5.9 diagnostic-only addition: read-only, already-computed by
    # the existing `self._joint_pos_action.process_actions(final_raw)`
    # call inside `process_actions` (called via env.step above, BEFORE
    # this point in do_step). NOT fed back into control. ---
    calf_pos_target_hypothetical = float(
      act._joint_pos_action._processed_actions[0, calf_local_action_idx]
    )

    if calf_clearance_achieved and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if foot_crossed_achieved and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step

    return {
      "step": step,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "foot_z": float(foot_pos[2]),
      "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
      "foot_height_above_wire": float(act.foot_height_above_wire[0]),
      "hip_pos": float(qpos[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "base_z": float(base_pos[2]),
      "base_quat_w": float(base_quat[0]),
      "wire_in_contact": wire_in_contact,
      "fl_ground_contact": fl_ground_found,
      "fr_wire_contact": fr_wire_found,
      "fl_hip_wire_contact_raw": hip_wire_contact_now,
      "fl_calf_wire_contact_raw": calf_wire_contact_now,
      "pos_thigh_actuator_force_global": float(
        env.sim.data.actuator_force[0, act._pos_thigh_ctrl_id]
      ),
      "aux_thigh_actuator_force_global": float(
        env.sim.data.actuator_force[0, act._aux_thigh_ctrl_id]
      ),
      "pos_calf_actuator_force_global": float(
        env.sim.data.actuator_force[0, act._pos_calf_ctrl_id]
      ),
      "aux_calf_actuator_force_global": float(
        env.sim.data.actuator_force[0, act._aux_calf_ctrl_id]
      ),
      "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
      "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
      "calf_forward_dist": calf_forward_dist,
      "calf_height_above_wire": calf_height_above_wire,
      "calf_side_far": calf_side_far,
      "foot_forward_dist": foot_forward_dist,
      "calf_clearance_achieved": calf_clearance_achieved,
      "foot_crossed_achieved": foot_crossed_achieved,
      "crossing_achieved": crossing_achieved,
      "release_counter": release_counter,
      "recovery_failed_timeout_flag": recovery_failed_timeout_flag,
      "released_sustained": released_sustained,
      "genuine_recovery_complete": genuine_recovery_complete,
      # D5.9 addition:
      "calf_pos_target_hypothetical": calf_pos_target_hypothetical,
    }

  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    rows.append(do_step(step, action))

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"ablate_hip_wire={cfg.ablate_hip_wire}\n")
    f.write(f"max_recovery_steps_override={cfg.max_recovery_steps_override}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
  print(f"[D5.9] wrote {csv_path}")
  print(f"  {state['enter_line']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D59Config)
  run(cfg)


if __name__ == "__main__":
  main()
