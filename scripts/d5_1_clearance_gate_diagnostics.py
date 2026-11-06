"""D5.1 -- Recovery FSM clearance-gate instrumentation.

Extends D4.8's hip-wire-contact-ablation harness with READ-ONLY logging
of the recovery FSM's actual stage-1/stage-2 clearance-gate variables,
to determine (per D5.0's identified gap) whether stage 1 (calf
clearance) or stage 2 (foot/toe crossing) is the first gate that
demonstrably fails to latch, rather than relying on indirect proxies
(raw contact flags, toe height alone).

NOTHING about the controller, FSM, gains, thresholds, wire geometry, or
checkpoint is changed. This script is a near-verbatim copy of
`d4_8_hip_wire_contact_ablation.py` (same `build_ablation_env`, same
deterministic zero-residual-action rollout, same seeds/out-dir
structure) with ONLY the per-step logging dict extended to additionally
read the following existing attributes directly off the
`ResidualRecoveryAction` instance (`act`), none of which are
recomputed or altered here:

  - act.calf_forward_dist         (stage-1 input, signed, world frame
                                    projected onto latched forward dir)
  - act.calf_height_above_wire     (stage-1 input, calf z above wire top)
  - act.calf_side_far              (stage-1 input, calf_forward_dist>0)
  - act.foot_forward_dist          (stage-2 input)
  - act._calf_clearance_achieved   (stage-1 latch, persists once True)
  - act._foot_crossed_achieved     (stage-2 latch, persists once True)
  - act._crossing_achieved         (= stage1 latch AND stage2 latch)
  - act._phase                    (0=NORMAL_WALK,1=RECOVERY,2=POST_RECOVERY)
  - act._recovery_elapsed         (steps since entering RECOVERY)
  - act._release_counter          (consecutive no-contact steps counted
                                    only AFTER crossing_achieved)
  - act._recovery_failed_timeout  (True iff exit was via the 16-step cap)

ONE-STEP LOGGING ALIGNMENT (verified from source before writing this
script, not assumed):
  `ResidualRecoveryAction.process_actions()` calls `self._update_fsm()`
  FIRST (before computing/applying the actual control for that step),
  and `process_actions` itself runs INSIDE `env.step(action)` (as part
  of the action manager's pipeline), before the physics substep. So by
  the time `env.step(action)` RETURNS, `act._phase`/the gate variables/
  `act._recovery_elapsed` etc. all reflect the FSM's state AS OF the
  step that was just stepped (i.e., already advanced for this step).
  This script reads all per-step fields (both the original D4.8 fields
  like `foot_z`/`calf_pos` and the new gate variables) AFTER
  `env.step(action)` returns, exactly like D4.8 did for its own fields
  -- so the new columns are logged at the SAME point in the loop and
  are mutually consistent with the existing columns (no additional
  skew introduced). The ENTER/EXIT RECOVERY console-line detection
  (`newly_entangled` checked BEFORE `env.step`) retains the same
  one-step-early cosmetic offset already identified and verified
  harmless in D4.8 (confirmed there: underlying foot_z/base_z
  trajectories were bit-identical to the pre-existing D3 baseline
  despite that offset) -- this does not affect the new gate-variable
  columns, which are all read post-step.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from mjlab.tasks.recovery.env_cfg_residual import (  # noqa: E402
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)


def build_ablation_env(cfg: D3GroundConfig, device: str, ablate_hip_wire: bool):
  """Verbatim copy of D4.8's `build_ablation_env` (same sensors, same
  spec_fn composition). See that script's module docstring for the
  verified contact-exclusion mechanism and the composition requirement
  (naive spec_fn overwrite silently drops the wire -- confirmed in
  D4.8, not re-derived here)."""
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG if cfg.use_isolated_wire else None,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y if cfg.use_isolated_wire else None,
    experimental_peak_triggered_thigh_stop=cfg.experimental_peak_triggered_thigh_stop,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  leg = ["FL", "FR", "RL", "RR"][cfg.affected_leg_idx]
  other_leg = "FR" if leg == "FL" else "FL"

  fl_ground_cfg = ContactSensorCfg(
    name="fl_ground_contact_diag",
    primary=ContactMatch(mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  fl_ground_pergeom_cfg = ContactSensorCfg(
    name="fl_ground_pergeom_diag",
    primary=ContactMatch(mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  existing_wire_sensor = next(s for s in env_cfg.scene.sensors if s.name == "wire_contact")
  fr_wire_cfg = ContactSensorCfg(
    name="fr_wire_contact_diag",
    primary=existing_wire_sensor.primary,
    secondary=ContactMatch(
      mode="geom", pattern=f"{other_leg}_calf2_collision", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (
    fl_ground_cfg,
    fl_ground_pergeom_cfg,
    fr_wire_cfg,
  )

  existing_spec_fn = env_cfg.scene.spec_fn

  def composed_spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
    if existing_spec_fn is not None:
      existing_spec_fn(spec, num_envs)
    if ablate_hip_wire:
      wire_seg_bodies = [b.name for b in spec.bodies if b.name.startswith("wire_seg_")]
      assert wire_seg_bodies, (
        "No wire_seg_* bodies found after existing spec_fn -- composition order wrong."
      )
      for seg_body in wire_seg_bodies:
        spec.add_exclude(bodyname1="robot/FL_hip", bodyname2=seg_body)
      print(
        f"[D5.1-ABLATION] Added {len(wire_seg_bodies)} body-pair excludes: "
        f"robot/FL_hip <-> {wire_seg_bodies}"
      )

  env_cfg.scene.spec_fn = composed_spec_fn

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env


@dataclass
class D51Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 120
  ablate_hip_wire: bool = False
  out_dir: str = ""


def run(cfg: D51Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_ablation_env(d3_cfg, device, cfg.ablate_hip_wire)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

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
  enter_line = None
  exit_line = None
  stage1_first_step = None
  stage2_first_step = None

  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    if bool(act.newly_entangled[0]):
      enter_line = f"ENTER RECOVERY step={step}"
    prev_phase = int(act.phase[0])
    env.step(action)
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      exit_line = (
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
      if any("FL_hip_collision" in n for n in pair) and any("wire_seg" in n for n in pair):
        hip_wire_contact_now = True
      if any("FL_calf2_collision" in n for n in pair) and any("wire_seg" in n for n in pair):
        calf_wire_contact_now = True

    # --- D5.1 new: read the FSM's own clearance-gate state directly,
    # read-only, post-step (see module docstring for alignment note). ---
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

    if calf_clearance_achieved and stage1_first_step is None:
      stage1_first_step = step
    if foot_crossed_achieved and stage2_first_step is None:
      stage2_first_step = step

    rows.append(
      {
        "step": step,
        "phase": new_phase,
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
        # --- D5.1 new columns ---
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
      }
    )

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"ablate_hip_wire={cfg.ablate_hip_wire}\n")
    f.write(f"{enter_line}\n{exit_line}\n")
    f.write(f"stage1_first_latch_step={stage1_first_step}\n")
    f.write(f"stage2_first_latch_step={stage2_first_step}\n")
  print(f"[D5.1] wrote {csv_path}")
  print(f"  {enter_line}")
  print(f"  {exit_line}")
  print(f"  stage1_first_latch_step={stage1_first_step}")
  print(f"  stage2_first_latch_step={stage2_first_step}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D51Config)
  run(cfg)


if __name__ == "__main__":
  main()
