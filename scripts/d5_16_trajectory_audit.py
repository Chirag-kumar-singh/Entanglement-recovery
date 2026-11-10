"""D5.16 -- Baseline recovery trajectory audit.

Diagnostic-only script. Reuses D5.3's exact env-build path
(`build_ablation_env`, same `go2_wire_recovery_residual_env_cfg` factory
call, same isolated FL wire, same 48-step recovery cap, same frozen
walking checkpoint, same hybrid `_apply_hybrid_actuators` controller)
and the D5.15-corrected viewer stepping pattern (viewer callback never
calls `env.step()` itself -- see that script's `viewer_policy`
docstring for the full rationale), and ADDS diagnostic-only logging
columns (FR/RL/RR ground contact sensors, FL hip/thigh/calf joint
velocities, FL foot position in robot-base-relative coordinates, base
linear/angular velocity in body frame, full base orientation quaternion,
and the active actuator control-mode flag already tracked internally by
the action term as `hybrid_thigh_pos_gain_active`) needed to answer
D5.16's phase-timing/stability questions. NO controller output, gain,
threshold, FSM transition, or checkpoint is touched -- this script only
reads additional existing/derived state and writes it to CSV.

Does NOT modify scripts/d5_3_extended_recovery_window.py or any other
existing diagnostic script; prior logs under logs/d5_3_*/,
logs/d5_15_viewer_fix/, etc. are untouched.
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
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402


def build_env(
  cfg: D3GroundConfig,
  device: str,
  max_recovery_steps_override: int,
):
  """Same build path as D5.3's `build_ablation_env` (no hip-wire ablation
  branch needed for this audit -- baseline only), PLUS FR/RL/RR ground
  contact sensors (same `ContactSensorCfg` pattern D5.3 already uses for
  FL) so all four legs' ground contact can be logged, not just FL."""
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG if cfg.use_isolated_wire else None,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y if cfg.use_isolated_wire else None,
    experimental_peak_triggered_thigh_stop=cfg.experimental_peak_triggered_thigh_stop,
    max_recovery_steps=max_recovery_steps_override,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  leg = ["FL", "FR", "RL", "RR"][cfg.affected_leg_idx]
  other_legs = [l for l in ["FL", "FR", "RL", "RR"] if l != leg]

  ground_sensor_cfgs = []
  for leg_name in ["FL", "FR", "RL", "RR"]:
    ground_sensor_cfgs.append(
      ContactSensorCfg(
        name=f"{leg_name.lower()}_ground_contact_diag",
        primary=ContactMatch(
          mode="geom", pattern=rf"{leg_name}_.*_collision\d*$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
      )
    )

  existing_wire_sensor = next(
    s for s in env_cfg.scene.sensors if s.name == "wire_contact"
  )
  other_wire_cfgs = []
  for other_leg in other_legs:
    other_wire_cfgs.append(
      ContactSensorCfg(
        name=f"{other_leg.lower()}_wire_contact_diag",
        primary=existing_wire_sensor.primary,
        secondary=ContactMatch(
          mode="geom", pattern=f"{other_leg}_calf2_collision", entity="robot"
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
      )
    )

  env_cfg.scene.sensors = (
    tuple(env_cfg.scene.sensors) + tuple(ground_sensor_cfgs) + tuple(other_wire_cfgs)
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env


@dataclass
class D516Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 200
  max_recovery_steps_override: int = 48
  out_dir: str = ""
  debug: bool = False
  """If True, open the D5.15-corrected native MuJoCo viewer (real
  physics) instead of running headless. CSV/log output is identical
  either way."""


def run(cfg: D516Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_env(d3_cfg, device, cfg.max_recovery_steps_override)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index("FL_hip_joint")
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")

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

  def do_step_pre(step: int) -> int:
    """Pre-step bookkeeping (identical pattern to D5.15's D5.3 fix):
    must be called with env state reflecting the END of the PREVIOUS
    step, strictly BEFORE this step's own `env.step()` call."""
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    return int(act.phase[0])

  def do_step_post(step: int, prev_phase: int) -> dict:
    """Post-step row construction. Must be called with env state
    reflecting the state immediately AFTER this step's `env.step()` has
    executed exactly once."""
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      state["exit_line"] = (
        f"EXIT RECOVERY step={step} elapsed={int(act.recovery_elapsed[0])} "
        f"timeout={bool(act.recovery_failed_timeout[0])}"
      )

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos_w = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    base_pos_w = entity.data.root_link_pos_w[0].clone()
    base_quat_w = entity.data.root_link_quat_w[0].clone()
    base_lin_vel_b = entity.data.root_link_lin_vel_b[0].clone()
    base_ang_vel_b = entity.data.root_link_ang_vel_b[0].clone()

    foot_rel_w = (foot_pos_w - base_pos_w).unsqueeze(0)
    foot_pos_b = quat_apply_inverse(base_quat_w.unsqueeze(0), foot_rel_w)[0]

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())

    ground_contacts = {}
    for leg_name in ["FL", "FR", "RL", "RR"]:
      sensor = env.scene.sensors[f"{leg_name.lower()}_ground_contact_diag"]
      ground_contacts[leg_name] = bool((sensor.data.found[0].reshape(-1) > 0).any())

    other_wire_contacts = {}
    for leg_name in ["FR", "RL", "RR"]:
      sensor_name = f"{leg_name.lower()}_wire_contact_diag"
      if sensor_name in env.scene.sensors:
        sensor = env.scene.sensors[sensor_name]
        other_wire_contacts[leg_name] = bool(
          (sensor.data.found[0].reshape(-1) > 0).any()
        )
      else:
        other_wire_contacts[leg_name] = False

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
    foot_height_above_wire = float(act.foot_height_above_wire[0])
    calf_clearance_achieved = bool(act._calf_clearance_achieved[0])
    foot_crossed_achieved = bool(act._foot_crossed_achieved[0])
    crossing_achieved = bool(act._crossing_achieved[0])
    release_counter = int(act._release_counter[0])
    recovery_failed_timeout_flag = bool(act._recovery_failed_timeout[0])
    released_sustained = crossing_achieved and (
      release_counter >= act.cfg.release_required_steps
    )
    genuine_recovery_complete = crossing_achieved and released_sustained
    pos_actuators_active = bool(act.hybrid_thigh_pos_gain_active[0])

    if calf_clearance_achieved and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if foot_crossed_achieved and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step

    return {
      "step": step,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "pos_actuators_active": pos_actuators_active,
      "foot_z": float(foot_pos_w[2]),
      "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
      "foot_height_above_wire": foot_height_above_wire,
      "foot_x_base": float(foot_pos_b[0]),
      "foot_y_base": float(foot_pos_b[1]),
      "foot_z_base": float(foot_pos_b[2]),
      "hip_pos": float(qpos[hip_i]),
      "hip_vel": float(qvel[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "base_x": float(base_pos_w[0]),
      "base_y": float(base_pos_w[1]),
      "base_z": float(base_pos_w[2]),
      "base_quat_w": float(base_quat_w[0]),
      "base_quat_x": float(base_quat_w[1]),
      "base_quat_y": float(base_quat_w[2]),
      "base_quat_z": float(base_quat_w[3]),
      "base_lin_vel_x": float(base_lin_vel_b[0]),
      "base_lin_vel_y": float(base_lin_vel_b[1]),
      "base_lin_vel_z": float(base_lin_vel_b[2]),
      "base_ang_vel_x": float(base_ang_vel_b[0]),
      "base_ang_vel_y": float(base_ang_vel_b[1]),
      "base_ang_vel_z": float(base_ang_vel_b[2]),
      "wire_in_contact": wire_in_contact,
      "fl_ground_contact": ground_contacts["FL"],
      "fr_ground_contact": ground_contacts["FR"],
      "rl_ground_contact": ground_contacts["RL"],
      "rr_ground_contact": ground_contacts["RR"],
      "fr_wire_contact": other_wire_contacts["FR"],
      "rl_wire_contact": other_wire_contacts["RL"],
      "rr_wire_contact": other_wire_contacts["RR"],
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
    }

  def do_step(step: int, action: torch.Tensor) -> dict:
    """Headless-path wrapper: pre -> env.step() exactly once -> post."""
    prev_phase = do_step_pre(step)
    env.step(action)
    return do_step_post(step, prev_phase)

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    pending: dict = {"step": None, "prev_phase": None}

    def viewer_policy(_obs):
      """D5.15-pattern fix applied from the start: this callback NEVER
      calls `env.step()` itself. It finalizes the PREVIOUS step's row
      (via `do_step_post`, safe because the env state visible here is
      exactly what the viewer's own prior `env.step()` call produced),
      then prepares the NEXT step via `do_step_pre`, and returns a zero
      action for the viewer to apply via its own single `env.step()`
      call -- guaranteeing exactly one step per rendered tick."""
      if pending["step"] is not None:
        row = do_step_post(pending["step"], pending["prev_phase"])
        rows.append(row)
        if (
          pending["step"] % 10 == 0
          or row["calf_clearance_achieved"]
          or row["crossing_achieved"]
        ):
          print(
            f"[D5.16] step={pending['step']} phase={row['phase']} "
            f"calf_forward_dist={row['calf_forward_dist']:.4f} "
            f"calf_height_above_wire={row['calf_height_above_wire']:.4f} "
            f"stage1={row['calf_clearance_achieved']} stage2={row['foot_crossed_achieved']}"
          )
        if pending["step"] + 1 >= cfg.max_steps:
          raise SystemExit(0)

      step = 0 if pending["step"] is None else pending["step"] + 1
      prev_phase = do_step_pre(step)
      pending["step"] = step
      pending["prev_phase"] = prev_phase
      return torch.zeros(env.action_manager.action.shape, device=device)

    print(
      "[INFO] Debug mode: native viewer (DYNAMIC, real physics). Close window to end run."
    )
    try:
      NativeMujocoViewer(env, viewer_policy).run()
    except SystemExit:
      pass
  else:
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
    f.write(f"max_recovery_steps_override={cfg.max_recovery_steps_override}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
  print(f"[D5.16] wrote {csv_path}")
  print(f"  {state['enter_line']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D516Config)
  run(cfg)


if __name__ == "__main__":
  main()
