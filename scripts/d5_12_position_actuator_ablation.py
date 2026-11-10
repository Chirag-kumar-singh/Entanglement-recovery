"""D5.12 -- Diagnose the hybrid recovery activation transient (part 2):
does DISABLING THE FL POSITION ACTUATORS at recovery entry (losing the
leg's normal stance-stabilizing gain/bias) by itself contribute to the
base-height sag found in D5.10, independent of the auxiliary
thigh-velocity/calf-torque commands' magnitude or onset profile?

D5.10 compared "override fully active" vs. "override fully disabled"
(which changes BOTH the position-actuator disabling AND the auxiliary
commands at once). D5.11 held the position-actuator disabling fixed and
varied only the auxiliary commands' onset speed, finding no effect.
D5.12 now holds the auxiliary command application fixed (always
SUPPRESSED, i.e. zero) and compares WITH vs. WITHOUT the position-
actuator disabling -- the one remaining factor neither D5.10 nor D5.11
isolated on its own.

CONDITION A -- baseline (identical to D5.10/D5.11's Condition A):
`_apply_hybrid_actuators` runs completely unmodified. Position actuators
disabled AND full-magnitude auxiliary commands applied immediately.
Included here again purely as the reference point for the drop/timing
numbers (same env build, same seed) -- not a new condition.

CONDITION C -- position actuators disabled, auxiliary commands
suppressed (zero): the REAL, unmodified `_apply_hybrid_actuators` still
runs first and unconditionally -- this performs the SAME position-
actuator gain/bias zeroing and auxiliary-actuator gain-enabling as
Condition A, byte-for-byte. Immediately afterward, a thin wrapper
overwrites `hybrid_thigh_vel_ctrl`/`hybrid_calf_torque_ctrl` to exactly
zero for RECOVERY steps and re-issues `set_joint_velocity_target`/
`set_joint_effort_target` with zero targets -- so the FL thigh/calf
position actuators are disabled exactly as in Condition A (the leg has
no normal stance-stabilizing gain/bias throughout RECOVERY), but NO
auxiliary torque/velocity command is ever applied. Because the
auxiliary thigh actuator is a velocity actuator with damping
(`recovery_thigh_velocity_kd`) enabled by the unmodified
`_apply_hybrid_actuators` call, commanding zero velocity does not mean
"no force" -- it means the thigh is lightly damped toward zero velocity
(passive damping only), while the calf's auxiliary torque actuator
commanded to zero applies exactly zero torque. This isolates "actuators
disabled, no commanded assistance" as cleanly as the existing actuator
model allows, without inventing a new actuator mode.

HOW CONDITION C IS IMPLEMENTED (minimal, isolated, non-production):
  - Same monkeypatch pattern as D5.11's `install_ramped_aux_commands`:
    the real bound method is saved and called first, unconditionally,
    then only the two logged command attributes and their physical
    targets are overwritten -- for RECOVERY steps only (`in_recovery`
    mask recomputed from `self._phase`, matching D5.11's convention).
  - No FSM, threshold, gain, wire-geometry, checkpoint, or
    `max_recovery_steps` change. No velocity gate or calf waiting-state
    mechanism. No production file edited.

Reuses `build_d510_env` from d5_10_stumble_ablation.py UNCHANGED, so all
conditions here share identical initial state with D5.10/D5.11.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_10_stumble_ablation import build_d510_env  # noqa: E402


def install_zero_aux_commands(act) -> None:
  """Monkeypatch `act._apply_hybrid_actuators` (bound method on this
  action-term instance only) to suppress the auxiliary thigh-velocity/
  calf-torque commands to exactly zero during RECOVERY, while leaving
  every other side effect of the real method (position-actuator
  disabling/restoring, auxiliary-actuator gain toggling, peak-detector
  state) completely untouched -- isolating the position-actuator
  disabling as the only active factor relative to normal walking."""
  original = act._apply_hybrid_actuators

  def zero_aux_apply_hybrid_actuators() -> None:
    original()  # unmodified: gain toggling + full-magnitude command application
    in_recovery = act._phase == 1  # PHASE_RECOVERY
    zero_thigh = torch.where(
      in_recovery,
      torch.zeros_like(act.hybrid_thigh_vel_ctrl),
      act.hybrid_thigh_vel_ctrl,
    )
    zero_calf = torch.where(
      in_recovery,
      torch.zeros_like(act.hybrid_calf_torque_ctrl),
      act.hybrid_calf_torque_ctrl,
    )
    act.hybrid_thigh_vel_ctrl = zero_thigh
    act.hybrid_calf_torque_ctrl = zero_calf

    act._entity.set_joint_velocity_target(
      zero_thigh.unsqueeze(-1), joint_ids=[act._hybrid_thigh_joint_id]
    )
    act._entity.set_joint_effort_target(
      zero_calf.unsqueeze(-1), joint_ids=[act._hybrid_calf_joint_id]
    )

  act._apply_hybrid_actuators = zero_aux_apply_hybrid_actuators


@dataclass
class D512Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 150
  """Hard diagnostic-window cap, matching D5.10/D5.11 for direct comparability."""
  condition: Literal["baseline", "zero_aux"] = "baseline"
  """'baseline' = Condition A (existing hybrid controller, unmodified,
  identical to D5.10/D5.11's Condition A). 'zero_aux' = Condition C
  (position actuators disabled exactly as in baseline; auxiliary
  thigh-velocity/calf-torque commands suppressed to zero throughout
  RECOVERY)."""
  out_dir: str = ""
  debug: bool = False
  """If True, open a dynamic native MuJoCo viewer (real physics, same
  pattern as prior D5.x --debug flags) instead of running headless.
  NOTE: D5.11 found this viewer mode diverges from the headless rollout
  and crashes on teardown in this environment -- use with that caveat."""


def run(cfg: D512Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_d510_env(d3_cfg, device, max_recovery_steps_override=48)

  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  if cfg.condition == "zero_aux":
    install_zero_aux_commands(act)

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
    "terminated_step": None,
    "terminated_reason": None,
  }

  def read_row(step: int, new_phase: int) -> dict:
    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]
    base_lin_vel_w = entity.data.root_link_vel_w[0, :3].clone()
    base_ang_vel_w = entity.data.root_link_vel_w[0, 3:].clone()

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())
    fl_ground_found = bool(
      (env.scene.sensors["fl_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    fr_ground_found = bool(
      (env.scene.sensors["fr_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    rl_ground_found = bool(
      (env.scene.sensors["rl_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    rr_ground_found = bool(
      (env.scene.sensors["rr_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    fr_wire_found = bool(
      (env.scene.sensors["fr_wire_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )

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

    return {
      "step": step,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "hip_pos": float(qpos[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "foot_z": float(foot_pos[2]),
      "foot_height_above_wire": float(act.foot_height_above_wire[0]),
      "base_x": float(base_pos[0]),
      "base_y": float(base_pos[1]),
      "base_z": float(base_pos[2]),
      "base_vz_world": float(base_lin_vel_w[2]),
      "base_quat_w": float(base_quat[0]),
      "base_ang_vel_x_world": float(base_ang_vel_w[0]),
      "base_ang_vel_y_world": float(base_ang_vel_w[1]),
      "base_ang_vel_z_world": float(base_ang_vel_w[2]),
      "wire_in_contact": wire_in_contact,
      "fl_ground_contact": fl_ground_found,
      "fr_ground_contact": fr_ground_found,
      "rl_ground_contact": rl_ground_found,
      "rr_ground_contact": rr_ground_found,
      "fr_wire_contact": fr_wire_found,
      "fl_hip_wire_contact_raw": hip_wire_contact_now,
      "fl_calf_wire_contact_raw": calf_wire_contact_now,
      "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
      "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
      "aux_thigh_actuator_force_actual": float(
        env.sim.data.actuator_force[0, act._aux_thigh_ctrl_id]
      ),
      "aux_calf_actuator_force_actual": float(
        env.sim.data.actuator_force[0, act._aux_calf_ctrl_id]
      ),
      "calf_forward_dist": float(act.calf_forward_dist[0]),
      "calf_height_above_wire": float(act.calf_height_above_wire[0]),
      "calf_clearance_achieved": bool(act._calf_clearance_achieved[0]),
      "foot_crossed_achieved": bool(act._foot_crossed_achieved[0]),
      "crossing_achieved": bool(act._crossing_achieved[0]),
      "recovery_failed_timeout_flag": bool(act._recovery_failed_timeout[0]),
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
    row = read_row(step, new_phase)
    if act._calf_clearance_achieved[0] and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if act._foot_crossed_achieved[0] and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step
    return row

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    step_counter = {"i": 0}

    def viewer_policy(_obs):
      i = step_counter["i"]
      action = torch.zeros(env.action_manager.action.shape, device=device)
      row = do_step(i, action)
      rows.append(row)
      print(
        f"[D5.12] step={i} phase={row['phase']} base_z={row['base_z']:.4f} "
        f"thigh_vel_cmd={row['thigh_vel_cmd']:.3f} calf_torque_cmd={row['calf_torque_cmd']:.3f} "
        f"fl_ground={row['fl_ground_contact']}"
      )
      step_counter["i"] += 1
      if step_counter["i"] >= cfg.max_steps or bool(env.reset_buf[0]):
        raise SystemExit(0)
      return action

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
      if bool(env.reset_buf[0]):
        reason = "fell_over" if bool(env.reset_terminated[0]) else "time_out"
        state["terminated_step"] = step
        state["terminated_reason"] = reason
        print(f"[D5.12] env0 TERMINATED at step={step} reason={reason}")
        break

  env.close()

  if not rows:
    print(
      "[D5.12] WARNING: no rows recorded (viewer likely exited before any step completed)."
    )
    return out_dir

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"condition={cfg.condition}\n")
    f.write(f"seed={cfg.seed}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
    f.write(f"terminated_step={state['terminated_step']}\n")
    f.write(f"terminated_reason={state['terminated_reason']}\n")
  print(f"[D5.12] wrote {csv_path}")
  print(f"  {state['enter_line']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  print(
    f"  terminated_step={state['terminated_step']} reason={state['terminated_reason']}"
  )
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D512Config)
  run(cfg)


if __name__ == "__main__":
  main()
