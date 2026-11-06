"""D3: First real end-to-end wire-recovery experiment using the
Quad-SDK-inspired HYBRID actuator-level recovery controller.

Entry point modeled directly on the existing, already-validated D2
launcher (scripts/d2_trajectory_check.py), reusing the SAME environment
factory (`go2_wire_recovery_residual_env_cfg`), SAME frozen walking
checkpoint default, and SAME wire geometry (`DEFAULT_WIRE_CFG` in
env_cfg_residual.py) -- only `recovery_mode="hybrid"` differs. Does NOT
modify the controller, FSM, wire geometry, or walking policy. This is a
baseline measurement run, not a tuning run.

Examples
--------
Headless, CSV + plots only::

    uv run python scripts/d3_wire_recovery_experiment.py

Visual, native MuJoCo viewer::

    uv run python scripts/d3_wire_recovery_experiment.py --debug
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import (
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
)
from mjlab.utils.torch import configure_torch_backends

PHASE_NAMES = {
  PHASE_NORMAL_WALK: "NORMAL_WALK",
  PHASE_RECOVERY: "RECOVERY",
  PHASE_POST_RECOVERY: "POST_RECOVERY",
}
LEG_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class D3Config:
  affected_leg_idx: int = 0
  """0=FL (first experiment leg per spec)."""
  seed: int = 0
  device: str | None = None
  max_steps: int = 400
  stop_steps_after_post_recovery: int = 60
  out_dir: str = ""
  debug: bool = False
  plot: bool = True


def _affected_joint_names(leg: str) -> tuple[str, str, str]:
  return (f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint")


def build_row(
  step: int,
  env: ManagerBasedRlEnv,
  act,
  entry_foot_pos: torch.Tensor | None,
  running_max_height: float,
) -> tuple[dict, float]:
  leg = LEG_NAMES[act.cfg.affected_leg_idx]
  hip_n, thigh_n, calf_n = _affected_joint_names(leg)
  entity = act._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index(hip_n)
  thigh_i = joint_names.index(thigh_n)
  calf_i = joint_names.index(calf_n)

  qpos = entity.data.joint_pos[0]
  qvel = entity.data.joint_vel[0]
  foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
  phase = int(act.phase[0])

  sim = env.sim
  pos_thigh_gain = sim.model.actuator_gainprm[0, act._pos_thigh_ctrl_id, 0].item()
  aux_thigh_gain = sim.model.actuator_gainprm[0, act._aux_thigh_ctrl_id, 0].item()
  pos_calf_gain = sim.model.actuator_gainprm[0, act._pos_calf_ctrl_id, 0].item()

  # Contact/entanglement state: read the SAME sensor the FSM itself
  # uses (no second, inconsistent definition of "in contact").
  sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
  in_contact = bool((sensor.data.found[0] > 0).any())

  if entry_foot_pos is not None:
    running_max_height = max(running_max_height, float(foot_pos[2]))

  exit_reason = ""
  if phase == PHASE_POST_RECOVERY:
    exit_reason = "cap_timeout" if bool(act.recovery_failed_timeout[0]) else "crossing+release"

  row = {
    "step": step,
    "sim_time": step * env.step_dt,
    "phase": PHASE_NAMES[phase],
    "in_contact": in_contact,
    "foot_x": float(foot_pos[0]),
    "foot_y": float(foot_pos[1]),
    "foot_z": float(foot_pos[2]),
    "foot_height_above_wire": float(act.foot_height_above_wire[0]),
    "foot_forward_dist_to_wire": float(act.foot_forward_dist[0]),
    "foot_side_far": bool(act.foot_side_far[0]),
    "thigh_pos": float(qpos[thigh_i]),
    "thigh_vel": float(qvel[thigh_i]),
    "calf_pos": float(qpos[calf_i]),
    "calf_vel": float(qvel[calf_i]),
    "walk_raw_thigh": float(act._raw_actions[0, thigh_i]) if act._raw_actions is not None else 0.0,
    "walk_raw_calf": float(act._raw_actions[0, calf_i]) if act._raw_actions is not None else 0.0,
    "recovery_thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
    "recovery_calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
    "pos_thigh_actuator_enabled": pos_thigh_gain != 0.0,
    "aux_thigh_actuator_enabled": aux_thigh_gain != 0.0,
    "pos_calf_actuator_enabled": pos_calf_gain != 0.0,
    "aux_calf_actuator_enabled": bool(act.hybrid_calf_torque_ctrl[0] != 0.0),
    "calf_clearance_achieved": bool(act.calf_clearance_achieved[0]),
    "foot_crossed_achieved": bool(act.foot_crossed_achieved[0]),
    "crossing_achieved": bool(act.crossing_achieved[0]),
    "recovery_exit_reason": exit_reason,
    "running_max_foot_height": running_max_height,
  }
  return row, running_max_height


def run(cfg: D3Config) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  out_dir = Path(
    cfg.out_dir
    or f"logs/d3_wire_recovery_hybrid/{LEG_NAMES[cfg.affected_leg_idx]}_seed{cfg.seed}"
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  act = env.action_manager._terms["joint_pos"]

  print(f"[D3] recovery_mode = {act.cfg.recovery_mode}")
  print(
    f"[D3] recovery_thigh_velocity={act.cfg.recovery_thigh_velocity}  "
    f"recovery_thigh_velocity_kd={act.cfg.recovery_thigh_velocity_kd}  "
    f"recovery_calf_torque={act.cfg.recovery_calf_torque}  "
    f"max_recovery_steps={act.cfg.max_recovery_steps}  "
    f"affected_leg={LEG_NAMES[cfg.affected_leg_idx]}"
  )
  print(f"[D3] wire geom names: {act.cfg.wire_geom_names}")

  rows: list[dict] = []
  entry_foot_pos: torch.Tensor | None = None
  running_max_h = -1e9
  contacted_wire = False
  recovery_activated = False

  env.reset(seed=cfg.seed)

  def do_step(step_idx: int, action: torch.Tensor) -> dict:
    nonlocal entry_foot_pos, running_max_h, contacted_wire, recovery_activated
    env.step(action)
    if bool(act.newly_entangled[0]):
      entry_foot_pos = act._entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
      running_max_h = -1e9
      recovery_activated = True
    row, running_max_h = build_row(step_idx, env, act, entry_foot_pos, running_max_h)
    if row["in_contact"]:
      contacted_wire = True
    return row

  reached_post_recovery_at: int | None = None
  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    step_counter = {"i": 0}

    def viewer_policy(_obs):
      i = step_counter["i"]
      action = torch.zeros(1, act.action_dim, device=device)
      row = do_step(i, action)
      rows.append(row)
      if i % 20 == 0:
        print(
          f"[D3] step={i} phase={row['phase']} contact={row['in_contact']} "
          f"foot=({row['foot_x']:.3f},{row['foot_y']:.3f},{row['foot_z']:.3f}) "
          f"clearance={row['foot_height_above_wire']:.3f} "
          f"fwd_dist={row['foot_forward_dist_to_wire']:.3f}"
        )
      step_counter["i"] += 1
      return action

    print("[INFO] Debug mode: native viewer. Close window to end run.")
    NativeMujocoViewer(env, viewer_policy).run()
  else:
    for i in range(cfg.max_steps):
      action = torch.zeros(1, act.action_dim, device=device)
      row = do_step(i, action)
      rows.append(row)
      if row["phase"] == "POST_RECOVERY" and reached_post_recovery_at is None:
        reached_post_recovery_at = i
        print(f"[D3] Reached POST_RECOVERY at step {i} (reason={row['recovery_exit_reason']})")
      if (
        reached_post_recovery_at is not None
        and i - reached_post_recovery_at >= cfg.stop_steps_after_post_recovery
      ):
        break

  env.close()

  if not rows:
    raise RuntimeError("No steps logged.")

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[D3] CSV written: {csv_path} ({len(rows)} rows)")

  # --- Summary / success-criteria report. ---
  max_foot_h = max(r["foot_z"] for r in rows)
  recovery_rows = [r for r in rows if r["phase"] == "RECOVERY"]
  max_clearance_during_recovery = (
    max(r["foot_height_above_wire"] for r in recovery_rows) if recovery_rows else float("nan")
  )
  crossed = any(r["crossing_achieved"] for r in rows)
  final_phase = rows[-1]["phase"]
  exit_reason = next((r["recovery_exit_reason"] for r in reversed(rows) if r["recovery_exit_reason"]), "")
  resumed_walking = final_phase == "POST_RECOVERY"

  print("\n==== D3 WIRE EXPERIMENT SUMMARY ====")
  print(f"A. FL contacted wire: {contacted_wire}")
  print(f"B. Hybrid recovery activated: {recovery_activated}")
  print(f"C. Max FL foot height (world Z): {max_foot_h:.4f}")
  print(f"   Max clearance above wire DURING recovery: {max_clearance_during_recovery:.4f}")
  print(f"D. Crossing achieved (foot passed wire horiz+vert per env's own def): {crossed}")
  print(f"E. Wire cleared: {crossed}")
  print(f"F. Final phase reached: {final_phase}")
  print(f"G. Resumed walking (reached POST_RECOVERY): {resumed_walking}")
  print(f"   Exit reason: {exit_reason}")

  if cfg.plot:
    _make_plots(rows, out_dir)

  return csv_path


def _make_plots(rows: list[dict], out_dir: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  t = [r["sim_time"] for r in rows]
  phase = [r["phase"] for r in rows]
  recovery_mask = [p == "RECOVERY" for p in phase]

  def _recovery_span():
    idxs = [i for i, m in enumerate(recovery_mask) if m]
    if not idxs:
      return None
    return t[idxs[0]], t[idxs[-1]]

  span = _recovery_span()

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
  ax1.plot(t, [r["foot_x"] for r in rows], label="foot_x")
  ax2.plot(t, [r["foot_z"] for r in rows], label="foot_z", color="tab:orange")
  for ax in (ax1, ax2):
    if span:
      ax.axvspan(span[0], span[1], color="red", alpha=0.15, label="RECOVERY")
    ax.legend()
    ax.grid(True, alpha=0.3)
  ax1.set_ylabel("foot X (m)")
  ax2.set_ylabel("foot Z (m)")
  ax2.set_xlabel("sim time (s)")
  fig.suptitle("D3 Plot A: FL foot trajectory (hybrid controller)")
  fig.tight_layout()
  fig.savefig(out_dir / "plotA_foot_trajectory.png", dpi=120)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(t, [r["thigh_pos"] for r in rows], label="thigh_pos")
  ax.plot(t, [r["calf_pos"] for r in rows], label="calf_pos")
  ax.plot(t, [r["thigh_vel"] for r in rows], label="thigh_vel", linestyle=":")
  ax.plot(t, [r["calf_vel"] for r in rows], label="calf_vel", linestyle=":")
  if span:
    ax.axvspan(span[0], span[1], color="red", alpha=0.15, label="RECOVERY")
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("sim time (s)")
  ax.set_title("D3 Plot B: affected-leg joint motion (hybrid controller)")
  fig.tight_layout()
  fig.savefig(out_dir / "plotB_joint_motion.png", dpi=120)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(t, [r["foot_height_above_wire"] for r in rows], label="foot_height_above_wire")
  ax.plot(t, [r["foot_forward_dist_to_wire"] for r in rows], label="foot_forward_dist_to_wire", linestyle="--")
  ax.axhline(0.0, color="black", linewidth=0.8)
  if span:
    ax.axvspan(span[0], span[1], color="red", alpha=0.15, label="RECOVERY")
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("sim time (s)")
  ax.set_ylabel("meters")
  ax.set_title("D3 Plot C: geometric clearance / crossing (hybrid controller)")
  fig.tight_layout()
  fig.savefig(out_dir / "plotC_clearance.png", dpi=120)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(t, [r["recovery_thigh_vel_cmd"] for r in rows], label="recovery_thigh_vel_cmd")
  ax.plot(t, [r["recovery_calf_torque_cmd"] for r in rows], label="recovery_calf_torque_cmd")
  if span:
    ax.axvspan(span[0], span[1], color="red", alpha=0.15, label="RECOVERY")
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("sim time (s)")
  ax.set_title("D3 Plot D: actuator-level recovery commands")
  fig.tight_layout()
  fig.savefig(out_dir / "plotD_actuator_commands.png", dpi=120)
  plt.close(fig)

  print(f"[D3] Plots written to {out_dir}")


def main() -> None:
  args = tyro.cli(D3Config, args=sys.argv[1:])
  run(args)


if __name__ == "__main__":
  main()
