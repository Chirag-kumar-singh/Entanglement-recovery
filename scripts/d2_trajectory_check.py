"""D2: Deterministic Go2 wire-recovery trajectory validation.

INSTRUMENTATION / VISUALIZATION ONLY. Does not change the D1 controller:
``thigh_recovery_delta``/``calf_recovery_delta``/timing/exit logic are
read from the existing ``go2_wire_recovery_residual_env_cfg`` defaults
(deterministic mode) and are NOT modified here.

Purpose: run one deterministic recovery episode, record a per-step CSV
of joint/foot/wire geometry state (reusing the environment's OWN
existing clearance/crossing computations from
``ResidualRecoveryAction``/``wire_geometry`` -- no second, inconsistent
definition of "clearance" is invented here), optionally generate
trajectory plots, and optionally show the episode live in MuJoCo's
native viewer.

Since ``recovery_mode="deterministic"`` makes the action-term's own
input tensor IGNORED (see ``ResidualRecoveryAction.process_actions``),
the "action" fed to ``env.step()`` every step is just a zero 3-vector --
there is no PPO/learned policy involved anywhere in this script.

Examples
--------
Headless, CSV + plots only::

    uv run python scripts/d2_trajectory_check.py

Visual, native MuJoCo viewer (also logs CSV in the background)::

    uv run python scripts/d2_trajectory_check.py --debug

Custom affected leg / output dir / seed::

    uv run python scripts/d2_trajectory_check.py \\
        --affected-leg-idx 1 --seed 123 --out-dir logs/d2_recovery/fr_run1
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab.tasks  # noqa: F401  (populates the task registry)
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
class D2Config:
  affected_leg_idx: int = 0
  """0=FL, 1=FR, 2=RL, 3=RR. Default matches the D1 smoke test (FL)."""
  seed: int = 0
  """Fixed seed for reproducibility (passed straight to env reset)."""
  device: str | None = None
  max_steps: int = 400
  """Hard cap on total logged steps (safety bound in case the episode
  never reaches POST_RECOVERY, e.g. the robot falls)."""
  stop_steps_after_post_recovery: int = 60
  """Once POST_RECOVERY is reached, keep logging this many extra steps
  (to see the aftermath) then stop -- avoids running the full, long
  episode length unnecessarily."""
  out_dir: str = ""
  """Output directory for CSV/plots. Default: logs/d2_recovery_trajectory/
  <affected_leg>_seed<seed>."""
  debug: bool = False
  """Launch the native MuJoCo viewer for live visual inspection. Runs
  until the viewer window is closed; CSV/plots are still produced from
  whatever was logged up to that point."""
  plot: bool = True
  """Generate the 3 required trajectory plots (A/B/C) from the CSV."""


def _affected_joint_names(leg: str) -> tuple[str, str, str]:
  return (f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint")


def build_row(
  step: int,
  env: ManagerBasedRlEnv,
  action_term,
  entry_foot_pos: torch.Tensor | None,
  running_max_height: float,
  running_min_height: float,
) -> tuple[dict, float, float]:
  """Collect one step's worth of instrumentation, reading EXCLUSIVELY
  from the action term's own already-computed geometry/FSM state (see
  ``ResidualRecoveryAction._update_fsm``) -- no second clearance/crossing
  definition is introduced here."""
  leg = LEG_NAMES[action_term.cfg.affected_leg_idx]
  hip_n, thigh_n, calf_n = _affected_joint_names(leg)

  entity = action_term._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index(hip_n)
  thigh_i = joint_names.index(thigh_n)
  calf_i = joint_names.index(calf_n)

  qpos = entity.data.joint_pos[0]
  qvel = entity.data.joint_vel[0]

  foot_pos = entity.data.site_pos_w[0, action_term._foot_site_idx, :].clone()

  phase = int(action_term.phase[0])

  a_walk = action_term.last_a_walk_raw
  a_final = action_term.last_final_raw
  affected_idx = action_term._affected_idx.tolist()

  unaffected_mask = torch.ones(12, dtype=torch.bool)
  for i in affected_idx:
    unaffected_mask[i] = False
  if a_walk is not None and a_final is not None:
    other9_unchanged = bool(
      torch.allclose(
        a_walk[0, unaffected_mask], a_final[0, unaffected_mask], atol=1e-6
      )
    )
    walk_thigh = float(a_walk[0, affected_idx[1]])
    walk_calf = float(a_walk[0, affected_idx[2]])
    final_thigh = float(a_final[0, affected_idx[1]])
    final_calf = float(a_final[0, affected_idx[2]])
  else:
    other9_unchanged = True
    walk_thigh = walk_calf = final_thigh = final_calf = 0.0

  if entry_foot_pos is not None:
    disp = (foot_pos - entry_foot_pos).tolist()
  else:
    disp = [0.0, 0.0, 0.0]

  running_max_height = max(running_max_height, float(foot_pos[2]))
  running_min_height = min(running_min_height, float(foot_pos[2]))

  exit_reason = ""
  if phase == PHASE_POST_RECOVERY:
    exit_reason = (
      "cap_timeout" if bool(action_term.recovery_failed_timeout[0]) else "crossing+release"
    )

  row = {
    "step": step,
    "sim_time": step * env.step_dt,
    "phase": PHASE_NAMES[phase],
    "affected_leg": leg,
    "recovery_elapsed_steps": int(action_term.recovery_elapsed[0]),
    "recovery_exit_reason": exit_reason,
    "hip_pos": float(qpos[hip_i]),
    "thigh_pos": float(qpos[thigh_i]),
    "calf_pos": float(qpos[calf_i]),
    "hip_vel": float(qvel[hip_i]),
    "thigh_vel": float(qvel[thigh_i]),
    "calf_vel": float(qvel[calf_i]),
    "walk_raw_thigh": walk_thigh,
    "walk_raw_calf": walk_calf,
    "residual_thigh": final_thigh - walk_thigh,
    "residual_calf": final_calf - walk_calf,
    "final_raw_thigh": final_thigh,
    "final_raw_calf": final_calf,
    "other9_unchanged": other9_unchanged,
    "foot_x": float(foot_pos[0]),
    "foot_y": float(foot_pos[1]),
    "foot_z": float(foot_pos[2]),
    "foot_disp_x": disp[0],
    "foot_disp_y": disp[1],
    "foot_disp_z": disp[2],
    "foot_forward_dist_to_wire": float(action_term.foot_forward_dist[0]),
    "foot_height_above_wire": float(action_term.foot_height_above_wire[0]),
    "foot_side_far": bool(action_term.foot_side_far[0]),
    "calf_forward_dist_to_wire": float(action_term.calf_forward_dist[0]),
    "calf_height_above_wire": float(action_term.calf_height_above_wire[0]),
    "calf_side_far": bool(action_term.calf_side_far[0]),
    "calf_clearance_achieved": bool(action_term.calf_clearance_achieved[0]),
    "foot_crossed_achieved": bool(action_term.foot_crossed_achieved[0]),
    "crossing_achieved": bool(action_term.crossing_achieved[0]),
    "running_max_foot_height": running_max_height,
    "running_min_foot_height": running_min_height,
  }
  return row, running_max_height, running_min_height


def run(cfg: D2Config) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  out_dir = Path(
    cfg.out_dir
    or f"logs/d2_recovery_trajectory/{LEG_NAMES[cfg.affected_leg_idx]}_seed{cfg.seed}"
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="deterministic",
    policy_device=device,
    play=True,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  action_term = env.action_manager._terms["joint_pos"]

  print(f"[D2] recovery_mode = {action_term.cfg.recovery_mode}")
  print(
    f"[D2] thigh_recovery_delta={action_term.cfg.thigh_recovery_delta}  "
    f"calf_recovery_delta={action_term.cfg.calf_recovery_delta}  "
    f"max_recovery_steps={action_term.cfg.max_recovery_steps}  "
    f"affected_leg={LEG_NAMES[cfg.affected_leg_idx]}"
  )
  print(
    f"[D2] wire geom names: {action_term.cfg.wire_geom_names} "
    f"(static anchors: see DEFAULT_WIRE_CFG in env_cfg_residual.py)"
  )

  rows: list[dict] = []
  entry_foot_pos: torch.Tensor | None = None
  running_max_h = -1e9
  running_min_h = 1e9

  env.reset(seed=cfg.seed)

  def do_step(step_idx: int, action: torch.Tensor) -> dict:
    nonlocal entry_foot_pos, running_max_h, running_min_h
    env.step(action)
    if bool(action_term.newly_entangled[0]):
      entry_foot_pos = action_term._entity.data.site_pos_w[
        0, action_term._foot_site_idx, :
      ].clone()
      running_max_h = -1e9
      running_min_h = 1e9
    row, running_max_h, running_min_h = build_row(
      step_idx, env, action_term, entry_foot_pos, running_max_h, running_min_h
    )
    return row

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    step_counter = {"i": 0}

    def viewer_policy(_obs):
      i = step_counter["i"]
      action = torch.zeros(1, action_term.action_dim, device=device)
      row = do_step(i, action)
      rows.append(row)
      if i % 20 == 0:
        print(
          f"[D2] step={i} phase={row['phase']} "
          f"foot=({row['foot_x']:.3f},{row['foot_y']:.3f},{row['foot_z']:.3f}) "
          f"clearance={row['foot_height_above_wire']:.3f} "
          f"fwd_dist={row['foot_forward_dist_to_wire']:.3f}"
        )
      step_counter["i"] += 1
      return action

    print(
      "[INFO] Debug mode: launching native viewer with 1 environment. "
      "Close the viewer window to end the run."
    )
    NativeMujocoViewer(env, viewer_policy).run()
  else:
    reached_post_recovery_at: int | None = None
    for i in range(cfg.max_steps):
      action = torch.zeros(1, action_term.action_dim, device=device)
      row = do_step(i, action)
      rows.append(row)
      if row["phase"] == "POST_RECOVERY" and reached_post_recovery_at is None:
        reached_post_recovery_at = i
        print(
          f"[D2] Reached POST_RECOVERY at step {i} "
          f"(reason={row['recovery_exit_reason']})"
        )
      if (
        reached_post_recovery_at is not None
        and i - reached_post_recovery_at >= cfg.stop_steps_after_post_recovery
      ):
        break

  env.close()

  if not rows:
    raise RuntimeError("No steps logged (viewer closed before any step ran?).")

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[D2] CSV written: {csv_path} ({len(rows)} rows)")

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

  # Plot A: foot trajectory.
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
  fig.suptitle("Plot A: affected foot trajectory")
  fig.tight_layout()
  fig.savefig(out_dir / "plotA_foot_trajectory.png", dpi=120)
  plt.close(fig)

  # Plot B: joint motion.
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(t, [r["thigh_pos"] for r in rows], label="thigh_pos")
  ax.plot(t, [r["calf_pos"] for r in rows], label="calf_pos")
  if span:
    ax.axvspan(span[0], span[1], color="red", alpha=0.15, label="RECOVERY")
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("sim time (s)")
  ax.set_ylabel("joint angle (rad)")
  ax.set_title("Plot B: affected-leg joint motion")
  fig.tight_layout()
  fig.savefig(out_dir / "plotB_joint_motion.png", dpi=120)
  plt.close(fig)

  # Plot C: clearance.
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(t, [r["foot_height_above_wire"] for r in rows], label="foot_height_above_wire")
  ax.plot(t, [r["calf_height_above_wire"] for r in rows], label="calf_height_above_wire")
  ax.plot(
    t,
    [r["foot_forward_dist_to_wire"] for r in rows],
    label="foot_forward_dist_to_wire",
    linestyle="--",
  )
  ax.axhline(0.0, color="black", linewidth=0.8)
  if span:
    ax.axvspan(span[0], span[1], color="red", alpha=0.15, label="RECOVERY")
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("sim time (s)")
  ax.set_ylabel("meters")
  ax.set_title("Plot C: geometric clearance / crossing (existing env definitions)")
  fig.tight_layout()
  fig.savefig(out_dir / "plotC_clearance.png", dpi=120)
  plt.close(fig)

  print(f"[D2] Plots written to {out_dir}")


def main() -> None:
  args = tyro.cli(D2Config, args=sys.argv[1:])
  run(args)


if __name__ == "__main__":
  main()
