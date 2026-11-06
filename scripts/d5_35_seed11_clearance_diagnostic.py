#!/usr/bin/env python
"""D5.35 seed-11 FL clearance failure diagnostic.

This is a read-only, observational follow-up to D5.31. It reconstructs
seed 11's failed FL clearance attempt under both the baseline and the
phase-aware threshold settings, and uses seed 0 baseline as the known
crossing reference.

It does NOT change the controller, FSM, reward, termination, wire
geometry, checkpoint, or recovery limits. It only logs the live
geometry/FSM state needed to explain why stage 1 never latched in seed
11.

Outputs are written to a fresh directory under logs/d5_35/.

Usage:

    uv run python scripts/d5_35_seed11_clearance_diagnostic.py

Optional:

    uv run python scripts/d5_35_seed11_clearance_diagnostic.py \
      --device cpu --max-steps 300
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import mjlab.tasks  # noqa: F401  (populate task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.tasks.recovery.mdp.actions import (
  LEG_NAMES,
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


@dataclass(frozen=True)
class RunSpec:
  label: str
  seed: int
  variant: str


@dataclass(frozen=True)
class D535Config:
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = "logs/d5_35"
  runs: tuple[RunSpec, ...] = (
    RunSpec(label="seed0_baseline_reference", seed=0, variant="baseline"),
    RunSpec(label="seed11_baseline", seed=11, variant="baseline"),
    RunSpec(label="seed11_phase_aware", seed=11, variant="phase_aware"),
  )


def _quat_to_roll_pitch_yaw(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
  """Convert a MuJoCo quaternion (w, x, y, z) to roll/pitch/yaw.

  The project documents quaternions in (w, x, y, z) order.
  """
  w, x, y, z = [float(v) for v in quat_wxyz]

  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)

  sinp = 2.0 * (w * y - z * x)
  sinp = max(-1.0, min(1.0, sinp))
  pitch = math.asin(sinp)

  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)
  return roll, pitch, yaw


def _classify_phase_tier(
  entry_speed: float, phase_aware: bool, cfg
) -> tuple[str, float, float]:
  if not phase_aware:
    return "static_baseline", cfg.clearance_margin, cfg.foot_crossing_margin
  if entry_speed < cfg.phase_aware_speed_threshold_low:
    return (
      "slow",
      cfg.phase_aware_slow_clearance_margin,
      cfg.phase_aware_slow_foot_crossing_margin,
    )
  if entry_speed < cfg.phase_aware_speed_threshold_high:
    return (
      "medium",
      cfg.phase_aware_medium_clearance_margin,
      cfg.phase_aware_medium_foot_crossing_margin,
    )
  return (
    "fast",
    cfg.phase_aware_fast_clearance_margin,
    cfg.phase_aware_fast_foot_crossing_margin,
  )


def _build_env(run: RunSpec, device: str) -> ManagerBasedRlEnv:
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=(run.variant == "phase_aware"),
    max_recovery_steps=48,
  )
  env_cfg.seed = run.seed
  env_cfg.sim.device = device
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _first_step(rows: list[dict], predicate) -> int:
  for row in rows:
    if predicate(row):
      return int(row["step"])
  return -1


def run_one(run: RunSpec, device: str, max_steps: int, out_dir: Path) -> dict:
  env = _build_env(run, device)
  act = env.action_manager.get_term("joint_pos")
  entity = act._entity
  joint_names = list(entity.joint_names)
  site_names = list(entity.site_names)
  leg = LEG_NAMES[0]
  hip_i = joint_names.index(f"{leg}_hip_joint")
  thigh_i = joint_names.index(f"{leg}_thigh_joint")
  calf_i = joint_names.index(f"{leg}_calf_joint")
  foot_site_i = site_names.index(leg)

  env.reset(seed=run.seed)

  rows: list[dict] = []
  entry_step = -1
  exit_step = -1
  exit_reason = "horizon_exhausted"
  env_terminated = False
  env_truncated = False
  entry_tier = "static_baseline"
  entry_clearance_margin = float(act.cfg.clearance_margin)
  entry_crossing_margin = float(act.cfg.foot_crossing_margin)
  entry_leg_speed = 0.0

  entry_state = None

  for step in range(max_steps):
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    phase = int(act._phase[0].item())
    recovery_elapsed = int(act.recovery_elapsed[0].item())
    wire_contact = bool(
      (env.scene.sensors[act.cfg.wire_contact_sensor_name].data.found[0] > 0).any()
    )

    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]
    base_lin_vel_w = entity.data.root_link_lin_vel_w[0]
    base_lin_vel_b = entity.data.root_link_lin_vel_b[0]
    base_ang_vel_w = entity.data.root_link_ang_vel_w[0]
    roll, pitch, yaw = _quat_to_roll_pitch_yaw(base_quat)

    calf_side_far = bool(act.calf_side_far[0].item())
    foot_side_far = bool(act.foot_side_far[0].item())
    calf_lifted = bool(
      (act.calf_height_above_wire[0].item() > act._effective_clearance_margin[0].item())
    )
    calf_near_to_far_gate = (
      not bool(act._prev_calf_side_far[0].item())
    ) and calf_side_far
    calf_clearance_event = (
      bool(act.calf_clearance_event[0].item())
      if hasattr(act, "calf_clearance_event")
      else False
    )
    foot_crossed_event = (
      bool(act.foot_crossed_event[0].item())
      if hasattr(act, "foot_crossed_event")
      else False
    )

    if phase == PHASE_RECOVERY and entry_step == -1:
      entry_step = step
      entry_state = {
        "seed": run.seed,
        "variant": run.variant,
        "step": step,
        "sim_time_s": step * env.step_dt,
        "entry_leg_speed_rad_s": float(act._entry_leg_speed[0].item()),
        "tier": _classify_phase_tier(
          float(act._entry_leg_speed[0].item()), run.variant == "phase_aware", act.cfg
        )[0],
        "clearance_margin_m": float(act._effective_clearance_margin[0].item()),
        "foot_crossing_margin_m": float(act._effective_foot_crossing_margin[0].item()),
        "base_x_m": float(base_pos[0].item()),
        "base_y_m": float(base_pos[1].item()),
        "base_z_m": float(base_pos[2].item()),
        "base_roll_rad": roll,
        "base_pitch_rad": pitch,
        "base_yaw_rad": yaw,
        "base_lin_vel_world_x": float(base_lin_vel_w[0].item()),
        "base_lin_vel_world_y": float(base_lin_vel_w[1].item()),
        "base_lin_vel_world_z": float(base_lin_vel_w[2].item()),
        "base_lin_vel_body_x": float(base_lin_vel_b[0].item()),
        "base_lin_vel_body_y": float(base_lin_vel_b[1].item()),
        "base_lin_vel_body_z": float(base_lin_vel_b[2].item()),
        "base_ang_vel_world_x": float(base_ang_vel_w[0].item()),
        "base_ang_vel_world_y": float(base_ang_vel_w[1].item()),
        "base_ang_vel_world_z": float(base_ang_vel_w[2].item()),
        "wire_contact": wire_contact,
        "FL_foot_x_m": float(entity.data.site_pos_w[0, foot_site_i, 0].item()),
        "FL_foot_y_m": float(entity.data.site_pos_w[0, foot_site_i, 1].item()),
        "FL_foot_z_m": float(entity.data.site_pos_w[0, foot_site_i, 2].item()),
        "FL_foot_vx_m_s": float(entity.data.site_lin_vel_w[0, foot_site_i, 0].item()),
        "FL_foot_vy_m_s": float(entity.data.site_lin_vel_w[0, foot_site_i, 1].item()),
        "FL_foot_vz_m_s": float(entity.data.site_lin_vel_w[0, foot_site_i, 2].item()),
        "FL_hip_pos_rad": float(entity.data.joint_pos[0, hip_i].item()),
        "FL_thigh_pos_rad": float(entity.data.joint_pos[0, thigh_i].item()),
        "FL_calf_pos_rad": float(entity.data.joint_pos[0, calf_i].item()),
        "FL_hip_vel_rad_s": float(entity.data.joint_vel[0, hip_i].item()),
        "FL_thigh_vel_rad_s": float(entity.data.joint_vel[0, thigh_i].item()),
        "FL_calf_vel_rad_s": float(entity.data.joint_vel[0, calf_i].item()),
      }
      tier_name, clearance_margin, crossing_margin = _classify_phase_tier(
        entry_state["entry_leg_speed_rad_s"], run.variant == "phase_aware", act.cfg
      )
      entry_state["tier"] = tier_name
      entry_state["clearance_margin_m"] = clearance_margin
      entry_state["foot_crossing_margin_m"] = crossing_margin
      entry_tier = tier_name
      entry_clearance_margin = clearance_margin
      entry_crossing_margin = crossing_margin
      entry_leg_speed = entry_state["entry_leg_speed_rad_s"]

    row = {
      "step": step,
      "sim_time_s": step * env.step_dt,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed_steps": recovery_elapsed,
      "entry_step": entry_step,
      "entry_leg_speed_rad_s": float(act._entry_leg_speed[0].item()),
      "entry_tier": entry_tier,
      "clearance_margin_m": float(act._effective_clearance_margin[0].item()),
      "foot_crossing_margin_m": float(act._effective_foot_crossing_margin[0].item()),
      "calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
      "calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
      "calf_side_far": calf_side_far,
      "calf_lifted": calf_lifted,
      "calf_near_to_far_gate": calf_near_to_far_gate,
      "calf_clearance_achieved": bool(act._calf_clearance_achieved[0].item()),
      "calf_clearance_event": calf_clearance_event,
      "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
      "foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
      "foot_side_far": foot_side_far,
      "foot_crossed_achieved": bool(act._foot_crossed_achieved[0].item()),
      "foot_crossed_event": foot_crossed_event,
      "crossing_achieved": bool(act._crossing_achieved[0].item()),
      "release_counter": int(act._release_counter[0].item()),
      "recovery_failed_timeout": bool(act._recovery_failed_timeout[0].item()),
      "wire_contact": wire_contact,
      "base_x_m": float(base_pos[0].item()),
      "base_y_m": float(base_pos[1].item()),
      "base_z_m": float(base_pos[2].item()),
      "base_roll_rad": roll,
      "base_pitch_rad": pitch,
      "base_yaw_rad": yaw,
      "base_lin_vel_world_x": float(base_lin_vel_w[0].item()),
      "base_lin_vel_world_y": float(base_lin_vel_w[1].item()),
      "base_lin_vel_world_z": float(base_lin_vel_w[2].item()),
      "base_lin_vel_body_x": float(base_lin_vel_b[0].item()),
      "base_lin_vel_body_y": float(base_lin_vel_b[1].item()),
      "base_lin_vel_body_z": float(base_lin_vel_b[2].item()),
      "base_ang_vel_world_x": float(base_ang_vel_w[0].item()),
      "base_ang_vel_world_y": float(base_ang_vel_w[1].item()),
      "base_ang_vel_world_z": float(base_ang_vel_w[2].item()),
      "FL_foot_x_m": float(entity.data.site_pos_w[0, foot_site_i, 0].item()),
      "FL_foot_y_m": float(entity.data.site_pos_w[0, foot_site_i, 1].item()),
      "FL_foot_z_m": float(entity.data.site_pos_w[0, foot_site_i, 2].item()),
      "FL_foot_vx_m_s": float(entity.data.site_lin_vel_w[0, foot_site_i, 0].item()),
      "FL_foot_vy_m_s": float(entity.data.site_lin_vel_w[0, foot_site_i, 1].item()),
      "FL_foot_vz_m_s": float(entity.data.site_lin_vel_w[0, foot_site_i, 2].item()),
      "FL_hip_pos_rad": float(entity.data.joint_pos[0, hip_i].item()),
      "FL_thigh_pos_rad": float(entity.data.joint_pos[0, thigh_i].item()),
      "FL_calf_pos_rad": float(entity.data.joint_pos[0, calf_i].item()),
      "FL_hip_vel_rad_s": float(entity.data.joint_vel[0, hip_i].item()),
      "FL_thigh_vel_rad_s": float(entity.data.joint_vel[0, thigh_i].item()),
      "FL_calf_vel_rad_s": float(entity.data.joint_vel[0, calf_i].item()),
    }
    rows.append(row)

    if phase == PHASE_POST_RECOVERY and exit_step == -1:
      exit_step = step
      exit_reason = (
        "cap_timeout"
        if bool(act.recovery_failed_timeout[0].item())
        else "crossing_release"
      )

    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      env_terminated = bool(terminateds[0].item())
      env_truncated = bool(truncateds[0].item())
      exit_reason = "timeout" if env_truncated else "fell_over"
      break
  else:
    exit_reason = "horizon_exhausted"

  env.close()

  csv_path = out_dir / f"{run.label}.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  stage1_step = _first_step(rows, lambda r: r["calf_clearance_achieved"])
  stage2_step = _first_step(rows, lambda r: r["foot_crossed_achieved"])
  recovery_step = _first_step(rows, lambda r: r["phase"] == "RECOVERY")
  post_step = _first_step(rows, lambda r: r["phase"] == "POST_RECOVERY")
  entry_state = entry_state or {}
  recovery_attempt_exit_reason = "none"
  if post_step != -1:
    first_post = next(r for r in rows if r["step"] == post_step)
    recovery_attempt_exit_reason = (
      "cap_timeout" if first_post["recovery_failed_timeout"] else "crossing_release"
    )

  failure_reason = ""
  if stage1_step == -1:
    any_lift = any(r["calf_lifted"] for r in rows if r["phase"] == "RECOVERY")
    any_far = any(r["calf_side_far"] for r in rows if r["phase"] == "RECOVERY")
    if not any_far:
      failure_reason = "missing near-to-far transition"
    elif not any_lift:
      failure_reason = "insufficient vertical clearance"
    else:
      failure_reason = "other unmet FSM condition"

  def stats_for_key(
    key: str, lo_step: int | None = None, hi_step: int | None = None
  ) -> tuple[float, float]:
    selected = [
      r[key]
      for r in rows
      if (lo_step is None or r["step"] >= lo_step)
      and (hi_step is None or r["step"] < hi_step)
    ]
    if not selected:
      return float("nan"), float("nan")
    return sum(selected) / len(selected), min(selected)

  summary_lines = []
  summary_lines.append("=" * 78)
  summary_lines.append(f"D5.35 SEED 11 FL CLEARANCE DIAGNOSTIC  run={run.label}")
  summary_lines.append("=" * 78)
  summary_lines.append("")
  summary_lines.append("--- ENTRY STATE AT NORMAL_WALK -> RECOVERY ---")
  summary_lines.append(
    f"  entry step: {entry_step}  entry time: {entry_state.get('sim_time_s', float('nan')):.3f} s"
  )
  summary_lines.append(
    f"  entry leg speed: {entry_leg_speed:.6f} rad/s  tier: {entry_tier}"
  )
  summary_lines.append(
    f"  active margins: clearance={entry_clearance_margin:.6f} m  crossing={entry_crossing_margin:.6f} m"
  )
  summary_lines.append(
    f"  base pose: x={entry_state.get('base_x_m', float('nan')):.4f} y={entry_state.get('base_y_m', float('nan')):.4f} z={entry_state.get('base_z_m', float('nan')):.4f}"
  )
  summary_lines.append(
    f"  base attitude: roll={entry_state.get('base_roll_rad', float('nan')):.4f} rad  pitch={entry_state.get('base_pitch_rad', float('nan')):.4f} rad  yaw={entry_state.get('base_yaw_rad', float('nan')):.4f} rad"
  )
  summary_lines.append(
    f"  base vertical velocity: {entry_state.get('base_lin_vel_world_z', float('nan')):.4f} m/s"
  )
  summary_lines.append(
    f"  wire contact at entry: {entry_state.get('wire_contact', False)}"
  )
  summary_lines.append(
    f"  FL foot pose at entry: x={entry_state.get('FL_foot_x_m', float('nan')):.4f} y={entry_state.get('FL_foot_y_m', float('nan')):.4f} z={entry_state.get('FL_foot_z_m', float('nan')):.4f}"
  )
  summary_lines.append(
    f"  FL joint state at entry: hip={entry_state.get('FL_hip_pos_rad', float('nan')):.4f}/{entry_state.get('FL_hip_vel_rad_s', float('nan')):.4f}  thigh={entry_state.get('FL_thigh_pos_rad', float('nan')):.4f}/{entry_state.get('FL_thigh_vel_rad_s', float('nan')):.4f}  calf={entry_state.get('FL_calf_pos_rad', float('nan')):.4f}/{entry_state.get('FL_calf_vel_rad_s', float('nan')):.4f}"
  )
  summary_lines.append("")
  summary_lines.append("--- RECOVERY TRAJECTORY SUMMARY ---")
  summary_lines.append(f"  recovery entry step: {recovery_step}")
  summary_lines.append(f"  stage1 latch step: {stage1_step}")
  summary_lines.append(f"  stage2 latch step: {stage2_step}")
  summary_lines.append(
    f"  recovery exit step: {post_step}  reason: {recovery_attempt_exit_reason}"
  )
  summary_lines.append(
    f"  env_terminated={env_terminated}  env_truncated={env_truncated}"
  )
  summary_lines.append(f"  overall observation-window outcome: {exit_reason}")
  summary_lines.append(
    "  first divergence from seed0 reference should be interpreted from this run's trace, not guessed"
  )
  summary_lines.append("")
  summary_lines.append("--- STAGE-1 FAILURE DIAGNOSIS ---")
  summary_lines.append(f"  stage1 never latched: {stage1_step == -1}")
  summary_lines.append(
    f"  recovered-step any calf_lifted=true: {any(r['calf_lifted'] for r in rows if r['phase'] == 'RECOVERY')}"
  )
  summary_lines.append(
    f"  recovered-step any calf_side_far=true: {any(r['calf_side_far'] for r in rows if r['phase'] == 'RECOVERY')}"
  )
  summary_lines.append(f"  diagnosed reason: {failure_reason or 'stage1 latched'}")
  summary_lines.append("")
  summary_lines.append("--- PERIOD STATS (FL wire-relative signals) ---")
  if recovery_step != -1:
    rec_rows = [r for r in rows if r["phase"] == "RECOVERY"]
    if rec_rows:
      min_calf_h = min(r["calf_height_above_wire_m"] for r in rec_rows)
      max_calf_h = max(r["calf_height_above_wire_m"] for r in rec_rows)
      min_foot_h = min(r["foot_height_above_wire_m"] for r in rec_rows)
      max_foot_h = max(r["foot_height_above_wire_m"] for r in rec_rows)
      summary_lines.append(
        f"  recovery calf height above wire range: [{min_calf_h:.4f}, {max_calf_h:.4f}] m"
      )
      summary_lines.append(
        f"  recovery foot height above wire range: [{min_foot_h:.4f}, {max_foot_h:.4f}] m"
      )
      summary_lines.append(
        f"  recovery calf_forward_dist range: [{min(r['calf_forward_dist_m'] for r in rec_rows):.4f}, {max(r['calf_forward_dist_m'] for r in rec_rows):.4f}] m"
      )
      summary_lines.append(
        f"  recovery foot_forward_dist range: [{min(r['foot_forward_dist_m'] for r in rec_rows):.4f}, {max(r['foot_forward_dist_m'] for r in rec_rows):.4f}] m"
      )
  summary_lines.append("")
  summary_lines.append("--- COMPARISON METRICS ---")
  summary_lines.append(
    f"  mean FL foot height above wire during recovery: {stats_for_key('foot_height_above_wire_m', recovery_step, post_step if post_step != -1 else None)[0]:.4f} m"
  )
  summary_lines.append(
    f"  mean FL calf height above wire during recovery: {stats_for_key('calf_height_above_wire_m', recovery_step, post_step if post_step != -1 else None)[0]:.4f} m"
  )
  summary_lines.append(
    f"  seed0 reference stage1/stage2: {bool(_first_step([r for r in rows if run.label == 'seed0_baseline_reference'], lambda r: False) == -1)} (see CSV row in this run file)"
  )
  summary_lines.append("")
  summary_lines.append(
    "Interpretation guidance: if seed11's calf never achieves both (1) a"
    " near->far transition and (2) height above the active clearance"
    " margin, stage1 cannot latch in the actual FSM. If the calf does"
    " lift but never crosses from near to far, the failure is geometric"
    "/trajectory-related rather than threshold-only. If both conditions"
    " occur yet stage1 still does not latch, that would indicate an FSM"
    " inconsistency (unexpected here)."
  )
  summary_lines.append("")
  summary_lines.append(f"CSV: {csv_path}")

  summary_text = "\n".join(summary_lines)
  print(summary_text)
  (out_dir / f"{run.label}_summary.txt").write_text(summary_text + "\n")
  return {
    "run": run.label,
    "seed": run.seed,
    "variant": run.variant,
    "csv": str(csv_path),
    "summary": str(out_dir / f"{run.label}_summary.txt"),
    "entry_step": entry_step,
    "entry_leg_speed": entry_leg_speed,
    "entry_tier": entry_tier,
    "clearance_margin": entry_clearance_margin,
    "crossing_margin": entry_crossing_margin,
    "stage1_step": stage1_step,
    "stage2_step": stage2_step,
    "exit_step": exit_step,
    "exit_reason": exit_reason,
    "failure_reason": failure_reason,
  }


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="D5.35 seed-11 FL clearance diagnostic")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--max-steps", type=int, default=300)
  parser.add_argument("--out-dir", default="logs/d5_35")
  args = parser.parse_args()

  cfg = D535Config(device=args.device, max_steps=args.max_steps, out_dir=args.out_dir)
  configure_torch_backends()

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  run_dir = out_dir / "seed11_clearance_diagnostic"
  run_dir.mkdir(parents=True, exist_ok=True)

  results = []
  for run in cfg.runs:
    print("\n" + "=" * 80)
    print(f"Running {run.label} (seed={run.seed}, variant={run.variant})")
    print("=" * 80)
    results.append(run_one(run, cfg.device, cfg.max_steps, run_dir))

  summary_path = run_dir / "summary.csv"
  with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

  print("\n" + "=" * 80)
  print(f"Wrote summary: {summary_path}")
  print("=" * 80)


if __name__ == "__main__":
  main()
