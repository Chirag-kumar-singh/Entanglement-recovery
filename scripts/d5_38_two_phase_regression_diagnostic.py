#!/usr/bin/env python
"""D5.38 diagnose why the D5.37 two-phase command regresses seed 0.

This is a read-only diagnostic pass. It does not modify controller
behavior, checkpoints, thresholds, rewards, or terminations.

It replays four bounded CPU runs:
  - seed 0 baseline
  - seed 0 two-phase
  - seed 11 baseline
  - seed 11 two-phase

and logs the command/state/geometry path through `ResidualRecoveryAction`
with enough detail to answer:
  - exactly when the two-phase switch fires;
  - what actuator commands/forces and joint motions occur immediately
    before/after that switch;
  - whether the switch produces the intended joint motion; and
  - whether that motion advances or retracts the calf horizontally.

Outputs go under ``logs/d5_38/two_phase_regression_diagnostic/``.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).parent))

import mjlab.tasks  # noqa: F401
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
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
  ResidualRecoveryAction,
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
  experimental_two_phase: bool


@dataclass(frozen=True)
class D538Config:
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = "logs/d5_38/two_phase_regression_diagnostic"
  switch_window: int = 8
  runs: tuple[RunSpec, ...] = (
    RunSpec(label="seed0_baseline", seed=0, experimental_two_phase=False),
    RunSpec(label="seed0_two_phase", seed=0, experimental_two_phase=True),
    RunSpec(label="seed11_baseline", seed=11, experimental_two_phase=False),
    RunSpec(label="seed11_two_phase", seed=11, experimental_two_phase=True),
  )


def _build_env(
  seed: int, device: str, experimental_two_phase: bool
) -> ManagerBasedRlEnv:
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=False,
    max_recovery_steps=48,
    experimental_two_phase_hybrid_recovery=experimental_two_phase,
  )
  env_cfg.seed = seed
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _variant_name(experimental_two_phase: bool) -> str:
  return "two_phase" if experimental_two_phase else "baseline"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
  if not rows:
    return float("nan")
  return sum(float(row[key]) for row in rows) / len(rows)


def _run_one(run: RunSpec, device: str, max_steps: int, out_dir: Path) -> Path:
  env = _build_env(run.seed, device, run.experimental_two_phase)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  entity = act._entity
  joint_names = list(entity.joint_names)
  fl = LEG_NAMES[0]
  hip_idx = joint_names.index(f"{fl}_hip_joint")
  thigh_idx = joint_names.index(f"{fl}_thigh_joint")
  calf_idx = joint_names.index(f"{fl}_calf_joint")

  hip_pos_ctrl_id = act._find_actuator_ctrl_id(
    f"{fl}_hip_joint", BuiltinPositionActuator
  )
  thigh_pos_ctrl_id = act._pos_thigh_ctrl_id
  calf_pos_ctrl_id = act._pos_calf_ctrl_id
  thigh_aux_ctrl_id = act._aux_thigh_ctrl_id
  calf_aux_ctrl_id = act._aux_calf_ctrl_id

  env.reset(seed=run.seed)

  rows: list[dict[str, Any]] = []
  recovery_offset = -1
  for step in range(max_steps):
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    phase = int(act.phase[0].item())
    if phase == PHASE_RECOVERY:
      recovery_offset += 1

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    pos_target = entity.data.joint_pos_target[0]
    sim_ctrl = env.sim.data.ctrl[0]
    sim_force = env.sim.data.actuator_force[0]
    lin_vel_w = entity.data.root_link_lin_vel_w[0]
    lin_vel_b = entity.data.root_link_lin_vel_b[0]

    walk_raw = (
      act.last_a_walk_raw[0]
      if act.last_a_walk_raw is not None
      else torch.zeros(12, device=env.device)
    )
    final_raw = (
      act.last_final_raw[0]
      if act.last_final_raw is not None
      else torch.zeros(12, device=env.device)
    )
    processed = act._joint_pos_action._processed_actions[0]

    row = {
      "label": run.label,
      "seed": run.seed,
      "variant": _variant_name(run.experimental_two_phase),
      "experimental_two_phase": run.experimental_two_phase,
      "step": step,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": int(act.recovery_elapsed[0].item()),
      "recovery_offset": recovery_offset if phase == PHASE_RECOVERY else -1,
      "newly_entangled": bool(act.newly_entangled[0].item()),
      "newly_recovered": bool(act.newly_recovered[0].item()),
      "newly_succeeded": bool(act.newly_succeeded[0].item()),
      "calf_clearance_achieved": bool(act.calf_clearance_achieved[0].item()),
      "foot_crossed_achieved": bool(act.foot_crossed_achieved[0].item()),
      "crossing_achieved": bool(act.crossing_achieved[0].item()),
      "release_achieved": bool(act.release_achieved[0].item()),
      "recovery_failed_timeout": bool(act.recovery_failed_timeout[0].item()),
      "calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
      "calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
      "foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
      "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
      "effective_clearance_margin_m": float(act._effective_clearance_margin[0].item()),
      "calf_lifted": bool(
        (act.calf_height_above_wire[0] > act._effective_clearance_margin[0]).item()
      ),
      "base_forward_vel_body_x": float(lin_vel_b[0].item()),
      "base_vertical_vel_world_z": float(lin_vel_w[2].item()),
      "walk_raw_hip": float(walk_raw[hip_idx].item()),
      "walk_raw_thigh": float(walk_raw[thigh_idx].item()),
      "walk_raw_calf": float(walk_raw[calf_idx].item()),
      "final_raw_hip": float(final_raw[hip_idx].item()),
      "final_raw_thigh": float(final_raw[thigh_idx].item()),
      "final_raw_calf": float(final_raw[calf_idx].item()),
      "processed_target_hip": float(processed[hip_idx].item()),
      "processed_target_thigh": float(processed[thigh_idx].item()),
      "processed_target_calf": float(processed[calf_idx].item()),
      "joint_pos_target_hip": float(pos_target[hip_idx].item()),
      "joint_pos_target_thigh": float(pos_target[thigh_idx].item()),
      "joint_pos_target_calf": float(pos_target[calf_idx].item()),
      "ctrl_hip_pos": float(sim_ctrl[hip_pos_ctrl_id].item()),
      "ctrl_thigh_pos": float(sim_ctrl[thigh_pos_ctrl_id].item()),
      "ctrl_calf_pos": float(sim_ctrl[calf_pos_ctrl_id].item()),
      "ctrl_thigh_aux_vel": float(sim_ctrl[thigh_aux_ctrl_id].item()),
      "ctrl_calf_aux_motor": float(sim_ctrl[calf_aux_ctrl_id].item()),
      "force_hip_pos": float(sim_force[hip_pos_ctrl_id].item()),
      "force_thigh_pos": float(sim_force[thigh_pos_ctrl_id].item()),
      "force_calf_pos": float(sim_force[calf_pos_ctrl_id].item()),
      "force_thigh_aux_vel": float(sim_force[thigh_aux_ctrl_id].item()),
      "force_calf_aux_motor": float(sim_force[calf_aux_ctrl_id].item()),
      "qpos_hip": float(qpos[hip_idx].item()),
      "qpos_thigh": float(qpos[thigh_idx].item()),
      "qpos_calf": float(qpos[calf_idx].item()),
      "qvel_hip": float(qvel[hip_idx].item()),
      "qvel_thigh": float(qvel[thigh_idx].item()),
      "qvel_calf": float(qvel[calf_idx].item()),
      "hybrid_thigh_vel_ctrl": float(act.hybrid_thigh_vel_ctrl[0].item()),
      "hybrid_calf_torque_ctrl": float(act.hybrid_calf_torque_ctrl[0].item()),
      "hybrid_thigh_pos_gain_active": bool(act.hybrid_thigh_pos_gain_active[0].item()),
      "two_phase_advance_active": bool(act.hybrid_two_phase_advance_active[0].item()),
      "two_phase_switch_step": int(act.two_phase_switch_step[0].item()),
      "peak_trigger_step": int(act.peak_trigger_step[0].item()),
    }
    rows.append(row)

    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      break

  env.close()

  csv_path = out_dir / f"{run.label}.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  return csv_path


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
  with open(csv_path, newline="") as f:
    return list(csv.DictReader(f))


def _recovery_rows(csv_path: Path) -> list[dict[str, Any]]:
  return [row for row in _load_rows(csv_path) if row["phase"] == "RECOVERY"]


def _find_row_by_recovery_offset(
  rows: list[dict[str, Any]], recovery_offset: int
) -> dict[str, Any] | None:
  for row in rows:
    if int(row["recovery_offset"]) == recovery_offset:
      return row
  return None


def _build_switch_overlay(
  seed: int,
  baseline_csv: Path,
  two_phase_csv: Path,
  out_dir: Path,
  switch_window: int,
) -> dict[str, Any]:
  baseline_rows = _recovery_rows(baseline_csv)
  two_phase_rows = _recovery_rows(two_phase_csv)
  switch_step = max(int(row["two_phase_switch_step"]) for row in two_phase_rows)
  assert switch_step >= 0, f"two-phase switch never fired for seed {seed}"

  overlay_rows: list[dict[str, Any]] = []
  keys = [
    "hybrid_thigh_vel_ctrl",
    "hybrid_calf_torque_ctrl",
    "ctrl_thigh_aux_vel",
    "ctrl_calf_aux_motor",
    "force_thigh_aux_vel",
    "force_calf_aux_motor",
    "qpos_thigh",
    "qpos_calf",
    "qvel_thigh",
    "qvel_calf",
    "calf_forward_dist_m",
    "calf_height_above_wire_m",
    "foot_forward_dist_m",
    "foot_height_above_wire_m",
    "base_forward_vel_body_x",
    "walk_raw_thigh",
    "walk_raw_calf",
    "joint_pos_target_thigh",
    "joint_pos_target_calf",
  ]

  for rel in range(-switch_window, switch_window + 1):
    offset = switch_step + rel
    baseline_row = _find_row_by_recovery_offset(baseline_rows, offset)
    two_phase_row = _find_row_by_recovery_offset(two_phase_rows, offset)
    if baseline_row is None or two_phase_row is None:
      continue

    row: dict[str, Any] = {
      "seed": seed,
      "switch_step": switch_step,
      "recovery_offset": offset,
      "relative_to_switch": rel,
      "baseline_step": int(baseline_row["step"]),
      "two_phase_step": int(two_phase_row["step"]),
      "baseline_calf_lifted": baseline_row["calf_lifted"],
      "two_phase_calf_lifted": two_phase_row["calf_lifted"],
      "two_phase_advance_active": two_phase_row["two_phase_advance_active"],
      "two_phase_switch_step_logged": int(two_phase_row["two_phase_switch_step"]),
    }
    for key in keys:
      baseline_val = float(baseline_row[key])
      two_phase_val = float(two_phase_row[key])
      row[f"baseline_{key}"] = baseline_val
      row[f"two_phase_{key}"] = two_phase_val
      row[f"delta_{key}"] = two_phase_val - baseline_val
    overlay_rows.append(row)

  overlay_csv = out_dir / f"seed{seed}_baseline_vs_two_phase_switch_overlay.csv"
  with open(overlay_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(overlay_rows[0].keys()))
    writer.writeheader()
    writer.writerows(overlay_rows)

  pre_rows = [row for row in overlay_rows if -3 <= int(row["relative_to_switch"]) <= -1]
  post_rows = [row for row in overlay_rows if 0 <= int(row["relative_to_switch"]) <= 3]
  switch_row = next(row for row in overlay_rows if int(row["relative_to_switch"]) == 0)

  summary = {
    "seed": seed,
    "switch_step": switch_step,
    "switch_baseline_recovery_step": int(switch_row["baseline_step"]),
    "switch_two_phase_recovery_step": int(switch_row["two_phase_step"]),
    "pre_baseline_calf_forward_mean": _mean(pre_rows, "baseline_calf_forward_dist_m"),
    "pre_two_phase_calf_forward_mean": _mean(pre_rows, "two_phase_calf_forward_dist_m"),
    "post_baseline_calf_forward_mean": _mean(post_rows, "baseline_calf_forward_dist_m"),
    "post_two_phase_calf_forward_mean": _mean(
      post_rows, "two_phase_calf_forward_dist_m"
    ),
    "pre_baseline_calf_height_mean": _mean(
      pre_rows, "baseline_calf_height_above_wire_m"
    ),
    "pre_two_phase_calf_height_mean": _mean(
      pre_rows, "two_phase_calf_height_above_wire_m"
    ),
    "post_baseline_calf_height_mean": _mean(
      post_rows, "baseline_calf_height_above_wire_m"
    ),
    "post_two_phase_calf_height_mean": _mean(
      post_rows, "two_phase_calf_height_above_wire_m"
    ),
    "pre_baseline_qvel_thigh_mean": _mean(pre_rows, "baseline_qvel_thigh"),
    "pre_two_phase_qvel_thigh_mean": _mean(pre_rows, "two_phase_qvel_thigh"),
    "post_baseline_qvel_thigh_mean": _mean(post_rows, "baseline_qvel_thigh"),
    "post_two_phase_qvel_thigh_mean": _mean(post_rows, "two_phase_qvel_thigh"),
    "pre_baseline_force_thigh_aux_mean": _mean(
      pre_rows, "baseline_force_thigh_aux_vel"
    ),
    "pre_two_phase_force_thigh_aux_mean": _mean(
      pre_rows, "two_phase_force_thigh_aux_vel"
    ),
    "post_baseline_force_thigh_aux_mean": _mean(
      post_rows, "baseline_force_thigh_aux_vel"
    ),
    "post_two_phase_force_thigh_aux_mean": _mean(
      post_rows, "two_phase_force_thigh_aux_vel"
    ),
    "pre_baseline_foot_forward_mean": _mean(pre_rows, "baseline_foot_forward_dist_m"),
    "pre_two_phase_foot_forward_mean": _mean(pre_rows, "two_phase_foot_forward_dist_m"),
    "post_baseline_foot_forward_mean": _mean(post_rows, "baseline_foot_forward_dist_m"),
    "post_two_phase_foot_forward_mean": _mean(
      post_rows, "two_phase_foot_forward_dist_m"
    ),
    "switch_ctrl_thigh_baseline": float(switch_row["baseline_ctrl_thigh_aux_vel"]),
    "switch_ctrl_thigh_two_phase": float(switch_row["two_phase_ctrl_thigh_aux_vel"]),
    "switch_ctrl_calf_baseline": float(switch_row["baseline_ctrl_calf_aux_motor"]),
    "switch_ctrl_calf_two_phase": float(switch_row["two_phase_ctrl_calf_aux_motor"]),
    "switch_qvel_thigh_baseline": float(switch_row["baseline_qvel_thigh"]),
    "switch_qvel_thigh_two_phase": float(switch_row["two_phase_qvel_thigh"]),
    "switch_calf_forward_baseline": float(switch_row["baseline_calf_forward_dist_m"]),
    "switch_calf_forward_two_phase": float(switch_row["two_phase_calf_forward_dist_m"]),
    "switch_calf_height_baseline": float(
      switch_row["baseline_calf_height_above_wire_m"]
    ),
    "switch_calf_height_two_phase": float(
      switch_row["two_phase_calf_height_above_wire_m"]
    ),
  }
  return summary


def _write_summary(
  per_seed: list[dict[str, Any]],
  d537_summary_csv: Path,
  d535_overlay_summary: Path,
  d536_overlay_summary: Path,
  out_dir: Path,
) -> None:
  d537_rows = _load_rows(d537_summary_csv)
  lines = ["D5.38 two-phase regression diagnostic", ""]

  lines.append("Reference artifacts consulted")
  lines.append(f"- D5.37 summary: {d537_summary_csv}")
  lines.append(f"- D5.35 overlay summary: {d535_overlay_summary}")
  lines.append(f"- D5.36 action overlay summary: {d536_overlay_summary}")
  lines.append("")

  lines.append("Confirmed measurements")
  for seed_summary in per_seed:
    seed = seed_summary["seed"]
    d537_row = next(
      row
      for row in d537_rows
      if int(row["seed"]) == seed and row["experimental_two_phase"] == "True"
    )
    lines.append(
      f"- Seed {seed}: D5.37 logged the two-phase switch at recovery step {d537_row['two_phase_switch_step']}; "
      f"this diagnostic reproduced the switch at recovery step {seed_summary['switch_step']}."
    )
    lines.append(
      f"  At the switch, thigh aux ctrl flips from baseline retract {seed_summary['switch_ctrl_thigh_baseline']:.4f} "
      f"to two-phase advance {seed_summary['switch_ctrl_thigh_two_phase']:.4f}, while calf aux ctrl stays unchanged at "
      f"{seed_summary['switch_ctrl_calf_two_phase']:.4f}."
    )
    lines.append(
      f"  Immediate post-switch mean thigh qvel changes from baseline {seed_summary['post_baseline_qvel_thigh_mean']:.4f} rad/s "
      f"to two-phase {seed_summary['post_two_phase_qvel_thigh_mean']:.4f} rad/s."
    )
    lines.append(
      f"  Immediate post-switch calf-forward mean changes from baseline {seed_summary['post_baseline_calf_forward_mean']:.4f} m "
      f"to two-phase {seed_summary['post_two_phase_calf_forward_mean']:.4f} m; calf-height mean changes from "
      f"{seed_summary['post_baseline_calf_height_mean']:.4f} m to {seed_summary['post_two_phase_calf_height_mean']:.4f} m."
    )
  lines.append("")

  lines.append("Interpretation")
  lines.append(
    "- The switch does produce the intended actuator-level change: thigh auxiliary velocity reverses sign while calf auxiliary torque stays fixed."
  )
  lines.append(
    "- Seed 0 regresses because the switch happens during a trajectory that still relies on continued retract-driven lift. After the sign flip, calf height drops sharply and the calf never reaches the late positive-forward stage that baseline achieves."
  )
  lines.append(
    "- Seed 11 gains only a small amount of horizontal progress because the same sign flip slightly reduces the near-side horizontal deficit, but it also collapses lift early; the calf remains on the near side and never satisfies the stage-1 near-to-far crossing event."
  )
  lines.append(
    "- This is consistent with D5.35/D5.36: seed 11's blocker is horizontal calf advance, but D5.38 shows that flipping thigh direction as soon as `calf_lifted` becomes true removes too much lift reserve and hurts the known-good seed before meaningful forward carry is established."
  )
  lines.append("")

  lines.append("Confirmed vs possible")
  lines.append(
    "- Confirmed: the two-phase command reverses thigh aux ctrl at the logged switch step, keeps calf aux ctrl unchanged, changes measured thigh velocity direction/magnitude, and reduces post-switch calf height in both seeds."
  )
  lines.append(
    "- Confirmed: seed 0's post-switch calf-forward trajectory is worse than its baseline at the same recovery offsets, and seed 11's improvement is too small to make calf_forward_dist positive."
  )
  lines.append(
    "- Possible explanation: the controller needs an intermediate hold/continue-lift interval or a different state trigger before forward advance. D5.38 does not prove the exact best trigger yet; it only rules out the current immediate-on-`calf_lifted` flip as a good choice."
  )
  lines.append("")

  lines.append("Specific next control experiment")
  lines.append(
    "- Keep the calf auxiliary torque unchanged, but replace the immediate sign flip with a delayed forward-advance trigger that requires BOTH `calf_lifted` and a minimum sustained lift/forward-readiness condition, e.g. `calf_lifted` for N consecutive recovery steps or `calf_lifted` plus `foot_forward_dist > 0` before reversing thigh aux velocity."
  )
  lines.append(
    "- Rationale: D5.38 shows the current trigger fires too early; it removes lift in seed 0 and only marginally helps seed 11. The next experiment should delay the advance phase until the leg has more vertical clearance and some forward carry already established."
  )

  (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="D5.38 two-phase regression diagnostic")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--max-steps", type=int, default=300)
  parser.add_argument("--out-dir", default=D538Config.out_dir)
  parser.add_argument("--switch-window", type=int, default=D538Config.switch_window)
  args = parser.parse_args()

  configure_torch_backends()
  cfg = D538Config(
    device=args.device,
    max_steps=args.max_steps,
    out_dir=args.out_dir,
    switch_window=args.switch_window,
  )

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  run_outputs: dict[str, Path] = {}
  for run in cfg.runs:
    print(
      f"Running {run.label} (seed={run.seed}, two_phase={run.experimental_two_phase})"
    )
    run_outputs[run.label] = _run_one(run, cfg.device, cfg.max_steps, out_dir)

  per_seed = []
  for seed in (0, 11):
    per_seed.append(
      _build_switch_overlay(
        seed=seed,
        baseline_csv=run_outputs[f"seed{seed}_baseline"],
        two_phase_csv=run_outputs[f"seed{seed}_two_phase"],
        out_dir=out_dir,
        switch_window=cfg.switch_window,
      )
    )

  summary_csv = out_dir / "switch_summary.csv"
  with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
    writer.writeheader()
    writer.writerows(per_seed)

  _write_summary(
    per_seed=per_seed,
    d537_summary_csv=Path("logs/d5_37/two_phase_experiment/summary.csv"),
    d535_overlay_summary=Path(
      "logs/d5_35/seed11_clearance_diagnostic/seed0_vs_seed11_recovery_overlay_summary.txt"
    ),
    d536_overlay_summary=Path(
      "logs/d5_36/recovery_action_audit/seed0_vs_seed11_action_overlay_summary.txt"
    ),
    out_dir=out_dir,
  )

  print(f"Wrote D5.38 outputs to {out_dir}")
  for row in per_seed:
    print(
      f"seed{row['seed']}: switch={row['switch_step']} "
      f"post_calf_fd baseline={row['post_baseline_calf_forward_mean']:.4f} "
      f"two_phase={row['post_two_phase_calf_forward_mean']:.4f}"
    )


if __name__ == "__main__":
  main()
