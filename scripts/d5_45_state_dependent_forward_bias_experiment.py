#!/usr/bin/env python
"""D5.45 state-dependent forward-advance bias experiment.

Runs four bounded CPU episodes:
  - baseline seed 0
  - forward-bias seed 0
  - baseline seed 11
  - forward-bias seed 11

The experimental run keeps the existing contact gate, recovery FSM,
release logic, thresholds, and calf torque unchanged, but shrinks the
post-clearance forward-advance thigh target to a smaller signed bias
instead of a full sign reversal.

Outputs go under ``logs/d5_45/forward_bias_experiment/``.
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

FORWARD_BIAS_CONFIRM_STEPS = 4
FORWARD_BIAS_TRANSITION_STEPS = 4
FORWARD_BIAS_ADVANCE_FRACTION = 0.5


@dataclass(frozen=True)
class RunSpec:
  label: str
  seed: int
  forward_bias: bool
  confirm_steps: int = 1
  transition_steps: int = 0
  advance_fraction: float = 1.0


@dataclass(frozen=True)
class D545Config:
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = "logs/d5_45/forward_bias_experiment"
  runs: tuple[RunSpec, ...] = (
    RunSpec(label="seed0_baseline", seed=0, forward_bias=False),
    RunSpec(
      label="seed0_forward_bias",
      seed=0,
      forward_bias=True,
      confirm_steps=FORWARD_BIAS_CONFIRM_STEPS,
      transition_steps=FORWARD_BIAS_TRANSITION_STEPS,
      advance_fraction=FORWARD_BIAS_ADVANCE_FRACTION,
    ),
    RunSpec(label="seed11_baseline", seed=11, forward_bias=False),
    RunSpec(
      label="seed11_forward_bias",
      seed=11,
      forward_bias=True,
      confirm_steps=FORWARD_BIAS_CONFIRM_STEPS,
      transition_steps=FORWARD_BIAS_TRANSITION_STEPS,
      advance_fraction=FORWARD_BIAS_ADVANCE_FRACTION,
    ),
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
    phase_aware_thresholds=False,
    max_recovery_steps=48,
    experimental_two_phase_hybrid_recovery=run.forward_bias,
    experimental_two_phase_lift_confirm_steps=run.confirm_steps,
    experimental_two_phase_transition_steps=run.transition_steps,
    experimental_two_phase_advance_fraction=run.advance_fraction,
  )
  env_cfg.seed = run.seed
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _variant_name(run: RunSpec) -> str:
  return "forward_bias" if run.forward_bias else "baseline"


def _run_one(
  run: RunSpec, device: str, max_steps: int, out_dir: Path
) -> dict[str, Any]:
  env = _build_env(run, device)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  entity = act._entity
  joint_names = list(entity.joint_names)
  fl = LEG_NAMES[0]
  hip_idx = joint_names.index(f"{fl}_hip_joint")
  thigh_idx = joint_names.index(f"{fl}_thigh_joint")
  calf_idx = joint_names.index(f"{fl}_calf_joint")
  env.reset(seed=run.seed)

  rows: list[dict[str, Any]] = []
  entered_recovery = False
  post_recovery_entered = False
  recovery_exit_reason = "not_entered"
  observation_outcome = "horizon_exhausted"
  first_positive_calf_forward_step = -1
  max_calf_height = float("-inf")
  max_calf_forward = float("-inf")
  max_foot_forward = float("-inf")
  recovery_offset = -1

  for step in range(max_steps):
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    phase = int(act.phase[0].item())
    if phase == PHASE_RECOVERY:
      recovery_offset += 1

    entered_recovery = (
      entered_recovery or phase == PHASE_RECOVERY or bool(act.newly_entangled[0].item())
    )
    post_recovery_entered = (
      post_recovery_entered
      or phase == PHASE_POST_RECOVERY
      or bool(act.newly_recovered[0].item())
    )
    if bool(act.newly_recovered[0].item()):
      recovery_exit_reason = (
        "crossing_release"
        if bool(act.crossing_achieved[0].item())
        and bool(act.release_achieved[0].item())
        else "cap_timeout"
      )

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    lin_vel_b = entity.data.root_link_lin_vel_b[0]

    calf_forward = float(act.calf_forward_dist[0].item())
    calf_height = float(act.calf_height_above_wire[0].item())
    foot_forward = float(act.foot_forward_dist[0].item())
    base_forward_vel = float(lin_vel_b[0].item())

    max_calf_height = max(max_calf_height, calf_height)
    max_calf_forward = max(max_calf_forward, calf_forward)
    max_foot_forward = max(max_foot_forward, foot_forward)
    if first_positive_calf_forward_step < 0 and calf_forward > 0.0:
      first_positive_calf_forward_step = step

    row = {
      "label": run.label,
      "seed": run.seed,
      "variant": _variant_name(run),
      "step": step,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": int(act.recovery_elapsed[0].item()),
      "recovery_offset": recovery_offset if phase == PHASE_RECOVERY else -1,
      "post_recovery_elapsed": int(act.post_recovery_elapsed[0].item()),
      "newly_entangled": bool(act.newly_entangled[0].item()),
      "newly_recovered": bool(act.newly_recovered[0].item()),
      "newly_succeeded": bool(act.newly_succeeded[0].item()),
      "calf_clearance_achieved": bool(act.calf_clearance_achieved[0].item()),
      "foot_crossed_achieved": bool(act.foot_crossed_achieved[0].item()),
      "crossing_achieved": bool(act.crossing_achieved[0].item()),
      "release_achieved": bool(act.release_achieved[0].item()),
      "recovery_failed_timeout": bool(act.recovery_failed_timeout[0].item()),
      "calf_forward_dist_m": calf_forward,
      "calf_height_above_wire_m": calf_height,
      "foot_forward_dist_m": foot_forward,
      "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
      "effective_clearance_margin_m": float(act._effective_clearance_margin[0].item()),
      "base_forward_vel_body_x": base_forward_vel,
      "qpos_hip": float(qpos[hip_idx].item()),
      "qpos_thigh": float(qpos[thigh_idx].item()),
      "qpos_calf": float(qpos[calf_idx].item()),
      "qvel_hip": float(qvel[hip_idx].item()),
      "qvel_thigh": float(qvel[thigh_idx].item()),
      "qvel_calf": float(qvel[calf_idx].item()),
      "hybrid_thigh_vel_ctrl": float(act.hybrid_thigh_vel_ctrl[0].item()),
      "hybrid_calf_torque_ctrl": float(act.hybrid_calf_torque_ctrl[0].item()),
      "two_phase_advance_active": bool(act.hybrid_two_phase_advance_active[0].item()),
      "two_phase_lift_counter": int(act.two_phase_lift_counter[0].item()),
      "two_phase_switch_step": int(act.two_phase_switch_step[0].item()),
      "two_phase_lift_confirm_steps": run.confirm_steps,
      "two_phase_transition_steps": run.transition_steps,
      "two_phase_advance_fraction": run.advance_fraction,
    }
    rows.append(row)

    if bool(act.newly_succeeded[0].item()):
      observation_outcome = "script_success"

    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      if bool(terminateds[0].item()):
        observation_outcome = "terminated"
      elif bool(truncateds[0].item()):
        observation_outcome = "truncated"
      break

  env.close()

  csv_path = out_dir / f"{run.label}.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  post_recovery_vels = [
    float(row["base_forward_vel_body_x"])
    for row in rows
    if row["phase"] == "POST_RECOVERY"
  ]
  summary = {
    "label": run.label,
    "seed": run.seed,
    "variant": _variant_name(run),
    "forward_bias": run.forward_bias,
    "confirm_steps": run.confirm_steps,
    "transition_steps": run.transition_steps,
    "advance_fraction": run.advance_fraction,
    "entered_recovery": entered_recovery,
    "post_recovery_entered": post_recovery_entered,
    "recovery_exit_reason": recovery_exit_reason,
    "observation_outcome": observation_outcome,
    "two_phase_triggered": any(int(row["two_phase_switch_step"]) >= 0 for row in rows),
    "two_phase_switch_step": max(int(row["two_phase_switch_step"]) for row in rows),
    "stage1_achieved": any(bool(row["calf_clearance_achieved"]) for row in rows),
    "stage2_achieved": any(bool(row["foot_crossed_achieved"]) for row in rows),
    "full_crossing_achieved": any(bool(row["crossing_achieved"]) for row in rows),
    "first_positive_calf_forward_step": first_positive_calf_forward_step,
    "max_calf_height_above_wire_m": max_calf_height,
    "max_calf_forward_dist_m": max_calf_forward,
    "max_foot_forward_dist_m": max_foot_forward,
    "final_base_forward_vel_body_x": rows[-1]["base_forward_vel_body_x"],
    "mean_post_recovery_forward_vel_body_x": (
      sum(post_recovery_vels) / len(post_recovery_vels)
      if post_recovery_vels
      else float("nan")
    ),
    "csv_path": str(csv_path),
  }
  return summary


def _write_outputs(summaries: list[dict[str, Any]], out_dir: Path) -> None:
  summary_csv = out_dir / "summary.csv"
  with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

  lines = ["D5.45 state-dependent forward-bias experiment summary", ""]
  for summary in summaries:
    lines.extend(
      [
        f"{summary['label']}:",
        (
          f"  seed={summary['seed']} variant={summary['variant']} "
          f"confirm_steps={summary['confirm_steps']} "
          f"transition_steps={summary['transition_steps']} "
          f"advance_fraction={summary['advance_fraction']}"
        ),
        f"  entered_recovery={summary['entered_recovery']} post_recovery_entered={summary['post_recovery_entered']}",
        f"  recovery_exit_reason={summary['recovery_exit_reason']} observation_outcome={summary['observation_outcome']}",
        f"  two_phase_triggered={summary['two_phase_triggered']} switch_step={summary['two_phase_switch_step']}",
        (
          f"  stage1_achieved={summary['stage1_achieved']} "
          f"stage2_achieved={summary['stage2_achieved']} "
          f"full_crossing_achieved={summary['full_crossing_achieved']}"
        ),
        f"  first_positive_calf_forward_step={summary['first_positive_calf_forward_step']}",
        f"  max_calf_height_above_wire_m={summary['max_calf_height_above_wire_m']:.6f}",
        f"  max_calf_forward_dist_m={summary['max_calf_forward_dist_m']:.6f}",
        f"  max_foot_forward_dist_m={summary['max_foot_forward_dist_m']:.6f}",
        f"  final_base_forward_vel_body_x={summary['final_base_forward_vel_body_x']:.6f}",
        f"  mean_post_recovery_forward_vel_body_x={summary['mean_post_recovery_forward_vel_body_x']:.6f}",
        "",
      ]
    )

  for seed in (0, 11):
    seed_runs = [summary for summary in summaries if summary["seed"] == seed]
    baseline = next(
      summary for summary in seed_runs if summary["variant"] == "baseline"
    )
    forward_bias = next(
      summary for summary in seed_runs if summary["variant"] == "forward_bias"
    )
    lines.extend(
      [
        f"seed {seed} comparison:",
        f"  exit baseline={baseline['recovery_exit_reason']} forward_bias={forward_bias['recovery_exit_reason']}",
        f"  stage1 baseline={baseline['stage1_achieved']} forward_bias={forward_bias['stage1_achieved']}",
        f"  stage2 baseline={baseline['stage2_achieved']} forward_bias={forward_bias['stage2_achieved']}",
        (
          f"  max_calf_forward baseline={baseline['max_calf_forward_dist_m']:.6f} "
          f"forward_bias={forward_bias['max_calf_forward_dist_m']:.6f}"
        ),
        (
          f"  max_calf_height baseline={baseline['max_calf_height_above_wire_m']:.6f} "
          f"forward_bias={forward_bias['max_calf_height_above_wire_m']:.6f}"
        ),
        (
          f"  post_recovery_forward_vel baseline={baseline['mean_post_recovery_forward_vel_body_x']:.6f} "
          f"forward_bias={forward_bias['mean_post_recovery_forward_vel_body_x']:.6f}"
        ),
        f"  switch_step forward_bias={forward_bias['two_phase_switch_step']}",
        "",
      ]
    )

  (out_dir / "summary.txt").write_text("\n".join(lines))


def main() -> None:
  configure_torch_backends()
  cfg = D545Config()
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summaries = [_run_one(run, cfg.device, cfg.max_steps, out_dir) for run in cfg.runs]
  _write_outputs(summaries, out_dir)

  print(f"Wrote D5.45 outputs to {out_dir}")
  for summary in summaries:
    print(
      f"{summary['label']}: exit={summary['recovery_exit_reason']} "
      f"stage1={summary['stage1_achieved']} stage2={summary['stage2_achieved']} "
      f"max_calf_forward={summary['max_calf_forward_dist_m']:.4f} "
      f"switch={summary['two_phase_switch_step']}"
    )


if __name__ == "__main__":
  main()
