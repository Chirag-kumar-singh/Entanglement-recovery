#!/usr/bin/env python
"""D5.37 two-phase FL hybrid recovery experiment.

Compares the unchanged hybrid baseline against one opt-in experimental
variant that:
  1. starts with the current lift/retract command; and
  2. once the calf is above the active clearance margin, flips ONLY the
     thigh velocity command to a forward-advance direction.

Scope is intentionally narrow:
  - FL isolated-wire setup only;
  - seeds 0 and 11 only;
  - CPU only;
  - no checkpoint, reward, termination, or threshold changes except the
    single opt-in hybrid subphase flag.

Outputs go under ``logs/d5_37/two_phase_experiment/``.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
class D537Config:
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = "logs/d5_37/two_phase_experiment"
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


def _run_one(run: RunSpec, device: str, max_steps: int, out_dir: Path) -> dict:
  env = _build_env(run.seed, device, run.experimental_two_phase)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  entity = act._entity
  env.reset(seed=run.seed)

  rows: list[dict] = []
  entered_recovery = False
  post_recovery_entered = False
  recovery_exit_reason = "not_entered"
  observation_outcome = "horizon_exhausted"
  first_positive_calf_forward_step = -1
  max_calf_height = float("-inf")
  max_calf_forward = float("-inf")
  max_foot_forward = float("-inf")

  for step in range(max_steps):
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    phase = int(act.phase[0].item())
    entered_recovery = (
      entered_recovery or phase == PHASE_RECOVERY or bool(act.newly_entangled[0].item())
    )
    post_recovery_entered = (
      post_recovery_entered
      or phase == PHASE_POST_RECOVERY
      or bool(act.newly_recovered[0].item())
    )

    calf_forward = float(act.calf_forward_dist[0].item())
    calf_height = float(act.calf_height_above_wire[0].item())
    foot_forward = float(act.foot_forward_dist[0].item())
    base_vel_x = float(entity.data.root_link_lin_vel_b[0, 0].item())

    max_calf_height = max(max_calf_height, calf_height)
    max_calf_forward = max(max_calf_forward, calf_forward)
    max_foot_forward = max(max_foot_forward, foot_forward)
    if first_positive_calf_forward_step < 0 and calf_forward > 0.0:
      first_positive_calf_forward_step = step

    if bool(act.newly_recovered[0].item()):
      recovery_exit_reason = (
        "crossing_release"
        if bool(act.crossing_achieved[0].item())
        and bool(act.release_achieved[0].item())
        else "cap_timeout"
      )

    row = {
      "step": step,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": int(act.recovery_elapsed[0].item()),
      "post_recovery_elapsed": int(act.post_recovery_elapsed[0].item()),
      "calf_clearance_achieved": bool(act.calf_clearance_achieved[0].item()),
      "foot_crossed_achieved": bool(act.foot_crossed_achieved[0].item()),
      "crossing_achieved": bool(act.crossing_achieved[0].item()),
      "release_achieved": bool(act.release_achieved[0].item()),
      "recovery_failed_timeout": bool(act.recovery_failed_timeout[0].item()),
      "newly_entangled": bool(act.newly_entangled[0].item()),
      "newly_recovered": bool(act.newly_recovered[0].item()),
      "newly_succeeded": bool(act.newly_succeeded[0].item()),
      "calf_forward_dist_m": calf_forward,
      "calf_height_above_wire_m": calf_height,
      "foot_forward_dist_m": foot_forward,
      "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
      "effective_clearance_margin_m": float(act._effective_clearance_margin[0].item()),
      "base_forward_vel_body_x": base_vel_x,
      "hybrid_thigh_vel_ctrl": float(act.hybrid_thigh_vel_ctrl[0].item()),
      "hybrid_calf_torque_ctrl": float(act.hybrid_calf_torque_ctrl[0].item()),
      "two_phase_advance_active": bool(act.hybrid_two_phase_advance_active[0].item()),
      "two_phase_switch_step": int(act.two_phase_switch_step[0].item()),
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

  summary = {
    "label": run.label,
    "seed": run.seed,
    "experimental_two_phase": run.experimental_two_phase,
    "entered_recovery": entered_recovery,
    "post_recovery_entered": post_recovery_entered,
    "recovery_exit_reason": recovery_exit_reason,
    "observation_outcome": observation_outcome,
    "two_phase_triggered": any(bool(r["two_phase_switch_step"] >= 0) for r in rows),
    "two_phase_switch_step": max(int(r["two_phase_switch_step"]) for r in rows),
    "stage1_achieved": any(bool(r["calf_clearance_achieved"]) for r in rows),
    "stage2_achieved": any(bool(r["foot_crossed_achieved"]) for r in rows),
    "full_crossing_achieved": any(bool(r["crossing_achieved"]) for r in rows),
    "first_positive_calf_forward_step": first_positive_calf_forward_step,
    "max_calf_height_above_wire_m": max_calf_height,
    "max_calf_forward_dist_m": max_calf_forward,
    "max_foot_forward_dist_m": max_foot_forward,
    "final_base_forward_vel_body_x": rows[-1]["base_forward_vel_body_x"],
    "csv_path": str(csv_path),
  }
  return summary


def _write_outputs(summaries: list[dict], out_dir: Path) -> None:
  summary_csv = out_dir / "summary.csv"
  with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
    writer.writeheader()
    writer.writerows(summaries)

  lines = ["D5.37 two-phase hybrid recovery summary", ""]
  for summary in summaries:
    lines.extend(
      [
        f"{summary['label']}:",
        f"  seed={summary['seed']} experimental_two_phase={summary['experimental_two_phase']}",
        f"  entered_recovery={summary['entered_recovery']} post_recovery_entered={summary['post_recovery_entered']}",
        f"  recovery_exit_reason={summary['recovery_exit_reason']} observation_outcome={summary['observation_outcome']}",
        f"  two_phase_triggered={summary['two_phase_triggered']} switch_step={summary['two_phase_switch_step']}",
        f"  stage1_achieved={summary['stage1_achieved']} stage2_achieved={summary['stage2_achieved']} full_crossing_achieved={summary['full_crossing_achieved']}",
        f"  first_positive_calf_forward_step={summary['first_positive_calf_forward_step']}",
        f"  max_calf_height_above_wire_m={summary['max_calf_height_above_wire_m']:.6f}",
        f"  max_calf_forward_dist_m={summary['max_calf_forward_dist_m']:.6f}",
        f"  max_foot_forward_dist_m={summary['max_foot_forward_dist_m']:.6f}",
        f"  final_base_forward_vel_body_x={summary['final_base_forward_vel_body_x']:.6f}",
        "",
      ]
    )

  for seed in (0, 11):
    baseline = next(
      s for s in summaries if s["seed"] == seed and not s["experimental_two_phase"]
    )
    variant = next(
      s for s in summaries if s["seed"] == seed and s["experimental_two_phase"]
    )
    lines.extend(
      [
        f"seed {seed} delta:",
        f"  baseline_exit={baseline['recovery_exit_reason']} variant_exit={variant['recovery_exit_reason']}",
        f"  baseline_stage1={baseline['stage1_achieved']} variant_stage1={variant['stage1_achieved']}",
        f"  baseline_stage2={baseline['stage2_achieved']} variant_stage2={variant['stage2_achieved']}",
        f"  baseline_max_calf_forward={baseline['max_calf_forward_dist_m']:.6f} variant_max_calf_forward={variant['max_calf_forward_dist_m']:.6f}",
        f"  baseline_max_foot_forward={baseline['max_foot_forward_dist_m']:.6f} variant_max_foot_forward={variant['max_foot_forward_dist_m']:.6f}",
        "",
      ]
    )

  (out_dir / "summary.txt").write_text("\n".join(lines))


def main() -> None:
  configure_torch_backends()
  cfg = D537Config()
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summaries = [_run_one(run, cfg.device, cfg.max_steps, out_dir) for run in cfg.runs]
  _write_outputs(summaries, out_dir)

  print(f"Wrote D5.37 outputs to {out_dir}")
  for summary in summaries:
    print(
      f"{summary['label']}: exit={summary['recovery_exit_reason']} "
      f"stage1={summary['stage1_achieved']} stage2={summary['stage2_achieved']} "
      f"max_calf_forward={summary['max_calf_forward_dist_m']:.4f} "
      f"two_phase={summary['two_phase_triggered']}"
    )


if __name__ == "__main__":
  main()
