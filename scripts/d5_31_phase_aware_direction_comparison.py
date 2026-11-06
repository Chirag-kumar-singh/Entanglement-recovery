#!/usr/bin/env python
"""
D5.31: Small controlled CPU comparison of baseline vs direction-corrected
phase-aware thresholds.

Scope (explicitly NOT a full sweep):
  - A small, fixed seed set (default: 0, 1, 11 -- 11 was flagged in
    D5.29's own script docstring as a "high-speed entry seed").
  - CPU only, identical environment settings/checkpoint/horizon for both
    variants.
  - No training, no retraining of the walking policy, no wire-geometry or
    hybrid-actuator changes -- this only toggles
    `phase_aware_thresholds` (now using the D5.31-corrected tier->margin
    direction baked into `ResidualRecoveryActionCfg`'s defaults).

For each seed/variant pair, records (ALL OBSERVED, not hypothesized):
  - stage1 (calf clearance) / stage2 (foot crossing) latch achievement
  - final exit reason (success / timeout / fell_over)
  - minimum foot height above the wire reached after recovery entry
    (most negative value observed; a proxy for "how close to/through the
    wire the foot actually got", independent of the margin used to latch)
  - body height drop: base link world-Z at recovery entry minus the
    minimum base link world-Z observed from recovery entry to episode end
  - contact loss: whether the wire-contact sensor ever reports zero
    contact after having reported contact (a simple founds->not-found
    transition count), reported as a simple bool + transition count

This is NOT a judgment of "better" -- it is a side-by-side numeric record
for human review, per the D5.31 task constraints.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)

PHASE_NORMAL_WALK = 0
PHASE_RECOVERY = 1
PHASE_POST_RECOVERY = 2


@dataclass
class D531Config:
  seeds: tuple[int, ...] = (0, 1, 11)
  variants: tuple[str, ...] = ("baseline", "phase_aware")
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = "logs/d5_31_phase_aware_direction_comparison"
  recovery_mode: str = "hybrid"


def run_single_trial(cfg: D531Config, seed: int, variant: str) -> dict:
  use_phase_aware = variant == "phase_aware"

  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode=cfg.recovery_mode,
    policy_device=cfg.device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=use_phase_aware,
    max_recovery_steps=48,
  )
  env_cfg.seed = seed
  env_cfg.sim.device = cfg.device

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env.reset(seed=seed)

  act = env.action_manager.get_term("joint_pos")
  ent = act._entity
  wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]

  entry_leg_speed = 0.0
  effective_clear = 0.0
  effective_cross = 0.0
  crossing_step = -1
  # `exit_reason` now distinguishes the environment's own terminal
  # outcomes (success / timeout / fell_over) from the case where the
  # SCRIPT's `max_steps` loop limit was reached without any environment
  # termination/truncation ("horizon_exhausted"). Horizon exhaustion is
  # explicitly NEITHER a recovery success NOR a recovery failure -- it
  # only means this observation window ended before the environment did
  # (see D5.32A audit). Default below is overwritten either inside the
  # termination/truncation branch OR, if the loop runs to completion,
  # after the loop.
  exit_reason = "horizon_exhausted"
  env_terminated = False
  env_truncated = False
  genuinely_recovered = False
  stage1_reached = False
  stage2_reached = False

  # --- Separately-logged recovery-FSM milestones (observed, not
  # hypothesized). These are tracked independently of `exit_reason` so
  # that horizon exhaustion never hides genuine FL recovery progress. ---
  recovery_exited = False  # FSM left RECOVERY (-> POST_RECOVERY) at all
  recovery_exit_reason = "none"  # "crossing_release" | "cap_timeout" | "none"
  post_recovery_entered = False  # FSM reached PHASE_POST_RECOVERY
  # Environment's OWN internal recovery-success bookkeeping (the FSM's
  # `newly_succeeded` / `_already_succeeded_this_episode` latch, which
  # additionally requires genuine-crossing eligibility + a sustained
  # post-recovery stability window -- STRICTER than the script's own
  # ad-hoc success check below). Logged separately so the two can never
  # be silently conflated.
  env_internal_success_fired = False
  # The comparison script's EXISTING (historical) success condition,
  # preserved verbatim so prior D5.31 results keep their exact meaning:
  # termination/truncation while phase==POST_RECOVERY AND
  # `_stable_since_post_recovery`. Tracked as its own flag.
  script_success_condition_fired = False

  recovery_entered = False
  base_height_at_entry = None
  min_base_height_post_entry = None
  min_foot_height_above_wire_post_entry = None

  prev_wire_contact = False
  contact_loss_transitions = 0
  ever_had_contact = False

  final_step = 0
  for step in range(cfg.max_steps):
    final_step = step
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    obs, rewards, terminateds, truncateds, info = env.step(actions)

    phase_now = int(act._phase[0].item())
    if phase_now == PHASE_RECOVERY and not recovery_entered:
      recovery_entered = True
      base_height_at_entry = float(ent.data.root_link_pos_w[0, 2].item())
      min_base_height_post_entry = base_height_at_entry
      min_foot_height_above_wire_post_entry = float(
        act.foot_height_above_wire[0].item()
      )
      entry_leg_speed = float(act._entry_leg_speed[0].item())
      effective_clear = float(act._effective_clearance_margin[0].item())
      effective_cross = float(act._effective_foot_crossing_margin[0].item())

    if recovery_entered:
      h = float(ent.data.root_link_pos_w[0, 2].item())
      if min_base_height_post_entry is None or h < min_base_height_post_entry:
        min_base_height_post_entry = h
      fh = float(act.foot_height_above_wire[0].item())
      if (
        min_foot_height_above_wire_post_entry is None
        or fh < min_foot_height_above_wire_post_entry
      ):
        min_foot_height_above_wire_post_entry = fh

    wire_contact_now = bool((wire_sensor.data.found[0] > 0).any().item())
    if wire_contact_now:
      ever_had_contact = True
    if prev_wire_contact and not wire_contact_now:
      contact_loss_transitions += 1
    prev_wire_contact = wire_contact_now

    if bool(act._crossing_achieved[0].item()) and crossing_step == -1:
      crossing_step = step
    if bool(act._calf_clearance_achieved[0].item()):
      stage1_reached = True
    if bool(act._foot_crossed_achieved[0].item()):
      stage2_reached = True

    # --- Recovery-FSM milestone tracking (independent of episode end).
    # Once POST_RECOVERY is reached, record WHY recovery exited using
    # the FSM's own latched `recovery_failed_timeout` flag (True =>
    # exited via the max_recovery_steps safety cap; False => genuine
    # crossing+release). ---
    if phase_now == PHASE_POST_RECOVERY:
      post_recovery_entered = True
      if not recovery_exited:
        recovery_exited = True
        recovery_exit_reason = (
          "cap_timeout"
          if bool(act._recovery_failed_timeout[0].item())
          else "crossing_release"
        )
    # Environment's OWN internal success latch (strict: genuine-
    # crossing eligibility + sustained post-recovery stability window).
    if bool(act._already_succeeded_this_episode[0].item()):
      env_internal_success_fired = True

    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      env_terminated = bool(terminateds[0].item())
      env_truncated = bool(truncateds[0].item())
      in_post_recovery = phase_now == PHASE_POST_RECOVERY
      # Historical script success condition, preserved verbatim.
      if in_post_recovery and bool(act._stable_since_post_recovery[0].item()):
        script_success_condition_fired = True
        genuinely_recovered = True
        exit_reason = "success"
      elif bool(truncateds[0].item()):
        exit_reason = "timeout"
      elif bool(terminateds[0].item()):
        exit_reason = "fell_over"
      break
  else:
    # The `for` loop ran to completion without hitting the
    # termination/truncation `break`: the SCRIPT's max_steps window
    # ended before the environment did. This is neither success nor
    # failure -- `exit_reason` retains its "horizon_exhausted" default.
    exit_reason = "horizon_exhausted"

  env.close()

  body_height_drop = (
    (base_height_at_entry - min_base_height_post_entry)
    if base_height_at_entry is not None
    else None
  )

  result = {
    "seed": seed,
    "variant": variant,
    "recovery_entered": recovery_entered,
    "entry_leg_speed": entry_leg_speed,
    "effective_clearance_margin": effective_clear,
    "effective_foot_crossing_margin": effective_cross,
    "stage1_reached": stage1_reached,
    "stage2_reached": stage2_reached,
    "crossing_achieved_at_step": crossing_step,
    "recovery_exited": recovery_exited,
    "recovery_exit_reason": recovery_exit_reason,
    "post_recovery_entered": post_recovery_entered,
    "env_internal_success_fired": env_internal_success_fired,
    "script_success_condition_fired": script_success_condition_fired,
    "env_terminated": env_terminated,
    "env_truncated": env_truncated,
    "genuinely_recovered": genuinely_recovered,
    "exit_reason": exit_reason,
    "final_step": final_step,
    "base_height_at_entry_m": base_height_at_entry,
    "min_base_height_post_entry_m": min_base_height_post_entry,
    "body_height_drop_m": body_height_drop,
    "min_foot_height_above_wire_post_entry_m": min_foot_height_above_wire_post_entry,
    "ever_had_wire_contact": ever_had_contact,
    "wire_contact_loss_transitions": contact_loss_transitions,
  }
  return result


def main():
  import argparse

  parser = argparse.ArgumentParser(
    description="D5.31 small controlled CPU comparison (baseline vs corrected phase_aware)"
  )
  parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 11])
  parser.add_argument(
    "--variants", type=str, nargs="+", default=["baseline", "phase_aware"]
  )
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--max-steps", type=int, default=300)
  parser.add_argument(
    "--out-dir", default="logs/d5_31_phase_aware_direction_comparison"
  )
  args = parser.parse_args()

  cfg = D531Config(
    seeds=tuple(args.seeds),
    variants=tuple(args.variants),
    device=args.device,
    max_steps=args.max_steps,
    out_dir=args.out_dir,
  )

  results = []
  for variant in cfg.variants:
    print(f"\n{'=' * 70}\nRunning {variant.upper()} variant\n{'=' * 70}")
    for seed in cfg.seeds:
      print(f"  Seed {seed:2d}...", end=" ", flush=True)
      r = run_single_trial(cfg, seed, variant)
      results.append(r)
      print(
        f"entered={r['recovery_entered']} s1={r['stage1_reached']} "
        f"s2={r['stage2_reached']} exit={r['exit_reason']} "
        f"drop={r['body_height_drop_m']}"
      )

  out_path = Path(cfg.out_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  csv_path = out_path / "d5_31_comparison.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

  print(f"\n{'=' * 70}\nRESULTS (observed, not hypothesized)\n{'=' * 70}")
  for r in results:
    print(r)
  print(f"\nWritten to: {csv_path}\n")


if __name__ == "__main__":
  main()
