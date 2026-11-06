#!/usr/bin/env python
"""
D5.29: Phase-aware FSM comparison experiment.

This script runs seeds 0-19 with both baseline and phase-aware variants,
using the same environment configuration and logging as D5.23, to measure
recovery success rates and compare the two FSM variants.

Results are saved to:
  logs/d5_29_phase_aware_comparison/{baseline,phase_aware}/seed_{seed}/

For each variant/seed pair, we log:
  - run_log.txt: Summary of recovery outcome and entry leg speed
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import subprocess

import torch

# Reuse D5.23 diagnostic infrastructure
sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig
from mjlab.tasks.recovery.env_cfg_residual import (
    ISOLATED_FL_WIRE_CFG,
    ISOLATED_FL_WIRE_SPAWN_Y,
    go2_wire_recovery_residual_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnv

PHASE_RECOVERY = 1
PHASE_POST_RECOVERY = 2


class TrialOutcome(NamedTuple):
    """Results of a single trial (seed + variant)."""
    seed: int
    variant: str
    genuinely_recovered: bool
    stage1_reached: bool
    stage2_reached: bool
    max_recovery_steps: int
    exit_reason: str
    entry_leg_speed: float
    effective_clearance_margin: float
    effective_foot_crossing_margin: float
    crossing_achieved_at_step: int


@dataclass
class D529Config:
    seeds: list[int] = None
    variants: list[str] = None
    device: str = "cuda:0"
    max_steps: int = 500
    out_dir: str = "logs/d5_29_phase_aware_comparison"
    recovery_mode: str = "hybrid"
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(range(20))  # Seeds 0-19
        if self.variants is None:
            self.variants = ["baseline", "phase_aware"]


def run_single_trial(
    cfg: D529Config,
    seed: int,
    variant: str,
) -> TrialOutcome:
    """
    Run a single recovery episode for one seed under one variant.
    Returns TrialOutcome with recovery success indicators.
    """
    use_phase_aware = (variant == "phase_aware")
    
    # Build environment with variant-specific configuration
    env_cfg = go2_wire_recovery_residual_env_cfg(
        affected_leg_idx=0,  # FL only
        num_envs=1,
        recovery_mode=cfg.recovery_mode,
        policy_device=cfg.device,
        play=True,  # No observation noise
        wire_cfg=ISOLATED_FL_WIRE_CFG,
        spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
        phase_aware_thresholds=use_phase_aware,  # CRITICAL: Enable for phase_aware variant
        max_recovery_steps=48,  # Same as D5.23 baseline
    )
    env_cfg.seed = seed
    env_cfg.sim.device = cfg.device
    
    # Create environment
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    
    # Get recovery action term via ActionManager public API.
    # The manager exposes `get_term(name)` and does not expose a `.terms`
    # attribute.
    recovery_action = env.action_manager.get_term("joint_pos")
    
    # Run episode
    obs, _ = env.reset()
    entry_leg_speed = 0.0
    effective_clear = 0.0
    effective_cross = 0.0
    crossing_step = -1
    exit_reason = "other"
    genuinely_recovered = False
    stage1_reached = False
    stage2_reached = False
    
    for step in range(cfg.max_steps):
        # No-op actions (zero residual)
        actions = torch.zeros(
            1, env.action_manager.total_action_dim, device=env.device
        )
        obs, rewards, terminateds, truncateds, info = env.step(actions)
        
        # Capture entry leg speed and effective thresholds on first recovery step
        if recovery_action._phase[0] == PHASE_RECOVERY:
            if step == 0 or entry_leg_speed == 0.0:
                entry_leg_speed = float(
                    recovery_action._entry_leg_speed[0].cpu().item()
                )
                effective_clear = float(
                    recovery_action._effective_clearance_margin[0].cpu().item()
                )
                effective_cross = float(
                    recovery_action._effective_foot_crossing_margin[0].cpu().item()
                )
        
        # Track crossing achievement
        if recovery_action._crossing_achieved[0] and crossing_step == -1:
            crossing_step = step
        
        # Track stage achievements
        if recovery_action._calf_clearance_achieved[0]:
            stage1_reached = True
        if recovery_action._foot_crossed_achieved[0]:
            stage2_reached = True
        
        # Check terminal conditions
        if terminateds[0] or truncateds[0]:
            in_post_recovery = recovery_action._phase[0] == PHASE_POST_RECOVERY
            
            if in_post_recovery and recovery_action._stable_since_post_recovery[0]:
                genuinely_recovered = True
                exit_reason = "success"
            elif truncateds[0]:
                exit_reason = "timeout"
            elif terminateds[0]:
                exit_reason = "fell_over"
            break
    
    env.close()
    
    # Write run log
    out_path = Path(cfg.out_dir) / variant / f"seed_{seed}"
    out_path.mkdir(parents=True, exist_ok=True)
    
    run_log_file = out_path / "run_log.txt"
    with open(run_log_file, 'w') as f:
        f.write(f"D5.29 Phase-aware FSM Comparison\n")
        f.write(f"{'='*60}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Variant: {variant}\n")
        f.write(f"Recovery mode: {cfg.recovery_mode}\n")
        f.write(f"\nEntry State:\n")
        f.write(f"  Entry leg speed: {entry_leg_speed:.4f} rad/s\n")
        f.write(f"  Entry leg speed tier: ")
        if entry_leg_speed < 4.5:
            f.write("SLOW\n")
        elif entry_leg_speed < 6.5:
            f.write("MEDIUM\n")
        else:
            f.write("FAST\n")
        f.write(f"\nEffective Thresholds:\n")
        f.write(f"  Clearance margin: {effective_clear:.4f} m\n")
        f.write(f"  Foot crossing margin: {effective_cross:.4f} m\n")
        f.write(f"\nRecovery Outcome:\n")
        f.write(f"  Genuinely recovered: {genuinely_recovered}\n")
        f.write(f"  Stage 1 reached: {stage1_reached}\n")
        f.write(f"  Stage 2 reached: {stage2_reached}\n")
        f.write(f"  Crossing achieved at step: {crossing_step}\n")
        f.write(f"  Exit reason: {exit_reason}\n")
    
    return TrialOutcome(
        seed=seed,
        variant=variant,
        genuinely_recovered=genuinely_recovered,
        stage1_reached=stage1_reached,
        stage2_reached=stage2_reached,
        max_recovery_steps=step + 1,
        exit_reason=exit_reason,
        entry_leg_speed=entry_leg_speed,
        effective_clearance_margin=effective_clear,
        effective_foot_crossing_margin=effective_cross,
        crossing_achieved_at_step=crossing_step,
    )


def main():
    """Run D5.29 comparison experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="D5.29 Phase-aware FSM Comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)),
                       help="Seeds to test (default: 0-19)")
    parser.add_argument("--variants", type=str, nargs="+",
                       default=["baseline", "phase_aware"],
                       help="Variants to test")
    parser.add_argument("--device", default="cuda:0",
                       help="Device for simulation")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Max steps per episode")
    parser.add_argument("--out-dir", default="logs/d5_29_phase_aware_comparison",
                       help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable MuJoCo viewer for visualization (slow)")
    
    args = parser.parse_args()
    
    cfg = D529Config(
        seeds=args.seeds,
        variants=args.variants,
        device=args.device,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
    )
    
    # Add debug flag to config (not part of dataclass but add as attribute)
    cfg.debug = args.debug
    
    outcomes = []
    
    for variant in cfg.variants:
        print(f"\n{'='*70}")
        print(f"Running {variant.upper()} variant")
        print(f"{'='*70}")
        
        for seed in cfg.seeds:
            print(f"  Seed {seed:2d}...", end=" ", flush=True)
            
            try:
                outcome = run_single_trial(cfg, seed, variant)
                outcomes.append(outcome)
                
                status = "✓" if outcome.genuinely_recovered else "✗"
                s2 = "✓" if outcome.stage2_reached else "✗"
                speed_tier = ("SLOW" if outcome.entry_leg_speed < 4.5
                             else "FAST" if outcome.entry_leg_speed >= 6.5
                             else "MED")
                print(f"{status} (s2:{s2}, speed:{speed_tier})")
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Write summary
    summary_file = Path(cfg.out_dir) / "summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "variant", "genuinely_recovered", "stage1_reached",
            "stage2_reached", "exit_reason", "entry_leg_speed",
            "effective_clearance_margin", "effective_foot_crossing_margin",
            "crossing_achieved_at_step"
        ])
        for o in outcomes:
            writer.writerow([
                o.seed, o.variant, o.genuinely_recovered, o.stage1_reached,
                o.stage2_reached, o.exit_reason, f"{o.entry_leg_speed:.4f}",
                f"{o.effective_clearance_margin:.4f}",
                f"{o.effective_foot_crossing_margin:.4f}",
                o.crossing_achieved_at_step,
            ])
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for variant in cfg.variants:
        var_outcomes = [o for o in outcomes if o.variant == variant]
        if not var_outcomes:
            continue
        
        success_count = sum(1 for o in var_outcomes if o.genuinely_recovered)
        stage2_count = sum(1 for o in var_outcomes if o.stage2_reached)
        
        print(f"\n{variant.upper()}:")
        print(f"  Total trials: {len(var_outcomes)}")
        print(f"  Genuine recoveries: {success_count}/{len(var_outcomes)} ({100*success_count/len(var_outcomes):.1f}%)")
        print(f"  Stage 2 reached: {stage2_count}/{len(var_outcomes)} ({100*stage2_count/len(var_outcomes):.1f}%)")
    
    print(f"\nSummary written to: {summary_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
