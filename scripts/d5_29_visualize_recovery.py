#!/usr/bin/env python
"""D5.29/D5.40 MuJoCo Visualization: watch recovery variants."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.viewer import EnvProtocol, NativeMujocoViewer, PolicyProtocol

GRADUAL_CONFIRM_STEPS = 4
GRADUAL_TRANSITION_STEPS = 4
FORWARD_BIAS_CONFIRM_STEPS = 4
FORWARD_BIAS_TRANSITION_STEPS = 4
FORWARD_BIAS_ADVANCE_FRACTION = 0.5


def visualize_recovery(
  seed: int, variant: str, device: str, max_steps: int | None = None
) -> None:
  use_phase_aware = variant == "phase_aware"
  use_gradual_transition = variant == "gradual_transition"
  use_forward_bias = variant == "forward_bias"

  print(f"\n{'=' * 70}")
  print(f"{variant.upper()} RECOVERY - Seed {seed}")
  print(f"{'=' * 70}")
  print("Opening MuJoCo viewer... watch the FL foot recover.")
  print("Key things to observe:")
  print("  - FL foot height during recovery")
  print("  - Whether foot crosses wire and stays on far side")
  print("  - Robot stability after recovery\n")

  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=use_phase_aware,
    max_recovery_steps=48,
    experimental_two_phase_hybrid_recovery=(
      use_gradual_transition or use_forward_bias
    ),
    experimental_two_phase_lift_confirm_steps=(
      GRADUAL_CONFIRM_STEPS
      if use_gradual_transition
      else FORWARD_BIAS_CONFIRM_STEPS
      if use_forward_bias
      else 1
    ),
    experimental_two_phase_transition_steps=(
      GRADUAL_TRANSITION_STEPS
      if use_gradual_transition
      else FORWARD_BIAS_TRANSITION_STEPS
      if use_forward_bias
      else 0
    ),
    experimental_two_phase_advance_fraction=(
      FORWARD_BIAS_ADVANCE_FRACTION if use_forward_bias else 1.0
    ),
  )
  env_cfg.seed = seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env.reset(seed=seed)

  print("Starting episode...\n")

  step_counter = {"step": 0}

  def viewer_policy(obs: torch.Tensor) -> torch.Tensor:
    del obs
    step = step_counter["step"]
    if step == 0:
      print("Viewer launched successfully.")
    step_counter["step"] += 1
    return torch.zeros(1, env.action_manager.total_action_dim, device=env.device)

  viewer = NativeMujocoViewer(
    cast(EnvProtocol, env), cast(PolicyProtocol, viewer_policy)
  )

  if max_steps is None:
    try:
      viewer.run()
    finally:
      env.close()
    print("Viewer closed.\n")
    return

  viewer.setup()
  step_count = 0
  while viewer.is_running() and step_count < max_steps:
    if not viewer.tick():
      time.sleep(0.001)
    step_count += 1

  print("Viewer closed.\n")


def main() -> None:
  parser = argparse.ArgumentParser(description="Recovery Visualization")
  parser.add_argument("--seed", type=int, default=0, help="Seed to visualize")
  parser.add_argument(
    "--variant",
    type=str,
    choices=["baseline", "phase_aware", "gradual_transition", "forward_bias"],
    default="baseline",
    help="Variant to visualize",
  )
  parser.add_argument(
    "--compare",
    action="store_true",
    help="Visualize both baseline and phase-aware for the same seed",
  )
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Simulation device (default: auto-detect cuda:0 if available, else cpu)",
  )
  parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="If set, auto-stop after this many viewer steps.",
  )

  args = parser.parse_args()

  device = args.device
  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

  if args.compare:
    print("\n" + "=" * 70)
    print(f"COMPARING BOTH VARIANTS FOR SEED {args.seed}")
    print("=" * 70)
    visualize_recovery(args.seed, "baseline", device, max_steps=args.max_steps)
    visualize_recovery(args.seed, "phase_aware", device, max_steps=args.max_steps)
  else:
    visualize_recovery(args.seed, args.variant, device, max_steps=args.max_steps)


if __name__ == "__main__":
  main()
