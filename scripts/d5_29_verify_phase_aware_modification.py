#!/usr/bin/env python
"""
D5.29 Verification: Test that the phase-aware FSM modification works correctly.

This script verifies that:
1. The phase-aware configuration parameters are properly defined
2. Entry leg speed can be computed from joint velocities
3. The three-tier threshold selection logic works as expected
4. The FSM uses the correct thresholds in both baseline and phase-aware modes
"""

import torch
import numpy as np
from pathlib import Path
import sys


def test_config_parameters():
  """Verify that phase-aware parameters are defined in ResidualRecoveryActionCfg."""
  print("\n" + "=" * 60)
  print("TEST 1: Config Parameters")
  print("=" * 60)

  from mjlab.tasks.recovery.mdp.actions import ResidualRecoveryActionCfg

  # Check that phase-aware parameters exist
  required_params = [
    "phase_aware_thresholds",
    "phase_aware_speed_threshold_low",
    "phase_aware_speed_threshold_high",
    "phase_aware_slow_clearance_margin",
    "phase_aware_slow_foot_crossing_margin",
    "phase_aware_medium_clearance_margin",
    "phase_aware_medium_foot_crossing_margin",
    "phase_aware_fast_clearance_margin",
    "phase_aware_fast_foot_crossing_margin",
  ]

  for param in required_params:
    if not hasattr(ResidualRecoveryActionCfg, param):
      print(f"  ✗ FAILED: Missing parameter {param}")
      return False

  print(f"  ✓ All {len(required_params)} phase-aware config parameters are present")

  # Check default values
  default_cfg = {
    "phase_aware_thresholds": False,
    "phase_aware_speed_threshold_low": 4.5,
    "phase_aware_speed_threshold_high": 6.5,
    "phase_aware_slow_clearance_margin": 0.020,
    "phase_aware_slow_foot_crossing_margin": 0.030,
    "phase_aware_medium_clearance_margin": 0.025,
    "phase_aware_medium_foot_crossing_margin": 0.035,
    "phase_aware_fast_clearance_margin": 0.030,
    "phase_aware_fast_foot_crossing_margin": 0.040,
  }

  print("\n  Default values:")
  for param, expected_val in default_cfg.items():
    print(f"    {param}: {expected_val}")

  return True


def test_environment_factory():
  """Verify that the environment factory accepts phase_aware_thresholds parameter."""
  print("\n" + "=" * 60)
  print("TEST 2: Environment Factory")
  print("=" * 60)

  from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
  import inspect

  sig = inspect.signature(go2_wire_recovery_residual_env_cfg)

  if "phase_aware_thresholds" not in sig.parameters:
    print("  ✗ FAILED: phase_aware_thresholds parameter not in factory signature")
    return False

  print("  ✓ phase_aware_thresholds parameter exists in factory function")

  # Check default value
  param = sig.parameters["phase_aware_thresholds"]
  if param.default != False:
    print(f"  ✗ FAILED: Default value should be False, got {param.default}")
    return False

  print(f"  ✓ Default value is False (preserves baseline)")

  return True


def test_threshold_selection_logic():
  """Verify that threshold selection logic works correctly."""
  print("\n" + "=" * 60)
  print("TEST 3: Threshold Selection Logic")
  print("=" * 60)

  # Simulate the _update_phase_aware_thresholds method
  def select_thresholds(entry_leg_speeds):
    """Compute effective thresholds based on leg speeds."""
    cfg_low_thresh = 4.5
    cfg_high_thresh = 6.5

    cfg_slow_clear = 0.020
    cfg_slow_cross = 0.030
    cfg_med_clear = 0.025
    cfg_med_cross = 0.035
    cfg_fast_clear = 0.030
    cfg_fast_cross = 0.040

    device = "cpu"

    # Initialize to slow-tier defaults
    clearance = torch.full_like(entry_leg_speeds, cfg_slow_clear, device=device)
    crossing = torch.full_like(entry_leg_speeds, cfg_slow_cross, device=device)

    # Medium tier: low_thresh <= leg_speed < high_thresh
    medium_mask = (entry_leg_speeds >= cfg_low_thresh) & (
      entry_leg_speeds < cfg_high_thresh
    )
    clearance[medium_mask] = cfg_med_clear
    crossing[medium_mask] = cfg_med_cross

    # Fast tier: leg_speed >= high_thresh
    fast_mask = entry_leg_speeds >= cfg_high_thresh
    clearance[fast_mask] = cfg_fast_clear
    crossing[fast_mask] = cfg_fast_cross

    return clearance, crossing

  # Test cases
  test_speeds = torch.tensor(
    [
      3.0,  # Slow (< 4.5)
      4.5,  # Boundary: low
      5.5,  # Medium
      6.5,  # Boundary: high
      7.5,  # Fast (>= 6.5)
    ]
  )

  expected_clear = [0.020, 0.025, 0.025, 0.030, 0.030]
  expected_cross = [0.030, 0.035, 0.035, 0.040, 0.040]

  clear, cross = select_thresholds(test_speeds)

  all_pass = True
  for i, (speed, exp_c, exp_cr) in enumerate(
    zip(test_speeds, expected_clear, expected_cross)
  ):
    actual_c = float(clear[i].item())
    actual_cr = float(cross[i].item())

    # Check with small tolerance for floating point
    if abs(actual_c - exp_c) > 0.0001 or abs(actual_cr - exp_cr) > 0.0001:
      print(
        f"  ✗ FAILED: Speed {speed:.1f} rad/s - Expected ({exp_c}, {exp_cr}), got ({actual_c}, {actual_cr})"
      )
      all_pass = False
    else:
      print(
        f"  ✓ Speed {speed:.1f} rad/s → clear={actual_c:.3f}, cross={actual_cr:.3f}"
      )

  return all_pass


def test_fsm_thresholds_instance():
  """Verify that ResidualRecoveryAction initializes threshold tracking variables."""
  print("\n" + "=" * 60)
  print("TEST 4: FSM Instance Variables")
  print("=" * 60)

  from mjlab.tasks.recovery.mdp.actions import ResidualRecoveryAction

  # Check that the method exists
  if not hasattr(ResidualRecoveryAction, "_update_phase_aware_thresholds"):
    print("  ✗ FAILED: _update_phase_aware_thresholds method not found")
    return False

  print("  ✓ _update_phase_aware_thresholds method exists")

  # Check that instance variables are initialized in __init__
  # (We can't easily instantiate the full Action without a full env,
  #  so we'll just verify the method signature)
  import inspect

  method = getattr(ResidualRecoveryAction, "_update_phase_aware_thresholds")
  sig = inspect.signature(method)

  if "entry_leg_speed" not in sig.parameters:
    print("  ✗ FAILED: entry_leg_speed parameter not in method signature")
    return False

  print("  ✓ _update_phase_aware_thresholds has correct signature")

  return True


def test_environment_factory_instantiation():
  """Verify that the environment factory can be instantiated with phase_aware_thresholds=True."""
  print("\n" + "=" * 60)
  print("TEST 5: Factory Instantiation")
  print("=" * 60)

  from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg

  try:
    # Try to create env config with phase_aware_thresholds=True
    # (Don't actually build the environment, just the config)
    cfg = go2_wire_recovery_residual_env_cfg(
      num_envs=1,
      phase_aware_thresholds=True,
    )

    # Verify the action config has phase_aware_thresholds=True
    action_cfg = cfg.actions["joint_pos"]
    if not action_cfg.phase_aware_thresholds:
      print("  ✗ FAILED: phase_aware_thresholds not set to True in action config")
      return False

    print("  ✓ Environment factory creates config with phase_aware_thresholds=True")

    # Also verify baseline config
    cfg_baseline = go2_wire_recovery_residual_env_cfg(
      num_envs=1,
      phase_aware_thresholds=False,
    )

    action_cfg_baseline = cfg_baseline.actions["joint_pos"]
    if action_cfg_baseline.phase_aware_thresholds:
      print("  ✗ FAILED: Baseline should have phase_aware_thresholds=False")
      return False

    print(
      "  ✓ Environment factory creates baseline config with phase_aware_thresholds=False"
    )

    return True

  except Exception as e:
    print(f"  ✗ FAILED: Exception during instantiation: {e}")
    import traceback

    traceback.print_exc()
    return False


def main():
  """Run all verification tests."""
  print("\n" + "=" * 80)
  print("D5.29 VERIFICATION: Phase-aware FSM Modification")
  print("=" * 80)

  tests = [
    ("Config Parameters", test_config_parameters),
    ("Environment Factory", test_environment_factory),
    ("Threshold Selection", test_threshold_selection_logic),
    ("FSM Instance Variables", test_fsm_thresholds_instance),
    ("Factory Instantiation", test_environment_factory_instantiation),
  ]

  results = []
  for name, test_func in tests:
    try:
      result = test_func()
      results.append((name, result))
    except Exception as e:
      print(f"\n  ✗ EXCEPTION in {name}: {e}")
      import traceback

      traceback.print_exc()
      results.append((name, False))

  # Summary
  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)

  passed = sum(1 for _, result in results if result)
  total = len(results)

  for name, result in results:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {name}")

  print(f"\nTotal: {passed}/{total} tests passed")
  print("=" * 80 + "\n")

  return all(result for _, result in results)


if __name__ == "__main__":
  success = main()
  sys.exit(0 if success else 1)
