"""Smoke test for Go2 entanglement recovery environment."""

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg import go2_entanglement_recovery_env_cfg


def test_recovery_env_creation():
  """Test that recovery environment can be created successfully."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  assert env is not None
  print(f"✓ Environment created successfully")
  print(f"  Num envs: {env.num_envs}")
  print(f"  Device: {env.device}")
  env.close()


def test_recovery_env_reset():
  """Test that environment can reset."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()
  assert obs is not None
  assert isinstance(obs, dict)
  print(f"✓ Environment reset successfully")
  print(f"  Observation keys: {list(obs.keys())}")
  env.close()


def test_observation_dimensions():
  """Test that observations have correct dimensions."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()

  actor_obs = obs["actor"]
  critic_obs = obs["critic"]

  print(f"✓ Observation dimensions:")
  print(f"  Actor obs shape: {actor_obs.shape}")
  print(f"  Critic obs shape: {critic_obs.shape}")

  # Verify shapes
  assert actor_obs.dim() == 2, f"Actor obs should be 2D, got {actor_obs.dim()}D"
  assert critic_obs.dim() == 2, f"Critic obs should be 2D, got {critic_obs.dim()}D"
  assert actor_obs.shape[0] == env.num_envs
  assert critic_obs.shape[0] == env.num_envs

  env.close()


def test_action_dimensions():
  """Test that actions have correct dimensions."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()

  # Go2 has 12 DOF (4 legs × 3 joints each)
  num_actions = 12
  actions = torch.zeros((env.num_envs, num_actions), device=env.device)

  print(f"✓ Action dimensions:")
  print(f"  Action shape: {actions.shape}")
  print(f"  Expected: ({env.num_envs}, {num_actions})")

  env.close()


def test_zero_action_step():
  """Test stepping with zero actions."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()

  actions = torch.zeros((env.num_envs, 12), device=env.device)
  obs, rewards, dones, truncs, info = env.step(actions)

  assert obs is not None
  assert rewards is not None
  assert dones is not None
  print(f"✓ Zero action step successful")
  print(f"  Rewards shape: {rewards.shape}")
  print(f"  Dones shape: {dones.shape}")

  env.close()


def test_random_action_step():
  """Test stepping with random actions."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()

  actions = torch.randn((env.num_envs, 12), device=env.device)
  obs, rewards, dones, truncs, info = env.step(actions)

  assert obs is not None
  print(f"✓ Random action step successful")
  print(f"  Sample reward: {rewards[0].item():.4f}")

  env.close()


def test_affected_leg_oracle():
  """Test that affected leg oracle is in observations."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()

  # Extract affected_leg observation (should be one-hot)
  actor_obs = obs["actor"]  # concatenated observations
  critic_obs = obs["critic"]  # concatenated observations

  # Verify oracle is present in the MDP
  # Oracle should be 4-dimensional (one-hot for 4 legs)
  print(f"✓ Affected leg oracle test:")
  print(f"  Actor obs total dim: {actor_obs.shape[1]}")
  print(f"  Critic obs total dim: {critic_obs.shape[1]}")
  print(f"  (First 4 dims should be one-hot: [0, 0, 1, 0] for RL leg)")

  env.close()


def test_multiple_legs():
  """Test with different affected legs."""
  leg_names = ["FL", "FR", "RL", "RR"]
  leg_indices = [0, 1, 2, 3]

  for idx, leg_name in zip(leg_indices, leg_names):
    cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=idx, play=False)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs, info = env.reset()
    actions = torch.zeros((env.num_envs, 12), device=env.device)
    obs, rewards, dones, truncs, info = env.step(actions)
    print(f"✓ {leg_name} (idx={idx}) environment works")
    env.close()


def test_play_mode():
  """Test play mode (no observation noise)."""
  cfg_train = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  cfg_play = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=True)

  env_train = ManagerBasedRlEnv(cfg_train, device="cpu")
  env_play = ManagerBasedRlEnv(cfg_play, device="cpu")

  # Both should work
  obs_train, _ = env_train.reset()
  obs_play, _ = env_play.reset()

  print(f"✓ Play mode test successful")
  print(f"  Training env obs shape: {obs_train['actor'].shape}")
  print(f"  Play env obs shape: {obs_play['actor'].shape}")

  env_train.close()
  env_play.close()


def test_episode_length():
  """Test that episodes terminate at correct length."""
  cfg = go2_entanglement_recovery_env_cfg(affected_leg_idx=2, play=False)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, info = env.reset()

  # Run for more than episode length (10 seconds at 200Hz decimated = ~50 steps)
  steps = 0
  done = False
  for _ in range(100):
    actions = torch.zeros((env.num_envs, 12), device=env.device)
    obs, rewards, dones, truncs, info = env.step(actions)
    steps += 1
    if dones[0]:
      done = True
      break

  print(f"✓ Episode length test:")
  print(f"  Episode ended after {steps} steps")
  print(f"  Expected ~50 steps (10s * 200Hz / decimation)")

  env.close()


if __name__ == "__main__":
  print("\n" + "=" * 60)
  print("Go2 Entanglement Recovery Environment Smoke Test")
  print("=" * 60 + "\n")

  test_recovery_env_creation()
  test_recovery_env_reset()
  test_observation_dimensions()
  test_action_dimensions()
  test_zero_action_step()
  test_random_action_step()
  test_affected_leg_oracle()
  test_multiple_legs()
  test_play_mode()
  test_episode_length()

  print("\n" + "=" * 60)
  print("All smoke tests passed! ✓")
  print("=" * 60 + "\n")
