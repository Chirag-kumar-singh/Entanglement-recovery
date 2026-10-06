"""RL configuration for Go2 entanglement recovery task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def go2_entanglement_recovery_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Go2 entanglement recovery task.

  Uses PPO with network architecture and hyperparameters tuned for
  recovery maneuvers.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,  # Lower exploration for precise recovery
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,  # Lower LR for stable recovery
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_recovery_oracle",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3000,  # Shorter for initial tests
  )


def go2_wire_recovery_residual_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """RL runner configuration for the residual-action wire-recovery task.

  Much smaller action space (3-dim FL residual, only active during the
  bounded RECOVERY window) than the oracle tasks above, so a smaller
  network and higher init_std (more exploration useful since the residual
  only gets to act for a few steps per episode) are used. `max_iterations`
  here is intentionally small -- this config is meant for the first
  smoke-training run, not a full training; override via
  `--agent.max-iterations` for a real run.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_wire_recovery_residual",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=20,  # Smoke-test default; override for real training.
  )
