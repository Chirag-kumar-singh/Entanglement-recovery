"""Go2 cable-entanglement test task registration.

Phase 3A test harness only: registers a task that lets the *existing*
trained Go2 walking policy be driven through an environment containing a
thin, anchored, articulated test cable. No new training is performed and no
recovery action/reward is added here.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import go2_cable_entanglement_test_env_cfg

register_mjlab_task(
  task_id="Mjlab-Go2-Cable-Entanglement-Test",
  env_cfg=go2_cable_entanglement_test_env_cfg(),
  play_env_cfg=go2_cable_entanglement_test_env_cfg(),
  # Reused only so the task registry / checkpoint-loading utilities
  # (obs_normalization, model architecture, experiment_name for checkpoint
  # lookup) match the existing trained Go2 velocity policy. No training is
  # run against this rl_cfg.
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
