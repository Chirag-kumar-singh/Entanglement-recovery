"""Go2 entanglement recovery task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .env_cfg import go2_entanglement_recovery_env_cfg
from .env_cfg_wire import go2_wire_entanglement_recovery_env_cfg
from .env_cfg_residual import go2_wire_recovery_residual_env_cfg
from .rl_cfg import (
  go2_entanglement_recovery_ppo_runner_cfg,
  go2_wire_recovery_residual_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Go2-Entanglement-Recovery-Oracle",
  env_cfg=go2_entanglement_recovery_env_cfg(),
  play_env_cfg=go2_entanglement_recovery_env_cfg(play=True),
  rl_cfg=go2_entanglement_recovery_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Go2-Wire-Entanglement-Recovery-Oracle",
  env_cfg=go2_wire_entanglement_recovery_env_cfg(),
  play_env_cfg=go2_wire_entanglement_recovery_env_cfg(play=True),
  rl_cfg=go2_entanglement_recovery_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Go2-Wire-Recovery-Residual",
  env_cfg=go2_wire_recovery_residual_env_cfg(),
  play_env_cfg=go2_wire_recovery_residual_env_cfg(play=True),
  rl_cfg=go2_wire_recovery_residual_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

