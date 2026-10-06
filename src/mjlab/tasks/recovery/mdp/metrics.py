"""Behavioral metrics for the residual recovery task (training-time
diagnostics only -- no effect on observations/rewards/termination).

Registered via `ManagerBasedRlEnvCfg.metrics` and logged automatically
under the `Episode_Metrics/` prefix (see mjlab.managers.metrics_manager).
Minimal, additive instrumentation requested for the 500-1000 iteration
real training run: recovery trigger/success/fall counts, residual
magnitude stats, forward progress, and post-recovery stability -- nothing
here feeds back into the policy's objective.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, PHASE_POST_RECOVERY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_action_term(env: ManagerBasedRlEnv, action_term_name: str):
  return env.action_manager.get_term(action_term_name)


def metric_recovery_triggered(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  """1.0 on the exact step a recovery episode begins (NORMAL_WALK ->
  RECOVERY), else 0.0. Use ``reduce="sum"`` to get a per-episode count of
  recovery triggers (almost always 0 or 1 given the FSM only ever enters
  RECOVERY once per episode)."""
  term = _get_action_term(env, action_term_name)
  return term.newly_entangled.float()


def metric_recovery_success(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  """1.0 on the exact step the FULL recovery sequence is confirmed --
  genuine geometric crossing + sustained release + post-recovery
  stability window (see ``ResidualRecoveryAction.newly_succeeded``), else
  0.0. Identical definition to the ``wire_recovery_final_success_bonus``
  reward term (post-audit fix -- previously this fired on contact release
  alone, which was the root cause of a reward exploit; see chat
  history). Use ``reduce="sum"`` for a per-episode success count."""
  term = _get_action_term(env, action_term_name)
  return term.newly_succeeded.float()


def metric_crossing_achieved(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  """1.0 on the exact step the affected foot genuinely crosses the wire
  while above it (see ``ResidualRecoveryAction.crossing_event``), else
  0.0. Use ``reduce="sum"`` for a per-episode count -- this is a STRICTLY
  easier condition than full success (no release/stability requirement
  yet), useful for diagnosing whether failures are crossing-side or
  stability-side."""
  term = _get_action_term(env, action_term_name)
  return term.crossing_event.float()


def metric_recovery_cap_hit(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  """1.0 on the exact step RECOVERY exits via the hard max_recovery_steps
  safety cap WITHOUT genuine crossing+release (i.e. a failed recovery
  attempt), else 0.0. Use ``reduce="sum"`` for a per-episode count."""
  term = _get_action_term(env, action_term_name)
  return (term.newly_recovered & term.recovery_failed_timeout).float()


def metric_fell_over(
  env: ManagerBasedRlEnv,
  limit_angle: float = math.radians(70.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """1.0 on any step the robot is tilted beyond ``limit_angle`` (mirrors
  the ``fell_over`` termination exactly), else 0.0. Use ``reduce="max"``
  for a per-episode fall indicator (0/1)."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  cos_angle = torch.abs(gravity_b[:, 2])
  return (cos_angle < math.cos(limit_angle)).float()


def metric_residual_magnitude(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  """L2 norm of the actual GATED residual actually applied this step
  (zero outside RECOVERY by construction). Use ``reduce="mean"`` for the
  episode-average magnitude, ``reduce="max"`` for the peak."""
  term = _get_action_term(env, action_term_name)
  if term.last_final_raw is None or term.last_a_walk_raw is None:
    return torch.zeros(env.num_envs, device=env.device)
  residual = term.last_final_raw - term.last_a_walk_raw
  return torch.norm(residual, dim=-1)


def metric_forward_progress(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Per-step forward (world x) linear velocity, used as a rolling
  forward-progress proxy. Use ``reduce="mean"`` for episode-average
  forward speed (a drop vs. the commanded 0.3 m/s during/after recovery
  would indicate the recovery intervention is costing locomotion)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 0]


def metric_post_recovery_tilt(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tilt-from-upright magnitude (0 = perfectly upright), reported ONLY
  during POST_RECOVERY steps (0.0 elsewhere so it does not bias the
  episode average with NORMAL_WALK data). Use ``reduce="mean"`` as a
  post-recovery walking-stability proxy -- large values indicate the
  policy destabilizes the gait on the way out of RECOVERY."""
  term = _get_action_term(env, action_term_name)
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  tilt = 1.0 - torch.abs(gravity_b[:, 2])
  in_post_recovery = term.phase == PHASE_POST_RECOVERY
  return torch.where(in_post_recovery, tilt, torch.zeros_like(tilt))


def metric_in_recovery(
  env: ManagerBasedRlEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
  """1.0 while the FSM is currently in RECOVERY, else 0.0. Use
  ``reduce="sum"`` for a per-episode recovery-duration-in-steps proxy."""
  term = _get_action_term(env, action_term_name)
  return (term.phase == PHASE_RECOVERY).float()
