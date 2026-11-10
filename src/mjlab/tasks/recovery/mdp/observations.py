"""Observations for recovery task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def affected_leg_oracle(
  env: ManagerBasedRlEnv,
  affected_leg_idx: int,
) -> torch.Tensor:
  """One-hot oracle observation of affected leg.

  Args:
    env: The RL environment.
    affected_leg_idx: Index of affected leg (0=FL, 1=FR, 2=RL, 3=RR).

  Returns:
    One-hot encoded tensor of shape [B, 4] where B is batch size.
    Example: affected_leg_idx=2 (RL) → [0, 0, 1, 0]
  """
  one_hot = torch.zeros(
    (env.num_envs, 4),
    device=env.device,
    dtype=torch.float32,
  )
  one_hot[:, affected_leg_idx] = 1.0
  return one_hot


# ---------------------------------------------------------------------------
# Residual-recovery-task observations. All of these read state from the
# ResidualRecoveryAction term (named `action_term_name` in the env's
# actions dict, default "joint_pos" to match the existing recovery env
# convention) -- the FSM phase, elapsed counters, previous residual, and
# pi_walk's own affected-leg sub-action all live there (see
# mjlab.tasks.recovery.mdp.actions), since that's the one component that
# already owns the wire-contact sensor lookup and the walk-policy call.
# ---------------------------------------------------------------------------


def _get_recovery_action_term(env: ManagerBasedRlEnv, action_term_name: str):
  return env.action_manager.get_term(action_term_name)


def recovery_phase_onehot(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """One-hot encoding of the recovery FSM phase: [normal_walk, recovery,
  post_recovery]. Shape [B, 3]."""
  term = _get_recovery_action_term(env, action_term_name)
  phase = term.phase  # [B], long, values in {0, 1, 2}
  return torch.nn.functional.one_hot(phase, num_classes=3).float()


def recovery_elapsed_norm(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Fraction of the max recovery window elapsed (0 outside RECOVERY).
  Shape [B, 1]."""
  term = _get_recovery_action_term(env, action_term_name)
  frac = term.recovery_elapsed.float() / float(term.cfg.max_recovery_steps)
  return frac.unsqueeze(-1)


def previous_residual(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """The previous step's Delta_a (raw policy-action-space units, affected
  leg only), as actually applied (pre-clamp/pre-gate value is NOT used --
  this is the raw action the policy itself emitted, since PPO's own
  action-rate penalties should see what it asked for, not what was
  allowed through). Shape [B, 3]."""
  term = _get_recovery_action_term(env, action_term_name)
  return term.raw_action


def affected_leg_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wire_contact",
) -> torch.Tensor:
  """Whether the affected leg is currently in contact with the wire.
  Shape [B, 1]."""
  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found  # [B, P]
  in_contact = (found > 0).any(dim=-1, keepdim=True)  # [B, 1]
  return in_contact.float()


def affected_foot_pos_rel_body(
  env: ManagerBasedRlEnv,
  leg_name: str,
  asset_cfg_name: str = "robot",
) -> torch.Tensor:
  """Affected leg's foot site position relative to the robot's base
  (world-frame vector, foot_pos_w - root_pos_w). Shape [B, 3]."""
  asset = env.scene[asset_cfg_name]
  site_names = list(asset.site_names)
  idx = site_names.index(leg_name)
  foot_pos_w = asset.data.site_pos_w[:, idx, :]
  root_pos_w = asset.data.root_link_pos_w
  return foot_pos_w - root_pos_w


def affected_foot_vel_rel_body(
  env: ManagerBasedRlEnv,
  leg_name: str,
  asset_cfg_name: str = "robot",
) -> torch.Tensor:
  """Affected leg's foot site velocity relative to the robot's base,
  computed as a finite difference of `affected_foot_pos_rel_body` cached
  across steps (stateful -- stores previous value as an attribute on the
  env for simplicity, since this is a plain function term)."""
  asset = env.scene[asset_cfg_name]
  site_names = list(asset.site_names)
  idx = site_names.index(leg_name)
  foot_pos_w = asset.data.site_pos_w[:, idx, :]
  root_pos_w = asset.data.root_link_pos_w
  rel_pos = foot_pos_w - root_pos_w

  cache_name = f"_prev_foot_rel_pos_{leg_name}"
  prev = getattr(env, cache_name, None)
  if prev is None or prev.shape[0] != rel_pos.shape[0]:
    prev = rel_pos.clone()
  vel = (rel_pos - prev) / env.step_dt
  setattr(env, cache_name, rel_pos.clone())
  return vel


def walk_affected_action(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """The frozen walking policy's OWN raw action for the affected leg's 3
  joints this step (a_walk_raw at the affected indices) -- i.e. what
  pi_walk itself "wanted" to do before any residual was added. Lets the
  recovery policy see (and potentially counteract or cooperate with) the
  walking policy's own intent for that leg. Shape [B, 3]."""
  term = _get_recovery_action_term(env, action_term_name)
  if term.last_a_walk_raw is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return term.last_a_walk_raw[:, term._affected_idx]


# ---------------------------------------------------------------------------
# Wire-relative geometric observations (post-audit addition, see chat
# history): the FSM/reward already compute `calf_height_above_wire`,
# `calf_forward_dist`, `foot_height_above_wire` (aliased here as "toe"
# height, see `ResidualRecoveryAction` module docstring -- the `foot`
# site IS the toe), and `foot_forward_dist` every step, but prior to this
# change the actor observation vector never saw any of them -- an audit
# found this as the PRIMARY learning bottleneck (the policy had no direct
# signal for where the wire is relative to the affected leg). These four
# terms read those SAME live-geometry values directly off the action
# term (no duplicate wire-geometry computation) and normalize them to
# bounded [-1, 1] ranges using the SAME scale constants
# (`WIRE_HEIGHT_NORM_SCALE`, `WIRE_FORWARD_NORM_SCALE`) used by the
# potential-shaping reward in rewards.py, so observation and reward stay
# on an identical, already-audited scale.
#
# Outside RECOVERY these are zeroed -- NOT left at their last in-recovery
# value -- so there is no ambiguity between "not currently in recovery"
# and a valid recovery-geometry reading; `recovery_phase_onehot` already
# tells the policy which phase it's in, so a consistent zero here cannot
# be confused with a genuine zero-crossing reading taken DURING recovery
# (phase one-hot disambiguates the two).
# ---------------------------------------------------------------------------


def calf_height_above_wire_norm(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Normalized calf height above the wire's live top surface, clamped to
  [-1, 1] via `WIRE_HEIGHT_NORM_SCALE` (0.05 m). Negative = calf below
  the wire (entangled/approaching), positive = calf above it (lifted
  clear). Zero outside RECOVERY. Shape [B, 1]."""
  from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, WIRE_HEIGHT_NORM_SCALE

  term = _get_recovery_action_term(env, action_term_name)
  val = torch.clamp(term.calf_height_above_wire / WIRE_HEIGHT_NORM_SCALE, -1.0, 1.0)
  active = term.phase == PHASE_RECOVERY
  return torch.where(active, val, torch.zeros_like(val)).unsqueeze(-1)


def calf_forward_dist_norm(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Normalized calf forward distance past the wire's local reference
  (robot's own latched forward direction), clamped to [-1, 1] via
  `WIRE_FORWARD_NORM_SCALE` (0.1 m). Negative = near side, positive = far
  side. Zero outside RECOVERY. Shape [B, 1]."""
  from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, WIRE_FORWARD_NORM_SCALE

  term = _get_recovery_action_term(env, action_term_name)
  val = torch.clamp(term.calf_forward_dist / WIRE_FORWARD_NORM_SCALE, -1.0, 1.0)
  active = term.phase == PHASE_RECOVERY
  return torch.where(active, val, torch.zeros_like(val)).unsqueeze(-1)


def toe_height_above_wire_norm(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Normalized toe/foot height above the wire's live top surface
  (`ResidualRecoveryAction.foot_height_above_wire` -- the `foot` site IS
  the toe, see module docstring), clamped to [-1, 1] via
  `WIRE_HEIGHT_NORM_SCALE`. Zero outside RECOVERY. Shape [B, 1]."""
  from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, WIRE_HEIGHT_NORM_SCALE

  term = _get_recovery_action_term(env, action_term_name)
  val = torch.clamp(term.foot_height_above_wire / WIRE_HEIGHT_NORM_SCALE, -1.0, 1.0)
  active = term.phase == PHASE_RECOVERY
  return torch.where(active, val, torch.zeros_like(val)).unsqueeze(-1)


def toe_forward_dist_norm(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Normalized toe/foot forward distance past the wire's local
  reference (`ResidualRecoveryAction.foot_forward_dist`), clamped to
  [-1, 1] via `WIRE_FORWARD_NORM_SCALE`. Zero outside RECOVERY.
  Shape [B, 1]."""
  from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, WIRE_FORWARD_NORM_SCALE

  term = _get_recovery_action_term(env, action_term_name)
  val = torch.clamp(term.foot_forward_dist / WIRE_FORWARD_NORM_SCALE, -1.0, 1.0)
  active = term.phase == PHASE_RECOVERY
  return torch.where(active, val, torch.zeros_like(val)).unsqueeze(-1)
