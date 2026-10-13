"""Reward functions for recovery task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _foot_clearance_xz(
  env: ManagerBasedRlEnv,
  rod_pos_xz: tuple[float, float] | torch.Tensor,
  leg_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Perpendicular (x-z) distance from a leg's foot site to the rod axis.

  The rod runs along world Y at a fixed (x, z) *relative to each
  environment's own grid origin*, so the relevant clearance distance for a
  leg is measured in the x-z plane only (distance along the rod's own axis
  does not un-obstruct the leg). ``rod_pos_xz`` is env-local (env 0's
  origin); with ``num_envs > 1`` each environment's rod copy -- and its
  robot's reset position -- are both offset by that environment's grid
  origin (see ``env_cfg.py``'s per-env rod replication), so the env's own
  x-origin offset is added here before computing distance. The grid never
  offsets z, so no z-correction is needed.

  Returns:
    Distance tensor of shape [B] (meters, always >= 0).
  """
  asset: Entity = env.scene[asset_cfg.name]

  if isinstance(rod_pos_xz, (tuple, list)):
    rod_xz = torch.tensor(rod_pos_xz, device=env.device, dtype=torch.float32)
  else:
    rod_xz = rod_pos_xz.to(env.device)

  # Per-env grid x-offset (z is never offset by the grid layout).
  env_origin_x = env.scene.env_origins[:, 0]  # [B]
  rod_xz_per_env = rod_xz.unsqueeze(0).expand(env.num_envs, -1).clone()  # [B, 2]
  rod_xz_per_env[:, 0] = rod_xz_per_env[:, 0] + env_origin_x

  site_names = list(asset.site_names)
  idx = site_names.index(leg_name)
  foot_pos = asset.data.site_pos_w[:, idx, :]  # [B, 3]
  foot_xz = foot_pos[:, [0, 2]]  # [B, 2]
  return torch.norm(foot_xz - rod_xz_per_env, dim=-1)  # [B]


def affected_leg_clearance(
  env: ManagerBasedRlEnv,
  rod_pos_xz: tuple[float, float] | torch.Tensor,
  rod_radius: float,
  leg_name: str,
  clearance_margin: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Shaped reward for the AFFECTED leg's foot clearing the rod.

  Unlike root/body distance from the obstacle, this rewards only the
  specific affected leg's foot moving its perpendicular (x-z) distance from
  the rod's axis beyond the rod radius, saturating at 1.0 once cleared by
  `clearance_margin`.

  Returns:
    Reward tensor of shape [B] in [0, 1].
  """
  dist = _foot_clearance_xz(env, rod_pos_xz, leg_name, asset_cfg)
  clearance = dist - rod_radius
  return torch.clamp(clearance / clearance_margin, min=0.0, max=1.0)


def recovery_success_bonus(
  env: ManagerBasedRlEnv,
  rod_pos_xz: tuple[float, float] | torch.Tensor,
  rod_radius: float,
  leg_name: str,
  clearance_threshold: float = 0.15,
  limit_angle: float = 0.5235987755982988,  # 30 degrees
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sparse bonus: 1.0 while the affected leg is fully clear of the rod
  AND the robot remains upright, else 0.0. This is the reward-side mirror
  of the `recovery_success` termination condition -- "successful clearing"
  as opposed to partial shaping progress.

  Returns:
    Reward tensor of shape [B] in {0.0, 1.0}.
  """
  asset: Entity = env.scene[asset_cfg.name]

  dist = _foot_clearance_xz(env, rod_pos_xz, leg_name, asset_cfg)
  cleared = (dist - rod_radius) > clearance_threshold

  gravity_b = asset.data.projected_gravity_b
  cos_angle = torch.abs(gravity_b[:, 2])

  upright = cos_angle > math.cos(limit_angle)

  return (cleared & upright).float()


def uprightness_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for staying upright (as opposed to only penalizing falling).

  Uses the same projected-gravity z-component as `fell_over`/
  `recovery_success`: ~1.0 when perfectly upright, decreasing toward 0 as
  the robot tilts.

  Returns:
    Reward tensor of shape [B] in [0, 1] (approximately).
  """
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  return torch.clamp(-gravity_b[:, 2], min=0.0, max=1.0)


def clearance_from_obstacle(
  env: ManagerBasedRlEnv,
  obstacle_pos: tuple[float, float, float] | torch.Tensor,
  obstacle_radius: float,
  affected_leg_joint_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for maintaining distance from obstacle with affected leg.

  Args:
    env: The RL environment.
    obstacle_pos: Obstacle center position [x, y, z] tuple or tensor.
    obstacle_radius: Obstacle radius/size in meters.
    affected_leg_joint_name: Name or regex of the affected leg joint (e.g., "RL_calf_joint").
    asset_cfg: Scene entity config for the robot.

  Returns:
    Reward tensor of shape [B]. Higher reward = farther from obstacle.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Convert obstacle_pos to tensor if needed
  if isinstance(obstacle_pos, (tuple, list)):
    obs_pos = torch.tensor(obstacle_pos, device=env.device, dtype=torch.float32)
  else:
    obs_pos = obstacle_pos.to(env.device)

  # Get root link position as proxy for body position
  root_pos = asset.data.root_link_pos_w  # [B, 3]

  # Distance from obstacle center
  distance = torch.norm(root_pos - obs_pos, dim=-1)  # [B]

  # Reward: exponential decay with distance (higher reward when far)
  # If distance >= obstacle_radius, reward is 1.0
  # If distance < obstacle_radius, reward decays toward 0
  clearance_reward = torch.exp(-torch.clamp(obstacle_radius - distance, min=0.0) / 0.1)
  return clearance_reward


def body_stability_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for unstable body orientation/angular velocity.

  Args:
    env: The RL environment.
    asset_cfg: Scene entity config for the robot.

  Returns:
    Penalty tensor of shape [B]. Higher = more unstable.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Penalize high angular velocity (instability)
  ang_vel = asset.data.root_link_ang_vel_b
  penalty = torch.norm(ang_vel, dim=-1)  # [B]
  return penalty


def joint_torque_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for high joint accelerations (proxy for torques).

  Args:
    env: The RL environment.
    asset_cfg: Scene entity config for the robot.

  Returns:
    Penalty tensor of shape [B].
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Use joint acceleration as proxy (computed from finite differences in step)
  # For now, just use a small constant penalty for stability
  return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)


def joint_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for high joint velocities (promoting smooth motion).

  Args:
    env: The RL environment.
    asset_cfg: Scene entity config for the robot.

  Returns:
    Penalty tensor of shape [B].
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel  # [B, num_joints]
  penalty = torch.norm(joint_vel, dim=-1)  # [B]
  return penalty


# ---------------------------------------------------------------------------
# Wire-interaction reward terms (ball-joint articulated wire, NOT the rigid
# rod). See ``mjlab.tasks.cable_test.ball_cable`` for the wire model and the
# multi-stage read-only diagnostic that validated it. Unlike the rod's
# geometric x-z clearance metric, "recovered" here is defined purely from
# the actual wire interaction: contact released + affected leg back in a
# normal/safe joint configuration + upright -- deliberately NOT requiring
# the foot to be above/past the wire, since a flexible tether has no fixed
# obstacle plane to clear.
# ---------------------------------------------------------------------------


def _affected_leg_joint_ids(
  env: ManagerBasedRlEnv, leg_name: str, asset_cfg: SceneEntityCfg
) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  suffixes = ("hip_joint", "thigh_joint", "calf_joint")
  names = [f"{leg_name}_{s}" for s in suffixes]
  return [joint_names.index(n) for n in names if n in joint_names]


def wire_contact_release_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wire_contact",
) -> torch.Tensor:
  """Dense SHAPING reward for the affected leg not currently being in
  contact with the wire -- NOT a success reward.

  Returns 1.0 while released, 0.0 while in contact, every step. This
  gives the policy a small, continuous gradient toward reducing/avoiding
  contact even before the persistence-gated `wire_recovery_success`
  condition can possibly be satisfied. Its weight must stay small relative
  to the sparse `wire_recovery_success_bonus` (see env_cfg_wire.py): a
  reward audit found that a large weight here (previously 1.0, alongside
  an over-eager instantaneous success bonus) made ordinary intermittent
  contact gaps pay out almost as much total reward as genuine recovery,
  which would encourage "waiting for a lucky gap" rather than deliberate
  recovery behavior.

  Returns:
    Reward tensor of shape [B] in {0.0, 1.0}.
  """
  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found  # [B, P]
  in_contact = (found > 0).any(dim=-1)  # [B]
  return (~in_contact).float()


def affected_leg_return_to_stance_reward(
  env: ManagerBasedRlEnv,
  leg_name: str,
  joint_pos_tolerance: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Shaped reward for the affected leg's joints returning toward their
  default (nominal standing/walking) configuration.

  This replaces the rod's foot-vs-obstacle x-z clearance shaping term: for
  a flexible wire there is no fixed obstacle plane to "clear", so instead
  we reward the affected leg's joints moving back within
  `joint_pos_tolerance` (radians, L2 over hip/thigh/calf) of their default
  pose -- a general "leg back in a safe/normal configuration" signal, not
  tied to any specific wire geometry.

  Returns:
    Reward tensor of shape [B] in [0, 1] (1.0 = at/within tolerance of
    default, decaying to 0.0 as the offset grows to 2x tolerance).
  """
  asset: Entity = env.scene[asset_cfg.name]
  jids = _affected_leg_joint_ids(env, leg_name, asset_cfg)
  jids_t = torch.as_tensor(jids, device=env.device, dtype=torch.long)

  joint_pos = asset.data.joint_pos[:, jids_t]  # [B, 3]
  default_pos = asset.data.default_joint_pos[:, jids_t]  # [B, 3]
  offset = torch.norm(joint_pos - default_pos, dim=-1)  # [B]

  return torch.clamp(1.0 - offset / (2.0 * joint_pos_tolerance), min=0.0, max=1.0)


def wire_recovery_success_bonus(
  env: ManagerBasedRlEnv,
  termination_term_name: str = "wire_recovery_success",
) -> torch.Tensor:
  """Sparse, ONE-TIME event bonus: 1.0 exactly on the step the
  ``wire_recovery_success`` termination's persistence-gated state machine
  transitions to success (contact_seen -> sustained release for
  ``required_consecutive_steps`` -> upright), else 0.0.

  This does NOT independently recompute "released + upright" (an earlier
  version did, and a no-PPO reward audit showed that let the +5 bonus fire
  on nearly every step with a transient one-step contact gap -- ~94% of
  steps in a 300-step random-action rollout -- making it indistinguishable
  from an idle/waiting reward). Instead this reads
  ``last_newly_succeeded`` directly off the SAME stateful termination
  instance used by `mjlab.tasks.recovery.mdp.terminations.wire_recovery_success`,
  guaranteeing the reward and termination can never disagree about what
  counts as "success" and that the bonus is a true one-shot event per
  episode (identical latch, identical persistence gate).

  Termination is always computed before rewards within `env.step()`, so
  `last_newly_succeeded` is guaranteed fresh for the current step.

  Returns:
    Reward tensor of shape [B] in {0.0, 1.0}, nonzero only on the exact
    step of genuine (persistence-gated) recovery success.
  """
  term_mgr = env.termination_manager
  term_idx = term_mgr.active_terms.index(termination_term_name)
  term_cfg = term_mgr._term_cfgs[term_idx]
  wire_term = term_cfg.func  # the wire_recovery_success ManagerTermBase instance
  return wire_term.last_newly_succeeded.float()


# ---------------------------------------------------------------------------
# Residual-recovery-task rewards (FSM-gated ResidualRecoveryAction, see
# mjlab.tasks.recovery.mdp.actions). Simple initial reward set per spec:
# forward velocity tracking + wire-clearance progress + stability + sparse
# success bonus, plus minimal residual-specific penalties (magnitude,
# affected-leg joint velocity, action rate) -- NOT the rod/wire oracle
# task's clearance-plane or return-to-stance shaping (those assume a fixed
# geometric target, inapplicable here since the walking policy itself
# already drives most of the body).
# ---------------------------------------------------------------------------


def _get_action_term(env: ManagerBasedRlEnv, action_term_name: str):
  return env.action_manager.get_term(action_term_name)


def forward_velocity_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  action_term_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exponential tracking reward for base linear velocity (xy) vs. the
  commanded twist -- same style as the velocity task's own
  `track_linear_velocity`, reused here since the walking policy itself is
  still what drives locomotion; the recovery policy's residual should not
  come at the cost of forward progress whenever possible.

  GATED TO ZERO DURING RECOVERY (post-audit, see chat history): this term
  was found to contribute ~0.97/step UNCONDITIONALLY, dwarfing the
  geometric progress signal by ~5-8x during the exact window where the
  policy needs to learn the lift/advance maneuver (which may legitimately
  require temporarily reducing forward speed). Active during NORMAL_WALK
  and POST_RECOVERY (restoring the normal walking objective once the
  maneuver concludes), zero during RECOVERY only.

  Returns:
    Reward tensor of shape [B] in [0, 1] (0 during RECOVERY).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  err = torch.sum((command[:, :2] - lin_vel_b) ** 2, dim=-1)
  reward = torch.exp(-err / 0.25)
  term = _get_action_term(env, action_term_name)
  active = term.phase != PHASE_RECOVERY
  return torch.where(active, reward, torch.zeros_like(reward))


def residual_magnitude_penalty(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """L2 penalty on the recovery policy's raw residual magnitude (as
  emitted, pre-gate) -- discourages unnecessarily large corrections.

  Returns:
    Penalty tensor of shape [B] (always >= 0).
  """
  term = _get_action_term(env, action_term_name)
  return torch.sum(term.raw_action ** 2, dim=-1)


def affected_leg_joint_vel_penalty(
  env: ManagerBasedRlEnv,
  leg_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L2 penalty on the affected leg's own joint velocities -- discourages
  wild flailing while still allowing the residual to move the leg
  purposefully.

  Returns:
    Penalty tensor of shape [B] (always >= 0).
  """
  asset: Entity = env.scene[asset_cfg.name]
  jids = _affected_leg_joint_ids(env, leg_name, asset_cfg)
  jids_t = torch.as_tensor(jids, device=env.device, dtype=torch.long)
  joint_vel = asset.data.joint_vel[:, jids_t]
  return torch.sum(joint_vel ** 2, dim=-1)


def residual_rate_penalty(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """L2 penalty on the change in raw residual action from the previous
  step (smoothness), mirroring `envs_mdp.action_rate_l2` but scoped to
  just the 3-dim residual rather than the full composite 12-dim action.

  Returns:
    Penalty tensor of shape [B] (always >= 0).
  """
  term = _get_action_term(env, action_term_name)
  prev = getattr(env, "_prev_residual_action", None)
  cur = term.raw_action
  if prev is None or prev.shape[0] != cur.shape[0]:
    prev = cur.clone()
  penalty = torch.sum((cur - prev) ** 2, dim=-1)
  env._prev_residual_action = cur.clone()
  return penalty


def recovery_phase_success_bonus(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """DEPRECATED (post audit, see chat history): this was the sole
  contact-release-based "success" bonus that let the policy learn a
  nudge-and-release exploit instead of genuinely clearing the wire. Kept
  only for backward compatibility / historical reference -- the residual
  recovery env no longer wires this into its active rewards dict. Use
  `wire_recovery_final_success_bonus` instead, which requires geometric
  crossing+clearance (not just contact release) AND a sustained
  post-recovery stability window.

  Returns:
    Reward tensor of shape [B] in {0.0, 1.0}.
  """
  term = _get_action_term(env, action_term_name)
  released_sustained = term._release_counter >= term.cfg.release_required_steps
  return (term.newly_recovered & released_sustained).float()


# ---------------------------------------------------------------------------
# Geometry-aware recovery rewards (post-audit fix). These replace
# `wire_contact_release_reward` / `recovery_phase_success_bonus` as the
# active recovery-task objective: they read `ResidualRecoveryAction`'s
# live wire-geometry state (`foot_height_above_wire`, `foot_forward_dist`,
# `foot_side_far`, `crossing_event`, `crossing_achieved`, `newly_succeeded`
# -- see mjlab.tasks.recovery.mdp.wire_geometry and the action term's FSM)
# so that "recovered" requires the affected foot to ACTUALLY go over the
# wire, not merely for the contact sensor to stop firing.
# ---------------------------------------------------------------------------


def wire_recovery_progress_reward(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Dense SHAPING reward active ONLY during RECOVERY, before calf
  clearance has been achieved. POTENTIAL-DIFFERENCE (delta) shaping,
  post-audit rework (see chat history): rather than the raw clamped
  absolute position (which had a large, constant negative offset at the
  start of every recovery episode, dwarfed by the always-on forward-
  velocity/uprightness rewards), this reads
  `ResidualRecoveryAction.stage1_progress_reward` -- the per-step
  NONNEGATIVE increase in a monotonic running-best potential
  Phi1 = norm(calf_height_above_wire) + norm(calf_forward_dist) (each
  clamped to [-1, 1]; see `WIRE_HEIGHT_NORM_SCALE`/
  `WIRE_FORWARD_NORM_SCALE` in actions.py). Monotonic tracking means
  oscillating back toward the baseline or cycling between two states
  earns exactly zero (never negative, never double-paid) -- see Part 5
  "no wiggle for reward" requirement. Already computed/gated by the
  action term itself (zero outside RECOVERY and zero once
  `calf_clearance_achieved`, since the action term only updates the
  stage-1 potential while that stage is active) -- this function does
  not duplicate any geometry computation.

  Returns:
    Reward tensor of shape [B], >= 0.
  """
  term = _get_action_term(env, action_term_name)
  return term.stage1_progress_reward


def wire_toe_crossing_progress_reward(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Dense SHAPING reward active ONLY once `calf_clearance_achieved` is
  True but `foot_crossed_achieved` is NOT yet True -- encourages the
  SECONDARY toe/foot confirmation stage (advancing the toe to the far
  side) once the primary calf-clearance condition has already been met.
  POTENTIAL-DIFFERENCE (delta) shaping, same scheme as
  `wire_recovery_progress_reward`: reads
  `ResidualRecoveryAction.stage2_progress_reward`, the per-step
  nonnegative increase in a monotonic running-best
  Phi2 = norm(foot_forward_dist) (clamped to [-1, 1]), re-baselined to
  the current value at the exact step calf-clearance latches (so stage 2
  never inherits a spurious head start from wherever the toe happened to
  already be). Zero before calf clearance (that's
  `wire_recovery_progress_reward`'s job) and zero once the toe has
  actually crossed (at that point `affected_leg_return_to_stance_reward_
  gated` takes over).

  Returns:
    Reward tensor of shape [B], >= 0.
  """
  term = _get_action_term(env, action_term_name)
  return term.stage2_progress_reward


def wire_crossing_event_bonus(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Sparse, ONE-TIME event bonus: 1.0 exactly on the step the FULL
  two-stage crossing condition completes -- calf clearance (above the
  wire, near->far) AND subsequent toe/foot far-side confirmation (see
  `ResidualRecoveryAction.crossing_event`). This is intentionally smaller
  than the final success bonus, and exists purely to shorten the
  credit-assignment horizon for PPO (the final success bonus can fire 15+
  steps later, after the post-recovery stability window) -- it does NOT
  by itself reward contact release or idle waiting, since `crossing_event`
  can only latch true given a real geometric calf clearance AND a real
  toe crossing.

  Returns:
    Reward tensor of shape [B] in {0.0, 1.0}.
  """
  term = _get_action_term(env, action_term_name)
  return term.crossing_event.float()


def affected_leg_return_to_stance_reward_gated(
  env: ManagerBasedRlEnv,
  leg_name: str,
  action_term_name: str = "joint_pos",
  joint_pos_tolerance: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Same shaping as `affected_leg_return_to_stance_reward`, but gated to
  only be active once `crossing_achieved` is True (the DESCEND / settle
  stage of the target maneuver, after the foot has genuinely crossed) --
  zero before that, so it cannot reward "settle the leg back down without
  ever crossing", which would otherwise re-introduce a close cousin of the
  original contact-release exploit.

  Returns:
    Reward tensor of shape [B] in [0, 1].
  """
  term = _get_action_term(env, action_term_name)
  base_reward = affected_leg_return_to_stance_reward(
    env, leg_name, joint_pos_tolerance, asset_cfg
  )
  return torch.where(term.crossing_achieved, base_reward, torch.zeros_like(base_reward))


def wire_recovery_final_success_bonus(
  env: ManagerBasedRlEnv,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """Sparse, ONE-TIME event bonus: 1.0 exactly on the step the FULL
  recovery sequence is confirmed -- genuine geometric crossing (above the
  wire, near->far) AND sustained contact release (gated to only count
  after that crossing) AND a subsequent `post_recovery_stable_steps`
  window of uninterrupted upright walking (see `ResidualRecoveryAction
  .newly_succeeded`, which is the single source of truth this reads
  directly off of -- reward/metric/FSM can never disagree about what
  counts as success).

  This is the SOLE replacement for the old contact-release-only
  `recovery_phase_success_bonus`.

  Returns:
    Reward tensor of shape [B] in {0.0, 1.0}.
  """
  term = _get_action_term(env, action_term_name)
  return term.newly_succeeded.float()

