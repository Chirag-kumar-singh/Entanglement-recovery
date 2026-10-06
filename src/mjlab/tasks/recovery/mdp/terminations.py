"""Termination and success conditions for recovery task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fell_over(
  env: ManagerBasedRlEnv,
  limit_angle: float = math.radians(70.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if robot tilts beyond limit_angle from upright."""

  asset: Entity = env.scene[asset_cfg.name]

  # Gravity projected into the robot's body frame.
  # For an upright robot this is approximately [0, 0, -1].
  gravity_b = asset.data.projected_gravity_b

  # The z-component tells us how much gravity points along
  # the robot's vertical body axis.
  #
  # Upright: gravity_b.z ~= -1
  # 70-degree tilt: gravity_b.z ~= -cos(70°)
  #
  # We use the absolute value because we only care about
  # deviation from the upright orientation.
  cos_angle = torch.abs(gravity_b[:, 2])

  # cos(theta) decreases as the robot tilts.
  cos_threshold = math.cos(limit_angle)

  return cos_angle < cos_threshold


def recovery_timeout(
  env: ManagerBasedRlEnv,
  time_out: bool = False,
) -> torch.Tensor:
  """Terminate when episode time limit is reached (handled by environment).

  This function is mostly for consistency with MDP framework.

  Args:
    env: The RL environment.
    time_out: If True, this is a timeout termination (for truncation vs. termination).

  Returns:
    Boolean tensor of shape [B]. True = timeout.
  """
  # Handled by the environment episode length
  return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)


def recovery_success(
  env: ManagerBasedRlEnv,
  rod_pos_xz: tuple[float, float] | torch.Tensor,
  rod_radius: float,
  leg_name: str,
  clearance_threshold: float = 0.15,
  limit_angle: float = math.radians(30.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Check if the AFFECTED leg has successfully cleared the rod while the
  robot remains upright.

  Success = the affected leg's foot is > (rod_radius + clearance_threshold)
  away from the rod's axis (measured in the x-z plane perpendicular to the
  rod, which runs along world Y) AND the robot's tilt from upright is
  within `limit_angle`. This is deliberately NOT based on root/body
  distance from the obstacle.

  Args:
    env: The RL environment.
    rod_pos_xz: Rod axis position as (x, z) tuple or tensor (rod runs
      along world Y at this fixed x, z).
    rod_radius: Rod radius in meters.
    leg_name: Foot site name of the affected leg (e.g., "FL").
    clearance_threshold: Minimum distance beyond the rod radius to count
      as cleared (meters).
    limit_angle: Maximum tilt from upright (radians) still counted as
      "remains upright" for success purposes.
    asset_cfg: Scene entity config for the robot.

  Returns:
    Boolean tensor of shape [B]. True = successfully cleared and upright.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if isinstance(rod_pos_xz, (tuple, list)):
    rod_xz = torch.tensor(rod_pos_xz, device=env.device, dtype=torch.float32)
  else:
    rod_xz = rod_pos_xz.to(env.device)

  # Per-env grid x-offset, matching the per-env rod replication in
  # env_cfg.py (each environment's rod copy and reset pose share the same
  # grid origin; z is never offset by the grid layout).
  env_origin_x = env.scene.env_origins[:, 0]  # [B]
  rod_xz_per_env = rod_xz.unsqueeze(0).expand(env.num_envs, -1).clone()  # [B, 2]
  rod_xz_per_env[:, 0] = rod_xz_per_env[:, 0] + env_origin_x

  site_names = list(asset.site_names)
  idx = site_names.index(leg_name)
  foot_pos = asset.data.site_pos_w[:, idx, :]  # [B, 3]
  foot_xz = foot_pos[:, [0, 2]]
  dist = torch.norm(foot_xz - rod_xz_per_env, dim=-1)  # [B]
  cleared = (dist - rod_radius) > clearance_threshold

  gravity_b = asset.data.projected_gravity_b
  cos_angle = torch.abs(gravity_b[:, 2])
  upright = cos_angle > math.cos(limit_angle)

  return cleared & upright


# ---------------------------------------------------------------------------
# Wire-interaction success termination (ball-joint articulated wire, NOT the
# rigid rod). See mjlab.tasks.recovery.mdp.rewards for the reward-side
# mirror and rationale: success is defined from the actual wire
# interaction (contact released + affected leg back in a safe/normal joint
# configuration + upright), NOT a fixed geometric clearance plane, and must
# additionally persist for several consecutive steps before counting as a
# real termination (a single noisy instantaneous release/re-contact
# shouldn't end the episode).
# ---------------------------------------------------------------------------


def _affected_leg_joint_ids(
  env: ManagerBasedRlEnv, leg_name: str, asset_cfg: SceneEntityCfg
) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  suffixes = ("hip_joint", "thigh_joint", "calf_joint")
  names = [f"{leg_name}_{s}" for s in suffixes]
  return [joint_names.index(n) for n in names if n in joint_names]


class wire_recovery_success(ManagerTermBase):
  """Terminate (success) via an explicit contact -> release state machine,
  NOT a fixed joint-pose tolerance.

  Rationale: an earlier version gated success primarily on the affected
  leg's joints being within a fixed ``joint_pos_tolerance`` (0.3 rad) of
  their default pose. For this wire, the measured contact-reset joint
  offset from default is tiny (~0.002-0.025 rad) and FL-wire contact is
  intermittent by nature (see the wire diagnostic in
  ``mjlab.tasks.cable_test``), so pure random noise satisfies
  "no contact + near-default joints" almost immediately -- confirmed by a
  no-PPO validation run where success fired within ~10 steps under random
  actions. A joint-pose tolerance is therefore, at best, a secondary
  sanity check here, never the primary success signal.

  State machine (per environment), advanced once per environment step:

  1. ``contact_seen``: a one-way latch that becomes ``True`` the first
     time genuine FL-wire contact is observed after the last reset. Success
     is impossible before this is ``True`` -- there is nothing to
     "recover" from otherwise.
  2. ``release_counter``: counts consecutive steps with NO FL-wire contact
     since ``contact_seen`` became ``True``. Reset to 0 immediately
     whenever contact reappears (so a single flicker of contact restarts
     the count -- release must be sustained, not just eventually true).
  3. Success requires: ``contact_seen`` AND
     ``release_counter >= required_consecutive_steps`` AND upright AND
     (optionally) the secondary joint-pose sanity check AND the success
     has not already fired this episode (``succeeded`` latch, so the
     termination can only trigger once per episode even though the
     underlying instantaneous condition may remain true across many
     subsequent steps).

  All three pieces of state (``contact_seen``, ``release_counter``,
  ``succeeded``) are reset to their initial values on episode reset via
  :meth:`reset`.
  """

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self.contact_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.release_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self.succeeded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Exact newly_succeeded event from the most recent __call__, exposed so
    # the reward-side bonus (mdp.rewards.recovery_success_bonus) can reuse
    # the IDENTICAL one-shot success definition instead of independently
    # recomputing "released + upright", which previously let the reward
    # fire on any transient one-step contact gap. Termination is always
    # computed before rewards within env.step(), so this is guaranteed to
    # be fresh by the time the reward reads it on the same step.
    self.last_newly_succeeded = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self.contact_seen.zero_()
      self.release_counter.zero_()
      self.succeeded.zero_()
      self.last_newly_succeeded.zero_()
    else:
      self.contact_seen[env_ids] = False
      self.release_counter[env_ids] = 0
      self.succeeded[env_ids] = False
      self.last_newly_succeeded[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    leg_name: str,
    required_consecutive_steps: int = 20,
    limit_angle: float = math.radians(30.0),
    sensor_name: str = "wire_contact",
    joint_pos_sanity_tolerance: float | None = 0.6,
    joint_vel_sanity_tolerance: float | None = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # [B, P]
    in_contact = (found > 0).any(dim=-1)  # [B]

    # 1. contact_seen: one-way latch, updated first so it can go true and
    #    be used within the SAME step that contact first appears.
    self.contact_seen = self.contact_seen | in_contact

    # 2. release_counter: reset on any contact, else increment (but only
    #    once contact has actually been seen -- before that it stays 0).
    self.release_counter = torch.where(
      in_contact,
      torch.zeros_like(self.release_counter),
      torch.where(
        self.contact_seen,
        self.release_counter + 1,
        self.release_counter,
      ),
    )

    released_sustained = self.release_counter >= required_consecutive_steps

    gravity_b = asset.data.projected_gravity_b
    cos_angle = torch.abs(gravity_b[:, 2])
    upright = cos_angle > math.cos(limit_angle)

    # 5. Optional secondary sanity check: NOT the primary criterion. Uses
    # a much looser threshold than the rod task's, since the recovery
    # policy may legitimately move the affected leg far from its default
    # walking pose while resolving the entanglement -- this only guards
    # against a degenerate "released" state where the leg is still wildly
    # off (e.g. flailing) even though it is no longer touching the wire.
    if joint_pos_sanity_tolerance is not None or joint_vel_sanity_tolerance is not None:
      jids = _affected_leg_joint_ids(env, leg_name, asset_cfg)
      jids_t = torch.as_tensor(jids, device=env.device, dtype=torch.long)
      sanity = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
      if joint_pos_sanity_tolerance is not None:
        joint_pos = asset.data.joint_pos[:, jids_t]
        default_pos = asset.data.default_joint_pos[:, jids_t]
        sanity = sanity & (
          torch.norm(joint_pos - default_pos, dim=-1) < joint_pos_sanity_tolerance
        )
      if joint_vel_sanity_tolerance is not None:
        joint_vel = asset.data.joint_vel[:, jids_t]
        sanity = sanity & (
          torch.norm(joint_vel, dim=-1) < joint_vel_sanity_tolerance
        )
    else:
      sanity = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    instantaneous_success = (
      self.contact_seen & released_sustained & upright & sanity
    )

    # 7. Latch: only ever fire once per episode. Once succeeded, stays
    # True (and hence keeps returning True as a termination) but the
    # CALLER (env.step) only acts on the rising edge via termination_manager,
    # so this simply guarantees a stable one-shot signal that does not
    # "un-fire" on a later transient contact re-appearance.
    newly_succeeded = instantaneous_success & ~self.succeeded
    self.succeeded = self.succeeded | instantaneous_success
    self.last_newly_succeeded = newly_succeeded
    return newly_succeeded
