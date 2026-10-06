"""Go2 cable-entanglement test environment.

Phase 3A test harness: verifies that the *existing trained* Go2 walking
policy (``Mjlab-Velocity-Flat-Unitree-Go2``) can physically encounter and
become constrained by a thin, ground-level cable while walking in +X.

This module does **not** modify the existing walking task, the Go2 robot
asset, or the existing recovery PPO task. It builds a new environment by
reusing ``unitree_go2_flat_env_cfg`` verbatim (same observations/actions, so
the trained policy's weights load and behave exactly as during training) and
then:

- Replaces nothing on the robot itself; adds a *separate*, static-anchored,
  articulated cable object to the scene via ``spec_fn``.
- Forces the velocity command to a constant +X walking command.
- Fixes the robot's reset position behind the cable so it walks into it.
- Adds contact sensors so cable/robot contact (especially the RL foot) can
  be measured explicitly, rather than inferred from visual proximity.

No recovery action/reward is added and no PPO training is performed here;
this is purely a simulation/instrumentation test harness (see
``scripts/cable_entanglement_test.py`` and ``tests/test_cable_entanglement_env.py``).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.cable_test.cable import CableCfg, build_cable_spec_fn
from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

# Rear-left (RL) hip is at approx x=-0.19, y=+0.05 in the Go2 body frame, so
# the +Y side of the robot corresponds to the *left* legs (FL, RL). Placing
# the cable only on the +Y side of the walking path means the right-side legs
# (FR, RR) never reach it, so the two rear legs can never be simultaneously
# trapped (requirement: never trap both rear legs at once).
FOOT_GEOM_NAMES = {
  "FL": "FL_foot_collision",
  "FR": "FR_foot_collision",
  "RL": "RL_foot_collision",
  "RR": "RR_foot_collision",
}

# Robot starts walking from here (world +X direction) toward the cable.
ROBOT_START_X = -1.2

# Cable height: raised off the floor so it isn't resting flush on the ground
# plane (which made entanglement hard to trigger since the foot would just
# slide over it). The cable still sags somewhat at its midpoint under
# gravity (it's a chain of free bodies pinned together, not a rigid rod), so
# the anchors are set higher than the desired ground clearance to compensate.
DEFAULT_CABLE_CFG = CableCfg(
  anchor_start=(0.0, 0.0, 0.10),
  anchor_end=(0.0, 0.5, 0.10),
  num_segments=10,
  radius=0.01,
)


def go2_cable_entanglement_test_env_cfg(
  play: bool = True,
  num_envs: int = 1,
  cable_cfg: CableCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the Go2 cable-entanglement test environment.

  Args:
    play: Kept for interface symmetry with other tasks; this environment is
      a test harness so it always behaves like "play" mode (no domain
      randomization surprises, fixed command). Both values behave the same.
    num_envs: Number of parallel environments. Because the cable is added as
      a single fixed-position object directly to the world spec (not
      replicated per-environment like scene entities), this environment is
      intended to be run with a small number of environments; ``num_envs=1``
      is required for the debug/visual-inspection mode.
    cable_cfg: Optional override for the cable geometry/placement.

  Returns:
    A ``ManagerBasedRlEnvCfg`` compatible with the existing trained Go2
    velocity policy (same observation/action layout as
    ``Mjlab-Velocity-Flat-Unitree-Go2``).
  """
  del play  # Test harness always runs in a deterministic, play-like mode.

  cable = cable_cfg or DEFAULT_CABLE_CFG

  # Start from the *existing* flat Go2 walking config, unmodified in source.
  # We only mutate the returned (deep-copyable) dataclass instance here, never
  # the original registered task config.
  cfg = unitree_go2_flat_env_cfg(play=True)

  cfg.scene.num_envs = num_envs

  # More collidable geoms (cable segments) than the base flat task, so bump
  # the contact/constraint buffers to avoid narrowphase overflow warnings.
  cfg.sim.nconmax = 64
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 400)

  # Step 3B stability pass: the articulated cable adds ~11 extra equality
  # constraints plus per-segment contacts on top of the walking task's
  # existing constraint set. Give the solver more iterations than the base
  # flat-terrain walking config (10) to converge on this stiffer system.
  cfg.sim.mujoco.iterations = 30

  # --- Add the test cable to the world. -------------------------------
  cfg.scene.spec_fn = build_cable_spec_fn(cable)

  # --- Force a constant +X walking command. ---------------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = (0.0, 0.0)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 1.0
  twist_cmd.rel_forward_envs = 1.0

  # --- Fixed deterministic reset: robot starts behind the cable. ------
  cfg.events["reset_base"] = EventTermCfg(
    func=cfg.events["reset_base"].func,
    mode="reset",
    params={
      "pose_range": {
        "x": (ROBOT_START_X, ROBOT_START_X),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      },
      "velocity_range": {},
    },
  )

  # --- Contact sensors: RL/FR/FL/RR foot vs. the cable. ---------------
  # `secondary` must resolve to a single element, so we use one sensor per
  # foot with the cable segments (multiple) as the `primary` side.
  cable_contact_sensors = []
  for leg, foot_geom in FOOT_GEOM_NAMES.items():
    cable_contact_sensors.append(
      ContactSensorCfg(
        name=f"cable_contact_{leg}",
        primary=ContactMatch(mode="geom", pattern=cable.geom_names, entity=None),
        secondary=ContactMatch(mode="geom", pattern=foot_geom, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
      )
    )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(cable_contact_sensors)

  # --- Keep the existing action/observation layout untouched. --------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  # Debug-friendly viewer defaults (single env visual inspection).
  cfg.viewer.distance = 1.8
  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 90.0

  # No recovery action, no recovery reward: rewards/observations/actions are
  # exactly those of the trained walking policy. This environment is for
  # inference-only testing, so a long episode is fine.
  cfg.episode_length_s = 15.0

  return cfg


def cable_geom_names(cable_cfg: CableCfg | None = None) -> tuple[str, ...]:
  """Convenience accessor for the cable segment geom names."""
  return (cable_cfg or DEFAULT_CABLE_CFG).geom_names
