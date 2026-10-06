"""TEMPORARY diagnostic env config: Go2 walking policy vs. a rigid rod.

CONTROL experiment only -- do not use for training or as a replacement for
the real cable-entanglement test environment. This module does not import
or mutate ``env_cfg.py``/``cable.py``; it independently re-derives an
equivalent environment from the same unmodified base
(``unitree_go2_flat_env_cfg``) used by ``env_cfg.py``, so it keeps:

- the exact same trained Go2 walking policy observation/action layout
- the same walking command (+0.8 m/s forward)
- the same fixed robot reset (starts behind the obstacle)
- the same simulation timestep/decimation (inherited untouched from the
  base flat-terrain config)
- the same terrain
- the same viewer/camera settings

...but replaces the articulated 10-segment cable with a single rigid
capsule "rod" (see ``rod_diagnostic.py``): same center position,
orientation, total length, radius (0.01 m), center height (0.10 m), and
collision masks as the cable, but no equality constraints, no free-jointed
segments, and no cable dynamics.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.cable_test.env_cfg import ROBOT_START_X
from mjlab.tasks.cable_test.rod_diagnostic import RodCfg, build_rod_spec_fn
from mjlab.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

# Same center height (0.10 m) / endpoints as env_cfg.py's DEFAULT_CABLE_CFG.
DEFAULT_ROD_CFG = RodCfg(
  anchor_start=(0.0, 0.0, 0.10),
  anchor_end=(0.0, 0.5, 0.10),
  radius=0.01,
)

# Regex matching every robot collision geom (feet, calves, trunk, etc.), so
# the diagnostic can report the *exact* robot geom that first touches the
# rod, not just the feet.
ROBOT_COLLISION_GEOM_PATTERN = r".*_collision"


def go2_rod_diagnostic_env_cfg(
  num_envs: int = 1,
  rod_cfg: RodCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the temporary Go2-vs-rigid-rod diagnostic environment.

  Mirrors ``go2_cable_entanglement_test_env_cfg`` in
  ``mjlab.tasks.cable_test.env_cfg`` as closely as possible (same base
  config, command, reset, viewer, episode length), swapping only the
  cable's ``spec_fn`` for the rigid rod's and using a general robot-vs-rod
  contact sensor instead of the cable's per-foot sensors.
  """
  rod = rod_cfg or DEFAULT_ROD_CFG

  # Start from the *existing* flat Go2 walking config, unmodified in source.
  cfg = unitree_go2_flat_env_cfg(play=True)

  cfg.scene.num_envs = num_envs

  # A single rigid body/geom needs far less contact/constraint headroom than
  # the articulated cable, but keep the same solver iteration bump as the
  # cable test for a fair, apples-to-apples control comparison.
  cfg.sim.nconmax = 64
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 400)
  cfg.sim.mujoco.iterations = 30

  # --- Add the rigid rod to the world (in place of the cable). ---------
  cfg.scene.spec_fn = build_rod_spec_fn(rod)

  # --- Same constant +X walking command as the cable test. -------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = (0.0, 0.0)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 1.0
  twist_cmd.rel_forward_envs = 1.0

  # --- Same fixed deterministic reset as the cable test. ----------------
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

  # --- General robot-geom vs. rod contact sensor. -----------------------
  # One primary per matched robot collision geom, so `found`/`force` are
  # reported per-geom and the exact contacting geom can be identified from
  # `ContactSensor.primary_names`.
  rod_contact_sensor = ContactSensorCfg(
    name="rod_contact",
    primary=ContactMatch(
      mode="geom", pattern=ROBOT_COLLISION_GEOM_PATTERN, entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern=rod.geom_names[0], entity=None),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (rod_contact_sensor,)

  # --- Keep the existing action/observation layout untouched. ----------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  # Same viewer/camera defaults as the cable test.
  cfg.viewer.distance = 1.8
  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 90.0

  cfg.episode_length_s = 15.0

  return cfg


def rod_geom_names(rod_cfg: RodCfg | None = None) -> tuple[str, ...]:
  """Convenience accessor for the rod's geom name(s)."""
  return (rod_cfg or DEFAULT_ROD_CFG).geom_names
