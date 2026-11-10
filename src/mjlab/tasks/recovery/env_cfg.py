"""Go2 entanglement recovery environment configuration.

This task trains a recovery policy for a Go2 robot entangled in a static obstacle.
The affected leg is provided as an oracle observation.
"""

from __future__ import annotations

import math
import mujoco
import torch
from dataclasses import dataclass, field

from mjlab.asset_zoo.robots import get_go2_robot_cfg, GO2_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.recovery import mdp
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Leg name lookup, matching the Go2 asset's site/joint naming convention
# (see mjlab.tasks.cable_test.instrumentation.LEG_NAMES) and the verified
# rod control experiment (mjlab.tasks.cable_test.rod_diagnostic).
LEG_NAMES = ("FL", "FR", "RL", "RR")

# Rigid-rod geometry, copied verbatim from the verified rod control
# experiment (mjlab.tasks.cable_test.env_cfg_rod_diagnostic.DEFAULT_ROD_CFG):
# same radius, same center height, same collision masks. Only the rod's
# world position differs here (see ROD_ANCHOR_START/END below), chosen to
# match the exact FL-rod contact snapshot observed when the existing
# trained Go2 walking policy was driven into this rod (see
# scripts/rod_diagnostic_test.py); first contact geom was
# FL_calf2_collision, no other leg made contact during the run.
ROD_RADIUS = 0.01
ROD_CENTER_HEIGHT = 0.10
ROD_CONTYPE = 1
ROD_CONAFFINITY = 1
# Same rod length/orientation (spans world Y) as the control experiment.
ROD_ANCHOR_START: tuple[float, float, float] = (0.0, 0.0, ROD_CENTER_HEIGHT)
ROD_ANCHOR_END: tuple[float, float, float] = (0.0, 0.5, ROD_CENTER_HEIGHT)

# Robot root pose AND affected-leg (FL) joint configuration at the moment of
# first FL/rod contact, measured directly by driving the existing trained
# Go2 walking policy into this exact rod placement (see
# scripts/rod_diagnostic_test.py). A default *standing* joint pose does
# NOT reproduce contact at this root position -- contact happens mid-swing,
# with FL's thigh/calf away from their standing angles -- so the FL joints
# are also biased at reset, not just the root pose.
CONTACT_SNAPSHOT_X = -0.144
CONTACT_SNAPSHOT_Y = -0.0085
# (measured default -> contact offsets: hip +0.0056, thigh +0.1159, calf -0.0224)
FL_HIP_CONTACT_OFFSET = 0.006
FL_THIGH_CONTACT_OFFSET = 0.116
FL_CALF_CONTACT_OFFSET = -0.022


@dataclass
class EntanglementObstacleCfg:
  """Configuration for the entanglement obstacle (rigid rod)."""

  # Obstacle position in world frame [x, y, z]. Defaults match the rigid
  # rod used in the verified control experiment.
  position: tuple[float, float, float] = ROD_ANCHOR_START

  # Obstacle radius (meters). Same as the verified rod control experiment.
  radius: float = ROD_RADIUS

  # Which leg is affected: 0=FL, 1=FR, 2=RL, 3=RR
  affected_leg_idx: int = 0  # Default: FL (front-left) -- the leg observed
  # to naturally make first contact when the existing trained Go2 walking
  # policy walks into this rod placement.

  # Affected leg calf joint name
  affected_leg_joint: str = "FL_calf_joint"

  # Distance threshold for successful clearance (meters)
  clearance_threshold: float = 0.15


def go2_entanglement_recovery_env_cfg(
  affected_leg_idx: int = 0,
  affected_leg_joint: str = "FL_calf_joint",
  obstacle_pos: tuple[float, float, float] = ROD_ANCHOR_START,
  obstacle_radius: float = ROD_RADIUS,
  num_envs: int = 1,
  env_spacing: float = 2.0,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Go2 entanglement recovery environment configuration.

  First-version scenario: the naturally-occurring FL/rod interaction
  observed when the existing trained Go2 walking policy is driven into a
  rigid rod placed across its walking path (see
  scripts/rod_diagnostic_test.py). No RL/RR-specific geometry is used;
  the rod is the same straight rigid capsule as the verified control
  experiment, and episodes start at the empirically-measured moment of
  first FL contact rather than re-simulating the walk-in.

  Supports ``num_envs > 1``: since mjlab/mujoco_warp simulate one shared
  compiled model batched over ``nworld = num_envs`` independent physics
  worlds, a per-env rigid rod is achieved by adding ``num_envs`` separate
  static rod bodies to the spec, each translated by that environment's
  grid origin (identical grid layout to mjlab's own "plane" terrain,
  see ``mjlab.terrains.terrain_entity.TerrainEntity._compute_env_origins_grid``).
  Each simulated world only ever registers physical contact with the one
  rod copy near its own grid cell -- the other copies are placed
  ``env_spacing`` meters away and are never within contact range.

  Args:
    affected_leg_idx: Index of affected leg (0=FL, 1=FR, 2=RL, 3=RR).
      Default is FL, matching the naturally observed first contact.
    affected_leg_joint: Joint name regex of affected leg (e.g., "FL_calf_joint").
    obstacle_pos: [x, y, z] position of the rod's start anchor in world frame
      (env-local, i.e. relative to each environment's own grid origin).
    obstacle_radius: Rod radius in meters (same as the control experiment: 0.01).
    num_envs: Number of parallel environments. Each gets its own
      independent rigid rod copy (same geometry/collision masks) at its
      own grid origin.
    env_spacing: Distance (meters) between environment grid origins. Must
      be large enough that adjacent rod copies (rod length ~0.5 m) and
      robots never physically overlap; default 2.0 m has ample margin.
    play: If True, disable noise/randomization for clean playback.

  Returns:
    ManagerBasedRlEnvCfg configured for recovery task.
  """
  affected_leg_name = LEG_NAMES[affected_leg_idx]

  ##
  # Observations
  ##

  # Noise config for training
  lin_vel_noise = Unoise(n_min=-0.2, n_max=0.2)
  ang_vel_noise = Unoise(n_min=-0.1, n_max=0.1)
  gravity_noise = Unoise(n_min=-0.05, n_max=0.05)
  joint_pos_noise = Unoise(n_min=-0.01, n_max=0.01)
  joint_vel_noise = Unoise(n_min=-0.5, n_max=0.5)

  # Actor receives noisy proprioceptive observations + oracle affected leg
  actor_terms = {
    "affected_leg": ObservationTermCfg(
      func=mdp.affected_leg_oracle,
      params={"affected_leg_idx": affected_leg_idx},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=lin_vel_noise if not play else None,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=ang_vel_noise if not play else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=gravity_noise if not play else None,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"biased": True},
      noise=joint_pos_noise if not play else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=joint_vel_noise if not play else None,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  # Critic gets clean proprioceptive observations + oracle
  critic_terms = {
    "affected_leg": ObservationTermCfg(
      func=mdp.affected_leg_oracle,
      params={"affected_leg_idx": affected_leg_idx},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO2_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  # Reset near the empirically-measured moment of first FL/rod contact
  # (see CONTACT_SNAPSHOT_X/Y above), with small jitter, rather than a
  # near-origin pose disconnected from the rod. This reproduces the
  # naturally-occurring walk-in contact state verified via
  # scripts/rod_diagnostic_test.py without re-simulating the walk-in
  # itself every episode.
  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (CONTACT_SNAPSHOT_X - 0.02, CONTACT_SNAPSHOT_X + 0.02),
          "y": (CONTACT_SNAPSHOT_Y - 0.02, CONTACT_SNAPSHOT_Y + 0.02),
          "z": (0.0, 0.0),  # Fixed height
          "yaw": (-0.05, 0.05),  # Small yaw variation
        },
        "velocity_range": {},  # No initial velocity
      },
    ),
    # All non-FL joints: reset to default stance (no offset).
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("(?!FL_).*",)),
      },
    ),
    # FL joints: biased toward the empirically-measured mid-swing
    # configuration at the moment of first FL/rod contact (see above),
    # with small jitter, instead of default standing angles. This is what
    # makes FL/rod contact happen deterministically at reset rather than
    # only if the policy happens to swing FL forward on its own.
    "reset_fl_hip": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (
          FL_HIP_CONTACT_OFFSET - 0.008,
          FL_HIP_CONTACT_OFFSET + 0.008,
        ),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("FL_hip_joint",)),
      },
    ),
    "reset_fl_thigh": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (
          FL_THIGH_CONTACT_OFFSET - 0.012,
          FL_THIGH_CONTACT_OFFSET + 0.012,
        ),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("FL_thigh_joint",)),
      },
    ),
    "reset_fl_calf": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (
          FL_CALF_CONTACT_OFFSET - 0.008,
          FL_CALF_CONTACT_OFFSET + 0.008,
        ),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("FL_calf_joint",)),
      },
    ),
  }

  ##
  # Rewards
  ##

  # Rod position as an (x, z) pair in the plane perpendicular to its axis
  # (the rod runs along world Y, so clearance is measured in x-z).
  rod_pos_xz = (obstacle_pos[0], obstacle_pos[2])

  rewards = {
    # Primary shaping term: reward the AFFECTED leg's foot for increasing
    # its perpendicular (x-z) distance from the rod's axis, saturating
    # once clear by `clearance_threshold` beyond the rod radius. This is
    # deliberately NOT based on root/body distance from the rod.
    "affected_leg_clearance": RewardTermCfg(
      func=mdp.affected_leg_clearance,
      weight=2.0,
      params={
        "rod_pos_xz": rod_pos_xz,
        "rod_radius": obstacle_radius,
        "leg_name": affected_leg_name,
        "clearance_margin": 0.15,
      },
    ),
    # Sparse bonus while the affected leg is fully clear AND the robot is
    # upright -- i.e. "successful clearing", not just partial progress.
    "recovery_success_bonus": RewardTermCfg(
      func=mdp.recovery_success_bonus,
      weight=5.0,
      params={
        "rod_pos_xz": rod_pos_xz,
        "rod_radius": obstacle_radius,
        "leg_name": affected_leg_name,
        "clearance_threshold": 0.15,
        "limit_angle": math.radians(30.0),
      },
    ),
    "uprightness": RewardTermCfg(
      func=mdp.uprightness_reward,
      weight=1.0,
    ),
    "body_stability": RewardTermCfg(
      func=mdp.body_stability_penalty,
      weight=-0.2,
    ),
    "joint_vel": RewardTermCfg(
      func=mdp.joint_vel_penalty,
      weight=-0.1,
    ),
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.05,
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"limit_angle": math.radians(70.0)},
    ),
    "recovery_success": TerminationTermCfg(
      func=mdp.recovery_success,
      params={
        "rod_pos_xz": rod_pos_xz,
        "rod_radius": obstacle_radius,
        "leg_name": affected_leg_name,
        "clearance_threshold": 0.15,
        "limit_angle": math.radians(30.0),
      },
    ),
  }

  ##
  # Environment config
  ##

  rod_geom_names = tuple(f"recovery_rod_{i}" for i in range(num_envs))
  rod_anchor_end = (obstacle_pos[0], obstacle_pos[1] + 0.5, obstacle_pos[2])

  def _compute_grid_env_origins_xy(n: int, spacing: float) -> list[tuple[float, float]]:
    """Pure-Python reimplementation of
    ``TerrainEntity._compute_env_origins_grid``'s (x, y) layout for
    ``terrain_type="plane"``, so the rod copies land exactly on the same
    grid cells that ``reset_root_state_uniform`` (via
    ``env.scene.env_origins``) will place each environment's robot on.
    Reimplemented locally (no torch dependency, no Scene/TerrainEntity
    instance available yet at spec-build time) but numerically identical.
    """
    num_rows = math.ceil(n / int(math.isqrt(n)) if n > 0 else 1)
    num_cols = math.ceil(n / num_rows)
    origins = []
    for idx in range(n):
      ii = idx // num_cols
      jj = idx % num_cols
      ox = -(ii - (num_rows - 1) / 2) * spacing
      oy = (jj - (num_cols - 1) / 2) * spacing
      origins.append((ox, oy))
    return origins

  def _add_rigid_rod(spec: mujoco.MjSpec, _num_envs: int) -> None:
    """Add ``num_envs`` independent copies of the SAME rigid rod used in
    the verified control experiment, one per environment grid cell.

    Note: `_num_envs` (the scene's authoritative runtime env count) is
    intentionally unused here -- this oracle task is preserved exactly
    as-is (not in scope for the wire-sensor-construction fix); it still
    closes over the outer `num_envs` parameter captured at config-
    construction time, same as before.

    Each copy is a single static body/geom (no joint, no equality
    constraint, no cable dynamics): same radius, same center height, same
    collision masks as
    ``mjlab.tasks.cable_test.rod_diagnostic.build_rod_spec_fn``. Copies are
    ``env_spacing`` meters apart (matching the grid layout mjlab's own
    "plane" terrain uses for ``env_origins``), so each simulated world
    (independent nworld physics instance) only ever geometrically reaches
    its own rod copy.
    """
    half_vec = tuple((rod_anchor_end[k] - obstacle_pos[k]) / 2.0 for k in range(3))
    base_center = tuple(obstacle_pos[k] + half_vec[k] for k in range(3))
    fromto = (
      -half_vec[0],
      -half_vec[1],
      -half_vec[2],
      half_vec[0],
      half_vec[1],
      half_vec[2],
    )
    grid_origins = _compute_grid_env_origins_xy(num_envs, env_spacing)
    for env_id, (ox, oy) in enumerate(grid_origins):
      center = (base_center[0] + ox, base_center[1] + oy, base_center[2])
      body = spec.worldbody.add_body(name=f"recovery_rod_{env_id}", pos=center)
      body.add_geom(
        name=rod_geom_names[env_id],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(obstacle_radius, 0.0, 0.0),
        fromto=fromto,
        rgba=(0.1, 0.8, 0.95, 1.0),
        contype=ROD_CONTYPE,
        conaffinity=ROD_CONAFFINITY,
        mass=0.2,
      )

  # Ground-truth per-env robot-geom vs. rod contact sensor. Primary = all
  # replicated rod copies (P = num_envs); secondary = the affected leg's
  # calf collision geom -- the SAME geom confirmed by the verified rod
  # diagnostic to be the actual first-contact geom (FL_calf2_collision),
  # not the foot-tip sphere (whose site position is only used as a rough
  # clearance *proxy* for the reward, not the true contact geometry).
  # Because each simulated world only ever physically reaches its OWN rod
  # copy, ``found``/``force`` of shape [B=num_envs, P=num_envs] is
  # diagonal: only `found[b, b]` should ever be nonzero. Off-diagonal
  # entries being zero is the verification check for "no
  # cross-environment interaction".
  rod_contact_sensor = ContactSensorCfg(
    name="rod_contact",
    primary=ContactMatch(mode="geom", pattern=rod_geom_names, entity=None),
    secondary=ContactMatch(
      mode="geom", pattern=f"{affected_leg_name}_calf2_collision", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={"robot": get_go2_robot_cfg()},
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      env_spacing=env_spacing,
      extent=2.0,
      spec_fn=_add_rigid_rod,
      sensors=(rod_contact_sensor,),
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.5,
      elevation=-30.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,  # 10 second episodes
  )
