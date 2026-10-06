"""Go2 entanglement recovery environment configuration -- WIRE variant.

Uses the validated ball-joint articulated wire
(:mod:`mjlab.tasks.cable_test.ball_cable`) as the entanglement obstacle,
instead of the rigid rod used by :mod:`mjlab.tasks.recovery.env_cfg`.

Architecture (unchanged from the rod version):

    Frozen walking policy -> normal walking -> FL contacts wire ->
    recovery policy activates -> recover FL -> walking can resume

The existing trained Go2 walking policy is NOT used or modified here (this
recovery task, like the rod version, learns the recovery policy from a
fixed post-contact reset snapshot -- it does not itself run the walking
policy). Recovery activation trigger for this first experiment is
ground-truth FL-wire contact (the reset snapshot below is taken at that
exact moment), not TCN detection.

Differences from the rod version, driven directly by the physical
diagnostic (see chat history / mjlab.tasks.cable_test test scripts) that
validated this wire against the existing trained Go2 walking policy:

- Obstacle is the compliant ball-joint wire, not a rigid capsule: contact
  is a traveling, intermittent series of touches rather than a single
  fixed penetration test, and the wire itself can be pulled taut (its tip
  pin equality residual/force measurably rises while the leg is in
  contact).
- Reward/termination "recovered" condition no longer uses the rod's fixed
  x-z geometric clearance plane (`affected_leg_clearance`,
  `recovery_success`/`recovery_success_bonus` in the rod's
  mdp.rewards/terminations). A flexible tether has no such fixed plane to
  clear. Instead, success requires: (1) FL-wire contact released, (2) FL
  joints back within tolerance of their default/nominal configuration,
  (3) robot upright, (4) that combined condition persisting for several
  consecutive steps (see `mdp.wire_recovery_success`) -- not merely
  instantaneous, since contact was observed to be intermittent even before
  "real" recovery.
- Reset snapshot (root pose + FL joint offsets) was measured directly
  against THIS wire (not re-used from the rod's snapshot), by driving the
  existing trained Go2 walking policy into the wire with the exact
  anti-tunneling simulation parameters validated in the diagnostic
  (timestep=0.0025, decimation=8): see CONTACT_SNAPSHOT_X/Y and
  FL_*_CONTACT_OFFSET below.
- Simulation timestep/decimation/nconmax/njmax/iterations match the
  validated diagnostic configuration (halved timestep + doubled
  decimation for anti-tunneling, larger contact/constraint buffers for
  the wire's many segments), not the rod's defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
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
from mjlab.tasks.cable_test.ball_cable import BallCableCfg, build_ball_cable_spec_fn
from mjlab.tasks.recovery import mdp
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

LEG_NAMES = ("FL", "FR", "RL", "RR")

# Wire geometry: identical to the validated diagnostic configuration (do
# not change without re-running that diagnostic against the existing
# trained walking policy -- see mjlab.tasks.cable_test.ball_cable).
DEFAULT_WIRE_CFG = BallCableCfg(
  anchor_start=(0.0, 0.0, 0.10),
  anchor_end=(0.0, 0.5, 0.10),
  radius=0.006,
)

# Root pose AND affected-leg (FL) joint configuration at the moment of
# first FL/wire contact, measured directly against THIS wire by driving
# the existing trained Go2 walking policy into it with the validated
# anti-tunneling sim parameters (see module docstring). Analogous to, but
# NOT copied from, the rod recovery env's CONTACT_SNAPSHOT_X/Y /
# FL_*_CONTACT_OFFSET (those were measured against the rigid rod at a
# different timestep/decimation and are not valid for this wire).
CONTACT_SNAPSHOT_X = -0.1802
CONTACT_SNAPSHOT_Y = -0.0344
FL_HIP_CONTACT_OFFSET = 0.0022
FL_THIGH_CONTACT_OFFSET = 0.0251
FL_CALF_CONTACT_OFFSET = -0.0108


def go2_wire_entanglement_recovery_env_cfg(
  affected_leg_idx: int = 0,
  affected_leg_joint: str = "FL_calf_joint",
  wire_cfg: BallCableCfg | None = None,
  num_envs: int = 1,
  env_spacing: float = 2.0,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Go2 wire-entanglement recovery environment configuration.

  Args:
    affected_leg_idx: Index of affected leg (0=FL, 1=FR, 2=RL, 3=RR).
      Default is FL, matching both the rod version and the observed
      first-contact leg for this wire.
    affected_leg_joint: Joint name of affected leg (e.g., "FL_calf_joint").
    wire_cfg: Optional override for the wire geometry/placement. Defaults
      to the validated configuration (see module docstring); changing this
      invalidates the reset snapshot below without re-calibration.
    num_envs: Number of parallel environments. Each gets its own
      independent wire copy at its own grid origin (same pattern as the
      rod recovery env).
    env_spacing: Distance (meters) between environment grid origins.
    play: If True, disable observation noise for clean playback.

  Returns:
    ManagerBasedRlEnvCfg configured for wire-based recovery training.
  """
  affected_leg_name = LEG_NAMES[affected_leg_idx]
  wire = wire_cfg or DEFAULT_WIRE_CFG

  ##
  # Observations (unchanged proprioceptive set + affected-leg oracle; NO
  # explicit wire position/contact-force observation, per the first
  # experiment's spec).
  ##

  lin_vel_noise = Unoise(n_min=-0.2, n_max=0.2)
  ang_vel_noise = Unoise(n_min=-0.1, n_max=0.1)
  gravity_noise = Unoise(n_min=-0.05, n_max=0.05)
  joint_pos_noise = Unoise(n_min=-0.01, n_max=0.01)
  joint_vel_noise = Unoise(n_min=-0.5, n_max=0.5)

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
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
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
  # Events: reset at the empirically-measured moment of first FL/wire
  # contact (see CONTACT_SNAPSHOT_X/Y above), with small jitter.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (CONTACT_SNAPSHOT_X - 0.02, CONTACT_SNAPSHOT_X + 0.02),
          "y": (CONTACT_SNAPSHOT_Y - 0.02, CONTACT_SNAPSHOT_Y + 0.02),
          "z": (0.0, 0.0),
          "yaw": (-0.05, 0.05),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("(?!FL_).*",)),
      },
    ),
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
  # Rewards: wire-interaction-based, NOT the rod's geometric clearance.
  #
  # `recovery_success_bonus` reuses the EXACT persistence-gated
  # `newly_succeeded` event from the `wire_recovery_success` termination's
  # state machine (contact_seen -> sustained release -> upright), so it
  # can only ever fire once per episode, in lockstep with that
  # termination -- see mdp.rewards.wire_recovery_success_bonus.
  #
  # `wire_contact_release` is a small DENSE shaping term only (weight
  # 0.2, deliberately << the sparse success bonus's weight 5.0): a reward
  # audit found the earlier weight (1.0) combined with an
  # over-eager instantaneous success bonus made ordinary intermittent
  # contact gaps pay out almost as much total reward as genuine recovery
  # success (+8.02 either way), i.e. "waiting" was reward-equivalent to
  # actually recovering. `affected_leg_return_to_stance` has been REMOVED
  # entirely for this first PPO sanity run: since the FL reset
  # configuration is already only ~0.002-0.025 rad from default, that
  # term let a policy collect nearly the maximum reward without doing any
  # recovery at all. The recovery objective here is freeing the affected
  # leg from the wire while remaining stable -- not returning immediately
  # to the nominal walking pose.
  ##

  rewards = {
    "wire_contact_release": RewardTermCfg(
      func=mdp.wire_contact_release_reward,
      weight=0.2,
      params={"sensor_name": "wire_contact"},
    ),
    "recovery_success_bonus": RewardTermCfg(
      func=mdp.wire_recovery_success_bonus,
      weight=5.0,
      params={"termination_term_name": "wire_recovery_success"},
    ),
    "uprightness": RewardTermCfg(func=mdp.uprightness_reward, weight=1.0),
    "body_stability": RewardTermCfg(func=mdp.body_stability_penalty, weight=-0.2),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_penalty, weight=-0.1),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
  }

  ##
  # Terminations
  ##

  terminations = {
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over, params={"limit_angle": math.radians(70.0)}
    ),
    "wire_recovery_success": TerminationTermCfg(
      func=mdp.wire_recovery_success,
      params={
        "leg_name": affected_leg_name,
        "limit_angle": math.radians(30.0),
        "required_consecutive_steps": 20,
        "sensor_name": "wire_contact",
        "joint_pos_sanity_tolerance": 0.6,
        "joint_vel_sanity_tolerance": 8.0,
      },
    ),
  }

  ##
  # Scene: wire (one copy per env) + contact sensor.
  ##

  all_wire_geom_names = tuple(
    name for env_id in range(num_envs) for name in wire.geom_names(env_id)
  )

  # Ground-truth FL-wire contact sensor. Primary = every wire segment
  # geom across ALL per-env wire copies; secondary = the affected leg's
  # calf collision geom (the geom confirmed by the diagnostic to make
  # first/repeated contact, not the foot-tip sphere). Physics isolation
  # between mjwarp worlds means env b's robot can only ever contact its
  # OWN wire copy, so `found`/`force` aggregated with `.any()`/reduction
  # over the full primary axis is already correctly scoped per env -- no
  # manual per-env slicing needed (same principle as the rod recovery
  # env's diagonal contact matrix, just aggregated via `.any()` here
  # instead of relying on a 1:1 diagonal).
  wire_contact_sensor = ContactSensorCfg(
    name="wire_contact",
    primary=ContactMatch(mode="geom", pattern=all_wire_geom_names, entity=None),
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
      spec_fn=build_ball_cable_spec_fn(wire, env_spacing=env_spacing),
      sensors=(wire_contact_sensor,),
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
      # Scaled per env: each env's wire has ~15-20 capsule segments, each
      # a potential contact/constraint source, plus the robot's own
      # self-contacts. Fixed rod-task-sized defaults (nconmax=128,
      # njmax=800) overflow once num_envs is more than a handful (empty
      # verified: 8 envs already required >244 active contacts). Per-env
      # allowances below were sized generously against that measurement.
      nconmax=max(128, 64 * num_envs),
      njmax=max(800, 128 * num_envs),
      mujoco=MujocoCfg(
        timestep=0.0025,
        iterations=30,
        ls_iterations=20,
      ),
    ),
    decimation=8,
    episode_length_s=10.0,
  )
