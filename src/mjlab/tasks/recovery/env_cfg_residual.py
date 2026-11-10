"""Go2 wire-entanglement recovery environment -- RESIDUAL action variant.

This is the RL-ready formulation built on top of the validated
ResidualRecoveryAction architecture (see
mjlab.tasks.recovery.mdp.actions): the frozen
Mjlab-Velocity-Flat-Unitree-Go2 walking policy drives the robot at all
times, and a small 3-dim PPO policy only ever controls a residual on the
affected leg's (FL) raw action, and ONLY during the FSM's RECOVERY phase
(see ResidualRecoveryActionCfg docstring for the full phase-transition
rules). Outside RECOVERY, PPO's output is computed but gated to zero
effect -- the frozen policy walks completely unperturbed.

Episode structure (NOT hardcoded step counts -- driven by real signals):
  NORMAL_WALK (pi_walk only) -> FL contacts wire (real contact sensor) ->
  RECOVERY (pi_walk + Delta_a on FL, up to max_recovery_steps) ->
  released (sustained) or cap reached -> POST_RECOVERY (pi_walk only,
  continues until episode time-out or fall).

This task reuses the SAME wire (BallCableCfg) and contact sensor pattern
validated in mjlab.tasks.recovery.env_cfg_wire's physical verification
(see chat history): default placement, anti-tunneling sim params
(timestep=0.0025, decimation=8), pinned walk-into-wire spawn pose.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import dataclasses

from mjlab.asset_zoo.robots import get_go2_robot_cfg
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_ACTUATOR_CALF,
  GO2_ACTUATOR_THIGH,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.cable_test.ball_cable import BallCableCfg, build_ball_cable_spec_fn
from mjlab.tasks.recovery import mdp
from mjlab.tasks.recovery.mdp.actions import ResidualRecoveryActionCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

LEG_NAMES = ("FL", "FR", "RL", "RR")

# Same wire geometry validated in the physical entanglement verification
# (sustained FL-wire contact runs of 37-95 consecutive steps across
# several nearby placements). Do not change without re-verifying.
DEFAULT_WIRE_CFG = BallCableCfg(
  anchor_start=(0.0, 0.0, 0.10),
  anchor_end=(0.0, 0.5, 0.10),
  radius=0.006,
)

# Deconfounded, single-leg-isolated variant of DEFAULT_WIRE_CFG. Measured
# (not guessed) from an actual dynamic hybrid-recovery run at seed=0,
# affected_leg_idx=0 (FL), steps 0-69, world frame, robot walking toward
# +X (FORWARD_CMD_VAL > 0 drives lin_vel_x):
#   FL foot y range during swing/contact window (steps 33-55): ~0.36-0.40
#   FR foot y range during the SAME window:                    ~0.11-0.17
# i.e. roughly a 0.19 m gap between the two legs' lateral corridors,
# straddled by DEFAULT_WIRE_CFG's full y in [0, 0.5] span -- which is why
# FR was also contacting the wire (confound). This variant shortens the
# wire so it only spans y in [0.30, 0.50]: comfortably covers FL's
# corridor (0.36-0.40) with 0.06 m margin on the near side and 0.10 m on
# the far side, while its nearest point (y=0.30) is a full 0.13-0.14 m
# clear of FR's observed max y (~0.17) -- FR should not be able to reach
# it even with some per-episode swing variance. Must be paired with
# `spawn_y_override=0.25` (the ORIGINAL DEFAULT_WIRE_CFG's midpoint) so
# the robot's spawn/body position -- and therefore both legs' absolute
# corridors -- is UNCHANGED from the baseline; only the wire's own
# physical extent is narrowed/shifted away from FR.
ISOLATED_FL_WIRE_CFG = BallCableCfg(
  anchor_start=(0.0, 0.30, 0.10),
  anchor_end=(0.0, 0.50, 0.10),
  radius=0.006,
)
# Spawn y-center to use alongside ISOLATED_FL_WIRE_CFG (see above).
ISOLATED_FL_WIRE_SPAWN_Y = 0.25

FORWARD_CMD_VAL = 0.3

# Default checkpoint for the frozen walking policy. Override via
# `walk_checkpoint` if training a new base policy.
DEFAULT_WALK_CHECKPOINT = "logs/rsl_rl/go2_velocity/2026-09-17_13-00-16/model_9900.pt"


def _build_recovery_robot_cfg(
  affected_leg_idx: int,
  recovery_mode: str,
  recovery_thigh_velocity_kd: float,
  recovery_calf_torque: float,
):
  """Build the recovery task's Go2 entity config.

  For ``recovery_mode in ("ppo_residual", "deterministic")`` this is just
  ``get_go2_robot_cfg()`` -- IDENTICAL to the normal walking task's robot,
  unmodified.

  For ``recovery_mode == "hybrid"`` (D3), TWO auxiliary actuators are
  added to the affected leg's thigh/calf joints ONLY, on a COPY of the
  base Go2 articulation (the normal walking task's entity, built
  separately by velocity/config/go2, is never touched):
    - a named velocity actuator on the thigh joint, used for the
      Quad-SDK-inspired damped-velocity retract command. Its damping is
      initialized to 0.0 (inert) -- it is turned on/off at runtime per
      env by ResidualRecoveryAction._apply_hybrid_actuators, never active
      during normal walking.
    - a named motor (pure feedforward torque) actuator on the calf
      joint, used for the Quad-SDK-inspired open-loop push torque.
      Motor actuators are inert whenever ctrl=0 (no velocity feedback
      term), so no additional on/off toggling is needed for this one.
  Both auxiliary actuators' effort_limit reuses the SAME bound as the
  existing walking actuator on that joint (GO2_ACTUATOR_THIGH/CALF's
  effort_limit), i.e. the recovery controller can never exceed the
  torque bound the joint's own normal-walking actuator already respects.
  """
  base = get_go2_robot_cfg()
  if recovery_mode != "hybrid":
    return base

  from mjlab.tasks.recovery.mdp.hybrid_actuators import (
    NamedMotorActuatorCfg,
    NamedVelocityActuatorCfg,
  )

  leg = LEG_NAMES[affected_leg_idx]
  thigh_joint = f"{leg}_thigh_joint"
  calf_joint = f"{leg}_calf_joint"

  aux_thigh_vel = NamedVelocityActuatorCfg(
    target_names_expr=(thigh_joint,),
    actuator_name=f"{thigh_joint}_recovery_vel",
    damping=0.0,  # inert by default; toggled on only during RECOVERY.
    effort_limit=GO2_ACTUATOR_THIGH.effort_limit,
  )
  aux_calf_motor = NamedMotorActuatorCfg(
    target_names_expr=(calf_joint,),
    actuator_name=f"{calf_joint}_recovery_motor",
    effort_limit=GO2_ACTUATOR_CALF.effort_limit,
  )

  assert base.articulation is not None
  new_actuators = base.articulation.actuators + (aux_thigh_vel, aux_calf_motor)
  new_articulation = dataclasses.replace(base.articulation, actuators=new_actuators)
  return dataclasses.replace(base, articulation=new_articulation)


def go2_wire_recovery_residual_env_cfg(
  affected_leg_idx: int = 0,
  walk_checkpoint: str = DEFAULT_WALK_CHECKPOINT,
  wire_cfg: BallCableCfg | None = None,
  num_envs: int = 1,
  env_spacing: float = 0.0,
  max_residual: float = 1.0,
  max_recovery_steps: int = 16,
  release_required_steps: int = 5,
  clearance_margin: float = 0.02,
  post_recovery_stable_steps: int = 15,
  play: bool = False,
  recovery_mode: str = "ppo_residual",
  thigh_recovery_delta: float = 0.3,
  calf_recovery_delta: float = -0.3,
  recovery_thigh_velocity: float = -2.0,
  recovery_thigh_velocity_kd: float = 2.0,
  recovery_calf_torque: float = 5.0,
  policy_device: str = "cuda:0",
  spawn_y_override: float | None = None,
  experimental_peak_triggered_thigh_stop: bool = False,
  peak_detector_rise_threshold: float = 0.01,
  peak_detector_decline_eps: float = 0.003,
  peak_detector_confirm_steps: int = 2,
  phase_aware_thresholds: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the residual-action wire-entanglement recovery environment.

  Args:
    affected_leg_idx: Index of affected leg (0=FL, 1=FR, 2=RL, 3=RR).
      FL-first per the staged implementation plan.
    walk_checkpoint: Path to the frozen walking policy checkpoint.
    wire_cfg: Optional override for the wire geometry/placement.
    num_envs: Number of parallel environments.
    env_spacing: Distance (meters) between environment grid origins. Each
      ``nworld`` world is an independent, non-interacting physics copy,
      so this only matters for native-viewer visual separation; keep at
      0.0 (default) so every world's robot resets to the same local
      (x, y), lining up with the single wire fixed at that same local
      position in every world copy.
    max_residual: Clamp on |Delta_a| in raw policy-action-space units.
      Only used when ``recovery_mode == "ppo_residual"``.
    max_recovery_steps: Hard cap on RECOVERY phase duration (steps).
    release_required_steps: Consecutive no-contact steps required to
      count as a genuine (non-cap-forced) release.
    play: If True, disable observation noise for clean playback.
    recovery_mode: ``"ppo_residual"`` (default, UNCHANGED prior behavior
      -- the env's own action input drives the affected leg's residual)
      or ``"deterministic"`` (D1: the env's action input is ignored for
      the affected leg; a fixed thigh/calf delta is applied instead, see
      ``ResidualRecoveryActionCfg`` docstring).
    thigh_recovery_delta: Constant thigh delta used only in
      ``"deterministic"`` mode. See ``ResidualRecoveryActionCfg``.
    calf_recovery_delta: Constant calf delta used only in
      ``"deterministic"`` mode. See ``ResidualRecoveryActionCfg``.
    policy_device: Device the frozen walking policy is loaded/run on.
      Exposed (rather than hardcoded) so D1 smoke tests can run on CPU
      without a GPU available.
    spawn_y_override: If set, overrides the spawn pose's lateral (y)
      centering, which otherwise defaults to the wire's own midpoint
      (``0.5 * (wire.anchor_start[1] + wire.anchor_end[1])``). This
      exists so a deliberately SHORT or laterally-OFFSET wire (e.g. one
      designed to intersect only one leg's swing corridor) does not
      also drag the robot's spawn position sideways with it -- the
      robot's spawn/body position and the wire's physical placement are
      independent degrees of freedom. Defaults to None, which preserves
      the original, validated wire-midpoint-centered spawn behavior
      exactly (no behavior change for any existing caller).
    experimental_peak_triggered_thigh_stop: Opt-in (default False,
      preserves baseline exactly) -- see `ResidualRecoveryActionCfg`
      docstring. Zeros ONLY the thigh velocity command once a genuine
      post-entry foot-height peak is detected; calf torque, ab/ad, and
      the FSM are unaffected.
    peak_detector_rise_threshold: See `ResidualRecoveryActionCfg`.
    peak_detector_decline_eps: See `ResidualRecoveryActionCfg`.
    peak_detector_confirm_steps: See `ResidualRecoveryActionCfg`.
    phase_aware_thresholds: D5.29 opt-in (default False, preserves
      baseline exactly) -- if True, enables leg-speed-dependent dynamic
      adjustment of crossing-latch thresholds (clearance_margin,
      foot_crossing_margin) based on affected leg velocity at recovery
      entry. D5.31 corrected the tier-to-margin direction (faster entry
      now gets smaller, not larger, margins, since both latch conditions
      are strict `>` comparisons). See `ResidualRecoveryActionCfg`
      docstring for the three-tier threshold logic.
  """
  affected_leg_name = LEG_NAMES[affected_leg_idx]
  wire = wire_cfg or DEFAULT_WIRE_CFG

  ##
  # Actions: single ResidualRecoveryAction term (3-dim, FL only).
  ##

  actions = {
    "joint_pos": ResidualRecoveryActionCfg(
      entity_name="robot",
      walk_checkpoint=walk_checkpoint,
      affected_leg_idx=affected_leg_idx,
      max_residual=max_residual,
      command_name="twist",
      device=policy_device,
      wire_contact_sensor_name="wire_contact",
      wire_geom_names=wire.geom_names(),
      wire_radius=wire.radius,
      clearance_margin=clearance_margin,
      max_recovery_steps=max_recovery_steps,
      release_required_steps=release_required_steps,
      post_recovery_stable_steps=post_recovery_stable_steps,
      recovery_mode=recovery_mode,
      thigh_recovery_delta=thigh_recovery_delta,
      calf_recovery_delta=calf_recovery_delta,
      recovery_thigh_velocity=recovery_thigh_velocity,
      recovery_thigh_velocity_kd=recovery_thigh_velocity_kd,
      recovery_calf_torque=recovery_calf_torque,
      experimental_peak_triggered_thigh_stop=experimental_peak_triggered_thigh_stop,
      peak_detector_rise_threshold=peak_detector_rise_threshold,
      peak_detector_decline_eps=peak_detector_decline_eps,
      peak_detector_confirm_steps=peak_detector_confirm_steps,
      phase_aware_thresholds=phase_aware_thresholds,
    )
  }

  ##
  # Commands: fixed forward velocity (no randomization for this first
  # experiment -- the FSM trigger/recovery dynamics are already
  # non-trivial; command randomization is deferred).

  ##

  commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),  # effectively never resample
      rel_standing_envs=0.0,
      heading_command=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(FORWARD_CMD_VAL, FORWARD_CMD_VAL),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
    )
  }

  ##
  # Observations: compact actor set per spec (one-hot affected leg,
  # affected-leg joint pos/vel, affected-foot pos/vel rel to body, base
  # lin/ang vel, projected gravity, affected-leg contact, recovery-phase
  # elapsed, previous residual, a_walk_affected). Critic gets privileged
  # info (full joint state, wire contact force/position proxy via the
  # contact sensor, base pose).
  ##

  lin_vel_noise = Unoise(n_min=-0.2, n_max=0.2)
  ang_vel_noise = Unoise(n_min=-0.1, n_max=0.1)
  gravity_noise = Unoise(n_min=-0.05, n_max=0.05)
  joint_pos_noise = Unoise(n_min=-0.01, n_max=0.01)
  joint_vel_noise = Unoise(n_min=-0.5, n_max=0.5)

  actor_terms = {
    "affected_leg_onehot": ObservationTermCfg(
      func=mdp.affected_leg_oracle,
      params={"affected_leg_idx": affected_leg_idx},
    ),
    "recovery_phase_onehot": ObservationTermCfg(
      func=mdp.recovery_phase_onehot,
      params={"action_term_name": "joint_pos"},
    ),
    "recovery_elapsed_norm": ObservationTermCfg(
      func=mdp.recovery_elapsed_norm,
      params={"action_term_name": "joint_pos"},
    ),
    "affected_leg_contact": ObservationTermCfg(
      func=mdp.affected_leg_contact,
      params={"sensor_name": "wire_contact"},
    ),
    "affected_leg_joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={
        "biased": True,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(f"{affected_leg_name}_.*_joint",)
        ),
      },
      noise=joint_pos_noise if not play else None,
    ),
    "affected_leg_joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(f"{affected_leg_name}_.*_joint",)
        ),
      },
      noise=joint_vel_noise if not play else None,
    ),
    "affected_foot_pos_rel_body": ObservationTermCfg(
      func=mdp.affected_foot_pos_rel_body,
      params={"leg_name": affected_leg_name},
    ),
    "affected_foot_vel_rel_body": ObservationTermCfg(
      func=mdp.affected_foot_vel_rel_body,
      params={"leg_name": affected_leg_name},
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
    "previous_residual": ObservationTermCfg(
      func=mdp.previous_residual,
      params={"action_term_name": "joint_pos"},
    ),
    "walk_affected_action": ObservationTermCfg(
      func=mdp.walk_affected_action,
      params={"action_term_name": "joint_pos"},
    ),
    "calf_height_above_wire_norm": ObservationTermCfg(
      func=mdp.calf_height_above_wire_norm,
      params={"action_term_name": "joint_pos"},
    ),
    "calf_forward_dist_norm": ObservationTermCfg(
      func=mdp.calf_forward_dist_norm,
      params={"action_term_name": "joint_pos"},
    ),
    "toe_height_above_wire_norm": ObservationTermCfg(
      func=mdp.toe_height_above_wire_norm,
      params={"action_term_name": "joint_pos"},
    ),
    "toe_forward_dist_norm": ObservationTermCfg(
      func=mdp.toe_forward_dist_norm,
      params={"action_term_name": "joint_pos"},
    ),
  }

  critic_terms = {
    **actor_terms,
    "joint_pos_full": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel_full": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Events: pinned spawn walking straight into the wire (same approach
  # validated in the physical entanglement verification), with small
  # jitter for diversity across envs/episodes.
  ##

  wire_mid_y = (
    spawn_y_override
    if spawn_y_override is not None
    else 0.5 * (wire.anchor_start[1] + wire.anchor_end[1])
  )
  spawn_x = wire.anchor_start[0] - 0.35

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (spawn_x - 0.02, spawn_x + 0.02),
          "y": (wire_mid_y - 0.03, wire_mid_y + 0.03),
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
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ##
  # Rewards: simple initial set per spec -- forward velocity tracking,
  # wire-clearance progress (dense release shaping + sparse phase-success
  # bonus), stability, and minimal residual-specific penalties.
  ##

  rewards = {
    "forward_velocity_tracking": RewardTermCfg(
      func=mdp.forward_velocity_tracking_reward,
      weight=1.0,
      params={"command_name": "twist", "action_term_name": "joint_pos"},
    ),
    "wire_recovery_progress": RewardTermCfg(
      func=mdp.wire_recovery_progress_reward,
      weight=0.5,
      params={"action_term_name": "joint_pos"},
    ),
    "wire_toe_crossing_progress": RewardTermCfg(
      func=mdp.wire_toe_crossing_progress_reward,
      weight=0.5,
      params={"action_term_name": "joint_pos"},
    ),
    "affected_leg_return_to_stance": RewardTermCfg(
      func=mdp.affected_leg_return_to_stance_reward_gated,
      weight=0.3,
      params={"leg_name": affected_leg_name, "action_term_name": "joint_pos"},
    ),
    "wire_crossing_event_bonus": RewardTermCfg(
      func=mdp.wire_crossing_event_bonus,
      weight=1.0,
      params={"action_term_name": "joint_pos"},
    ),
    "wire_recovery_final_success_bonus": RewardTermCfg(
      func=mdp.wire_recovery_final_success_bonus,
      weight=5.0,
      params={"action_term_name": "joint_pos"},
    ),
    "uprightness": RewardTermCfg(func=mdp.uprightness_reward, weight=1.0),
    "body_stability": RewardTermCfg(func=mdp.body_stability_penalty, weight=-0.2),
    "residual_magnitude": RewardTermCfg(
      func=mdp.residual_magnitude_penalty,
      weight=-0.02,
      params={"action_term_name": "joint_pos"},
    ),
    "affected_leg_joint_vel": RewardTermCfg(
      func=mdp.affected_leg_joint_vel_penalty,
      weight=-0.003,
      params={"leg_name": affected_leg_name},
    ),
    "residual_rate": RewardTermCfg(
      func=mdp.residual_rate_penalty,
      weight=-0.005,
      params={"action_term_name": "joint_pos"},
    ),
  }

  ##
  # Terminations: fell_over + time_out only. Deliberately NO success
  # termination -- the episode must continue through POST_RECOVERY so the
  # reward can observe sustained post-recovery walking, per the 4-phase
  # episode structure requirement.
  ##

  terminations = {
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over, params={"limit_angle": math.radians(70.0)}
    ),
  }

  ##
  # Metrics: training-time-only behavioral diagnostics (do NOT affect
  # observations/rewards/termination). Logged automatically under
  # ``Episode_Metrics/`` (TensorBoard/W&B) by the existing MetricsManager
  # -- see mjlab.tasks.recovery.mdp.metrics for definitions.
  ##

  metrics = {
    "recovery_triggered": MetricsTermCfg(
      func=mdp.metric_recovery_triggered,
      params={"action_term_name": "joint_pos"},
      reduce="sum",
    ),
    "crossing_achieved": MetricsTermCfg(
      func=mdp.metric_crossing_achieved,
      params={"action_term_name": "joint_pos"},
      reduce="sum",
    ),
    "recovery_success": MetricsTermCfg(
      func=mdp.metric_recovery_success,
      params={"action_term_name": "joint_pos"},
      reduce="sum",
    ),
    "recovery_cap_hit": MetricsTermCfg(
      func=mdp.metric_recovery_cap_hit,
      params={"action_term_name": "joint_pos"},
      reduce="sum",
    ),
    "fell_over": MetricsTermCfg(func=mdp.metric_fell_over, params={}, reduce="max"),
    "residual_magnitude_mean": MetricsTermCfg(
      func=mdp.metric_residual_magnitude,
      params={"action_term_name": "joint_pos"},
      reduce="mean",
    ),
    "residual_magnitude_peak": MetricsTermCfg(
      func=mdp.metric_residual_magnitude,
      params={"action_term_name": "joint_pos"},
      reduce="max",
    ),
    "forward_progress": MetricsTermCfg(
      func=mdp.metric_forward_progress, params={}, reduce="mean"
    ),
    "post_recovery_tilt": MetricsTermCfg(
      func=mdp.metric_post_recovery_tilt,
      params={"action_term_name": "joint_pos"},
      reduce="mean",
    ),
    "recovery_duration_steps": MetricsTermCfg(
      func=mdp.metric_in_recovery,
      params={"action_term_name": "joint_pos"},
      reduce="sum",
    ),
  }

  ##
  # Scene: ONE wire (native `nworld=num_envs` replication gives every
  # environment its own independent copy, exactly like the single Go2
  # robot entity) + contact sensor.
  #
  # IMPORTANT ARCHITECTURE NOTE (post wire-replication fix): an earlier
  # version of this task manually baked `num_envs` separate wire copies
  # into the single shared `MjSpec` (one per grid cell), on the mistaken
  # assumption that mujoco_warp's `nworld` batching would NOT otherwise
  # replicate the wire per-env. In fact `nworld` replication applies to
  # the WHOLE spec -- so that produced `num_envs` wires PER world (one
  # "real" copy near that world's robot, plus `num_envs - 1` dead-weight
  # copies also being fully simulated), i.e. O(num_envs^2) total wire
  # bodies/contacts. This silently capped the practical num_envs far
  # below what the GPU should otherwise support and made per-step time
  # scale quadratically. The fix: exactly ONE wire in the spec (see
  # `build_ball_cable_spec_fn`), combined with `env_spacing=0.0` above so
  # every independent per-world robot resets to the same local (x, y) the
  # single wire is fixed at. `wire_contact`'s pattern is now a plain,
  # static tuple of literal geom names (`wire._0_geom`, ... -- no env-id
  # suffix): `ContactSensor`/mujoco_warp resolve a literal geom name
  # against the model ONCE, and native `nworld` batching automatically
  # gives every world its own independent contact-sensor reading for that
  # same name, exactly like the robot's own per-foot contact sensors
  # (see e.g. `mjlab.tasks.velocity.config.go2.env_cfgs`'s static
  # `FOOT_GEOM_NAMES`-based `ContactMatch`, which needs no per-env
  # suffixing either).
  ##

  wire_contact_sensor = ContactSensorCfg(
    name="wire_contact",
    primary=ContactMatch(mode="geom", pattern=wire.geom_names(), entity=None),
    secondary=ContactMatch(
      mode="geom", pattern=f"{affected_leg_name}_calf2_collision", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  robot_cfg = _build_recovery_robot_cfg(
    affected_leg_idx=affected_leg_idx,
    recovery_mode=recovery_mode,
    recovery_thigh_velocity_kd=recovery_thigh_velocity_kd,
    recovery_calf_torque=recovery_calf_torque,
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={"robot": robot_cfg},
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      env_spacing=env_spacing,
      extent=2.0,
      spec_fn=build_ball_cable_spec_fn(wire),
      sensors=(wire_contact_sensor,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.5,
      elevation=-30.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # Fixed, NOT scaled by num_envs: with the wire-replication fix
      # (exactly one wire per independent `nworld` world, see
      # `build_ball_cable_spec_fn`), per-world contact/constraint counts
      # are small and constant regardless of num_envs -- scaling these
      # with num_envs was a symptom of the old O(num_envs^2) multi-wire
      # bug (every world simulating `num_envs` wire copies), not a
      # genuine requirement. These values were sized generously for one
      # robot + one wire's worst-case simultaneous contact/constraint
      # count and verified across num_envs in {1, 8, 32, 64}.
      nconmax=128,
      njmax=800,
      mujoco=MujocoCfg(
        timestep=0.0025,
        iterations=30,
        ls_iterations=20,
      ),
    ),
    decimation=8,
    # Generous episode length to accommodate normal-walk (variable, until
    # contact) + recovery (<=12 steps) + post-recovery walk (60-100
    # steps) comfortably, with margin: ~250 steps * 0.02s/step = 5s.
    episode_length_s=8.0,
  )


def verify_wire_sensor_coverage(env) -> None:
  """Explicit, loud sanity check for the corrected single-wire
  architecture: fails if the `wire_contact` sensor's resolved primary-geom
  set does not exactly match the ONE wire's segment geoms.

  Call this right after constructing a `ManagerBasedRlEnv` for this task
  (before trusting any rollout involving wire contact) -- e.g. in the
  staged validation scripts, or ad hoc after changing `num_envs`.

  Unlike the pre-fix version of this check (which expected
  `num_envs * segments_per_env` primaries, one set per env baked into the
  spec), the corrected architecture has exactly ONE wire in the shared
  spec; native `nworld=num_envs` replication is what gives every
  environment its own independent contact reading for that same static
  set of geom names. So the expected primary count is simply
  `wire.num_segments`, regardless of `num_envs`.

  Raises:
    RuntimeError: if the sensor's resolved primary names do not exactly
      equal the single wire's expected segment-geom names (indicating
      either a naming mismatch or a reintroduction of the old per-env
      multi-wire-copy bug).
  """
  scene = env.scene
  sensor = scene.sensors.get("wire_contact")
  if sensor is None:
    raise RuntimeError(
      "No 'wire_contact' sensor found on scene. Available sensors: "
      f"{list(scene.sensors.keys())}"
    )
  n_envs = scene.num_envs
  if n_envs < 1:
    raise RuntimeError(f"scene.num_envs={n_envs} is invalid.")

  from mjlab.tasks.cable_test.ball_cable import BALL_CABLE_SEG_PREFIX

  resolved_names = set(sensor.primary_names)
  # Infer segment count from the resolved names themselves (robust to
  # wire geometry changes), matching the single-wire naming convention
  # (`{BALL_CABLE_SEG_PREFIX}_{i}_geom`, no env-id suffix).
  prefix = f"{BALL_CABLE_SEG_PREFIX}_"
  segs = sorted(
    int(name[len(prefix) : -len("_geom")])
    for name in resolved_names
    if name.startswith(prefix) and name.endswith("_geom")
  )
  if not segs or segs != list(range(len(segs))):
    raise RuntimeError(
      "wire_contact sensor's resolved primary geom names do not match the "
      f"expected single-wire naming convention '{prefix}{{i}}_geom'. "
      f"Resolved names: {sorted(resolved_names)}. This may indicate the "
      "O(num_envs^2) multi-wire-per-world bug has been reintroduced (one "
      "wire copy PER ENV baked into the spec again), or a naming mismatch."
    )
  expected_names = {f"{prefix}{i}_geom" for i in segs}
  if resolved_names != expected_names:
    extra = resolved_names - expected_names
    raise RuntimeError(
      "wire_contact sensor resolved unexpected extra primary geom names "
      f"beyond the single wire's own segments: {sorted(extra)}."
    )
  print(
    f"[OK] wire_contact sensor covers exactly ONE wire "
    f"({len(segs)} segments, {len(resolved_names)} total primaries), "
    f"correctly relying on native nworld={n_envs} replication for "
    "per-environment contact detection."
  )
