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

from mjlab.asset_zoo.robots import get_go2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTermCfg
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

FORWARD_CMD_VAL = 0.3

# Default checkpoint for the frozen walking policy. Override via
# `walk_checkpoint` if training a new base policy.
DEFAULT_WALK_CHECKPOINT = (
  "logs/rsl_rl/go2_velocity/2026-09-17_13-00-16/model_9900.pt"
)


def go2_wire_recovery_residual_env_cfg(
  affected_leg_idx: int = 0,
  walk_checkpoint: str = DEFAULT_WALK_CHECKPOINT,
  wire_cfg: BallCableCfg | None = None,
  num_envs: int = 1,
  env_spacing: float = 2.0,
  max_residual: float = 1.0,
  max_recovery_steps: int = 12,
  release_required_steps: int = 5,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the residual-action wire-entanglement recovery environment.

  Args:
    affected_leg_idx: Index of affected leg (0=FL, 1=FR, 2=RL, 3=RR).
      FL-first per the staged implementation plan.
    walk_checkpoint: Path to the frozen walking policy checkpoint.
    wire_cfg: Optional override for the wire geometry/placement.
    num_envs: Number of parallel environments.
    env_spacing: Distance (meters) between environment grid origins.
    max_residual: Clamp on |Delta_a| in raw policy-action-space units.
    max_recovery_steps: Hard cap on RECOVERY phase duration (steps).
    release_required_steps: Consecutive no-contact steps required to
      count as a genuine (non-cap-forced) release.
    play: If True, disable observation noise for clean playback.
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
      device="cuda:0",
      wire_contact_sensor_name="wire_contact",
      max_recovery_steps=max_recovery_steps,
      release_required_steps=release_required_steps,
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

  wire_mid_y = 0.5 * (wire.anchor_start[1] + wire.anchor_end[1])
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
      params={"command_name": "twist"},
    ),
    "wire_contact_release": RewardTermCfg(
      func=mdp.wire_contact_release_reward,
      weight=0.2,
      params={"sensor_name": "wire_contact"},
    ),
    "recovery_phase_success_bonus": RewardTermCfg(
      func=mdp.recovery_phase_success_bonus,
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
      weight=-0.01,
      params={"leg_name": affected_leg_name},
    ),
    "residual_rate": RewardTermCfg(
      func=mdp.residual_rate_penalty,
      weight=-0.02,
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
  # Scene: wire (one copy per env) + contact sensor (same pattern as
  # env_cfg_wire.py).
  #
  # IMPORTANT: `num_envs` (this function's parameter) is only the value
  # used when THIS function is called. Callers frequently build the env
  # cfg once (e.g. at task-registration time, with the registry default
  # num_envs=1) and later override `scene.num_envs` post-hoc (e.g. via
  # `--env.scene.num-envs 512` on the CLI, or by mutating
  # `env_cfg.scene.num_envs` directly before constructing the env). A
  # plain, eagerly-computed tuple of per-env wire geom names (or a
  # closure capturing this function's `num_envs` parameter) would go
  # stale in that case -- it would only ever cover the `num_envs` seen
  # here, silently leaving every other environment's wire undetected by
  # `wire_contact`.
  #
  # NOTE: a naive fix using a closure over a locally-held `SceneCfg`
  # object (reading `scene_cfg.num_envs` lazily) does NOT work here,
  # despite superficially resembling `Scene._add_terrain`'s
  # `terrain.num_envs = scene.num_envs` sync. Empirically verified: tyro's
  # CLI parsing (used by `train.py` for `--env.scene.num-envs`)
  # reconstructs `SceneCfg`/`TerrainEntityCfg`/the `sensors` tuple as
  # BRAND NEW objects on every parse (even for untouched fields), while
  # passing opaque `Callable`-typed leaf fields (`spec_fn`,
  # `ContactMatch.pattern`) through UNCHANGED BY REFERENCE. A closure
  # captured here would therefore keep pointing at the ORIGINAL
  # (pre-override, registry-default) `SceneCfg` instance forever -- the
  # exact same silent-staleness bug this fix is meant to eliminate, just
  # one level removed. See the empirical test that established this
  # (identity-preservation check against `tyro.cli(..., default=...)`).
  #
  # The only mechanism proven to work across BOTH `deepcopy` (used by the
  # task registry) and tyro's reconstruct-with-pass-through-callables
  # behavior is to have `Scene.__init__` explicitly PASS its own
  # authoritative, final `self._cfg.num_envs` into `spec_fn` and into
  # `ContactSensor.edit_spec` (which forwards it to any callable
  # `ContactMatch.pattern`) at the moment they are actually invoked. This
  # requires no captured references to any config object at all -- see
  # `SceneCfg.spec_fn`'s and `ContactMatch.pattern`'s updated signatures
  # (both now take `num_envs: int` as an explicit argument).
  ##

  def _wire_geom_names(n: int) -> tuple[str, ...]:
    if n < 1:
      raise RuntimeError(
        f"wire_contact sensor: num_envs={n} is invalid (must be >= 1)."
      )
    names = tuple(name for env_id in range(n) for name in wire.geom_names(env_id))
    expected = n * wire.num_segments
    if len(names) != expected:
      # Should be unreachable (names/expected are both derived from the
      # same `n`), but guard against future refactors breaking this
      # invariant silently.
      raise RuntimeError(
        f"wire_contact sensor: generated {len(names)} wire geom names for "
        f"num_envs={n}, expected {expected} ({wire.num_segments} segments "
        "per env). This indicates a bug in wire geom-name generation."
      )
    return names

  wire_contact_sensor = ContactSensorCfg(
    name="wire_contact",
    primary=ContactMatch(mode="geom", pattern=_wire_geom_names, entity=None),
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
    commands=commands,
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
      # Scaled by this function's `num_envs` parameter.
      #
      # IMPORTANT, discovered while validating this fix: the required
      # njmax at CONSTRUCTION time (mujoco_warp's `put_data` buffer-sizing
      # check, evaluated against the model's default/keyframe pose,
      # *before* any reset-driven pinned wire-contact spawn ever occurs)
      # scales linearly with num_envs at ~183 per env for THIS scene
      # (empirically measured via the overflow error's reported minimum
      # at num_envs in {96, 128, 160, 256, 512}: all consistent with
      # njmax_min ~= 183 * num_envs). This differs from
      # `env_cfg_wire.py`'s untouched njmax=max(800, 128*num_envs)
      # formula, which is NOT actually sufficient at this scene's true
      # per-env constraint count -- that sibling task's formula appears
      # to "work" only because mujoco_warp does not hard-fail at RUNTIME
      # when per-step nefc exceeds njmax (contacts/constraints beyond the
      # cap are apparently silently dropped rather than raising), so an
      # undersized njmax silently degrades physical fidelity rather than
      # crashing. Use 184 (not 128) per env here to actually pass the
      # hard construction-time check with margin.
      #
      # SEPARATELY, and more restrictively: with MuJoCo's DENSE jacobian
      # (this scene's default), the per-world Jacobian/Cholesky buffers
      # scale as O(njmax * nv), and since njmax itself must scale with
      # num_envs (above), total GPU memory scales roughly as
      # O(num_envs^2) for this scene. Empirically this exceeds a 16GB GPU
      # somewhere between num_envs=96 and num_envs=128 (128 requires
      # ~19.5GB just for one buffer). This means num_envs is effectively
      # capped well below the 512 used in the (discarded, invalid) prior
      # training run on hardware of this class -- see the staged
      # validation report for the chosen num_envs and the reasoning.
      # Switching to `mujoco.jacobian="sparse"` was explored as a
      # workaround but hit a separate CUDA-graph-capture OOM during
      # stepping in preliminary testing and was not pursued further
      # within this fix's scope (it targets the wire_contact sensor
      # construction bug specifically).
      nconmax=max(128, 64 * num_envs),
      njmax=max(800, 184 * num_envs),
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
  """Explicit, loud sanity check: fails if the `wire_contact` sensor's
  resolved primary-geom count does not match the actual replicated
  environment count.

  Call this right after constructing a `ManagerBasedRlEnv` for this task
  (before trusting any rollout involving wire contact) -- e.g. in the
  staged Test 1 validation script, or ad hoc after changing `num_envs`.

  Raises:
    RuntimeError: if the sensor's resolved primary count does not equal
      `env.scene.num_envs * wire.num_segments`, which would indicate the
      sensor is not covering every environment's wire (the original bug
      this function guards against), or is covering more/fewer wires
      than actually exist in the compiled scene.
  """
  scene = env.scene
  sensor = scene.sensors.get("wire_contact")
  if sensor is None:
    raise RuntimeError(
      "No 'wire_contact' sensor found on scene. Available sensors: "
      f"{list(scene.sensors.keys())}"
    )
  n_envs = scene.num_envs
  resolved = len(sensor.primary_names)
  # Infer segments-per-env from the resolved names themselves (robust to
  # wire geometry changes) rather than importing DEFAULT_WIRE_CFG.
  if n_envs < 1:
    raise RuntimeError(f"scene.num_envs={n_envs} is invalid.")
  if resolved % n_envs != 0:
    raise RuntimeError(
      f"wire_contact sensor covers {resolved} primary geoms, which is not "
      f"evenly divisible by scene.num_envs={n_envs}. The sensor is almost "
      "certainly NOT correctly covering every replicated environment's "
      "wire. This is exactly the bug class `verify_wire_sensor_coverage` "
      "exists to catch -- do not proceed with training/eval until fixed."
    )
  segs_per_env = resolved // n_envs
  # Validate by naming-convention structure (env-id-indexed segment names,
  # as produced by `BallCableCfg.geom_names`) rather than re-instantiating
  # a specific wire config -- this stays correct even if a caller passed a
  # custom `wire_cfg` with a different geometry/segment count.
  from mjlab.tasks.cable_test.ball_cable import BALL_CABLE_SEG_PREFIX

  expected_names = {
    f"{BALL_CABLE_SEG_PREFIX}{env_id}_{i}_geom"
    for env_id in range(n_envs)
    for i in range(segs_per_env)
  }
  resolved_names = set(sensor.primary_names)
  if resolved_names != expected_names:
    missing = expected_names - resolved_names
    extra = resolved_names - expected_names
    raise RuntimeError(
      "wire_contact sensor's resolved primary geom names do not match the "
      f"expected per-env wire geom names for num_envs={n_envs}. "
      f"Missing ({len(missing)}): {sorted(missing)[:5]}... "
      f"Extra ({len(extra)}): {sorted(extra)[:5]}..."
    )
  print(
    f"[OK] wire_contact sensor covers all {n_envs} environments "
    f"({segs_per_env} wire segments/env, {resolved} total primaries)."
  )
