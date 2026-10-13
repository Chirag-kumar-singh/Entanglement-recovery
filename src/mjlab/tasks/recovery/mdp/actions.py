"""Residual recovery action term: composes a FROZEN walking policy's raw
12-joint action with a small learned residual on ONE affected leg (3
joints), while leaving the other 9 joints exactly as the frozen policy
produced them.

Architecture (see chat/audit history for full rationale):

    PPO outputs Delta_a in R^3 (affected leg only)
         |
         v
    obs_walk = <reconstruct the EXACT 48-dim observation vector that the
                frozen Mjlab-Velocity-Flat-Unitree-Go2 policy was trained
                with -- base_lin_vel, base_ang_vel, projected_gravity,
                joint_pos_rel(biased=True), joint_vel_rel, last_action,
                command -- see module docstring in
                ``verify_residual_action`` scripts for the term-by-term
                audit this was derived from>
         |
         v
    a_walk_raw = pi_walk(obs_walk)          # frozen, no_grad, eval mode
         |
         v
    final_raw = a_walk_raw.clone()
    final_raw[:, affected_leg_idx*3 : affected_leg_idx*3+3] += Delta_a
         |
         v
    (reuse the EXISTING, UNMODIFIED mjlab.envs.mdp.actions.actions.JointPositionAction
     pipeline: process_actions() applies scale/offset, apply_actions() sets
     the joint position target via entity.set_joint_position_target())

CRITICAL DESIGN DECISION -- residual space:
  Delta_a is added in RAW POLICY-ACTION SPACE (i.e. the same
  pre-scale/pre-offset units pi_walk itself outputs), NOT in physical
  joint-angle/target space. Rationale:
    1. With Delta_a=0, `final_raw` is BYTE-FOR-BYTE `a_walk_raw`, so it
       passes through the exact same, unmodified JointPositionAction
       scale/offset/encoder-bias pipeline as the real play-mode path --
       there is no risk of double-applying or re-deriving scale/offset
       logic for the residual itself.
    2. PPO's Gaussian action head naturally outputs values in a
       roughly unit-scaled space (matching GaussianDistribution's
       typical init_std); adding in raw-action space keeps the residual
       in that same natural scale, rather than requiring the policy to
       learn radian-scale corrections (GO2_ACTION_SCALE ~0.01-0.02 for
       Go2, i.e. physical-space residuals would need very large raw
       outputs to have any effect, which is a harder optimization
       target).
    3. The physical effect of a raw-space residual is still a joint
       angle change of `Delta_a * scale` radians (same scale dict as the
       walking policy, GO2_ACTION_SCALE) once passed through the
       existing pipeline -- so a `max_residual` clamp in raw units
       still corresponds to an easily-computed physical bound (see
       `max_residual_rad_equivalent` below).

CRITICAL DESIGN DECISION -- pi_walk's own observation history:
  `pi_walk`'s own "last_action" observation term must reflect ONLY
  `a_walk_raw` from the previous step (i.e. what pi_walk itself would
  have produced/perceived in a standalone deployment), NOT
  `final_raw` (which includes the recovery residual). This action term
  therefore maintains its OWN internal buffer
  (`self._last_walk_raw_action`), separate from
  `env.action_manager.action` (which holds the RECOVERY policy's own
  3-dim Delta_a, as seen by the recovery task's observation manager).
  This is essential for honoring "same physical state + same command ->
  exactly same pi_walk action": if pi_walk's own history included the
  residual-perturbed action, its behavior would silently diverge from
  the frozen play-mode policy's true behavior even when Delta_a=0 is
  held constant for long enough that this would otherwise matter.

Does NOT modify mjlab.envs.mdp.actions.actions.JointPositionAction (reused
unmodified, composed via a private internal instance) and does NOT
modify/retrain the frozen walking policy (loaded once, eval mode, every
call wrapped in torch.no_grad()).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

LEG_NAMES = ("FL", "FR", "RL", "RR")

# Recovery FSM phase ids (see ResidualRecoveryActionCfg docstring).
PHASE_NORMAL_WALK = 0
PHASE_RECOVERY = 1
PHASE_POST_RECOVERY = 2

# Normalization scales for wire-relative geometric quantities (observations
# AND potential-shaping reward both reuse these SAME constants -- see
# `wire_recovery_progress_reward`/`wire_toe_crossing_progress_reward` in
# rewards.py, which used these exact values as `height_scale`/
# `forward_scale` defaults prior to the potential-shaping rework, and the
# new `calf_height_above_wire_norm`/etc. observations in observations.py.
# Reused rather than re-invented so observations and reward stay on an
# identical, already-audited scale.
WIRE_HEIGHT_NORM_SCALE = 0.05
WIRE_FORWARD_NORM_SCALE = 0.1

# Exact joint order the frozen Mjlab-Velocity-Flat-Unitree-Go2 policy was
# trained with (confirmed live in multiple prior diagnostic scripts this
# session via `action_manager.get_term("joint_pos").target_names`).
WALK_JOINT_ORDER = (
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


def _load_frozen_walk_policy(checkpoint: str, device: str):
  """Loads the frozen Go2 walking policy via the EXACT same path used by
  scripts/play.py and every manual diagnostic script this session:
  load_env_cfg(task_id, play=True) -> ManagerBasedRlEnv ->
  RslRlVecEnvWrapper -> MjlabOnPolicyRunner.load(checkpoint) ->
  get_inference_policy(). The throwaway velocity-task env is closed
  immediately after extracting the policy callable (rsl_rl's
  `get_inference_policy` returns a bound, stateless-w.r.t.-env inference
  function -- verified empirically in the accompanying verification
  script, not merely assumed).
  """
  import mjlab.tasks  # noqa: F401  (populates task registry)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go2"
  configure_torch_backends()
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(TASK_ID)
  throwaway_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(throwaway_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  throwaway_env.close()
  return policy


@dataclass(kw_only=True)
class ResidualRecoveryActionCfg(ActionTermCfg):
  """Configuration for the residual recovery action term."""

  walk_checkpoint: str
  """Path to the frozen walking policy checkpoint."""

  affected_leg_idx: int
  """Index of the affected leg (0=FL, 1=FR, 2=RL, 3=RR). Only this leg's
  3 joints (hip/thigh/calf) receive the learned residual."""

  max_residual: float = 1.0
  """Clamp on |Delta_a| in RAW policy-action-space units (same space
  pi_walk's own output lives in, pre scale/offset). Conservative initial
  bound; see module docstring for why raw-space was chosen. Combined with
  GO2_ACTION_SCALE (~0.01-0.02 per joint), this corresponds to a physical
  joint-angle bound of roughly
  `max_residual * GO2_ACTION_SCALE[joint]` radians -- NOT the same
  as the previous experiments' +-0.5 rad posture clamp, and must be
  re-derived from the real Go2 joint ranges/effort limits (see
  `max_residual_rad_hint` below), not reused blindly.
  """

  command_name: str = "twist"
  """Name of the command term supplying the (lin_vel_x, lin_vel_y,
  ang_vel_z) command pi_walk expects as its own observation input."""

  device: str = "cuda:0"

  # --- D1: deterministic recovery mode (Quad-SDK-inspired binary
  # normal/obstructed swing approximation, see module docstring addendum
  # below) ---
  #
  # `recovery_mode` selects WHICH SOURCE provides the 3-dim recovery
  # delta applied to the affected leg during the RECOVERY phase. This is
  # the smallest possible extension point: everything else (FSM entry/
  # exit via the real wire-contact sensor + geometric crossing, the
  # affected-leg index mapping, the frozen-policy composition, and the
  # existing JointPositionAction safety pipeline) is unchanged and fully
  # reused by both modes.
  #
  #   "ppo_residual" (default, UNCHANGED existing behavior): the 3-dim
  #     `actions` tensor passed into `process_actions` (normally a PPO
  #     policy's output) is clamped to `max_residual` and used as
  #     Delta_a, exactly as before this change.
  #   "deterministic": `actions` is IGNORED entirely (no PPO/learned
  #     component at all). Delta_a is instead a fixed, per-step-constant
  #     3-vector `[0, thigh_recovery_delta, calf_recovery_delta]` --
  #     zero on the ab/ad (hip) joint (continue the frozen walking
  #     policy's own command, per the Quad-SDK-inspired binary
  #     normal/obstructed-swing behavior: "affected leg ab/ad -> normal
  #     walking behavior"), a constant retract delta on thigh, and a
  #     constant push/extend delta on calf. This is intentionally NOT a
  #     multi-stage scripted trajectory (no RETRACT/ADVANCE/DESCEND
  #     states) -- it is a single constant command held for the entire
  #     RECOVERY phase, gated on/off by the exact same phase mask used
  #     by the PPO path.
  recovery_mode: Literal["ppo_residual", "deterministic", "hybrid"] = "ppo_residual"
  """Source of the affected-leg recovery behavior during RECOVERY. See
  class docstring addendum above. Default preserves all prior behavior
  unchanged (existing PPO-residual tasks are unaffected).

  ``"hybrid"`` (D3): a genuine Quad-SDK-inspired hybrid controller, NOT a
  position residual. During RECOVERY, the affected leg's thigh and calf
  POSITION actuators are runtime-disabled (their MuJoCo actuator_gainprm/
  actuator_biasprm are zeroed for that env+joint only) and two AUXILIARY
  actuators (added to the recovery task's Go2 entity only -- see
  ``env_cfg_residual.py`` -- never the normal walking task) take over:
  a velocity actuator on the thigh joint (pure damped-velocity command,
  analogous to Quad-SDK's hip retract: kp=0, kd=swing_kd, vel_setpoint=
  -retract_vel) and a motor actuator on the calf joint (pure open-loop
  feedforward torque, analogous to Quad-SDK's knee push: kp=0, kd=0,
  torque_ff=-tau_push). The ab/ad (Go2's *_hip_joint) continues to
  receive the frozen walking policy's own position command, unchanged,
  exactly as in "deterministic" mode. See ``_apply_hybrid_actuators``."""

  thigh_recovery_delta: float = 0.3
  """Constant RAW policy-action-space delta applied to the affected
  leg's thigh joint throughout RECOVERY, used only when
  ``recovery_mode == "deterministic"``. Sign convention verified
  empirically against the compiled Go2 model (see D0.5 audit /
  scripts/verify_fl_joint_signs.py-style probe): a POSITIVE thigh delta
  moves the foot in -X (backward) in the robot's own frame, i.e. a
  retracting motion -- matching the Quad-SDK reference behavior's
  "affected leg hip/thigh -> retract/backward behavior". Combined with
  `GO2_ACTION_SCALE` (~0.01-0.02 rad per raw unit for the thigh
  actuator), this corresponds to a small, conservative physical angle
  change (a few degrees) -- deliberately NOT an aggressive value; tune
  via this parameter, not by editing the controller."""

  calf_recovery_delta: float = -0.3
  """Constant RAW policy-action-space delta applied to the affected
  leg's calf joint throughout RECOVERY, used only when
  ``recovery_mode == "deterministic"``. Sign convention verified
  empirically: a NEGATIVE calf delta moves the foot in +X and +Z
  (forward AND up) in the robot's own frame. This sign was chosen
  (rather than the naively "positive extends the joint toward its
  less-flexed limit" direction, which empirically moves the foot
  backward+down) specifically because the existing wire geometry has
  the robot approaching the wire from -X and walking in +X, so clearing
  it requires the foot to advance forward AND lift upward, not retreat
  further. This is the D0.5-audit-flagged geometry-dependent choice,
  resolved here by checking the actual `DEFAULT_WIRE_CFG`
  anchor/approach direction (see env_cfg_residual.py), not assumed from
  the raw joint axis alone. Deliberately small/conservative; tune via
  this parameter."""

  # --- D3: hybrid actuator-level recovery parameters (used only when
  # recovery_mode == "hybrid"). D2 established that a position residual
  # on the calf is overwhelmed by the frozen walking policy's own
  # gait-phase-driven calf command; these parameters instead define a
  # genuine actuator-level override (see recovery_mode docstring). ---

  recovery_thigh_velocity: float = -2.0
  """Signed magnitude (rad/s) of the thigh joint's damped-velocity
  retract command during RECOVERY (hybrid mode only). `_apply_hybrid_
  actuators` writes `thigh_vel_cmd = -cfg.recovery_thigh_velocity`, so
  this value's sign is NOT the sign of the actual ctrl -- it is
  deliberately negative here so the resulting ctrl is +2.0 rad/s.

  Sign resolved empirically via the refined D3 free-space verification
  script (scripts/d3_test_b_refined_thigh_sign.py), which measures
  instantaneous qvel/actuator_force (not net position drift) under both
  candidate signs using the real Go2 MuJoCo model:
    - cfg=+2.0 -> ctrl=-2.0 -> measured qvel < 0 (thigh angle decreases)
    - cfg=-2.0 -> ctrl=+2.0 -> measured qvel > 0 (thigh angle increases)
  D0.5's independently-verified Go2 joint convention established that a
  POSITIVE thigh angle change moves the foot in -X (backward), i.e. the
  retract direction. Therefore cfg=-2.0 (ctrl=+2.0, qvel>0) is the
  physically-correct retract configuration; cfg=+2.0 was the wrong
  (extend) direction and has been corrected here. This is a sign
  correction, not a tuned/swept value -- the 2.0 rad/s magnitude is
  unchanged from its original conservative derivation: Go2's own
  walking gait (see D2 CSV, logs/d2_recovery_trajectory/FL_seed0/
  trajectory.csv) exhibits thigh joint velocities on the order of 1-3
  rad/s during normal swing, unlike Quad-SDK's Spirit-specific
  retract_vel=15 rad/s which is not applicable to Go2 and is not copied
  here."""

  recovery_thigh_velocity_kd: float = 2.0
  """Damping gain (N*m*s/rad) of the auxiliary thigh velocity actuator,
  active only when RECOVERY is active for the affected leg (zeroed
  otherwise, see _apply_hybrid_actuators). Chosen conservatively: with a
  typical velocity tracking error of a few rad/s, torque = kd * error is
  a few N*m, well within GO2_ACTUATOR_THIGH's existing effort_limit of
  23.5 N*m (the same safety bound the normal walking thigh actuator
  already respects -- see go2_constants.py). This is a genuinely
  different physical quantity from Quad-SDK's Spirit-specific swing_kd
  and is not copied from it."""

  recovery_calf_torque: float = 5.0
  """Magnitude (N*m) of the calf joint's open-loop feedforward push
  torque during RECOVERY (hybrid mode only). Sign resolved empirically
  via the D3 free-space verification script, not assumed from the old
  D1 calf-delta sign (D2 established that a POSITION delta and a direct
  TORQUE command can require opposite signs for the same intended
  physical motion, since one acts through the gait-phase-coupled
  position target and the other bypasses it entirely). Conservative
  initial magnitude: GO2_ACTUATOR_CALF's effort_limit is 45 N*m; 5.0 N*m
  (~11%) is a modest fraction, chosen to be clearly non-negligible but
  far from saturating the actuator or destabilizing the leg, consistent
  with "architectural validation, not optimization" for this first
  hybrid-mode run. NOT Quad-SDK's Spirit-specific tau_push=2.0 N*m
  (different robot, different actuator, not copied)."""

  # --- EXPERIMENTAL (opt-in, default OFF): peak-triggered thigh stop.
  #
  # Follow-up to the D3 ground-contact diagnostic (clean, deconfounded
  # FL-only isolated-wire baseline): in all 4 tested seeds, the
  # recovery-phase foot height peaks well below the wire and then
  # DESCENDS, and the FL foot subsequently strikes the ground
  # (FL_foot_collision vs terrain, 100-276 N peak force) a few steps
  # later while the thigh/calf auxiliary actuators are still actively
  # driving. This experiment tests ONE isolated hypothesis: does zeroing
  # ONLY the thigh velocity command (leaving calf torque, ab/ad, FSM,
  # and everything else untouched) as soon as a genuine foot-height peak
  # is detected reduce or prevent that ground impact?
  #
  # Disabled by default (experimental_peak_triggered_thigh_stop=False)
  # -- this preserves the validated baseline _apply_hybrid_actuators
  # behavior EXACTLY for every existing caller/config. Must be
  # explicitly opted into (see scripts/d3_peak_trigger_experiment.py).
  experimental_peak_triggered_thigh_stop: bool = False
  """If True, latch `hybrid_thigh_vel_ctrl` to 0.0 (via the SAME
  existing `set_joint_velocity_target` call already used for the normal
  recovery command -- no new control mechanism) for the remainder of
  RECOVERY once the internal peak detector fires (see
  `peak_detector_rise_threshold`/`peak_detector_decline_eps`/
  `peak_detector_confirm_steps` below). Does NOT touch calf torque,
  ab/ad, or any FSM transition -- recovery continues exactly as before,
  only the thigh velocity target changes. Default False: zero behavior
  change for the validated baseline."""

  peak_detector_rise_threshold: float = 0.01
  """Minimum rise (meters) of `foot_height_above_wire` above its value
  AT recovery entry required before the peak detector is allowed to
  "arm" (start tracking a running max / looking for a decline) for that
  env. Chosen conservatively from the measured clean isolated-wire
  baseline (logs/d3_clean_fl_seed{0..3}/trajectory.csv): the observed
  recovery-phase foot_z rise from entry to peak was ~0.012-0.019 m in
  all 4 seeds. 0.01 m is set deliberately BELOW every observed real
  rise (so the detector always has a chance to arm before the real
  peak) while still being clearly above sensor/integration noise at this
  timestep -- NOT tuned to any single seed's exact value."""

  peak_detector_decline_eps: float = 0.003
  """Minimum drop (meters) of `foot_height_above_wire` below its running
  max (since arming) required to count as a single "declining" sample.
  Chosen as ~3 mm: small enough to catch the real descent promptly
  after the peak, large enough to not be triggered by single-step
  simulation/contact-solver jitter (which is far below 1 mm at this
  timestep based on the baseline CSVs' step-to-step noise floor)."""

  peak_detector_confirm_steps: int = 2
  """Number of CONSECUTIVE declining samples (see
  `peak_detector_decline_eps`) required before the peak is considered
  CONFIRMED (not a single noisy sample). 2 is the smallest value that
  satisfies "require a genuine decrease after a preceding rise, rather
  than triggering on a single noisy sample" while still reacting
  promptly (at 0.02s/step, this is a 0.02-0.04s detection latency)."""

  # --- Recovery FSM (phase gating) ---
  #
  # The residual Delta_a is only ever allowed to affect the physical
  # action while the FSM is in the RECOVERY phase. During NORMAL_WALK and
  # POST_RECOVERY, Delta_a is forced to zero regardless of what the
  # recovery policy outputs -- i.e. PPO only ever controls the robot
  # during the actual recovery window, exactly like a human wouldn't be
  # "steering" the frozen gait before/after the incident. This gating
  # lives here (not in a separate observation/reward term) because the
  # action term is the one single place that already composes a_walk_raw
  # with Delta_a every step, and it already owns the wire-contact sensor
  # lookup needed to detect the entanglement trigger.
  #
  # Phase transitions (per-env, driven by real signals -- no hardcoded
  # step counts other than a hard safety cap on recovery duration):
  #   NORMAL_WALK -> RECOVERY: the instant genuine FL-wire contact is
  #     observed (via `wire_contact_sensor_name`).
  #   RECOVERY -> POST_RECOVERY: EITHER (a) contact has been continuously
  #     released for `release_required_steps` consecutive steps (a real
  #     "cleared" signal, not a step-count guess), OR (b)
  #     `max_recovery_steps` elapsed since entering RECOVERY (a hard cap,
  #     not a target -- prevents an episode getting stuck in RECOVERY
  #     forever if the policy never manages to release the leg).
  #   POST_RECOVERY: persists until episode end (handled by the
  #     env's own time_out/fell_over terminations).
  #
  # GEOMETRIC CLEARANCE (post-audit, see chat history): the above
  # contact-only release condition let the policy "nudge until contact
  # disappears" WITHOUT ever lifting the foot above or advancing it past
  # the wire -- a genuine reward-exploit loophole. The exit condition now
  # ADDITIONALLY requires `crossing_achieved`: a one-shot latch set only
  # when the affected foot transitions from the wire's near side to its
  # far side WHILE simultaneously above the wire's live (deformed)
  # centerline height by `clearance_margin`. `release_required_steps` of
  # sustained no-contact only starts being counted AFTER that latch is
  # set. See `mjlab.tasks.recovery.mdp.wire_geometry` for how the live
  # wire centerline position is derived from the actual ball-joint
  # segment geoms (not the wire's static initial placement).
  wire_contact_sensor_name: str = "wire_contact"

  wire_geom_names: tuple[str, ...] = ()
  """Static geom names of the wire's segments (BallCableCfg.geom_names()),
  used to read the wire's LIVE centerline position each step. Must be set
  by the env_cfg factory."""

  wire_radius: float = 0.006
  """Wire capsule radius (meters); added to the interpolated centerline z
  to get the wire's top surface height for clearance checks."""

  clearance_margin: float = 0.02
  """Minimum height (meters) the CALF (primary entanglement-clearance
  reference, see module docstring) must be above the wire's live top
  surface, AT THE INSTANT it crosses from near to far side, for that
  crossing to count as genuine calf clearance. NEEDS HUMAN CONFIRMATION
  (see audit report) -- chosen modestly since the prior broken policy's
  tracked point stayed 6-9cm BELOW the wire the whole episode, so even
  this small margin is a qualitative, unambiguous improvement over that
  failure mode."""

  foot_crossing_margin: float = 0.02
  """Minimum forward distance (meters, along the robot's own latched
  forward direction) the toe/foot must be past the wire's local
  reference, AFTER calf clearance has been achieved, for the two-stage
  crossing condition to be considered complete. Unlike
  `clearance_margin`, this is a purely horizontal/forward margin -- the
  toe is NOT required to be above the wire at all (see module docstring:
  the toe is a lower kinematic point than the calf and is not the body
  segment that gets entangled). NEEDS HUMAN CONFIRMATION -- chosen as a
  small, non-arbitrary starting point equal to `clearance_margin` for
  consistency; not yet empirically tuned."""

  # --- D5: Phase-aware dynamic thresholds (experimental, opt-in, default OFF) ---
  #
  # D5.25-D5.28 identified that 40% of seeds never reach crossing-stage
  # latches, and found that "neither-stage-reaching" seeds have 27% higher
  # entry leg speed (5.801 vs 4.562 rad/s). Hypothesis: fast-moving legs
  # are kinematically incompatible with fixed crossing-latch thresholds.
  #
  # D5.29's original three-tier mapping had the direction backwards: both
  # latch conditions are strict `>` comparisons (see `_update_fsm` --
  # `calf_lifted = calf_height_above_wire > effective_clearance_margin`,
  # `foot_forward_dist > effective_foot_crossing_margin`), so a LARGER
  # margin makes the latch HARDER to satisfy, not easier. D5.29 assigned
  # the LARGEST margins (0.030/0.040) to the FAST tier -- exactly the
  # population D5.28 found was already failing to latch -- which would
  # make fast-entry failures *more* likely, not less. D5.31 corrects this:
  # margins are now NON-INCREASING as entry speed increases, so a faster
  # entry is never held to a stricter (larger-margin) latch requirement
  # than a slower one.
  #
  # This experiment tests THREE-TIER LEG-SPEED-DEPENDENT THRESHOLD ADJUSTMENT:
  #   - Slow (leg_speed < 4.5): 0.030/0.040 (largest margins -- slow legs
  #     have the most control authority/time within the fixed
  #     `max_recovery_steps` cap, so the strictest genuine-clearance bar
  #     is kept here, not loosened).
  #   - Medium (4.5 <= leg_speed < 6.5): 0.025/0.035 (unchanged midpoint).
  #   - Fast (leg_speed >= 6.5): 0.020/0.030 (smallest margins -- fast
  #     entries get the least strict latch bar, since D5.28 found this is
  #     the population already struggling to latch at all under the
  #     static baseline).
  # These are the SAME three magnitude pairs D5.29 used, only reassigned
  # to the opposite speed tiers -- this is a direction correction, NOT a
  # blanket loosening of all thresholds (the slow tier is in fact now
  # stricter than D5.29's original slow tier). All six margin constants
  # below remain EXPERIMENTAL, opt-in starting points (see
  # `phase_aware_thresholds`) -- not re-tuned or validated against new
  # seed data by this correction; only the tier assignment direction was
  # fixed. When enabled, thresholds are dynamically selected at recovery
  # entry based on the affected leg's instantaneous velocity (thigh +
  # calf).
  #
  # Disabled by default (phase_aware_thresholds=False) -- preserves the
  # validated baseline FSM behavior EXACTLY for every existing caller/config.
  # Must be explicitly opted into for D5.29/D5.31 comparison runs.
  phase_aware_thresholds: bool = False
  """If True, dynamically adjust crossing-latch thresholds
  (clearance_margin and foot_crossing_margin) based on the affected leg's
  velocity at recovery entry, using three tiers as defined above. If False
  (default), use the static baseline thresholds. See D5.28 for the
  hypothesis/rationale and D5.31 for the direction correction."""

  # Thresholds for phase-aware tier selection (leg speed in rad/s).
  phase_aware_speed_threshold_low: float = 4.5
  """Leg speed threshold (rad/s) below which 'slow' tier is selected."""

  phase_aware_speed_threshold_high: float = 6.5
  """Leg speed threshold (rad/s) above which 'fast' tier is selected.
  Between low and high uses 'medium' tier."""

  # Slow tier thresholds (leg_speed < 4.5 rad/s). EXPERIMENTAL: D5.31
  # reassigned this tier the LARGEST margin pair (was D5.29's fast tier).
  phase_aware_slow_clearance_margin: float = 0.030
  """Clearance margin for slow-entry legs. EXPERIMENTAL starting point,
  not re-tuned by the D5.31 direction correction."""

  phase_aware_slow_foot_crossing_margin: float = 0.040
  """Foot crossing margin for slow-entry legs. EXPERIMENTAL starting
  point, not re-tuned by the D5.31 direction correction."""

  # Medium tier thresholds (4.5 <= leg_speed < 6.5 rad/s). Unchanged by
  # the D5.31 direction correction (symmetric midpoint tier).
  phase_aware_medium_clearance_margin: float = 0.025
  """Clearance margin for medium-speed-entry legs. EXPERIMENTAL starting
  point."""

  phase_aware_medium_foot_crossing_margin: float = 0.035
  """Foot crossing margin for medium-speed-entry legs. EXPERIMENTAL
  starting point."""

  # Fast tier thresholds (leg_speed >= 6.5 rad/s). EXPERIMENTAL: D5.31
  # reassigned this tier the SMALLEST margin pair (was D5.29's slow
  # tier), so fast entries are no longer held to the strictest latch bar.
  phase_aware_fast_clearance_margin: float = 0.020
  """Clearance margin for fast-entry legs. EXPERIMENTAL starting point,
  not re-tuned by the D5.31 direction correction."""

  phase_aware_fast_foot_crossing_margin: float = 0.030
  """Foot crossing margin for fast-entry legs. EXPERIMENTAL starting
  point, not re-tuned by the D5.31 direction correction."""

  max_recovery_steps: int = 16
  """Hard safety-cap on RECOVERY phase duration (steps). NEEDS HUMAN
  CONFIRMATION (see audit report) -- chosen as roughly 2x the ~9-step
  natural FL swing duration measured for plain walking, to budget for the
  strictly harder retract+lift+advance+descend maneuver; this has NOT
  been directly profiled (only plain-swing timing has). If reached
  without the full two-stage `crossing_achieved` condition, the attempt
  is classified FAILURE (no success bonus), not success."""

  release_required_steps: int = 5

  post_recovery_stable_steps: int = 15
  """Consecutive POST_RECOVERY steps the robot must remain upright after a
  genuine crossing+release before final success is declared. NEEDS HUMAN
  CONFIRMATION (see audit report) -- chosen as roughly one Go2 trot gait
  cycle at this control rate, but not directly profiled for this task."""

  def build(self, env: ManagerBasedRlEnv) -> "ResidualRecoveryAction":
    return ResidualRecoveryAction(self, env)


class ResidualRecoveryAction(ActionTerm):
  """Residual recovery action term. See module docstring."""

  cfg: ResidualRecoveryActionCfg

  def __init__(self, cfg: ResidualRecoveryActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # self._entity already set by ActionTerm.__init__ via cfg.entity_name
    # (must be "robot" for this task).
    self._entity: Entity = self._entity

    # Reuse the EXISTING, UNMODIFIED JointPositionAction pipeline for
    # applying the final combined 12-dim raw action (scale/offset/
    # encoder-bias handling identical to the real velocity task).
    from mjlab.asset_zoo.robots import GO2_ACTION_SCALE

    inner_cfg = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO2_ACTION_SCALE,
      use_default_offset=True,
    )
    self._joint_pos_action = JointPositionAction(inner_cfg, env)

    # Verify joint order matches the frozen policy's training-time order.
    assert list(self._joint_pos_action.target_names) == list(WALK_JOINT_ORDER), (
      f"Unexpected joint order, refusing to compose blindly: "
      f"{self._joint_pos_action.target_names}"
    )

    leg = LEG_NAMES[cfg.affected_leg_idx]
    self._affected_idx = torch.tensor(
      [WALK_JOINT_ORDER.index(f"{leg}_hip_joint"),
       WALK_JOINT_ORDER.index(f"{leg}_thigh_joint"),
       WALK_JOINT_ORDER.index(f"{leg}_calf_joint")],
      device=self.device, dtype=torch.long,
    )

    self._policy = _load_frozen_walk_policy(cfg.walk_checkpoint, cfg.device)

    # pi_walk's OWN previous raw action (12-dim), tracked independently
    # of env.action_manager.action (which holds the recovery policy's
    # own 3-dim Delta_a). See module docstring: critical for faithful
    # observation reconstruction.
    self._last_walk_raw_action = torch.zeros(
      self.num_envs, 12, device=self.device
    )

    self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
    self._action_dim = 3

    # Last a_walk_raw / final_raw, exposed for verification/logging.
    self.last_a_walk_raw: torch.Tensor | None = None
    self.last_final_raw: torch.Tensor | None = None

    self._imu_lin_vel: BuiltinSensor = env.scene["robot/imu_lin_vel"]
    self._imu_ang_vel: BuiltinSensor = env.scene["robot/imu_ang_vel"]

    # --- Wire geometry (live, not static) ---
    from mjlab.tasks.recovery.mdp import wire_geometry

    assert cfg.wire_geom_names, (
      "ResidualRecoveryActionCfg.wire_geom_names must be set by the env_cfg "
      "factory (see BallCableCfg.geom_names())."
    )
    self._wire_geom_ids = wire_geometry.resolve_wire_segment_geom_ids(
      env, cfg.wire_geom_names
    )
    foot_site_names = list(self._entity.site_names)
    self._foot_site_idx = foot_site_names.index(LEG_NAMES[cfg.affected_leg_idx])

    # PRIMARY entanglement-clearance reference: the actual calf/shin geom
    # that the contact sensor's secondary match targets (see
    # `wire_contact_sensor_name` config in env_cfg_residual.py --
    # `FL_calf2_collision` etc.), NOT the toe site. An earlier version of
    # this detector used the toe site as the sole clearance reference and
    # was found (empirical probe, see chat history) to sit ~5.3cm BELOW
    # the wire's height even when the calf was genuinely lifted clear --
    # the toe is simply a lower point on the kinematic chain than the
    # body segment that actually snags the wire.
    # NOTE: raw `mujoco.mj_name2id` lookups fail here because compiled
    # geom names are namespaced with the entity's attachment prefix (e.g.
    # 'robot/FL_calf2_collision'), not the bare name. Resolve via the
    # entity's own name-matching API instead (same approach used above
    # for the toe site), which returns an ENTITY-LOCAL index usable with
    # `self._entity.data.geom_pos_w` (entity-scoped, analogous to
    # `site_pos_w`).
    calf_geom_name = f"{LEG_NAMES[cfg.affected_leg_idx]}_calf2_collision"
    calf_geom_ids, _ = self._entity.find_geoms(calf_geom_name)
    assert len(calf_geom_ids) == 1, (
      f"Calf collision geom '{calf_geom_name}' not found (or ambiguous) on "
      f"entity '{self._entity}' -- cannot track the primary "
      "entanglement-clearance reference."
    )
    self._calf_geom_idx = calf_geom_ids[0]

    # Per-step geometric quantities, recomputed every `_update_fsm` call,
    # exposed for reward/observation/diagnostic terms (see module
    # docstring and mjlab.tasks.recovery.mdp.wire_geometry).
    self.calf_forward_dist = torch.zeros(self.num_envs, device=self.device)
    self.calf_height_above_wire = torch.zeros(self.num_envs, device=self.device)
    self.calf_side_far = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.calf_clearance_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    self.foot_forward_dist = torch.zeros(self.num_envs, device=self.device)
    self.foot_height_above_wire = torch.zeros(self.num_envs, device=self.device)
    self.foot_side_far = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.foot_crossed_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    # Backward-compat alias (old single-stage name some callers may use).
    self.crossing_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # --- Potential-difference (delta) shaping state (post-audit rework,
    # see chat history: replaces the old absolute-clamp progress reward,
    # which had a large negative constant offset dwarfed by always-on
    # forward-velocity/uprightness rewards). Two independent potentials,
    # one per crossing stage:
    #   stage1 (pre calf-clearance): Phi1 = norm(calf_height) +
    #     norm(calf_forward_dist), both clamped to [-1, 1].
    #   stage2 (post calf-clearance, pre foot-crossing): Phi2 =
    #     norm(foot_forward_dist), clamped to [-1, 1].
    # Each potential is tracked as a MONOTONIC running best (not the raw
    # instantaneous value) so the reward only ever pays for genuine new
    # progress -- oscillating back toward the baseline or cycling between
    # two states earns zero, never negative, and never double-pays for
    # revisiting already-reached progress (see rewards.py docstrings and
    # the Part 5 "no wiggle" requirement). `stageN_progress_reward` is the
    # per-step nonnegative delta of the running best, exposed directly for
    # the reward functions to read (no duplicate geometry computation).
    self._stage1_best_potential = torch.zeros(self.num_envs, device=self.device)
    self._stage2_best_potential = torch.zeros(self.num_envs, device=self.device)
    self.stage1_progress_reward = torch.zeros(self.num_envs, device=self.device)
    self.stage2_progress_reward = torch.zeros(self.num_envs, device=self.device)

    # Robot's own intended forward direction in the world XY plane,
    # LATCHED at the instant of entering RECOVERY (from
    # `entity.data.heading_w`), so a later in-recovery yaw perturbation
    # (e.g. from flailing while entangled) does not redefine "forward"
    # mid-maneuver. See `wire_geometry.signed_forward_distance`.
    self._forward_dir_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self._forward_dir_xy[:, 0] = 1.0  # default +X until first latch

    # --- Recovery FSM state (per-env) ---
    self._phase = torch.full(
      (self.num_envs,), PHASE_NORMAL_WALK, device=self.device, dtype=torch.long
    )
    self._recovery_elapsed = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._release_counter = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._calf_clearance_achieved = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._foot_crossed_achieved = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    # Backward-compat alias: `crossing_achieved` now means the FULL
    # two-stage condition (calf clearance AND foot crossed).
    self._crossing_achieved = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._prev_calf_side_far = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._recovery_failed_timeout = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._post_recovery_elapsed = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._stable_since_post_recovery = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._post_recovery_eligible = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._already_succeeded_this_episode = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    # One-shot flags for the step just completed, exposed for reward/
    # observation terms to read (newly_entangled: NORMAL_WALK->RECOVERY
    # this step; newly_recovered: RECOVERY->POST_RECOVERY this step,
    # regardless of success/failure classification; newly_succeeded: the
    # FINAL geometric-clearance + release + post-recovery-stability
    # success event, which may fire many steps after newly_recovered).
    self.newly_entangled = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self.newly_recovered = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self.newly_succeeded = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )

    # --- D5: Phase-aware threshold tracking (only used when cfg.
    # phase_aware_thresholds=True; inert/unused otherwise so baseline
    # behavior is completely unaffected). ---
    # Entry leg speed (computed at recovery trigger, stored for duration
    # of recovery phase): leg_speed = |thigh_vel| + |calf_vel|
    self._entry_leg_speed = torch.zeros(self.num_envs, device=self.device)
    # Effective thresholds (per-env, recomputed at entering_recovery based
    # on _entry_leg_speed): one of three tiers selected.
    self._effective_clearance_margin = torch.zeros(self.num_envs, device=self.device)
    self._effective_foot_crossing_margin = torch.zeros(self.num_envs, device=self.device)

    # --- D1 deterministic-mode fixed delta (built once; zero on the
    # ab/ad/hip index per the binary normal/obstructed-swing spec: the
    # affected leg's ab/ad continues to receive the frozen walking
    # policy's own command, unmodified). Broadcast over the batch each
    # step rather than re-built every call. ---
    self._deterministic_delta = torch.tensor(
      [0.0, cfg.thigh_recovery_delta, cfg.calf_recovery_delta],
      device=self.device,
    )

    # --- D3 hybrid-mode actuator bookkeeping (only set up when
    # recovery_mode == "hybrid"; no-op/unallocated otherwise so D1/D2
    # ppo_residual/deterministic paths are completely unaffected). ---
    self._hybrid_initialized = False
    if cfg.recovery_mode == "hybrid":
      self._init_hybrid_actuators(env)

    # Lightweight debug state: only print on phase-transition EVENTS (not
    # every step), per the "do not spam logs" requirement.
    self._logged_entry_step: int = -1

  def _find_actuator_ctrl_id(self, joint_name: str, actuator_cls=None) -> int:
    """Resolve the global MuJoCo ctrl id of the actuator targeting
    ``joint_name``. If ``actuator_cls`` is given, only actuators of that
    exact class are considered (needed to disambiguate the existing
    position actuator from the new auxiliary actuator on the same
    joint)."""
    for act in self._entity.actuators:
      if actuator_cls is not None and type(act) is not actuator_cls:
        continue
      if joint_name in act.target_names:
        idx = act.target_names.index(joint_name)
        return int(act.global_ctrl_ids[idx].item())
    raise RuntimeError(
      f"No {'matching' if actuator_cls is None else actuator_cls.__name__} "
      f"actuator found targeting joint '{joint_name}'."
    )

  def _update_phase_aware_thresholds(self, entry_leg_speed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """D5/D5.31: Compute effective crossing-latch thresholds based on
    entry leg speed (thigh + calf velocity) using three tiers, with
    margins selected from ``cfg`` so the direction correction lives in
    one place (the cfg defaults above):
      - Slow (leg_speed < 4.5): largest margins (cfg default 0.030/0.040)
      - Medium (4.5 <= leg_speed < 6.5): intermediate (cfg default 0.025/0.035)
      - Fast (leg_speed >= 6.5): smallest margins (cfg default 0.020/0.030)

    Both latch conditions in `_update_fsm` are strict `>` comparisons, so
    a LARGER margin is a STRICTER (harder to satisfy) requirement. D5.31
    assigns the largest margins to the slow tier and the smallest to the
    fast tier so that faster entries are never held to a stricter latch
    bar than slower ones (see the `phase_aware_thresholds` docstring for
    the full rationale and the D5.29 bug this corrects).

    Args:
      entry_leg_speed: [B] tensor of leg speed (rad/s) at recovery trigger

    Returns:
      (clearance_margin, foot_crossing_margin) tuple of [B] tensors with
      effective thresholds for each env selected based on speed tier.
    """
    cfg = self.cfg
    device = self.device

    # Initialize to slow-tier defaults
    clearance = torch.full_like(
      entry_leg_speed,
      cfg.phase_aware_slow_clearance_margin,
      device=device
    )
    crossing = torch.full_like(
      entry_leg_speed,
      cfg.phase_aware_slow_foot_crossing_margin,
      device=device
    )

    # Medium tier: 4.5 <= leg_speed < 6.5
    medium_mask = (entry_leg_speed >= cfg.phase_aware_speed_threshold_low) & \
                  (entry_leg_speed < cfg.phase_aware_speed_threshold_high)
    clearance[medium_mask] = cfg.phase_aware_medium_clearance_margin
    crossing[medium_mask] = cfg.phase_aware_medium_foot_crossing_margin

    # Fast tier: leg_speed >= 6.5
    fast_mask = entry_leg_speed >= cfg.phase_aware_speed_threshold_high
    clearance[fast_mask] = cfg.phase_aware_fast_clearance_margin
    crossing[fast_mask] = cfg.phase_aware_fast_foot_crossing_margin

    return clearance, crossing

  def _init_hybrid_actuators(self, env: ManagerBasedRlEnv) -> None:
    """D3: resolve ctrl ids for the affected leg's existing position
    actuators (thigh/calf) and the new auxiliary actuators (velocity on
    thigh, motor on calf -- added to the recovery task's Go2 entity by
    env_cfg_residual.py, NEVER to the normal walking task), and capture
    the ORIGINAL position-actuator gain/bias so they can be restored
    EXACTLY (not hardcoded) on recovery exit.

    This requires ``actuator_gainprm``/``actuator_biasprm`` to have real
    per-world (per-env) memory -- ``expand_model_fields`` is the same
    primitive the existing ``dr.pd_gains`` domain-randomization function
    uses (mjlab.envs.mdp.dr.actuator) for per-env PD gain overrides.
    """
    from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
    from mjlab.tasks.recovery.mdp.hybrid_actuators import (
      NamedMotorActuator,
      NamedVelocityActuator,
    )

    env.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm"))

    leg = LEG_NAMES[self.cfg.affected_leg_idx]
    thigh_joint = f"{leg}_thigh_joint"
    calf_joint = f"{leg}_calf_joint"

    self._pos_thigh_ctrl_id = self._find_actuator_ctrl_id(
      thigh_joint, BuiltinPositionActuator
    )
    self._pos_calf_ctrl_id = self._find_actuator_ctrl_id(
      calf_joint, BuiltinPositionActuator
    )
    self._aux_thigh_ctrl_id = self._find_actuator_ctrl_id(
      thigh_joint, NamedVelocityActuator
    )
    self._aux_calf_ctrl_id = self._find_actuator_ctrl_id(
      calf_joint, NamedMotorActuator
    )

    # Local (entity-scoped) joint indices, for set_joint_velocity_target/
    # set_joint_effort_target (which index by JOINT, not ctrl id).
    joint_names = list(self._entity.joint_names)
    self._hybrid_thigh_joint_id = joint_names.index(thigh_joint)
    self._hybrid_calf_joint_id = joint_names.index(calf_joint)

    # Capture ORIGINAL position-actuator gain/bias EXACTLY (per-world,
    # since expand_model_fields allocates one copy per env) -- restored
    # verbatim on recovery exit, never hardcoded/re-derived.
    self._orig_thigh_gainprm = env.sim.model.actuator_gainprm[
      :, self._pos_thigh_ctrl_id, :
    ].clone()
    self._orig_thigh_biasprm = env.sim.model.actuator_biasprm[
      :, self._pos_thigh_ctrl_id, :
    ].clone()
    self._orig_calf_gainprm = env.sim.model.actuator_gainprm[
      :, self._pos_calf_ctrl_id, :
    ].clone()
    self._orig_calf_biasprm = env.sim.model.actuator_biasprm[
      :, self._pos_calf_ctrl_id, :
    ].clone()

    # Per-env instrumentation (exposed for the D3 trajectory logger).
    self.hybrid_thigh_vel_ctrl = torch.zeros(self.num_envs, device=self.device)
    self.hybrid_calf_torque_ctrl = torch.zeros(self.num_envs, device=self.device)
    self.hybrid_thigh_pos_gain_active = torch.ones(
      self.num_envs, device=self.device, dtype=torch.bool
    )

    # EXPERIMENTAL peak-detector state (inert unless
    # cfg.experimental_peak_triggered_thigh_stop=True; see cfg docstring
    # for the rule). Allocated unconditionally (cheap, 5 small tensors)
    # so toggling the flag never requires re-initialization.
    self._peak_armed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._peak_running_max = torch.zeros(self.num_envs, device=self.device)
    self._peak_entry_height = torch.zeros(self.num_envs, device=self.device)
    self._peak_decline_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._peak_triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    # Debug/logging only: recovery_elapsed step at which the trigger
    # fired for each env (-1 = not yet triggered this recovery attempt).
    self.peak_trigger_step = torch.full(
      (self.num_envs,), -1, device=self.device, dtype=torch.long
    )

    self._hybrid_initialized = True

  def _apply_hybrid_actuators(self) -> None:
    """D3: per-step hybrid actuator-level control. Must be called AFTER
    `_update_fsm` (so `self._phase`/`newly_entangled`/`newly_recovered`
    reflect this step) and is independent of `process_actions`'s
    PPO/deterministic branch (hybrid mode ignores `actions` entirely --
    the recovery behavior lives entirely at the actuator level, not in
    the composed `final_raw` 12-vector)."""
    in_recovery = self._phase == PHASE_RECOVERY
    sim = self._env.sim

    enter_ids = self.newly_entangled.nonzero(as_tuple=True)[0]
    exit_ids = self.newly_recovered.nonzero(as_tuple=True)[0]

    if enter_ids.numel() > 0:
      gainprm_width = sim.model.actuator_gainprm.shape[-1]
      biasprm_width = sim.model.actuator_biasprm.shape[-1]
      zero_gain = torch.zeros(enter_ids.numel(), gainprm_width, device=self.device)
      zero_bias = torch.zeros(enter_ids.numel(), biasprm_width, device=self.device)
      # Disable the affected leg's thigh/calf POSITION actuators (zero
      # gain AND bias -- see docstring warning below: zeroing gainprm
      # alone is NOT sufficient).
      sim.model.actuator_gainprm[enter_ids, self._pos_thigh_ctrl_id, :] = zero_gain
      sim.model.actuator_biasprm[enter_ids, self._pos_thigh_ctrl_id, :] = zero_bias
      sim.model.actuator_gainprm[enter_ids, self._pos_calf_ctrl_id, :] = zero_gain
      sim.model.actuator_biasprm[enter_ids, self._pos_calf_ctrl_id, :] = zero_bias
      # Enable the auxiliary thigh VELOCITY actuator's damping (it must
      # be OFF outside RECOVERY -- a velocity actuator with ctrl=0 but
      # nonzero kd still applies -kd*vel passive damping at all times,
      # which would corrupt normal walking; see module docstring).
      sim.model.actuator_gainprm[enter_ids, self._aux_thigh_ctrl_id, 0] = (
        self.cfg.recovery_thigh_velocity_kd
      )
      sim.model.actuator_biasprm[enter_ids, self._aux_thigh_ctrl_id, 2] = (
        -self.cfg.recovery_thigh_velocity_kd
      )
      self.hybrid_thigh_pos_gain_active[enter_ids] = False

      # EXPERIMENTAL peak-detector: reset per-env state on every fresh
      # RECOVERY entry (inert unless cfg.experimental_peak_triggered_
      # thigh_stop=True, but reset unconditionally so the flag can be
      # toggled without stale state).
      self._peak_armed[enter_ids] = False
      self._peak_running_max[enter_ids] = self.foot_height_above_wire[enter_ids]
      self._peak_entry_height[enter_ids] = self.foot_height_above_wire[enter_ids]
      self._peak_decline_counter[enter_ids] = 0
      self._peak_triggered[enter_ids] = False
      self.peak_trigger_step[enter_ids] = -1

    if exit_ids.numel() > 0:
      # Restore EXACT original position-actuator gain/bias (not
      # hardcoded -- captured once at init in _init_hybrid_actuators).
      sim.model.actuator_gainprm[exit_ids, self._pos_thigh_ctrl_id, :] = (
        self._orig_thigh_gainprm[exit_ids]
      )
      sim.model.actuator_biasprm[exit_ids, self._pos_thigh_ctrl_id, :] = (
        self._orig_thigh_biasprm[exit_ids]
      )
      sim.model.actuator_gainprm[exit_ids, self._pos_calf_ctrl_id, :] = (
        self._orig_calf_gainprm[exit_ids]
      )
      sim.model.actuator_biasprm[exit_ids, self._pos_calf_ctrl_id, :] = (
        self._orig_calf_biasprm[exit_ids]
      )
      # Disable the auxiliary velocity actuator again.
      sim.model.actuator_gainprm[exit_ids, self._aux_thigh_ctrl_id, 0] = 0.0
      sim.model.actuator_biasprm[exit_ids, self._aux_thigh_ctrl_id, 2] = 0.0
      self.hybrid_thigh_pos_gain_active[exit_ids] = True

    # EXPERIMENTAL peak-detector update (no-op unless cfg.
    # experimental_peak_triggered_thigh_stop=True). Runs every step
    # while in_recovery, using ONLY the current and past samples of
    # self.foot_height_above_wire (already computed this step by
    # _update_fsm, which always runs before this method) -- no future
    # samples or offline trajectory knowledge are used.
    #
    # Rule (see cfg docstrings for threshold rationale):
    #   1. ARM once foot_height_above_wire has risen >=
    #      peak_detector_rise_threshold above its value at recovery
    #      entry (prevents triggering before any meaningful rise).
    #   2. Once armed, track a running max of foot_height_above_wire.
    #   3. A step counts as "declining" if foot_height_above_wire is >=
    #      peak_detector_decline_eps below that running max.
    #   4. TRIGGER (latched) after peak_detector_confirm_steps
    #      CONSECUTIVE declining steps.
    if self.cfg.experimental_peak_triggered_thigh_stop:
      h = self.foot_height_above_wire
      active = in_recovery & (~self._peak_triggered)

      rise = h - self._peak_entry_height
      newly_armed = active & (~self._peak_armed) & (
        rise >= self.cfg.peak_detector_rise_threshold
      )
      armed_now = self._peak_armed | newly_armed

      prev_running_max = self._peak_running_max
      declined = active & armed_now & (
        h < prev_running_max - self.cfg.peak_detector_decline_eps
      )
      self._peak_decline_counter = torch.where(
        declined,
        self._peak_decline_counter + 1,
        torch.where(
          active & armed_now,
          torch.zeros_like(self._peak_decline_counter),
          self._peak_decline_counter,
        ),
      )

      trigger_now = (
        active
        & armed_now
        & (self._peak_decline_counter >= self.cfg.peak_detector_confirm_steps)
      )
      if bool(trigger_now.any()):
        trig_ids = trigger_now.nonzero(as_tuple=True)[0]
        self.peak_trigger_step[trig_ids] = self._recovery_elapsed[trig_ids]
        print(
          f"[peak-detector] env{trig_ids.tolist()} TRIGGERED at "
          f"recovery_elapsed={self._recovery_elapsed[trig_ids].tolist()} "
          f"foot_height_above_wire={h[trig_ids].tolist()} "
          f"running_max={prev_running_max[trig_ids].tolist()} "
          f"(entry_height={self._peak_entry_height[trig_ids].tolist()})"
        )

      self._peak_running_max = torch.where(
        active & armed_now, torch.maximum(prev_running_max, h), prev_running_max
      )
      self._peak_armed = armed_now
      self._peak_triggered = self._peak_triggered | trigger_now

    # Per-step command: nonzero only while in_recovery, zero otherwise
    # (no stale recovery command persists once RECOVERY ends -- the
    # auxiliary velocity actuator's gain is ALSO zeroed above on exit,
    # so even if a stale nonzero target lingered for one step it would
    # already be physically inert; writing zero here is a second,
    # redundant safeguard).
    thigh_vel_cmd = torch.where(
      in_recovery,
      torch.full_like(self.hybrid_thigh_vel_ctrl, -self.cfg.recovery_thigh_velocity),
      torch.zeros_like(self.hybrid_thigh_vel_ctrl),
    )
    if self.cfg.experimental_peak_triggered_thigh_stop:
      # ONLY the thigh velocity command is affected -- calf torque,
      # ab/ad, and all FSM/phase logic are untouched. Uses the SAME
      # existing set_joint_velocity_target call below (no new control
      # mechanism); zeroing the target does not forcibly stop the joint
      # (no brake/position-hold is applied), so the thigh may continue
      # to coast under residual momentum and/or the damping term already
      # present in the auxiliary velocity actuator (kd * (0 - qvel)) --
      # this is intentional and exactly what Step 6's "momentum or other
      # dynamic effects" classification is designed to detect.
      thigh_vel_cmd = torch.where(
        self._peak_triggered, torch.zeros_like(thigh_vel_cmd), thigh_vel_cmd
      )
    calf_torque_cmd = torch.where(
      in_recovery,
      torch.full_like(self.hybrid_calf_torque_ctrl, -self.cfg.recovery_calf_torque),
      torch.zeros_like(self.hybrid_calf_torque_ctrl),
    )
    self.hybrid_thigh_vel_ctrl = thigh_vel_cmd
    self.hybrid_calf_torque_ctrl = calf_torque_cmd

    self._entity.set_joint_velocity_target(
      thigh_vel_cmd.unsqueeze(-1),
      joint_ids=[self._hybrid_thigh_joint_id],
    )
    self._entity.set_joint_effort_target(
      calf_torque_cmd.unsqueeze(-1),
      joint_ids=[self._hybrid_calf_joint_id],
    )

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def phase(self) -> torch.Tensor:
    """Current FSM phase per env: 0=NORMAL_WALK, 1=RECOVERY,
    2=POST_RECOVERY. Long tensor, shape [B]."""
    return self._phase

  @property
  def recovery_elapsed(self) -> torch.Tensor:
    """Steps elapsed since entering RECOVERY (0 outside RECOVERY). Long
    tensor, shape [B]."""
    return self._recovery_elapsed

  @property
  def crossing_achieved(self) -> torch.Tensor:
    """Latched bool, shape [B]: True once the FULL two-stage crossing
    condition has been met during the current recovery attempt -- calf
    clearance (above wire + near->far) AND subsequent toe/foot far-side
    crossing. Reset on each new NORMAL_WALK->RECOVERY entry."""
    return self._crossing_achieved

  @property
  def calf_clearance_achieved(self) -> torch.Tensor:
    """Latched bool, shape [B]: True once the calf (PRIMARY entanglement
    reference) has genuinely cleared the wire -- above it by
    ``clearance_margin`` AND crossed near->far -- during the current
    recovery attempt."""
    return self._calf_clearance_achieved

  @property
  def foot_crossed_achieved(self) -> torch.Tensor:
    """Latched bool, shape [B]: True once the toe/foot has reached the
    far side of the wire by ``foot_crossing_margin``, AFTER
    ``calf_clearance_achieved`` latched (secondary confirmation)."""
    return self._foot_crossed_achieved

  @property
  def release_achieved(self) -> torch.Tensor:
    """Bool, shape [B]: True once sustained no-contact has been observed
    for ``release_required_steps`` AFTER ``crossing_achieved`` latched
    (i.e. the genuine, geometry-gated release condition)."""
    return self._crossing_achieved & (self._release_counter >= self.cfg.release_required_steps)

  @property
  def recovery_failed_timeout(self) -> torch.Tensor:
    """Latched bool, shape [B]: True if the most recent RECOVERY phase
    exited via the ``max_recovery_steps`` safety cap WITHOUT genuine
    crossing+release (i.e. a failed attempt), else False."""
    return self._recovery_failed_timeout

  @property
  def post_recovery_elapsed(self) -> torch.Tensor:
    """Steps elapsed since entering POST_RECOVERY (0 outside it). Long
    tensor, shape [B]."""
    return self._post_recovery_elapsed

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._last_walk_raw_action.zero_()
      self._phase.fill_(PHASE_NORMAL_WALK)
      self._recovery_elapsed.zero_()
      self._release_counter.zero_()
      self._calf_clearance_achieved.zero_()
      self._foot_crossed_achieved.zero_()
      self._crossing_achieved.zero_()
      self._prev_calf_side_far.zero_()
      self._recovery_failed_timeout.zero_()
      self._post_recovery_elapsed.zero_()
      self._stable_since_post_recovery.zero_()
      self._post_recovery_eligible.zero_()
      self._already_succeeded_this_episode.zero_()
      self.newly_entangled.zero_()
      self.newly_recovered.zero_()
      self.newly_succeeded.zero_()
      self._stage1_best_potential.zero_()
      self._stage2_best_potential.zero_()
      self.stage1_progress_reward.zero_()
      self.stage2_progress_reward.zero_()
    else:
      self._last_walk_raw_action[env_ids] = 0.0
      self._phase[env_ids] = PHASE_NORMAL_WALK
      self._recovery_elapsed[env_ids] = 0
      self._release_counter[env_ids] = 0
      self._calf_clearance_achieved[env_ids] = False
      self._foot_crossed_achieved[env_ids] = False
      self._crossing_achieved[env_ids] = False
      self._prev_calf_side_far[env_ids] = False
      self._recovery_failed_timeout[env_ids] = False
      self._post_recovery_elapsed[env_ids] = 0
      self._stable_since_post_recovery[env_ids] = False
      self._post_recovery_eligible[env_ids] = False
      self._already_succeeded_this_episode[env_ids] = False
      self.newly_entangled[env_ids] = False
      self.newly_recovered[env_ids] = False
      self.newly_succeeded[env_ids] = False
      self._stage1_best_potential[env_ids] = 0.0
      self._stage2_best_potential[env_ids] = 0.0
      self.stage1_progress_reward[env_ids] = 0.0
      self.stage2_progress_reward[env_ids] = 0.0
    self._joint_pos_action.reset(env_ids)

  def _update_fsm(self) -> None:
    """Advances the recovery FSM by one step, using the real FL-wire
    contact sensor as the entanglement/release signal, and the TWO-STAGE
    geometric crossing condition (calf clearance -> toe/foot far-side
    confirmation) as the recovery-completion signal (see module docstring
    and cfg docstring for the full transition rules). Must be called
    exactly once per environment step, before the Delta_a gate is applied
    in `process_actions`.
    """
    from mjlab.tasks.recovery.mdp import wire_geometry

    sensor = self._env.scene.sensors[self.cfg.wire_contact_sensor_name]
    found = sensor.data.found  # [B, P]
    in_contact = (found > 0).any(dim=-1)  # [B]

    is_normal = self._phase == PHASE_NORMAL_WALK
    is_recovery = self._phase == PHASE_RECOVERY

    # NORMAL_WALK -> RECOVERY on first genuine contact. Latch the robot's
    # OWN intended forward direction (world XY heading) AT THIS INSTANT,
    # so later in-recovery yaw perturbations (flailing while entangled)
    # don't redefine "forward" mid-maneuver (see
    # `wire_geometry.signed_forward_distance`).
    entering_recovery = is_normal & in_contact

    # --- D5: Compute entry leg speed at recovery trigger (for phase-aware
    # threshold selection). Leg speed = |thigh_vel| + |calf_vel|. ---
    if self.cfg.phase_aware_thresholds:
      # Get the affected leg's joint velocities from the entity data
      # (joint_vel is [B, num_joints], extract thigh and calf indices)
      leg = LEG_NAMES[self.cfg.affected_leg_idx]
      thigh_name = f"{leg}_thigh_joint"
      calf_name = f"{leg}_calf_joint"
      joint_names = list(self._entity.joint_names)
      thigh_idx = joint_names.index(thigh_name)
      calf_idx = joint_names.index(calf_name)

      thigh_vel = torch.abs(self._entity.data.joint_vel[:, thigh_idx])  # [B]
      calf_vel = torch.abs(self._entity.data.joint_vel[:, calf_idx])    # [B]
      entry_leg_speed = thigh_vel + calf_vel  # [B]

      # Update entry leg speed when entering recovery
      self._entry_leg_speed = torch.where(
        entering_recovery, entry_leg_speed, self._entry_leg_speed
      )

      # Compute effective thresholds based on entry leg speed
      clearance_margins, crossing_margins = self._update_phase_aware_thresholds(
        self._entry_leg_speed
      )
      self._effective_clearance_margin = clearance_margins
      self._effective_foot_crossing_margin = crossing_margins
    else:
      # Baseline: use static thresholds from config
      self._effective_clearance_margin[:] = self.cfg.clearance_margin
      self._effective_foot_crossing_margin[:] = self.cfg.foot_crossing_margin

    self._phase = torch.where(
      entering_recovery, torch.full_like(self._phase, PHASE_RECOVERY), self._phase
    )
    heading = self._entity.data.heading_w  # [B], radians
    new_forward_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    self._forward_dir_xy = torch.where(
      entering_recovery.unsqueeze(-1), new_forward_dir, self._forward_dir_xy
    )

    # --- Live wire-geometry-aware state for BOTH tracked points (every
    # step, regardless of phase, so observations/rewards always have a
    # fresh value). ---
    seg_pos = wire_geometry.wire_segment_world_pos(self._env, self._wire_geom_ids)

    calf_pos = self._entity.data.geom_pos_w[:, self._calf_geom_idx, :]  # [B, 3]
    calf_local_x, calf_local_z = wire_geometry.local_wire_xz_at_y(seg_pos, calf_pos[:, 1])
    calf_forward_dist = wire_geometry.signed_forward_distance(
      calf_pos[:, :2],
      torch.stack([calf_local_x, calf_pos[:, 1]], dim=-1),
      self._forward_dir_xy,
    )
    calf_height_above_wire = calf_pos[:, 2] - (calf_local_z + self.cfg.wire_radius)
    calf_side_far = calf_forward_dist > 0.0
    calf_lifted = calf_height_above_wire > self._effective_clearance_margin

    foot_pos = self._entity.data.site_pos_w[:, self._foot_site_idx, :]  # [B, 3]
    foot_local_x, foot_local_z = wire_geometry.local_wire_xz_at_y(seg_pos, foot_pos[:, 1])
    foot_forward_dist = wire_geometry.signed_forward_distance(
      foot_pos[:, :2],
      torch.stack([foot_local_x, foot_pos[:, 1]], dim=-1),
      self._forward_dir_xy,
    )
    foot_height_above_wire = foot_pos[:, 2] - (foot_local_z + self.cfg.wire_radius)
    foot_side_far = foot_forward_dist > 0.0

    self.calf_forward_dist = calf_forward_dist
    self.calf_height_above_wire = calf_height_above_wire
    self.calf_side_far = calf_side_far
    self.foot_forward_dist = foot_forward_dist
    self.foot_height_above_wire = foot_height_above_wire
    self.foot_side_far = foot_side_far

    self._recovery_elapsed = torch.where(
      entering_recovery, torch.zeros_like(self._recovery_elapsed), self._recovery_elapsed
    )
    self._release_counter = torch.where(
      entering_recovery, torch.zeros_like(self._release_counter), self._release_counter
    )
    self._calf_clearance_achieved = torch.where(
      entering_recovery, torch.zeros_like(self._calf_clearance_achieved),
      self._calf_clearance_achieved,
    )
    self._foot_crossed_achieved = torch.where(
      entering_recovery, torch.zeros_like(self._foot_crossed_achieved),
      self._foot_crossed_achieved,
    )
    self._crossing_achieved = torch.where(
      entering_recovery, torch.zeros_like(self._crossing_achieved), self._crossing_achieved
    )
    self._recovery_failed_timeout = torch.where(
      entering_recovery, torch.zeros_like(self._recovery_failed_timeout),
      self._recovery_failed_timeout,
    )
    # Initialize side tracking on entry so the very first in-recovery
    # step can't spuriously register a calf-clearance event.
    self._prev_calf_side_far = torch.where(
      entering_recovery, calf_side_far.clone(), self._prev_calf_side_far
    )

    # Within RECOVERY: advance elapsed counter, detect the two-stage
    # crossing, and only THEN start counting sustained release.
    still_recovery_after_entry = is_recovery | entering_recovery
    self._recovery_elapsed = torch.where(
      still_recovery_after_entry, self._recovery_elapsed + 1, self._recovery_elapsed
    )

    # STAGE 1 -- calf clearance: NEAR -> FAR transition of the CALF
    # (primary entanglement reference) WHILE it is above the wire by
    # `clearance_margin`. A near->far transition while the calf is still
    # at/below the wire (sliding/scraping past at ground level, the
    # originally observed failure mode) does NOT latch.
    calf_clearance_event = (
      still_recovery_after_entry & (~self._prev_calf_side_far) & calf_side_far & calf_lifted
    )
    self.calf_clearance_event = calf_clearance_event
    self._calf_clearance_achieved = self._calf_clearance_achieved | calf_clearance_event

    # STAGE 2 -- foot/toe crossing confirmation: ONLY evaluated after
    # calf clearance has already latched. Requires the toe to reach the
    # far side by `foot_crossing_margin` -- NOT required to be above the
    # wire at all (see module docstring: the toe is a lower kinematic
    # point than the calf and is not what gets entangled).
    foot_crossed_event = (
      still_recovery_after_entry & self._calf_clearance_achieved
      & (~self._foot_crossed_achieved)
      & (foot_forward_dist > self._effective_foot_crossing_margin)
    )
    self.foot_crossed_event = foot_crossed_event
    self._foot_crossed_achieved = self._foot_crossed_achieved | foot_crossed_event

    # Backward-compat aliases: "crossing" now means the FULL two-stage
    # condition is satisfied.
    full_crossing_now = self._calf_clearance_achieved & self._foot_crossed_achieved
    crossing_event = full_crossing_now & (~self._crossing_achieved) & still_recovery_after_entry
    self.crossing_event = crossing_event
    self._crossing_achieved = self._crossing_achieved | full_crossing_now

    eligible_for_release = still_recovery_after_entry & self._crossing_achieved
    self._release_counter = torch.where(
      eligible_for_release & ~in_contact,
      self._release_counter + 1,
      torch.where(eligible_for_release & in_contact,
                  torch.zeros_like(self._release_counter), self._release_counter),
    )

    # --- Potential-difference (delta) shaping, stage 1 & 2 (see
    # `_stage1_best_potential`/`_stage2_best_potential` docstring above).
    # Computed AFTER the stage-1/stage-2 latches above so
    # `calf_clearance_achieved` already reflects this step's own event.
    norm_calf_h = torch.clamp(calf_height_above_wire / WIRE_HEIGHT_NORM_SCALE, -1.0, 1.0)
    norm_calf_f = torch.clamp(calf_forward_dist / WIRE_FORWARD_NORM_SCALE, -1.0, 1.0)
    norm_foot_f = torch.clamp(foot_forward_dist / WIRE_FORWARD_NORM_SCALE, -1.0, 1.0)
    phi1 = norm_calf_h + norm_calf_f
    phi2 = norm_foot_f

    stage1_active = still_recovery_after_entry & ~self._calf_clearance_achieved
    stage2_active = (
      still_recovery_after_entry & self._calf_clearance_achieved & ~self._foot_crossed_achieved
    )

    # Baseline the running-best potential to the CURRENT value (not zero)
    # the instant a stage becomes newly active, so the very first active
    # step never spuriously pays out a large one-shot "progress" reward
    # just from the stage switching on. Stage 1 baselines at
    # `entering_recovery`; stage 2 baselines at the step `calf_clearance_
    # event` latches (the transition INTO stage 2).
    self._stage1_best_potential = torch.where(
      entering_recovery, phi1, self._stage1_best_potential
    )
    self._stage2_best_potential = torch.where(
      calf_clearance_event, phi2, self._stage2_best_potential
    )

    new_best1 = torch.where(
      stage1_active, torch.maximum(self._stage1_best_potential, phi1), self._stage1_best_potential
    )
    new_best2 = torch.where(
      stage2_active, torch.maximum(self._stage2_best_potential, phi2), self._stage2_best_potential
    )
    self.stage1_progress_reward = torch.where(
      stage1_active, new_best1 - self._stage1_best_potential, torch.zeros_like(phi1)
    )
    self.stage2_progress_reward = torch.where(
      stage2_active, new_best2 - self._stage2_best_potential, torch.zeros_like(phi2)
    )
    self._stage1_best_potential = new_best1
    self._stage2_best_potential = new_best2

    released_sustained = self._release_counter >= self.cfg.release_required_steps
    genuine_recovery_complete = self._crossing_achieved & released_sustained
    cap_reached = self._recovery_elapsed >= self.cfg.max_recovery_steps
    exiting_recovery = still_recovery_after_entry & (genuine_recovery_complete | cap_reached)

    self._recovery_failed_timeout = torch.where(
      exiting_recovery & ~genuine_recovery_complete,
      torch.ones_like(self._recovery_failed_timeout),
      self._recovery_failed_timeout,
    )

    self._phase = torch.where(
      exiting_recovery, torch.full_like(self._phase, PHASE_POST_RECOVERY), self._phase
    )
    self._post_recovery_elapsed = torch.where(
      exiting_recovery, torch.zeros_like(self._post_recovery_elapsed), self._post_recovery_elapsed
    )
    self._stable_since_post_recovery = torch.where(
      exiting_recovery, torch.ones_like(self._stable_since_post_recovery), self._stable_since_post_recovery
    )

    self._prev_calf_side_far = calf_side_far.clone()

    self.newly_entangled = entering_recovery
    self.newly_recovered = exiting_recovery

    # --- POST_RECOVERY stability window -> FINAL success ---
    # CRITICAL: only an episode that exited RECOVERY via genuine
    # crossing+release (`genuine_recovery_complete`) is even ELIGIBLE for
    # the stability window to lead to success. A cap-timeout exit
    # (failure) must NOT be able to accrue success later just because the
    # frozen walking policy happens to keep walking stably afterward --
    # that would silently reintroduce a close cousin of the original
    # contact-release exploit (this was caught during Test A/D
    # validation: a zero-residual/cap-timeout episode was initially still
    # earning `newly_succeeded` purely from post-recovery stability).
    self._post_recovery_eligible = torch.where(
      exiting_recovery, genuine_recovery_complete, self._post_recovery_eligible
    )

    # Only steps strictly AFTER the transition step count toward the
    # window (the transition step itself starts the counter at 0).
    still_post = (self._phase == PHASE_POST_RECOVERY) & ~exiting_recovery
    gravity_b = self._entity.data.projected_gravity_b
    upright_now = torch.abs(gravity_b[:, 2]) > math.cos(math.radians(70.0))
    self._stable_since_post_recovery = torch.where(
      still_post, self._stable_since_post_recovery & upright_now, self._stable_since_post_recovery
    )
    self._post_recovery_elapsed = torch.where(
      still_post, self._post_recovery_elapsed + 1, self._post_recovery_elapsed
    )
    reached_stable_window = (
      still_post & self._post_recovery_eligible & self._stable_since_post_recovery
      & (self._post_recovery_elapsed >= self.cfg.post_recovery_stable_steps)
    )
    just_succeeded = reached_stable_window & ~self._already_succeeded_this_episode
    self.newly_succeeded = just_succeeded
    self._already_succeeded_this_episode = self._already_succeeded_this_episode | just_succeeded

  def _build_walk_obs(self) -> torch.Tensor:
    """Reconstructs the EXACT 48-dim observation vector the frozen
    Mjlab-Velocity-Flat-Unitree-Go2 policy expects, term-by-term audited
    against mjlab.tasks.velocity.velocity_env_cfg's actor group:
      base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
      + joint_pos_rel(biased=True)(12) + joint_vel_rel(12)
      + last_action(12) + command(3) = 48.
    Play-mode has enable_corruption=False (no noise), which this always
    matches (no noise added here either).
    """
    asset = self._entity

    base_lin_vel = self._imu_lin_vel.data
    base_ang_vel = self._imu_ang_vel.data
    projected_gravity = asset.data.projected_gravity_b

    default_joint_pos = asset.data.default_joint_pos
    joint_pos_biased = asset.data.joint_pos_biased
    joint_pos_rel = joint_pos_biased - default_joint_pos

    default_joint_vel = asset.data.default_joint_vel
    joint_vel_rel = asset.data.joint_vel - default_joint_vel

    last_action = self._last_walk_raw_action

    command = self._env.command_manager.get_command(self.cfg.command_name)
    assert command is not None, (
      f"ResidualRecoveryAction requires a '{self.cfg.command_name}' command "
      f"term to reconstruct pi_walk's observation."
    )

    return torch.cat(
      [base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel,
       joint_vel_rel, last_action, command],
      dim=-1,
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    """`actions` is Delta_a, shape [B, 3] (affected leg only, raw
    policy-action-space units). IGNORED entirely when
    ``cfg.recovery_mode == "deterministic"`` -- see cfg docstring."""
    # Advance the recovery FSM first, using the CURRENT (pre-this-step)
    # wire-contact state, then gate Delta_a by the resulting phase: only
    # RECOVERY allows the residual through. This must happen before the
    # physical action is computed/applied, since it determines whether
    # Delta_a has any effect this step at all. UNCHANGED for both modes:
    # D1's deterministic controller reuses the exact same oracle
    # contact-sensor entry trigger and geometric crossing+release exit
    # condition as the PPO-residual path (see D0/D0.5 audit -- this FSM
    # is phase/geometry/contact driven, not PPO-driven, so it is already
    # mode-agnostic).
    self._update_fsm()
    recovery_mask_2d = (self._phase == PHASE_RECOVERY).unsqueeze(-1).float()  # [B, 1]

    if self.cfg.recovery_mode == "hybrid":
      # D3: no position-space composition at all for the affected leg's
      # thigh/calf -- the recovery behavior lives ENTIRELY at the
      # actuator level (see _apply_hybrid_actuators). `actions` is
      # ignored; `delta_a` stays zero so `final_raw` below is BYTE-FOR-
      # BYTE `a_walk_raw` on every joint, including the affected leg's
      # ab/ad (which keeps walking normally) AND its thigh/calf (whose
      # POSITION actuator is physically disabled during RECOVERY, so
      # whatever position target is computed for them has zero effect
      # -- see _apply_hybrid_actuators for the actuator-level override).
      self._raw_actions.zero_()
      delta_a = torch.zeros(self.num_envs, 3, device=self.device)
      self._apply_hybrid_actuators()
    elif self.cfg.recovery_mode == "deterministic":
      # D1: no PPO/learned component. `actions` is ignored entirely.
      # `self._raw_actions` still records the deterministic command
      # actually in effect this step (zeroed outside RECOVERY), purely
      # for observation/metric/logging consistency with the PPO path's
      # `raw_action` property contract.
      delta_a = self._deterministic_delta.unsqueeze(0).expand(self.num_envs, -1)
      delta_a = delta_a * recovery_mask_2d
      self._raw_actions[:] = delta_a
    else:
      # "ppo_residual": UNCHANGED existing behavior.
      self._raw_actions[:] = actions
      delta_a = torch.clamp(actions, -self.cfg.max_residual, self.cfg.max_residual)
      delta_a = delta_a * recovery_mask_2d

    obs_walk = self._build_walk_obs()
    obs_walk_td = TensorDict(
      {"actor": obs_walk}, batch_size=[obs_walk.shape[0]]
    )
    with torch.no_grad():
      a_walk_raw = self._policy(obs_walk_td)  # frozen, no grad, [B, 12]

    final_raw = a_walk_raw.clone()
    final_raw[:, self._affected_idx] = final_raw[:, self._affected_idx] + delta_a

    self.last_a_walk_raw = a_walk_raw.detach().clone()
    self.last_final_raw = final_raw.detach().clone()

    # pi_walk's own next-step observation sees ONLY its own previous raw
    # action, never the recovery-modified one (see module docstring).
    self._last_walk_raw_action = a_walk_raw.detach().clone()

    self._joint_pos_action.process_actions(final_raw)
    self._log_transitions()

  def _log_transitions(self) -> None:
    """Lightweight, event-gated debug logging -- fires ONLY on the exact
    step(s) a phase transition happens for any env (never every step),
    per the "do not spam logs" requirement. Prints env 0's state as a
    representative sample (sufficient for D1 smoke-testing; not wired
    into any structured logger since none of the existing recovery
    tasks' MetricsManager terms are being modified here)."""
    if bool(self.newly_entangled[0]):
      extra = ""
      if self.cfg.recovery_mode == "hybrid":
        extra = (
          f", recovery_thigh_velocity={self.cfg.recovery_thigh_velocity}, "
          f"recovery_thigh_velocity_kd={self.cfg.recovery_thigh_velocity_kd}, "
          f"recovery_calf_torque={self.cfg.recovery_calf_torque}"
        )
      print(
        f"[recovery] env0 ENTER RECOVERY (mode={self.cfg.recovery_mode}, "
        f"affected_leg_idx={self.cfg.affected_leg_idx}, "
        f"thigh_delta={self.cfg.thigh_recovery_delta}, "
        f"calf_delta={self.cfg.calf_recovery_delta}{extra})"
      )
    if bool(self.newly_recovered[0]):
      reason = "cap_timeout" if bool(self._recovery_failed_timeout[0]) else "crossing+release"
      print(
        f"[recovery] env0 EXIT RECOVERY (reason={reason}, "
        f"elapsed={int(self._recovery_elapsed[0])}, "
        f"wire_contact={'yes' if bool(self._env.scene.sensors[self.cfg.wire_contact_sensor_name].data.found[0].gt(0).any()) else 'no'})"
      )
    if bool(self.newly_succeeded[0]):
      print("[recovery] env0 RECOVERY SUCCESS (post-recovery stability window reached)")

  def apply_actions(self) -> None:
    self._joint_pos_action.apply_actions()
