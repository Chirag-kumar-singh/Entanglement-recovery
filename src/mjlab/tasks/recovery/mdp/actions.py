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
from typing import TYPE_CHECKING

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
  wire_contact_sensor_name: str = "wire_contact"
  max_recovery_steps: int = 12
  release_required_steps: int = 5

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
    # One-shot flags for the step just completed, exposed for reward/
    # observation terms to read (newly_entangled: NORMAL_WALK->RECOVERY
    # this step; newly_recovered: RECOVERY->POST_RECOVERY this step).
    self.newly_entangled = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self.newly_recovered = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
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

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._last_walk_raw_action.zero_()
      self._phase.fill_(PHASE_NORMAL_WALK)
      self._recovery_elapsed.zero_()
      self._release_counter.zero_()
      self.newly_entangled.zero_()
      self.newly_recovered.zero_()
    else:
      self._last_walk_raw_action[env_ids] = 0.0
      self._phase[env_ids] = PHASE_NORMAL_WALK
      self._recovery_elapsed[env_ids] = 0
      self._release_counter[env_ids] = 0
      self.newly_entangled[env_ids] = False
      self.newly_recovered[env_ids] = False
    self._joint_pos_action.reset(env_ids)

  def _update_fsm(self) -> None:
    """Advances the recovery FSM by one step, using the real FL-wire
    contact sensor as the entanglement/release signal (see cfg docstring
    for the full transition rules). Must be called exactly once per
    environment step, before the Delta_a gate is applied in
    `process_actions`."""
    sensor = self._env.scene.sensors[self.cfg.wire_contact_sensor_name]
    found = sensor.data.found  # [B, P]
    in_contact = (found > 0).any(dim=-1)  # [B]

    is_normal = self._phase == PHASE_NORMAL_WALK
    is_recovery = self._phase == PHASE_RECOVERY

    # NORMAL_WALK -> RECOVERY on first genuine contact.
    entering_recovery = is_normal & in_contact
    self._phase = torch.where(
      entering_recovery, torch.full_like(self._phase, PHASE_RECOVERY), self._phase
    )
    self._recovery_elapsed = torch.where(
      entering_recovery, torch.zeros_like(self._recovery_elapsed), self._recovery_elapsed
    )
    self._release_counter = torch.where(
      entering_recovery, torch.zeros_like(self._release_counter), self._release_counter
    )

    # Within RECOVERY: advance elapsed counter and release counter.
    still_recovery_after_entry = is_recovery | entering_recovery
    self._recovery_elapsed = torch.where(
      still_recovery_after_entry, self._recovery_elapsed + 1, self._recovery_elapsed
    )
    self._release_counter = torch.where(
      still_recovery_after_entry & ~in_contact,
      self._release_counter + 1,
      torch.where(still_recovery_after_entry & in_contact,
                  torch.zeros_like(self._release_counter), self._release_counter),
    )

    released_sustained = self._release_counter >= self.cfg.release_required_steps
    cap_reached = self._recovery_elapsed >= self.cfg.max_recovery_steps
    exiting_recovery = still_recovery_after_entry & (released_sustained | cap_reached)

    self._phase = torch.where(
      exiting_recovery, torch.full_like(self._phase, PHASE_POST_RECOVERY), self._phase
    )

    self.newly_entangled = entering_recovery
    self.newly_recovered = exiting_recovery

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
    policy-action-space units)."""
    self._raw_actions[:] = actions

    # Advance the recovery FSM first, using the CURRENT (pre-this-step)
    # wire-contact state, then gate Delta_a by the resulting phase: only
    # RECOVERY allows the residual through. This must happen before the
    # physical action is computed/applied, since it determines whether
    # Delta_a has any effect this step at all.
    self._update_fsm()
    recovery_mask = (self._phase == PHASE_RECOVERY).unsqueeze(-1).float()  # [B, 1]

    delta_a = torch.clamp(actions, -self.cfg.max_residual, self.cfg.max_residual)
    delta_a = delta_a * recovery_mask

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

  def apply_actions(self) -> None:
    self._joint_pos_action.apply_actions()
