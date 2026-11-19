"""Multi-leg wire-entanglement recovery: one per-leg recovery controller per
entangled leg, composed into a single 12-joint command.

Each leg gets its own unmodified ``ResidualRecoveryAction`` (own wire-contact
sensor, own FSM, own crossing latches, own scripted step). All of them run the
same frozen walking policy on the same observation, so their ``a_walk_raw``
agree. The composed command is that walking action with each stepping leg's
thigh/calf replaced by its own controller's output. Only this term writes
actuator targets.

With ``serialize_steps`` a leg's scripted step does not start while another
leg is mid-step (lift, advance or descend), so the two front legs are never
both off the ground at once.

With ``step_min_body_vx`` a leg's step also waits until the body is moving
forward. Once one front leg has crossed, the other (still pinned) leg makes the
body rebound backward, and a step started during the rebound gets dragged back
behind the wire (D6.1).

A body hold (``hold_mode``) is active while an entangled leg is still waiting
to cross (see ``hold_window``) and ends once every entangled leg has crossed:
  - "cmd": the walking policy's forward command is lowered to ``hold_cmd_vx``.
  - "freeze": every joint holds the command it had when the hold began, except
    a leg that is mid-step, which follows its scripted step.
Neither fixes the FL+FR case (D6.1, 20 seeds, baseline 8/20): "cmd" at 0 m/s
gave 10/20, "freeze" 1-5/20. The body still slides back during the second
leg's step with every other joint frozen and no wire contact, so the slide is
not caused by the walking policy or the wire. Both modes are off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.tasks.recovery.mdp.actions import (
  PHASE_RECOVERY,
  ResidualRecoveryAction,
  ResidualRecoveryActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MultiLegRecoveryActionCfg(ActionTermCfg):
  leg_cfgs: tuple[ResidualRecoveryActionCfg, ...]
  """One per-leg controller config. ``recovery_mode`` must be "scripted_step"
  (the leg steps over the wire) or "ppo_residual" (observe only: with zero
  actions this is the walking policy, while the FSM still tracks contact and
  crossing)."""
  serialize_steps: bool = True
  """Hold a leg's step start while another leg is mid-step."""
  serialize_gap_steps: int = 0
  """Extra control steps after the other leg's touchdown before releasing."""
  step_min_body_vx: tuple[float | None, ...] = ()
  """Per leg (same order as ``leg_cfgs``): hold the step start until the
  smoothed body-frame forward velocity (m/s) exceeds this. None or missing =
  no velocity gate for that leg."""
  body_vx_smoothing: float = 0.5
  """EMA weight on the newest body vx sample."""
  step_max_roll_deg: tuple[float | None, ...] = ()
  """Per leg (same order as ``leg_cfgs``): hold the step start while the body's
  lateral tilt exceeds this many degrees, for at most ``roll_gate_max_steps``
  control steps. None or missing = no roll gate for that leg. D6.1: the falls
  are roll instabilities that start during the first leg's step."""
  roll_gate_max_steps: int = 30
  """Release the roll gate after this many held steps."""
  hold_mode: str = "off"
  """"off", "cmd" or "freeze" (see module docstring)."""
  hold_cmd_vx: float = 0.0
  """Forward command (m/s) given to the walking policy in "cmd" mode."""
  hold_window: str = "after_first"
  """"after_first": from the first leg's full crossing until every leg in
  RECOVERY has crossed. "entangled": whenever any leg is in RECOVERY and not
  yet crossed."""
  hold_ramp_steps: int = 1
  """Control steps to blend into/out of the hold."""
  hold_max_steps: int = 40
  """Release the hold after this many steps even if a leg has not crossed
  (0 = never), so a stuck leg does not freeze the robot for good."""

  def __getattr__(self, name: str) -> Any:
    # Single-leg observation/reward terms read `term.cfg.<field>`.
    leg_cfgs = self.__dict__.get("leg_cfgs")
    if not leg_cfgs:
      raise AttributeError(name)
    return getattr(leg_cfgs[0], name)

  def build(self, env: ManagerBasedRlEnv) -> "MultiLegRecoveryAction":
    return MultiLegRecoveryAction(self, env)


class MultiLegRecoveryAction(ActionTerm):
  cfg: MultiLegRecoveryActionCfg

  def __init__(self, cfg: MultiLegRecoveryActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    if not cfg.leg_cfgs:
      raise ValueError("MultiLegRecoveryActionCfg needs at least one leg.")
    for c in cfg.leg_cfgs:
      if c.recovery_mode not in ("scripted_step", "ppo_residual"):
        raise ValueError(f"Unsupported per-leg recovery_mode {c.recovery_mode!r}.")
    self.legs: list[ResidualRecoveryAction] = [c.build(env) for c in cfg.leg_cfgs]
    self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self.body_vx = torch.zeros(self.num_envs, device=self.device)
    self._roll_wait = torch.zeros(
      len(self.legs), self.num_envs, device=self.device, dtype=torch.long
    )
    self.hold_level = torch.zeros(self.num_envs, device=self.device)
    """0 = normal command, 1 = full hold; ramps by 1/hold_ramp_steps."""
    self._hold_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._hold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._latched_raw = torch.zeros(self.num_envs, 12, device=self.device)
    self._prev_final = torch.zeros(self.num_envs, 12, device=self.device)
    if cfg.hold_mode not in ("off", "cmd", "freeze"):
      raise ValueError(f"Unknown hold_mode {cfg.hold_mode!r}.")
    if cfg.hold_window not in ("after_first", "entangled"):
      raise ValueError(f"Unknown hold_window {cfg.hold_window!r}.")

  def __getattr__(self, name: str) -> Any:
    # Single-leg observation/reward terms read the first leg's state.
    legs = self.__dict__.get("legs")
    if legs is None:
      raise AttributeError(name)
    return getattr(legs[0], name)

  @property
  def action_dim(self) -> int:
    return 3 * len(self.legs)

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def leg_busy(self, leg: ResidualRecoveryAction) -> torch.Tensor:
    """[B] True while `leg` is lifting, advancing or descending (plus the
    configured gap after touchdown)."""
    if leg.cfg.recovery_mode != "scripted_step":
      return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    c = leg.cfg
    swing = c.step_lift_steps + c.step_advance_steps + c.step_descend_steps
    in_step = (leg.phase == PHASE_RECOVERY) & leg._step_started
    return in_step & (leg._step_clock() < swing + self.cfg.serialize_gap_steps)

  def hold_active(self) -> torch.Tensor:
    """[B] True while the body should be held (uses each leg's FSM state as
    of the previous step)."""
    in_rec = [leg.phase == PHASE_RECOVERY for leg in self.legs]
    crossed = [leg._crossing_achieved for leg in self.legs]
    pending = torch.stack([r & ~c for r, c in zip(in_rec, crossed, strict=True)]).any(0)
    if self.cfg.hold_window == "entangled":
      return pending
    return pending & torch.stack(crossed).any(0)

  def _update_hold(self) -> None:
    if self.cfg.hold_mode == "off":
      return
    step = 1.0 / max(self.cfg.hold_ramp_steps, 1)
    active = self.hold_active()
    self._hold_steps = torch.where(active, self._hold_steps + 1, 0)
    if self.cfg.hold_max_steps > 0:
      active = active & (self._hold_steps <= self.cfg.hold_max_steps)
    delta = (active.float() - self.hold_level).clamp(-step, step)
    self.hold_level = self.hold_level + delta
    rising = active & ~self._hold_prev
    self._latched_raw = torch.where(
      rising.unsqueeze(-1), self._prev_final, self._latched_raw
    )
    self._hold_prev = active
    if self.cfg.hold_mode != "cmd":
      return
    command = self._env.command_manager.get_command(self.legs[0].cfg.command_name)
    assert command is not None
    held = command.clone()
    lvl = self.hold_level
    held[:, 0] = (1.0 - lvl) * command[:, 0] + lvl * self.cfg.hold_cmd_vx
    for leg in self.legs:
      leg.walk_command_override = held

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    self._update_hold()
    a = self.cfg.body_vx_smoothing
    vx = self._entity.data.root_link_lin_vel_b[:, 0]
    self.body_vx = a * vx + (1.0 - a) * self.body_vx
    no_block = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    for i, leg in enumerate(self.legs):
      block = no_block
      if self.cfg.serialize_steps:
        for j, other in enumerate(self.legs):
          if j != i:
            block = block | self.leg_busy(other)
      max_roll = self.cfg.step_max_roll_deg[i : i + 1]
      if max_roll and max_roll[0] is not None:
        grav = self._entity.data.projected_gravity_b
        roll_deg = torch.rad2deg(torch.atan2(grav[:, 1], -grav[:, 2])).abs()
        waiting = (leg.phase == PHASE_RECOVERY) & ~leg._step_started
        over = waiting & (roll_deg > max_roll[0])
        gate = over & (self._roll_wait[i] < self.cfg.roll_gate_max_steps)
        self._roll_wait[i] = torch.where(
          over, self._roll_wait[i] + 1, torch.where(waiting, self._roll_wait[i], 0)
        )
        block = block | gate
      min_vx = self.cfg.step_min_body_vx[i : i + 1]
      if min_vx and min_vx[0] is not None:
        block = block | (self.body_vx <= min_vx[0])
      leg.step_block = block
      leg.process_actions(actions[:, 3 * i : 3 * i + 3])

    first = self.legs[0]
    assert first.last_a_walk_raw is not None
    final = first.last_a_walk_raw.clone()
    for leg in self.legs:
      if leg.cfg.recovery_mode == "scripted_step":
        assert leg.last_final_raw is not None
        idx = leg._affected_idx
        final[:, idx] = leg.last_final_raw[:, idx]
    if self.cfg.hold_mode == "freeze":
      frozen = self._latched_raw.clone()
      for leg in self.legs:
        busy = self.leg_busy(leg).unsqueeze(-1)
        idx = leg._affected_idx[1:3]
        frozen[:, idx] = torch.where(busy, final[:, idx], frozen[:, idx])
      lvl = self.hold_level.unsqueeze(-1)
      final = lvl * frozen + (1.0 - lvl) * final
    self._prev_final = final.clone()
    self.last_final_raw = final
    first._joint_pos_action.process_actions(final)

  def apply_actions(self) -> None:
    self.legs[0]._joint_pos_action.apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.body_vx[ids] = 0.0
    self.hold_level[ids] = 0.0
    self._roll_wait[:, ids] = 0
    self._hold_prev[ids] = False
    self._hold_steps[ids] = 0
    for leg in self.legs:
      leg.reset(env_ids)
