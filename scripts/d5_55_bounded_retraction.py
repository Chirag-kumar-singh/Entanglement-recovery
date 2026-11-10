#!/usr/bin/env python
"""D5.55 -- Bounded FL retraction before support transfer (opt-in variant).

Tests the D5.54 diagnosis: FL remains hybrid-retracted through the FR/RL
support transfer, and nothing takes its place, causing the second collapse.

New, separate variant: ``bounded_retraction``. Keeps the D5.53 touchdown guard
(imported, unmodified) for recovery entry. Adds a second, independent wrapper
around ``act._apply_hybrid_actuators`` that releases FL back to the normal
walking position-control path once the hybrid has held it for one FL swing,
then re-arms (via the same touchdown/lift-off sequence the entry guard uses)
and resumes the retract for the next swing. This repeats until the FSM's own,
unmodified crossing/release logic actually exits RECOVERY. No hybrid command,
gain, threshold, checkpoint, wire model, or exit/crossing logic is changed.

Swing-duration estimate ("gait-derived release condition"): the wrapper
continuously tracks FL's ground-contact transitions (the same
``fl_ground_contact_diag`` sensor D5.16/D5.52/D5.53 already use) regardless of
phase. Each time FL touches down *while still in NORMAL_WALK*, the just-ended
airborne run length is recorded as ``last_complete_swing_len`` (frozen once
recovery starts -- not re-measured from perturbed, mid-recovery steps, so the
estimate reflects the robot's actual unperturbed gait, not a guess). At hybrid
entry/resume, the wrapper also knows ``airborne_since_liftoff`` (steps FL has
already been off the ground, tracked the same way). The release budget is

    budget = max(1, last_complete_swing_len - airborne_since_liftoff)

i.e. "run hybrid for the rest of the swing FL is already in", not a fixed
number of steps from entry. If no complete NORMAL_WALK swing was observed
before entry (should not happen after 40+ steps of walking), a documented
fallback of 11 steps (D5.54's measured ~10-12 step swing) is used and logged
as a limitation.

Position-control restoration at release reuses the exact gain/bias values the
production code captured once at init (``act._orig_thigh_gainprm`` etc.,
``act._pos_thigh_ctrl_id`` etc.) and the exact restore/disable sequence
``_apply_hybrid_actuators`` itself uses on a genuine exit -- this script does
not invent new gain values.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

sys.path.insert(0, str(Path(__file__).parent))

from d5_52_wire_removed_forced_entry import (  # noqa: E402
  _first,
  _handoff_step,
  _write_csv,
)
from d5_53_touchdown_guard import D553Config, TouchdownGuard, _setup  # noqa: E402

from mjlab.tasks.recovery.mdp.actions import (  # noqa: E402
  PHASE_RECOVERY,
  ResidualRecoveryAction,
)

SEEDS = (0, 11)
FALLBACK_SWING_LEN = 11
RELEASE_DEBOUNCE = 2


@dataclass
class D555Config:
  variant: Literal["all", "bounded_retraction"] = "all"
  seed: int | None = None
  device: str = "cpu"
  max_steps: int = 250
  out_dir: str = "logs/d5_55"
  debug: bool = False


class BoundedRetraction:
  """Wraps ``act._apply_hybrid_actuators``. Independent of, and composes
  with, the D5.53 ``TouchdownGuard`` (which wraps ``_update_fsm`` instead)."""

  def __init__(self, env, act: ResidualRecoveryAction):
    self.env, self.act = env, act
    self.fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    self.airborne_run = 0
    self.last_complete_swing_len: int | None = None
    self.status = "IDLE"  # IDLE | ACTIVE | RELEASE_WAIT_TOUCHDOWN | RELEASE_STANCE
    self.budget = 0
    self.elapsed = 0
    self.consec = 0
    self.cycle = 0
    self.events: list[dict[str, Any]] = []
    self.step = -1
    self.original = act._apply_hybrid_actuators
    act._apply_hybrid_actuators = self._wrapped  # ty: ignore[invalid-assignment]

  def _log(self, msg: str) -> None:
    self.events.append({"step": self.step, "msg": msg})
    print(f"[D5.55] step={self.step} {msg}")

  def _reset_two_phase_state(self, *, reset_switch_step: bool) -> None:
    """Clear two-phase advance state if this build has it (it is optional)."""
    act = self.act
    for name, value in (
      ("hybrid_two_phase_advance_active", False),
      ("two_phase_lift_counter", 0),
    ):
      buf = getattr(act, name, None)
      if buf is not None:
        buf[0] = value
    if reset_switch_step:
      buf = getattr(act, "two_phase_switch_step", None)
      if buf is not None:
        buf[0] = -1

  def _enable_hybrid_gains(self) -> None:
    act, env = self.act, self.env
    m = env.sim.model
    gw = m.actuator_gainprm.shape[-1]
    bw = m.actuator_biasprm.shape[-1]
    ids = torch.zeros(1, dtype=torch.long, device=act.device)
    m.actuator_gainprm[ids, act._pos_thigh_ctrl_id, :] = torch.zeros(
      1, gw, device=act.device
    )
    m.actuator_biasprm[ids, act._pos_thigh_ctrl_id, :] = torch.zeros(
      1, bw, device=act.device
    )
    m.actuator_gainprm[ids, act._pos_calf_ctrl_id, :] = torch.zeros(
      1, gw, device=act.device
    )
    m.actuator_biasprm[ids, act._pos_calf_ctrl_id, :] = torch.zeros(
      1, bw, device=act.device
    )
    m.actuator_gainprm[ids, act._aux_thigh_ctrl_id, 0] = (
      act.cfg.recovery_thigh_velocity_kd
    )
    m.actuator_biasprm[
      ids, act._aux_thigh_ctrl_id, 2
    ] = -act.cfg.recovery_thigh_velocity_kd
    act.hybrid_thigh_pos_gain_active[ids] = False

  def _restore_walking_gains(self) -> None:
    act, env = self.act, self.env
    m = env.sim.model
    ids = torch.zeros(1, dtype=torch.long, device=act.device)
    m.actuator_gainprm[ids, act._pos_thigh_ctrl_id, :] = act._orig_thigh_gainprm[ids]
    m.actuator_biasprm[ids, act._pos_thigh_ctrl_id, :] = act._orig_thigh_biasprm[ids]
    m.actuator_gainprm[ids, act._pos_calf_ctrl_id, :] = act._orig_calf_gainprm[ids]
    m.actuator_biasprm[ids, act._pos_calf_ctrl_id, :] = act._orig_calf_biasprm[ids]
    m.actuator_gainprm[ids, act._aux_thigh_ctrl_id, 0] = 0.0
    m.actuator_biasprm[ids, act._aux_thigh_ctrl_id, 2] = 0.0
    act.hybrid_thigh_pos_gain_active[ids] = True

  def _wrapped(self) -> None:
    self.step += 1
    act = self.act
    in_recovery = bool((act._phase[0] == PHASE_RECOVERY).item())
    normal_walk = int(act._phase[0]) == 0
    fl_contact = bool((self.fl_ground.data.found[0] > 0).any())

    # --- FL swing-length tracking (every step, phase-gated on NORMAL_WALK
    # for the *estimate*, but the airborne counter itself runs always so
    # it reflects the real, current, possibly-perturbed swing in progress
    # at the moment hybrid activates). ---
    if not fl_contact:
      self.airborne_run += 1
    else:
      if self.airborne_run > 0 and normal_walk:
        self.last_complete_swing_len = self.airborne_run
      self.airborne_run = 0

    # --- Genuine FSM transitions: exactly the production enter/exit gain
    # bookkeeping, reproduced here because we cannot call the original
    # `_apply_hybrid_actuators` (it would unconditionally redrive the full
    # retract command based on `in_recovery` alone and undo our pause). ---
    if bool(act.newly_entangled[0]):
      self._enable_hybrid_gains()
      self._reset_two_phase_state(reset_switch_step=True)
      act._peak_armed[0] = False
      act._peak_running_max[0] = act.foot_height_above_wire[0]
      act._peak_entry_height[0] = act.foot_height_above_wire[0]
      act._peak_decline_counter[0] = 0
      act._peak_triggered[0] = False
      act.peak_trigger_step[0] = -1
      swing_est = self.last_complete_swing_len or FALLBACK_SWING_LEN
      self.budget = max(1, swing_est - self.airborne_run)
      self.elapsed = 0
      self.consec = 0
      self.status = "ACTIVE"
      self.cycle = 1
      fb = (
        ""
        if self.last_complete_swing_len
        else f" (FALLBACK, no NORMAL_WALK swing observed, default {FALLBACK_SWING_LEN})"
      )
      self._log(
        f"ENTER recovery: swing_est={swing_est}{fb} airborne_already={self.airborne_run} "
        f"-> budget={self.budget}"
      )
    if bool(act.newly_recovered[0]):
      if self.status != "IDLE":
        self._restore_walking_gains()
      self._reset_two_phase_state(reset_switch_step=False)
      self._log(f"EXIT recovery (status was {self.status}, cycle {self.cycle})")
      self.status = "IDLE"

    # --- Bounded-retraction sub-FSM (only while still inside the same
    # continuous RECOVERY phase; the FSM's own crossing/release/exit logic,
    # in `_update_fsm`, is untouched and runs independently every step). ---
    if in_recovery and self.status != "IDLE":
      if self.status == "ACTIVE":
        self.elapsed += 1
        if self.elapsed >= self.budget:
          self._restore_walking_gains()
          self.status, self.consec = "RELEASE_WAIT_TOUCHDOWN", 0
          self._log(
            f"RELEASE to walking control (elapsed={self.elapsed} budget={self.budget})"
          )
      elif self.status == "RELEASE_WAIT_TOUCHDOWN":
        self.consec = self.consec + 1 if fl_contact else 0
        if self.consec >= RELEASE_DEBOUNCE:
          self.status, self.consec = "RELEASE_STANCE", 0
          self._log("FL touchdown confirmed (released)")
      elif self.status == "RELEASE_STANCE":
        self.consec = self.consec + 1 if not fl_contact else 0
        if self.consec >= RELEASE_DEBOUNCE:
          self._enable_hybrid_gains()
          swing_est = self.last_complete_swing_len or FALLBACK_SWING_LEN
          self.budget, self.elapsed, self.consec = swing_est, 0, 0
          self.status = "ACTIVE"
          self.cycle += 1
          self._log(f"RESUME hybrid retract, cycle {self.cycle} (budget={self.budget})")

    active = in_recovery and self.status == "ACTIVE"
    thigh_vel_cmd = torch.where(
      torch.tensor([active], device=act.device),
      torch.full_like(act.hybrid_thigh_vel_ctrl, -act.cfg.recovery_thigh_velocity),
      torch.zeros_like(act.hybrid_thigh_vel_ctrl),
    )
    calf_torque_cmd = torch.where(
      torch.tensor([active], device=act.device),
      torch.full_like(act.hybrid_calf_torque_ctrl, -act.cfg.recovery_calf_torque),
      torch.zeros_like(act.hybrid_calf_torque_ctrl),
    )
    act.hybrid_thigh_vel_ctrl = thigh_vel_cmd
    act.hybrid_calf_torque_ctrl = calf_torque_cmd
    act._entity.set_joint_velocity_target(
      thigh_vel_cmd.unsqueeze(-1), joint_ids=[act._hybrid_thigh_joint_id]
    )
    act._entity.set_joint_effort_target(
      calf_torque_cmd.unsqueeze(-1), joint_ids=[act._hybrid_calf_joint_id]
    )


def _make_row(
  env, act, guard: TouchdownGuard, br: BoundedRetraction, step: int, seed: int
):
  from d5_53_touchdown_guard import Recorder

  if not hasattr(env, "_d555_rec"):
    env._d555_rec = Recorder(env, act)
  r = env._d555_rec.row(step, "bounded_retraction", -1)
  r["seed"] = seed
  r["variant"] = "bounded_retraction"
  r["guard_state"] = guard.state
  r["bounded_status"] = br.status
  r["bounded_cycle"] = br.cycle
  r["bounded_budget"] = br.budget
  r["newly_succeeded"] = bool(act.newly_succeeded[0])
  return r


def run_bounded(cfg: D555Config, seed: int) -> dict[str, Any]:
  d553 = D553Config(device=cfg.device)
  env, act, guard, probe, _ = _setup(d553, "guard_enabled", seed)
  assert guard is not None
  br = BoundedRetraction(env, act)
  env.reset(seed=seed)
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  rows, ever_succeeded = [], False
  for step in range(cfg.max_steps):
    probe.begin_step(step)
    env.step(zero)
    r = _make_row(env, act, guard, br, step, seed)
    ever_succeeded = ever_succeeded or r["newly_succeeded"]
    rows.append(r)
    if r["fell_over"]:
      print(
        f"[D5.55] bounded_retraction seed{seed}: fell_over at step {step}; stopping"
      )
      break
  nexclude = int(env.sim.mj_model.nexclude)
  env.close()
  return {
    "rows": rows,
    "events": br.events,
    "ever_succeeded": ever_succeeded,
    "nexclude": nexclude,
    "substeps": probe.rows,
  }


def run_viewer(cfg: D555Config, seed: int) -> Path:
  from mjlab.viewer import NativeMujocoViewer

  d553 = D553Config(device=cfg.device)
  env, act, guard, probe, _ = _setup(d553, "guard_enabled", seed)
  assert guard is not None
  br = BoundedRetraction(env, act)
  env.reset(seed=seed)
  rows: list[dict[str, Any]] = []
  pending: dict[str, Any] = {"step": None}

  def viewer_policy(_obs):  # noqa: ANN001
    if pending["step"] is not None:
      rows.append(_make_row(env, act, guard, br, pending["step"], seed))
      if pending["step"] % 25 == 0:
        print(
          f"[D5.55 viewer] seed{seed} step={pending['step']} phase={rows[-1]['phase']} "
          f"guard={rows[-1]['guard_state']} bounded={rows[-1]['bounded_status']}"
        )
      if pending["step"] + 1 >= cfg.max_steps or rows[-1]["fell_over"]:
        raise SystemExit(0)
    step = 0 if pending["step"] is None else pending["step"] + 1
    probe.begin_step(step)
    pending["step"] = step
    return torch.zeros(env.action_manager.action.shape, device=env.device)

  try:
    NativeMujocoViewer(cast(Any, env), cast(Any, viewer_policy)).run()
  except SystemExit:
    pass
  env.close()
  out = Path(cfg.out_dir) / f"viewer_bounded_retraction_seed{seed}"
  out.mkdir(parents=True, exist_ok=True)
  path = out / "trace.csv"
  _write_csv(path, rows)
  print(f"[D5.55 viewer] wrote {path} ({len(rows)} rows)")
  return path


def summarize(
  variant: str, seed: int, rows: list[dict[str, Any]], ever_succeeded
) -> dict[str, Any]:
  by = {r["step"]: r for r in rows}
  handoff = _handoff_step(rows)
  align = handoff if handoff is not None else 44

  def support(r):
    return sum(abs(r[f"{leg}_ground_force_z_n"]) for leg in ("fl", "fr", "rl", "rr"))

  win0 = [by[s] for s in range(align, align + 10) if s in by]
  exit_ = _first(rows, lambda r: r["newly_recovered"], 0)
  return {
    "variant": variant,
    "seed": seed,
    "rows": len(rows),
    "hybrid_handoff_step": handoff,
    "peak_abs_pitch_rate_k0_9_rps": max(abs(r["base_ang_vel_b_y_rps"]) for r in win0)
    if win0
    else None,
    "min_total_support_k0_9_n": min(support(r) for r in win0) if win0 else None,
    "min_base_z_full_run_m": min(r["base_z_m"] for r in rows),
    "peak_abs_pitch_rate_full_run_rps": max(
      abs(r["base_ang_vel_b_y_rps"]) for r in rows
    ),
    "min_total_support_full_run_n": min(support(r) for r in rows),
    "steps_min_total_support_lt10N": sum(1 for r in rows if support(r) < 10.0),
    "first_crossing_step": _first(rows, lambda r: r["crossing_achieved"], 0),
    "first_release_step": _first(rows, lambda r: r["release_achieved"], 0),
    "recovery_exit_step": exit_,
    "recovery_exit_timeout": bool(by[exit_]["recovery_failed_timeout"])
    if exit_ is not None
    else None,
    "genuine_recovery_complete": bool(ever_succeeded),
    "fell_over_step": _first(rows, lambda r: r["fell_over"], 0),
  }


def main() -> None:
  import tyro

  cfg = tyro.cli(D555Config)
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)

  if cfg.debug:
    assert cfg.seed is not None, "--debug requires --seed"
    run_viewer(cfg, cfg.seed)
    return

  seeds = (cfg.seed,) if cfg.seed is not None else SEEDS
  results = {}
  summaries = []
  checks: list[dict[str, Any]] = []

  for seed in seeds:
    res = run_bounded(cfg, seed)
    results[seed] = res
    summaries.append(
      summarize("bounded_retraction", seed, res["rows"], res["ever_succeeded"])
    )

  if cfg.seed is not None:
    _write_csv(
      out / f"trace_bounded_retraction_seed{cfg.seed}.csv", results[cfg.seed]["rows"]
    )
    print(f"[D5.55] wrote trace_bounded_retraction_seed{cfg.seed}.csv")
    return

  # --- Validation: re-run baseline/guard_enabled in-memory (no new trace
  # files -- D5.53's artifacts are reused, these runs only prove today's
  # d5_53_touchdown_guard.py, imported unmodified, still reproduces them). ---
  base_nex = None
  for seed in seeds:
    for variant in ("baseline", "guard_enabled"):
      d553 = D553Config(device=cfg.device, max_steps=cfg.max_steps)
      env, act, guard, probe, make_row = _setup(d553, variant, seed)
      env.reset(seed=seed)
      zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
      rows, ever = [], False
      for step in range(d553.max_steps):
        probe.begin_step(step)
        env.step(zero)
        r = make_row(step)
        ever = ever or bool(act.newly_succeeded[0])
        rows.append(r)
        if r["fell_over"]:
          break
      nexclude = int(env.sim.mj_model.nexclude)
      env.close()
      if base_nex is None:
        base_nex = nexclude
      ref = {
        int(r["step"]): r
        for r in csv.DictReader(open(f"logs/d5_53/trace_{variant}_seed{seed}.csv"))
      }
      cols = [
        "base_z_m",
        "base_ang_vel_b_y_rps",
        "FL_calf_pos_rad",
        "hybrid_mode_active",
        "phase",
      ]
      worst = 0.0
      for r in rows:
        x = ref[r["step"]]
        for c in cols:
          va, vb = x[c], r[c]
          if isinstance(vb, bool):
            assert (va == "True") == vb, (
              f"{variant} seed{seed} step{r['step']} {c} mismatch"
            )
          elif isinstance(vb, str):
            assert va == vb, (
              f"{variant} seed{seed} step{r['step']} {c} mismatch: {va} != {vb}"
            )
          else:
            worst = max(worst, abs(float(va) - float(vb)))
      checks.append(
        {
          "check": f"{variant}_seed{seed}_reproduces_d5_53",
          "value": f"max_abs_diff={worst:.3e}",
          "pass": worst <= 1e-5,
        }
      )
      summaries.append(summarize(variant, seed, rows, ever))

  for seed in seeds:
    r = results[seed]
    cnt = sum(s["robot_wire_contact_count"] for s in r["substeps"])
    checks.append(
      {
        "check": f"wire_collisions_enabled_bounded_retraction_seed{seed}",
        "value": f"nexclude={r['nexclude']} (baseline {base_nex}); robot-wire substep contacts={cnt}",
        "pass": r["nexclude"] == base_nex and cnt > 0,
      }
    )

  for seed in seeds:
    rows = results[seed]["rows"]
    seen_release = any(
      r["bounded_status"] != "ACTIVE" and r["phase"] == "RECOVERY" for r in rows
    )
    seen_resume = any(e["msg"].startswith("RESUME") for e in results[seed]["events"])
    checks.append(
      {
        "check": f"fl_released_to_walking_before_exit_seed{seed}",
        "value": f"released_at_least_once={seen_release} resumed_at_least_once={seen_resume}",
        "pass": seen_release,
      }
    )

  _write_csv(out / "validation.csv", checks)
  for c in checks:
    print(f"[D5.55] {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {c['value']}")

  _write_csv(out / "comparison_summary.csv", summaries)
  for s in summaries:
    print(s)
  if not all(c["pass"] for c in checks):
    print("[D5.55] PRECONDITION FAILURE -- results must not be interpreted")
    sys.exit(2)


if __name__ == "__main__":
  main()
