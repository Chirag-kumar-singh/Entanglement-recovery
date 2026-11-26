"""D6.1 -- Multi-leg entanglement: one wire across BOTH front legs.

D5.58 isolated the wire to one leg at a time. Here the wire spans the whole
front of the robot (the original DEFAULT_WIRE_CFG, y in [0, 0.5], spawn
centred at y=0.25), so FL and FR both meet it. The rear legs are still
contact-excluded from the wire (they come later), so this isolates the
front-pair problem.

Each leg in --legs gets its own unmodified D5.58 per-leg controller (own
contact sensor, FSM, crossing latches and scripted step, with its D5.58
per-leg start delay), composed by `MultiLegRecoveryAction`. Front legs not in
--legs still get an observe-only controller (walking policy, FSM tracking
only) so their contact and crossing are logged.

Usage
-----
    # Baseline: walking policy only, just watch what the wire does.
    uv run python scripts/d6_1_front_pair_wire.py --legs
    # Single-leg recovery with the wide wire (does the other leg cope?).
    uv run python scripts/d6_1_front_pair_wire.py --legs FL
    # Both front legs, steps serialized (default).
    uv run python scripts/d6_1_front_pair_wire.py --legs FL FR --seed 3 --debug
    # Seed sweep.
    uv run python scripts/d6_1_front_pair_wire.py --legs FL FR --seeds 0-19
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d5_58_per_leg_recovery_window import (  # noqa: E402
  CONTROL_DT,
  STEP_GATE_MAX_AGE,
  STEP_START_DELAY,
  VERDICT_MAX_DRIFT_Y,
  VERDICT_MIN_BASE_Z,
  VERDICT_MIN_WALKOFF_SPEED,
  VERDICT_WALKOFF_STEPS,
  VERDICT_WINDOW_STEPS,
  D558Config,
)
from per_leg_wire import LEG_IDX, LEG_NAMES, exclude_legs_from_wire  # noqa: E402

from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.recovery.env_cfg_residual import (  # noqa: E402
  DEFAULT_WIRE_CFG,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.tasks.recovery.mdp import wire_geometry  # noqa: E402
from mjlab.tasks.recovery.mdp.multi_leg import (  # noqa: E402
  MultiLegRecoveryAction,
  MultiLegRecoveryActionCfg,
)

FRONT_LEGS = ("FL", "FR")
# Per-leg default scripted-step reach (m) when it differs from the D5.58 value
# (0.30). FR 0.18: on 60 seeds (front legs on the wire, 2 attempts) falls drop
# 12 -> 2 and the front stage clears 68% -> 80%; single-leg FR is unchanged
# (37/40 at both reaches). Override with --step-advance.
DEFAULT_STEP_ADVANCE = {"FR": 0.18}
REAR_LEGS = ("RL", "RR")
PAIRS = {"front": FRONT_LEGS, "rear": REAR_LEGS, "all": FRONT_LEGS + REAR_LEGS}
WIRE_SPAWN_Y = 0.25
FAR_SIDE_MARGIN = 0.05
"""A leg counts as across when its calf and foot are this far (m) past the
wire's local x at the end of the rollout."""


@dataclass
class D61Config:
  pair: str = "front"
  """Which legs the wire entangles: "front" (FL, FR), "rear" (RL, RR) or "all"
  (all four: front legs meet the wire first, rear legs about 70 steps later).
  Legs outside the pair are contact-excluded from the wire."""
  legs: tuple[str, ...] | None = None
  """Legs that run the scripted step (default: the whole pair). Other legs of
  the pair are observe-only; --legs with no names runs the walking policy."""
  seed: int = 0
  seeds: str = ""
  """Seed sweep, e.g. "0-19" (overrides --seed, headless, summary only)."""
  jobs: int = 2
  """Sweep worker processes. Each peaks at ~1.7 GB; keep jobs * 1.7 GB well
  under free RAM."""
  device: str | None = None
  max_steps: int = 0
  """0 = auto: 220 for the front pair, 320 for the rear pair and for all four
  legs (the rear legs reach the wire about 70 steps later)."""
  max_recovery_steps: int = 48
  serialize_steps: bool = True
  serialize_gap_steps: int = 0
  support_gate_z: float = 0.0
  crossing_fallback: int = 0
  new_legs: tuple[str, ...] = LEG_NAMES
  """Legs that use the new settings (touchdown window, crossing fallback, support
  gate and gap). The other legs keep the D5.58 settings."""
  support_diagonal: bool = False
  start_delay: dict[str, int] = field(default_factory=dict)
  """Per-leg start delay override, e.g. --start-delay FL 23 FR 0. Missing
  legs use the D5.58 value."""
  step_advance: dict[str, float] = field(default_factory=dict)
  """Per-leg scripted-step forward reach (m) from the foot's start, e.g.
  --step-advance FR 0.30. Missing legs use DEFAULT_STEP_ADVANCE, else the
  D5.58 value (0.30)."""
  step_back: dict[str, float] = field(default_factory=dict)
  """Per-leg backward travel (m) of the lift phase, e.g. --step-back FL -0.04.
  Zero or negative = lift-and-slide: the foot rises with a forward press instead
  of retracting first (Yim et al. 2023). Missing legs use the D5.58 value."""
  max_attempts: dict[str, int] = field(default_factory=dict)
  """Per-leg scripted-step attempts per recovery, e.g. --max-attempts FL 3
  (see ResidualRecoveryActionCfg.step_max_attempts). Missing legs: 1."""
  retry_on: str = "uncrossed"
  """"uncrossed" or "contact" (see ResidualRecoveryActionCfg.step_retry_on)."""
  retry_delay: int = 0
  """Steps to wait after a finished attempt before re-arming."""
  step_gate: dict[str, int] = field(default_factory=dict)
  """Per-leg gait-phase start gate max age (control steps), e.g. --step-gate FL
  6: the step waits until the two feet that are neither this leg nor its
  diagonal partner have just landed (see ResidualRecoveryActionCfg). 0 = off.
  Missing legs use the D5.58 value (off for FL/FR)."""
  td_window: tuple[int, int] | None = None
  """Leg-independent start rule: every leg in the pair starts its step when its
  own foot has been planted between LO and HI steps (e.g. --td-window 17 19).
  Sets every start delay to 0 and turns the RR gait gate off unless given."""
  own_planted: dict[str, bool] = field(default_factory=dict)
  """Per-leg: hold the step start until the leg's own foot is planted, e.g.
  --own-planted FL True (see ResidualRecoveryActionCfg.step_own_planted)."""
  step_gate_timeout: int = 60
  """Start anyway after this many steps in RECOVERY if the gate never opens."""
  max_roll_deg: dict[str, float] = field(default_factory=dict)
  """Per-leg roll gate (deg), e.g. --max-roll-deg FL 8: hold that leg's step
  start while |lateral tilt| exceeds it (see MultiLegRecoveryActionCfg)."""
  roll_gate_max_steps: int = 30
  min_body_z: dict[str, float] = field(default_factory=dict)
  """Per-leg base-height start gate (m), e.g. --min-body-z RL 0.29 RR 0.29: hold
  that leg's step start while the body is lower than this (see
  MultiLegRecoveryActionCfg)."""
  height_gate_max_steps: int = 30
  min_body_vx: dict[str, float] = field(default_factory=dict)
  """Per-leg body forward-velocity start gate (m/s), e.g. --min-body-vx FL
  0.15. Missing legs have no gate."""
  partner_hold: bool = False
  """Hold the stepping leg's trot-diagonal partner at the standing pose during
  the step (see MultiLegRecoveryActionCfg.partner_hold)."""
  partner_hold_ramp_steps: int = 2
  support_lift_thigh: float = 0.0
  support_lift_calf: float = 0.0
  """Body support (D7): thigh/calf offset (rad) on the standing legs while a leg
  steps (see MultiLegRecoveryActionCfg.support_lift_thigh)."""
  support_lift_ramp_steps: int = 3
  support_lift_trigger: tuple[str, ...] = ()
  """Legs whose steps trigger the support lift, e.g. --support-lift-trigger FL FR.
  Empty = any stepping leg."""
  hold_mode: str = "off"
  """Body hold (option 2): "off", "cmd" or "freeze" (see multi_leg.py)."""
  hold_cmd_vx: float = 0.0
  """Forward command (m/s) during a "cmd" hold."""
  hold_window: str = "after_first"
  """"after_first" or "entangled" (see MultiLegRecoveryActionCfg)."""
  hold_ramp_steps: int = 1
  hold_max_steps: int = 40
  spawn_y_offset: float = 0.0
  start_x: float | None = None
  """Pin the start x (m, world frame), replacing the +-0.02 m jitter. The default
  spawn is wire x - 0.35; moving it changes the gait phase at wire contact and
  so which leg touches first."""
  start_yaw: float | None = None
  """Pin the start yaw (rad), replacing the +-0.05 rad jitter."""
  start_x_range: tuple[float, float] | None = None
  """Benchmark mode: give every seed its own start x in [lo, hi] (m), spread
  evenly over seeds (golden-ratio sequence), so a sweep covers many approach
  phases. One gait cycle is about 0.135 m of travel, so e.g. -0.485 -0.35.
  Ignored if --start-x is set."""
  out_dir: str = ""
  debug: bool = False


def _step_overrides(leg: str, cfg: D61Config) -> dict:
  base = D558Config()
  use_td = bool(cfg.td_window) and leg in cfg.new_legs
  return {
    "step_lift_height": base.step_lift_height,
    "step_back": cfg.step_back.get(leg, base.step_back),
    "step_advance": cfg.step_advance.get(
      leg, DEFAULT_STEP_ADVANCE.get(leg, base.step_advance)
    ),
    "step_lift_steps": base.step_lift_steps,
    "step_advance_steps": base.step_advance_steps,
    "step_descend_steps": base.step_descend_steps,
    "step_blend_steps": base.step_blend_steps,
    "step_start_delay_steps": cfg.start_delay.get(
      leg, 0 if use_td else STEP_START_DELAY[leg]
    ),
    "step_gate_max_age": cfg.step_gate.get(
      leg, 0 if use_td else STEP_GATE_MAX_AGE[leg]
    ),
    "crossing_fallback_steps": cfg.crossing_fallback if leg in cfg.new_legs else 0,
    "step_td_age_min": cfg.td_window[0] if use_td else 0,
    "step_td_age_max": cfg.td_window[1] if use_td else 0,
    "step_gate_timeout_steps": cfg.step_gate_timeout,
    "step_own_planted": cfg.own_planted.get(leg, False),
    "step_max_attempts": cfg.max_attempts.get(leg, 1),
    "step_retry_delay_steps": cfg.retry_delay,
    "step_retry_on": cfg.retry_on,
  }


GOLDEN = 0.6180339887498949


def effective_start_x(cfg: D61Config, seed: int) -> float | None:
  """Start x used for `seed`, or None for the default +-0.02 m jitter."""
  if cfg.start_x is not None:
    return cfg.start_x
  if cfg.start_x_range is not None:
    lo, hi = cfg.start_x_range
    return lo + (hi - lo) * ((seed * GOLDEN) % 1.0)
  return None


def build_env(cfg: D61Config, seed: int, device: str):
  wire = DEFAULT_WIRE_CFG
  pair = PAIRS[cfg.pair]
  active_legs = pair if cfg.legs is None else cfg.legs
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=LEG_IDX[pair[0]],
    num_envs=1,
    recovery_mode="scripted_step",
    policy_device=device,
    play=True,
    wire_cfg=wire,
    spawn_y_override=WIRE_SPAWN_Y + cfg.spawn_y_offset,
    max_recovery_steps=cfg.max_recovery_steps,
  )
  env_cfg.seed = seed
  env_cfg.sim.device = device
  pose_range = env_cfg.events["reset_base"].params["pose_range"]
  start_x = effective_start_x(cfg, seed)
  if start_x is not None:
    pose_range["x"] = (start_x, start_x)
  if cfg.start_yaw is not None:
    pose_range["yaw"] = (cfg.start_yaw, cfg.start_yaw)

  base_action = env_cfg.actions["joint_pos"]
  leg_cfgs = []
  sensors = list(env_cfg.scene.sensors)
  for leg in pair:
    sensor_name = f"wire_contact_{leg}"
    sensors.append(
      ContactSensorCfg(
        name=sensor_name,
        primary=ContactMatch(mode="geom", pattern=wire.geom_names(), entity=None),
        secondary=ContactMatch(
          mode="geom", pattern=f"{leg}_calf2_collision", entity="robot"
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
      )
    )
    active = leg in active_legs
    leg_cfgs.append(
      dataclasses.replace(
        base_action,
        affected_leg_idx=LEG_IDX[leg],
        wire_contact_sensor_name=sensor_name,
        recovery_mode="scripted_step" if active else "ppo_residual",
        **_step_overrides(leg, cfg),
      )
    )
  env_cfg.scene.sensors = tuple(sensors)
  env_cfg.actions["joint_pos"] = MultiLegRecoveryActionCfg(
    entity_name="robot",
    leg_cfgs=tuple(leg_cfgs),
    serialize_steps=cfg.serialize_steps,
    serialize_gap_steps=cfg.serialize_gap_steps,
    support_mask=tuple(leg in cfg.new_legs for leg in pair),
    support_gate_z=cfg.support_gate_z,
    support_diagonal=cfg.support_diagonal,
    step_min_body_vx=tuple(cfg.min_body_vx.get(leg) for leg in pair),
    step_max_roll_deg=tuple(cfg.max_roll_deg.get(leg) for leg in pair),
    step_min_body_z=tuple(cfg.min_body_z.get(leg) for leg in pair),
    height_gate_max_steps=cfg.height_gate_max_steps,
    roll_gate_max_steps=cfg.roll_gate_max_steps,
    partner_hold=cfg.partner_hold,
    partner_hold_ramp_steps=cfg.partner_hold_ramp_steps,
    support_lift_thigh=cfg.support_lift_thigh,
    support_lift_calf=cfg.support_lift_calf,
    support_lift_ramp_steps=cfg.support_lift_ramp_steps,
    support_lift_trigger=cfg.support_lift_trigger,
    hold_mode=cfg.hold_mode,
    hold_cmd_vx=cfg.hold_cmd_vx,
    hold_window=cfg.hold_window,
    hold_ramp_steps=cfg.hold_ramp_steps,
    hold_max_steps=cfg.hold_max_steps,
  )

  wire_spec_fn = env_cfg.scene.spec_fn

  def spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
    assert wire_spec_fn is not None
    wire_spec_fn(spec, num_envs)
    excluded = tuple(leg for leg in LEG_NAMES if leg not in pair)
    if excluded:
      exclude_legs_from_wire(spec, excluded, verbose=False)

  env_cfg.scene.spec_fn = spec_fn

  from mjlab.envs import ManagerBasedRlEnv

  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def first_reset(rows: list[dict], legs: tuple[str, ...] = FRONT_LEGS) -> int | None:
  """Step of the first episode reset (the env auto-resets after `fell_over`), seen
  as a recovery phase dropping back to 0, which only a reset does."""
  for i in range(1, len(rows)):
    if any(
      int(rows[i - 1][f"{g}_phase"]) > 0 and int(rows[i][f"{g}_phase"]) == 0
      for g in legs
    ):
      return i
  return None


def verdict(rows: list[dict], legs: tuple[str, ...] = FRONT_LEGS) -> tuple[bool, str]:
  """PASS: both legs of the pair end across the wire (geometric, independent of
  the FSM), no sidestep, no collapse, keeps walking. Same thresholds as the
  D5.58 strict verdict, with the window anchored at the first wire contact
  of either leg."""
  entry = next(
    (i for i, r in enumerate(rows) if any(r[f"{g}_contact"] for g in legs)),
    None,
  )
  if entry is None:
    return False, "no wire contact"
  fell = first_reset(rows, legs)
  if fell is not None:
    return False, f"fell over at step {fell} (episode reset)"
  last = rows[-1]
  window = rows[entry : entry + VERDICT_WINDOW_STEPS + 1]
  drift = abs(window[-1]["base_y"] - rows[entry]["base_y"])
  min_z = min(r["base_z"] for r in window)
  tail = rows[-(VERDICT_WALKOFF_STEPS + 1) :]
  speed = (tail[-1]["base_x"] - tail[0]["base_x"]) / ((len(tail) - 1) * CONTROL_DT)
  failures = []
  for g in legs:
    if not (
      last[f"{g}_calf_dx"] > FAR_SIDE_MARGIN and last[f"{g}_foot_dx"] > FAR_SIDE_MARGIN
    ):
      failures.append(f"{g} not across")
  if drift > VERDICT_MAX_DRIFT_Y:
    failures.append(f"sidestep drift {drift:.2f}m")
  if min_z < VERDICT_MIN_BASE_Z:
    failures.append(f"body collapse z={min_z:.2f}m")
  if speed < VERDICT_MIN_WALKOFF_SPEED:
    failures.append(f"stalled {speed:.2f}m/s")
  metrics = f"drift_y={drift:.3f} min_base_z={min_z:.3f} walkoff_vx={speed:.2f}"
  if failures:
    return False, "; ".join(failures) + f" ({metrics})"
  return True, metrics


def staged_verdict(rows: list[dict]) -> tuple[str, str]:
  """Four-leg verdict split by stage. The front legs meet the wire first, the
  rear legs about 70 steps later.

  FAIL_FRONT: the front stage did not clear -- the robot fell, collapsed or
    sidestepped before the rear legs reached the wire, or a front leg was not
    across by then (or the rear legs never reached the wire).
  FAIL_REAR: the front stage cleared but the rear stage (or a later fall, or a
    front leg dragged back) did not.
  PASS: all four legs end across, no fall, no sidestep or collapse, still walking.
  """
  legs = FRONT_LEGS + REAR_LEGS
  first = next(
    (i for i, r in enumerate(rows) if any(r[f"{g}_contact"] for g in legs)), None
  )
  if first is None:
    return "SKIP", "no wire contact"
  rear = next(
    (i for i, r in enumerate(rows) if any(r[f"{g}_contact"] for g in REAR_LEGS)), None
  )
  fell = first_reset(rows, legs)
  t_front = len(rows) - 1 if rear is None else rear

  def across(r: dict, g: str) -> bool:
    return r[f"{g}_calf_dx"] > FAR_SIDE_MARGIN and r[f"{g}_foot_dx"] > FAR_SIDE_MARGIN

  # Front stage: from the first wire contact to the rear legs' first contact;
  # drift and collapse use the same 60-step window as the other verdicts.
  window = rows[first : min(t_front, first + VERDICT_WINDOW_STEPS) + 1]
  front_bad = []
  if fell is not None and fell <= t_front:
    front_bad.append(f"fell over at step {fell} before the rear legs reached the wire")
  else:
    snap = rows[t_front]
    front_bad += [f"{g} not across" for g in FRONT_LEGS if not across(snap, g)]
    drift = abs(window[-1]["base_y"] - window[0]["base_y"])
    if drift > VERDICT_MAX_DRIFT_Y:
      front_bad.append(f"sidestep drift {drift:.2f}m")
    if min(r["base_z"] for r in window) < VERDICT_MIN_BASE_Z:
      front_bad.append(f"body collapse z={min(r['base_z'] for r in window):.2f}m")
    if rear is None:
      front_bad.append("rear legs never reached the wire")
  if front_bad:
    return "FAIL_FRONT", "; ".join(front_bad)

  # Rear stage: from the rear legs' first contact to the end.
  assert rear is not None
  if fell is not None:
    return "FAIL_REAR", f"fell over at step {fell} (front stage had cleared)"
  last = rows[-1]
  rwin = rows[rear : rear + VERDICT_WINDOW_STEPS + 1]
  rdrift = abs(rwin[-1]["base_y"] - rows[rear]["base_y"])
  rmin_z = min(r["base_z"] for r in rwin)
  tail = rows[-(VERDICT_WALKOFF_STEPS + 1) :]
  speed = (tail[-1]["base_x"] - tail[0]["base_x"]) / ((len(tail) - 1) * CONTROL_DT)
  bad = [f"{g} not across" for g in REAR_LEGS if not across(last, g)]
  bad += [f"{g} dragged back" for g in FRONT_LEGS if not across(last, g)]
  if rdrift > VERDICT_MAX_DRIFT_Y:
    bad.append(f"sidestep drift {rdrift:.2f}m")
  if rmin_z < VERDICT_MIN_BASE_Z:
    bad.append(f"body collapse z={rmin_z:.2f}m")
  if speed < VERDICT_MIN_WALKOFF_SPEED:
    bad.append(f"stalled {speed:.2f}m/s")
  metrics = f"drift_y={rdrift:.3f} min_base_z={rmin_z:.3f} walkoff_vx={speed:.2f}"
  if bad:
    return "FAIL_REAR", "; ".join(bad) + f" ({metrics})"
  return "PASS", metrics


def run(cfg: D61Config, seed: int, quiet: bool = False) -> tuple[str, str, dict]:
  pair = PAIRS[cfg.pair]
  legs = pair if cfg.legs is None else cfg.legs
  for leg in legs:
    if leg not in pair:
      raise ValueError(f"--legs {leg!r} is not in the {cfg.pair} pair {pair}")
  max_steps = cfg.max_steps or (220 if cfg.pair == "front" else 320)
  device = cfg.device or "cpu"
  env = build_env(cfg, seed, device)
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, MultiLegRecoveryAction)
  ctrl = dict(zip(pair, term.legs, strict=True))
  entity = term._entity
  site_names = list(entity.site_names)
  wire_ids = term.legs[0]._wire_geom_ids

  env.reset(seed=seed)
  rows: list[dict] = []
  events: dict[str, int] = {}

  def record(step: int) -> dict:
    base = entity.data.root_link_pos_w[0]
    seg = wire_geometry.wire_segment_world_pos(env, wire_ids)
    row: dict = {
      "step": step,
      "base_x": float(base[0]),
      "base_y": float(base[1]),
      "base_z": float(base[2]),
      "hold_level": float(term.hold_level[0]),
      "partner_level_max": float(term.partner_level[:, 0].max()),
      "lift_level_max": float(term.lift_level[:, 0].max()),
    }
    grav = entity.data.projected_gravity_b[0]
    angvel = entity.data.root_link_ang_vel_b[0]
    row.update(
      {
        "grav_x": float(grav[0]),
        "grav_y": float(grav[1]),
        "grav_z": float(grav[2]),
        "angvel_x": float(angvel[0]),
        "angvel_y": float(angvel[1]),
        "angvel_z": float(angvel[2]),
      }
    )
    for g in LEG_NAMES:
      row[f"{g}_foot_z"] = float(entity.data.site_pos_w[0, site_names.index(g), 2])
    for g, c in ctrl.items():
      calf = entity.data.geom_pos_w[:, c._calf_geom_idx, :]
      foot = entity.data.site_pos_w[:, c._foot_site_idx, :]
      calf_wx, _ = wire_geometry.local_wire_xz_at_y(seg, calf[:, 1])
      foot_wx, _ = wire_geometry.local_wire_xz_at_y(seg, foot[:, 1])
      sensor = env.scene.sensors[c.cfg.wire_contact_sensor_name]
      started = bool(c._step_started[0])
      jp = entity.data.joint_pos[:1]
      thigh_q = jp[:, c._step_thigh_jid]
      calf_q = jp[:, c._step_calf_jid]
      fk = c._leg_fk(thigh_q, calf_q)[0]
      row.update(
        {
          f"{g}_thigh": float(thigh_q[0]),
          f"{g}_calf_q": float(calf_q[0]),
          # Foot in the thigh-joint sagittal plane (x forward, z up, metres) and
          # the leg's angle from vertical (deg, positive = foot ahead of the hip).
          f"{g}_fk_x": float(fk[0]),
          f"{g}_fk_z": float(fk[1]),
          f"{g}_leg_angle_deg": math.degrees(math.atan2(float(fk[0]), -float(fk[1]))),
          f"{g}_phase": int(c.phase[0]),
          f"{g}_contact": bool((sensor.data.found[0] > 0).any()),
          f"{g}_calf_dx": float(calf[0, 0] - calf_wx[0]),
          f"{g}_calf_h": float(c.calf_height_above_wire[0]),
          f"{g}_foot_dx": float(foot[0, 0] - foot_wx[0]),
          f"{g}_calf_latch": bool(c._calf_clearance_achieved[0]),
          f"{g}_foot_latch": bool(c._foot_crossed_achieved[0]),
          f"{g}_step_started": started,
          f"{g}_step_k": int(c._step_clock()[0]) if started else -1,
          f"{g}_blocked": bool(c.step_block[0]) if c.step_block is not None else False,
        }
      )
      for key, flag in (
        ("contact", row[f"{g}_contact"]),
        ("enter", row[f"{g}_phase"] == 1),
        ("step", started),
        ("calf_latch", row[f"{g}_calf_latch"]),
        ("foot_latch", row[f"{g}_foot_latch"]),
      ):
        if flag and f"{g}_{key}" not in events:
          events[f"{g}_{key}"] = step
    return row

  zero = torch.zeros(env.action_manager.action.shape, device=device)
  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    counter = {"step": -1}

    def viewer_policy(_obs):
      if counter["step"] >= 0:
        rows.append(record(counter["step"]))
        if counter["step"] + 1 >= max_steps:
          raise SystemExit(0)
      counter["step"] += 1
      return zero

    try:
      NativeMujocoViewer(env, viewer_policy).run()
    except SystemExit:
      pass
  else:
    for step in range(max_steps):
      env.step(zero)
      rows.append(record(step))
  env.close()

  passed, detail = verdict(rows, pair)
  tag = "+".join(legs) or "walk"
  out_dir = Path(cfg.out_dir or f"logs/d6_1_front_pair_wire/{tag}_seed{seed}")
  out_dir.mkdir(parents=True, exist_ok=True)
  with open(out_dir / "trajectory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  label = "SKIP" if detail == "no wire contact" else ("PASS" if passed else "FAIL")
  if detail.startswith("fell over"):
    label = "FELL"
  if cfg.pair == "all":
    label, detail = staged_verdict(rows)
  summary = (
    f"legs={tag} seed={seed} serialize={cfg.serialize_steps} "
    f"start_x={effective_start_x(cfg, seed)} start_yaw={cfg.start_yaw} "
    f"start_delay={cfg.start_delay} step_advance={cfg.step_advance} "
    f"step_gate={cfg.step_gate} step_gate_timeout={cfg.step_gate_timeout} "
    f"own_planted={cfg.own_planted} td_window={cfg.td_window} "
    f"partner_hold={cfg.partner_hold}/{cfg.partner_hold_ramp_steps} "
    f"max_roll_deg={cfg.max_roll_deg} roll_gate_max_steps={cfg.roll_gate_max_steps} "
    f"max_attempts={cfg.max_attempts} retry_on={cfg.retry_on} "
    f"retry_delay={cfg.retry_delay} "
    f"min_body_vx={cfg.min_body_vx} "
    f"max_recovery_steps={cfg.max_recovery_steps} hold_mode={cfg.hold_mode} "
    f"hold_cmd_vx={cfg.hold_cmd_vx} "
    f"hold_window={cfg.hold_window} hold_ramp_steps={cfg.hold_ramp_steps} "
    f"hold_max_steps={cfg.hold_max_steps} "
    f"hold_steps={sum(r['hold_level'] > 0 for r in rows)} events={events}\n"
    f"FULL_CONFIG={dataclasses.asdict(cfg)}\n"
    f"VERDICT={label} {detail}"
  )
  (out_dir / "run_log.txt").write_text(summary + "\n")
  if not quiet:
    print(summary)
  return label, detail, events


def _parse_seeds(spec: str) -> list[int]:
  seeds: list[int] = []
  for part in spec.split(","):
    lo, _, hi = part.partition("-")
    seeds.extend(range(int(lo), int(hi or lo) + 1))
  return seeds


def _run_seed(args: tuple[D61Config, int]) -> tuple[int, str, str, dict]:
  cfg, seed = args
  if cfg.out_dir:
    cfg = dataclasses.replace(cfg, out_dir=f"{cfg.out_dir}/seed{seed}")
  label, detail, events = run(cfg, seed, quiet=True)
  return seed, label, detail, events


def main() -> None:
  import multiprocessing as mp

  import tyro

  cfg = tyro.cli(D61Config)
  if not cfg.seeds:
    run(cfg, cfg.seed)
    return
  cfg = dataclasses.replace(cfg, debug=False)
  # One fresh process per seed: building an env leaks ~180 MB that is never
  # returned, so an in-process 20-seed sweep grows to ~5 GB.
  ctx = mp.get_context("spawn")
  results = []
  with ctx.Pool(cfg.jobs, maxtasksperchild=1) as pool:
    tasks = [(cfg, s) for s in _parse_seeds(cfg.seeds)]
    for seed, label, detail, events in pool.imap(_run_seed, tasks):
      print(f"seed={seed} {label} {detail} {events}", flush=True)
      results.append(label)
  if cfg.pair == "all":
    counts = {k: results.count(k) for k in ("PASS", "FAIL_FRONT", "FAIL_REAR", "SKIP")}
    print(f"PASS {counts['PASS']}/{len(results)} {counts}")
    return
  n_pass = results.count("PASS")
  print(
    f"PASS {n_pass}/{len(results)} (FELL {results.count('FELL')}, "
    f"SKIP {results.count('SKIP')})"
  )


if __name__ == "__main__":
  main()
