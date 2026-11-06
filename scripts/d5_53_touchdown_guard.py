#!/usr/bin/env python
"""D5.53 -- Touchdown-guard recovery experiment (opt-in diagnostic variant).

Hypothesis (D5.52): activating the hybrid retract on FL one step before its
normal touchdown removes the support the trot support-transfer needs, causing
the body runaway. Test: defer hybrid activation while FL is about to touch
down, until FL has completed touchdown, stance, and lift-off.

No production file is modified. The guard wraps ``act._update_fsm`` and masks
only the ``found`` flag of the ``wire_contact`` sensor seen by the FSM while
the guard is deferring entry (the FSM is in NORMAL_WALK then, where that signal
is used only for ``entering_recovery``). The hybrid actuator path, commands,
gains, thresholds, FSM exit/crossing/release logic are the original code.

Guard state machine (one transition evaluation per policy step, using the state
at the end of the previous step -- the same information the FSM uses):

  IDLE            no entry candidate seen yet.
                  real contact AND FL foot height < h_enter AND FL foot vz < 0
                    -> DEFER_TOUCHDOWN  (entry masked)
                  real contact otherwise -> pass through (entry allowed;
                    state PASSED_THROUGH)
  DEFER_TOUCHDOWN FL ground contact for >= debounce consecutive steps
                    -> DEFER_STANCE
  DEFER_STANCE    FL ground contact absent for >= debounce consecutive steps
                    -> ARMED   (lift-off completed; guard disarmed)
  ARMED           real contact passes through: natural entry, hybrid unchanged.
  Any DEFER_* state also ends -> ARMED if ``max_defer_steps`` is exceeded
                    (safety timeout; logged as timeout, never silent).
  Guard states are only evaluated while the FSM phase is NORMAL_WALK.

Hysteresis/anti-chatter: the height/velocity test is evaluated only once per
entry candidate (in IDLE). All later transitions use the FL terrain-contact
sensor with a debounce, not the height threshold, so the guard cannot toggle
around the threshold.

Foot height = ``site_pos_w`` z of the FL foot site (sphere radius 0.022); the
terrain is a plane at z=0 (asserted at build). Foot vz = ``site_lin_vel_w`` z.
Quantities are those at the end of the previous policy step, like D5.16.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_16_trajectory_audit import build_env  # noqa: E402
from d5_52_wire_removed_forced_entry import (  # noqa: E402
  Recorder,
  SubstepWireProbe,
  _first,
  _handoff_step,
  _write_csv,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.recovery.mdp.actions import (  # noqa: E402
  PHASE_NORMAL_WALK,
  ResidualRecoveryAction,
)

VARIANTS = ("baseline", "guard_disabled", "guard_enabled")
GUARD_STATES = (
  "DISABLED",
  "IDLE",
  "DEFER_TOUCHDOWN",
  "DEFER_STANCE",
  "ARMED",
  "PASSED_THROUGH",
)
DEFERRING = ("DEFER_TOUCHDOWN", "DEFER_STANCE")


@dataclass
class D553Config:
  variant: Literal["all", "baseline", "guard_disabled", "guard_enabled"] = "all"
  seed: int | None = None
  """Single seed for viewer or single-variant runs; None = both 0 and 11."""
  device: str = "cpu"
  max_steps: int = 250
  max_recovery_steps_override: int = 48
  out_dir: str = "logs/d5_53"
  h_enter_m: float = 0.030
  """FL foot-site height (m) below which a descending foot counts as about to
  touch down. Foot radius is 0.022, resting site height 0.021-0.022, measured
  swing apex 0.039 (D5.16 seed 0, steps 0-43): 0.030 = radius + 8 mm, i.e. the
  last ~half of the descent."""
  debounce_steps: int = 2
  max_defer_steps: int = 75
  debug: bool = False
  """Open the D5.15-pattern native viewer (needs --variant and --seed)."""


class TouchdownGuard:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    act: ResidualRecoveryAction,
    enabled: bool,
    cfg: D553Config,
  ):
    self.env, self.act, self.enabled, self.cfg = env, act, enabled, cfg
    self.state = "IDLE" if enabled else "DISABLED"
    self.calls = 0
    self.consec = 0
    self.defer_count = 0
    self.deferred_now = False
    self.events: list[dict[str, Any]] = []
    robot = env.scene["robot"]
    self.foot_site = list(robot.site_names).index("FL")
    mjm = env.sim.mj_model
    gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert gid >= 0 and int(mjm.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    assert abs(float(mjm.geom_pos[gid, 2])) < 1e-9, "terrain plane not at z=0"
    self.name = act.cfg.wire_contact_sensor_name
    self.real_sensor = env.scene.sensors[self.name]
    self.fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    self.last = {"h": float("nan"), "vz": float("nan"), "contact": False}
    self.original = act._update_fsm
    act._update_fsm = self._wrapped  # ty: ignore[invalid-assignment]

  def _log(self, event: str) -> None:
    self.events.append({"step": self.calls - 1, "event": event, "state": self.state})
    print(f"[D5.53 guard] step={self.calls - 1} {event} -> {self.state}")

  def _wrapped(self) -> None:
    self.calls += 1
    robot = self.env.scene["robot"]
    h = float(robot.data.site_pos_w[0, self.foot_site, 2])
    vz = float(robot.data.site_lin_vel_w[0, self.foot_site, 2])
    contact = bool((self.real_sensor.data.found[0] > 0).any())
    fl_ground = bool((self.fl_ground.data.found[0] > 0).any())
    self.last = {"h": h, "vz": vz, "contact": contact}
    self.deferred_now = False
    normal = int(self.act.phase[0]) == PHASE_NORMAL_WALK
    if self.enabled and normal:
      cfg = self.cfg
      if self.state == "IDLE" and contact:
        if h < cfg.h_enter_m and vz < 0.0:
          self.state = "DEFER_TOUCHDOWN"
          self.consec = self.defer_count = 0
          self._log(f"defer entry (h={h:.4f} vz={vz:.3f})")
        else:
          self.state = "PASSED_THROUGH"
          self._log(f"entry allowed, not near touchdown (h={h:.4f} vz={vz:.3f})")
      elif self.state in DEFERRING:
        self.defer_count += 1
        if self.state == "DEFER_TOUCHDOWN":
          self.consec = self.consec + 1 if fl_ground else 0
          if self.consec >= cfg.debounce_steps:
            self.state, self.consec = "DEFER_STANCE", 0
            self._log("FL touchdown confirmed")
        else:
          self.consec = self.consec + 1 if not fl_ground else 0
          if self.consec >= cfg.debounce_steps:
            self.state, self.consec = "ARMED", 0
            self._log("FL lift-off confirmed; guard re-armed")
        if self.state in DEFERRING and self.defer_count > cfg.max_defer_steps:
          self.state = "ARMED"
          self._log("TIMEOUT: guard released without lift-off")
      self.deferred_now = self.state in DEFERRING
    if self.deferred_now:
      sensors = self.env.scene.sensors
      masked = SimpleNamespace(
        data=SimpleNamespace(found=torch.zeros_like(self.real_sensor.data.found))
      )
      sensors[self.name] = cast(Any, masked)
      try:
        self.original()
      finally:
        sensors[self.name] = self.real_sensor
    else:
      self.original()


def _setup(cfg: D553Config, variant: str, seed: int):
  d3 = D3GroundConfig(seed=seed, device=cfg.device, use_isolated_wire=True)
  env = build_env(d3, cfg.device, cfg.max_recovery_steps_override)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  guard = None
  if variant != "baseline":
    guard = TouchdownGuard(env, act, variant == "guard_enabled", cfg)
  probe = SubstepWireProbe(env)
  rec = Recorder(env, act)
  robot = env.scene["robot"]
  foot_site = list(robot.site_names).index("FL")

  def make_row(step: int) -> dict[str, Any]:
    r = rec.row(step, variant, -1)
    r["seed"] = seed
    r["variant"] = variant
    r["wire_contact_trigger_real"] = r["wire_contact_sensor"]
    r["fl_foot_world_z_m"] = float(robot.data.site_pos_w[0, foot_site, 2])
    r["fl_foot_world_vz_mps"] = float(robot.data.site_lin_vel_w[0, foot_site, 2])
    if guard is not None:
      r["guard_state"] = guard.state
      r["guard_deferred_this_step"] = guard.deferred_now
      r["guard_eval_foot_h_m"] = guard.last["h"]
      r["guard_eval_foot_vz_mps"] = guard.last["vz"]
    else:
      r["guard_state"] = "NONE"
      r["guard_deferred_this_step"] = False
      r["guard_eval_foot_h_m"] = float("nan")
      r["guard_eval_foot_vz_mps"] = float("nan")
    return r

  return env, act, guard, probe, make_row


def run_variant(cfg: D553Config, variant: str, seed: int) -> dict[str, Any]:
  env, act, guard, probe, make_row = _setup(cfg, variant, seed)
  env.reset(seed=seed)
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  rows = []
  for step in range(cfg.max_steps):
    probe.begin_step(step)
    env.step(zero)
    rows.append(make_row(step))
    if rows[-1]["fell_over"]:
      print(f"[D5.53] {variant} seed{seed}: fell_over at step {step}; stopping")
      break
  nexclude = int(env.sim.mj_model.nexclude)
  env.close()
  return {
    "rows": rows,
    "substeps": probe.rows,
    "events": guard.events if guard else [],
    "nexclude": nexclude,
  }


def run_viewer(cfg: D553Config, variant: str, seed: int) -> Path:
  from mjlab.viewer import NativeMujocoViewer

  env, act, guard, probe, make_row = _setup(cfg, variant, seed)
  env.reset(seed=seed)
  rows: list[dict[str, Any]] = []
  pending: dict[str, Any] = {"step": None}

  def viewer_policy(_obs):  # noqa: ANN001
    if pending["step"] is not None:
      rows.append(make_row(pending["step"]))
      if pending["step"] % 25 == 0:
        print(
          f"[D5.53 viewer] {variant} seed{seed} step={pending['step']} "
          f"phase={rows[-1]['phase']} guard={rows[-1]['guard_state']}"
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
  out = Path(cfg.out_dir) / f"viewer_{variant}_seed{seed}"
  out.mkdir(parents=True, exist_ok=True)
  path = out / "trace.csv"
  _write_csv(path, rows)
  print(f"[D5.53 viewer] wrote {path} ({len(rows)} rows)")
  return path


def summarize(variant: str, seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
  by = {r["step"]: r for r in rows}
  handoff = _handoff_step(rows)
  first_contact = _first(rows, lambda r: r["wire_contact_sensor"], 0)
  align = handoff if handoff is not None else (first_contact or 0)
  win = [by[s] for s in range(align, align + 10) if s in by]
  pre = by.get(align - 1)

  def support(r):
    return sum(abs(r[f"{leg}_ground_force_z_n"]) for leg in ("fl", "fr", "rl", "rr"))

  def stance_pair_lost(r):
    return not r["fr_ground_contact"] and not r["rl_ground_contact"]

  def all_support_lost(r):
    return all(not r[f"{leg}_ground_contact"] for leg in ("fr", "rl", "rr", "fl"))

  exit_ = _first(rows, lambda r: r["newly_recovered"], 0)
  gs = {
    e: _first(rows, lambda r, e=e: r["guard_state"] == e, 0)
    for e in ("DEFER_TOUCHDOWN", "DEFER_STANCE", "ARMED", "PASSED_THROUGH")
  }
  out: dict[str, Any] = {
    "seed": seed,
    "variant": variant,
    "rows": len(rows),
    "first_wire_contact_step": first_contact,
    "recovery_entry_step": _first(rows, lambda r: r["phase"] == "RECOVERY", 0),
    "hybrid_handoff_step": handoff,
    "guard_defer_start_step": gs["DEFER_TOUCHDOWN"],
    "guard_touchdown_step": gs["DEFER_STANCE"],
    "guard_rearm_step": gs["ARMED"],
    "guard_passed_through_step": gs["PASSED_THROUGH"],
    "deferred_steps": sum(1 for r in rows if r["guard_deferred_this_step"]),
    "alignment_step_k0": align,
    "pre_handoff_pitch_rate_rps": pre["base_ang_vel_b_y_rps"] if pre else None,
  }
  if win:
    out.update(
      {
        "peak_abs_pitch_rate_k0_9_rps": max(
          abs(r["base_ang_vel_b_y_rps"]) for r in win
        ),
        "base_z_drop_k0_9_m": (pre["base_z_m"] - min(r["base_z_m"] for r in win))
        if pre
        else None,
        "min_world_vz_k0_9_mps": min(r["base_lin_vel_w_z_mps"] for r in win),
        "min_total_support_k0_9_n": min(support(r) for r in win),
        "peak_abs_roll_rate_k0_9_rps": max(abs(r["base_ang_vel_b_x_rps"]) for r in win),
        "peak_abs_fl_calf_vel_k0_9_radps": max(
          abs(r["FL_calf_vel_radps"]) for r in win
        ),
        "max_fl_ground_force_k0_9_n": max(abs(r["fl_ground_force_z_n"]) for r in win),
        "all4_feet_off_ground_k": (
          (s - align)
          if (s := _first(rows, all_support_lost, align)) is not None
          else None
        ),
        "fr_rl_both_off_k": (
          (s - align)
          if (s := _first(rows, stance_pair_lost, align)) is not None
          else None
        ),
      }
    )
  out.update(
    {
      "min_base_z_full_run_m": min(r["base_z_m"] for r in rows),
      "peak_abs_pitch_rate_full_run_rps": max(
        abs(r["base_ang_vel_b_y_rps"]) for r in rows
      ),
      "first_crossing_step": _first(rows, lambda r: r["crossing_achieved"], 0),
      "first_release_step": _first(rows, lambda r: r["release_achieved"], 0),
      "recovery_exit_step": exit_,
      "recovery_exit_timeout": bool(by[exit_]["recovery_failed_timeout"])
      if exit_ is not None
      else None,
      "recovery_ever_triggered": first_contact is not None,
      "fell_over_step": _first(rows, lambda r: r["fell_over"], 0),
      "max_robot_wire_force_full_run_n": None,
    }
  )
  return out


def max_diff(a: list[dict], b: list[dict], skip: tuple[str, ...]) -> tuple[float, int]:
  worst, nonnum = 0.0, 0
  for ra, rb in zip(a, b, strict=False):
    for k, va in ra.items():
      if k in skip or k not in rb:
        continue
      vb = rb[k]
      if isinstance(va, float) and isinstance(vb, float):
        if va != va and vb != vb:
          continue
        worst = max(worst, abs(va - vb))
      elif va != vb:
        nonnum += 1
  return worst, nonnum


def main() -> None:
  import tyro

  cfg = tyro.cli(D553Config)
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  if cfg.debug:
    assert cfg.variant != "all" and cfg.seed is not None, (
      "--debug requires --variant and --seed"
    )
    run_viewer(cfg, cfg.variant, cfg.seed)
    return

  variants = VARIANTS if cfg.variant == "all" else (cfg.variant,)
  seeds = (0, 11) if cfg.seed is None else (cfg.seed,)
  res: dict[tuple[str, int], dict[str, Any]] = {}
  for seed in seeds:
    for v in variants:
      r = run_variant(cfg, v, seed)
      res[(v, seed)] = r
      _write_csv(out / f"trace_{v}_seed{seed}.csv", r["rows"])
      _write_csv(out / f"substeps_{v}_seed{seed}.csv", r["substeps"])
      if r["events"]:
        _write_csv(out / f"guard_events_{v}_seed{seed}.csv", r["events"])
      print(f"[D5.53] wrote trace_{v}_seed{seed}.csv ({len(r['rows'])} rows)")
  if cfg.variant != "all" or cfg.seed is not None:
    return

  checks: list[dict[str, Any]] = []

  def add(name: str, value: str, ok: bool) -> None:
    checks.append({"check": name, "value": value, "pass": bool(ok)})

  # 1. Seed-0 baseline reproduces D5.52 baseline (and thus D5.49).
  ref = {
    int(r["step"]): r for r in csv.DictReader(open("logs/d5_52/trace_baseline.csv"))
  }
  cols = [
    "base_z_m",
    "base_ang_vel_b_y_rps",
    "base_lin_vel_w_z_mps",
    "FL_thigh_pos_rad",
    "FL_calf_pos_rad",
    "FL_calf_vel_radps",
    "wire_contact_sensor_force_mag_n",
    "fr_ground_force_z_n",
    "FR_calf_a_walk_raw",
  ]
  mx = 0.0
  for r in res[("baseline", 0)]["rows"]:
    if r["step"] in ref:
      for c in cols:
        mx = max(mx, abs(float(ref[r["step"]][c]) - float(r[c])))
  add("seed0_baseline_reproduces_d5_52", f"max_abs_diff={mx:.3e} tol=1e-5", mx <= 1e-5)

  # 2. guard-disabled reproduces baseline exactly (all measurement columns).
  skip = ("variant", "condition", "guard_state")
  for seed in seeds:
    d, n = max_diff(
      res[("baseline", seed)]["rows"], res[("guard_disabled", seed)]["rows"], skip
    )
    add(
      f"guard_disabled_equals_baseline_seed{seed}",
      f"max_abs_numeric_diff={d:.3e} nonnumeric_mismatches={n}",
      d == 0.0 and n == 0,
    )

  # 3. Wire collisions enabled in all wire-on variants.
  base_nex = res[("baseline", 0)]["nexclude"]
  for seed in seeds:
    for v in variants:
      r = res[(v, seed)]
      cnt = sum(s["robot_wire_contact_count"] for s in r["substeps"])
      add(
        f"wire_collisions_enabled_{v}_seed{seed}",
        f"nexclude={r['nexclude']} (baseline {base_nex}); robot-wire substep contacts={cnt}",
        r["nexclude"] == base_nex and cnt > 0,
      )

  # 4. No hybrid activation during deferral.
  for seed in seeds:
    rows = res[("guard_enabled", seed)]["rows"]
    bad = [
      r["step"]
      for r in rows
      if r["guard_deferred_this_step"]
      and (
        r["phase"] != "NORMAL_WALK"
        or r["hybrid_mode_active"]
        or r["fl_thigh_pos_gain"] <= 0
        or r["fl_calf_pos_gain"] <= 0
        or r["thigh_vel_cmd"] != 0
        or r["calf_torque_cmd"] != 0
        or r["act_FL_calf_joint_recovery_motor_ctrl"] != 0
        or r["act_FL_thigh_joint_recovery_vel_ctrl"] != 0
      )
    ]
    n_def = sum(1 for r in rows if r["guard_deferred_this_step"])
    add(
      f"no_hybrid_during_deferral_seed{seed}",
      f"deferred_steps={n_def} violations={bad}",
      not bad,
    )

  # 5. Guard transitions and re-arm.
  for seed in seeds:
    ev = res[("guard_enabled", seed)]["events"]
    names = [e["event"].split(" (")[0] for e in ev]
    deferred = any(n.startswith("defer entry") for n in names)
    if deferred:
      ok = (
        "FL touchdown confirmed" in names
        and "FL lift-off confirmed; guard re-armed" in names
        and names.index("FL touchdown confirmed")
        < names.index("FL lift-off confirmed; guard re-armed")
        and not any(n.startswith("TIMEOUT") for n in names)
      )
      add(f"guard_sequence_touchdown_stance_liftoff_seed{seed}", " > ".join(names), ok)
    else:
      add(
        f"guard_sequence_seed{seed}",
        "guard never deferred: " + (" > ".join(names) or "no entry candidate"),
        True,
      )

  # 6. Seed 11 natural trigger reported, not forced.
  s11 = res[("baseline", 11)]["rows"]
  trig = _first(s11, lambda r: r["wire_contact_sensor"], 0)
  add("seed11_natural_contact_trigger", f"first_contact_step={trig}", trig is not None)

  _write_csv(out / "validation.csv", checks)
  for c in checks:
    print(f"[D5.53] {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {c['value']}")

  summaries = []
  for seed in seeds:
    for v in variants:
      s = summarize(v, seed, res[(v, seed)]["rows"])
      sub = res[(v, seed)]["substeps"]
      s["max_robot_wire_force_full_run_n"] = max(
        x["robot_wire_force_mag_n"] for x in sub
      )
      s["robot_wire_substeps_in_contact"] = sum(
        1 for x in sub if x["robot_wire_contact_count"] > 0
      )
      summaries.append(s)
  _write_csv(out / "comparison_summary.csv", summaries)

  aligned = []
  for seed in seeds:
    base_h = _handoff_step(res[("baseline", seed)]["rows"])
    for v in variants:
      h = _handoff_step(res[(v, seed)]["rows"])
      sub: dict[int, dict[str, Any]] = {}
      for s in res[(v, seed)]["substeps"]:
        d = sub.setdefault(s["step"], {"n": 0, "fmax": 0.0})
        d["n"] = max(d["n"], s["robot_wire_contact_count"])
        d["fmax"] = max(d["fmax"], s["robot_wire_force_mag_n"])
      for r in res[(v, seed)]["rows"]:
        lo = (base_h if base_h is not None else 40) - 10
        if lo <= r["step"] <= lo + 60:
          a = {
            "k_from_baseline_handoff": r["step"] - (base_h or 0),
            "k_from_own_handoff": (r["step"] - h) if h is not None else "",
            **r,
          }
          d = sub.get(r["step"], {"n": 0, "fmax": 0.0})
          a["substep_robot_wire_contact_count_max"] = d["n"]
          a["substep_robot_wire_force_max_n"] = d["fmax"]
          aligned.append(a)
  _write_csv(out / "aligned_trajectory.csv", aligned)
  for s in summaries:
    print(s)
  if not all(c["pass"] for c in checks):
    print("[D5.53] PRECONDITION FAILURE -- results must not be interpreted")
    sys.exit(2)


if __name__ == "__main__":
  main()
