#!/usr/bin/env python
"""D5.57 -- Wire-off replay of the D5.55 bounded release (stall diagnostic).

Question: does the post-release walking stall (D5.55/D5.56) need the continuous
FL calf-wire contact, or does it also occur with the wire removed?

Conditions (seeds 0 and 11; D5.52 wire-exclusion build, i.e. zero robot-wire
collisions; same checkpoint/config/reset as D5.55):

  wire_off_control            normal frozen walking policy, no hybrid entry.
  wire_off_bounded_retraction the D5.55 ``BoundedRetraction`` wrapper, imported
                              unchanged, with the existing FSM entry forced at
                              the matching FL gait phase.

Matching phase: in the wire-on D5.55 runs hybrid entry happens two steps after
FL lift-off (lift-off row 60, entry 62). Here the FSM's contact input is forced
true for exactly one step: the first step at which FL has been off the ground
for two consecutive steps (same terrain-contact sensor and same counting the
wrapper uses) after FL has touched down at/after step 44. Nothing else is
changed: hybrid commands, gains, budget, release, exit/crossing logic, policy,
checkpoint and observation processing are the existing code.

Everything is kept in memory unless ``--trace-dir`` is given (scratch space, for
the viewer-vs-headless byte comparison). ``summary.csv`` is the only artifact.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import d5_52_wire_removed_forced_entry as d552  # noqa: E402
from d5_55_bounded_retraction import BoundedRetraction  # noqa: E402
from d5_56_policy_stall_diag import Capture, p2p  # noqa: E402

from mjlab.tasks.recovery.mdp.actions import (  # noqa: E402
  PHASE_NORMAL_WALK,
  ResidualRecoveryAction,
)

CONDITIONS = ("wire_off_control", "wire_off_bounded_retraction")
SEEDS = (0, 11)
MIN_STANCE_STEP = 44
AIRBORNE_STEPS_AT_ENTRY = 2
SWING_Z_M = 0.03
LATE_LO, LATE_HI = 140, 250  # late window, after the FSM exit in every run
# D5.55 wire-on values the matched variant must reproduce.
D555_SWING_EST, D555_AIRBORNE, D555_BUDGET = 11, 2, 9


@dataclass
class D557Config:
  condition: Literal["all", "wire_off_control", "wire_off_bounded_retraction"] = "all"
  seed: int | None = None
  device: str = "cpu"
  max_steps: int = 250
  out_dir: str = "logs/d5_57"
  trace_dir: str = ""
  """Scratch directory for per-run traces (viewer-vs-headless comparison only)."""
  debug: bool = False
  """Open the D5.15-pattern native viewer (needs --condition and --seed)."""


class PhaseForcedEntry:
  """Force the existing FSM entry once, two steps after an FL lift-off."""

  def __init__(self, env, act: ResidualRecoveryAction):
    self.env, self.act = env, act
    self.fl = env.scene.sensors["fl_ground_contact_diag"]
    self.name = act.cfg.wire_contact_sensor_name
    self.real = env.scene.sensors[self.name]
    self.calls = 0
    self.airborne = 0
    self.stance_seen = False
    self.fired = False
    self.forced_step = -1
    self.first_airborne_step = -1
    self.original = act._update_fsm
    act._update_fsm = self._wrapped  # ty: ignore[invalid-assignment]

  def _wrapped(self) -> None:
    step = self.calls
    self.calls += 1
    contact = bool((self.fl.data.found[0] > 0).any())
    if contact:
      if step >= MIN_STANCE_STEP:
        self.stance_seen = True
      self.airborne = 0
    else:
      self.airborne += 1
      if self.stance_seen and self.airborne == 1 and self.first_airborne_step < 0:
        self.first_airborne_step = step
    normal = int(self.act._phase[0]) == PHASE_NORMAL_WALK
    fire = (
      not self.fired
      and normal
      and self.stance_seen
      and self.airborne == AIRBORNE_STEPS_AT_ENTRY
    )
    if not fire:
      self.original()
      return
    sensors = self.env.scene.sensors
    forced = SimpleNamespace(
      data=SimpleNamespace(found=torch.ones_like(self.real.data.found))
    )
    sensors[self.name] = cast(Any, forced)
    try:
      self.original()
    finally:
      sensors[self.name] = self.real
    self.fired = True
    self.forced_step = step
    print(
      f"[D5.57] forced FSM entry at step {step} (FL airborne {self.airborne} steps)"
    )


class Run:
  def __init__(self, cfg: D557Config, condition: str, seed: int):
    cfg552 = d552.D552Config(seed=seed, device=cfg.device, max_steps=cfg.max_steps)
    self.env = d552.build_condition_env(cfg552, wire_off=True)
    self.act = cast(
      ResidualRecoveryAction, self.env.action_manager.get_term("joint_pos")
    )
    self.condition, self.seed = condition, seed
    self.forced = self.br = None
    if condition == "wire_off_bounded_retraction":
      self.forced = PhaseForcedEntry(self.env, self.act)
      self.br = BoundedRetraction(self.env, self.act)
    self.probe = d552.SubstepWireProbe(self.env)
    self.cap = Capture(self.env, self.act)
    self.fl_sensor = self.env.scene.sensors["fl_ground_contact_diag"]
    self.rows: list[dict[str, Any]] = []
    self.n_sub = 0
    self.ever_succeeded = False

  def begin(self, step: int) -> None:
    self.probe.begin_step(step)

  def record(self, step: int) -> None:
    act, env = self.act, self.env
    self.cap.record()
    new = self.probe.rows[self.n_sub :]
    self.n_sub = len(self.probe.rows)
    m = env.sim.model
    self.ever_succeeded = self.ever_succeeded or bool(act.newly_succeeded[0])
    row: dict[str, Any] = {
      "step": step,
      "phase": int(act._phase[0]),
      "bounded_status": self.br.status if self.br else "NONE",
      "fl_contact": bool((self.fl_sensor.data.found[0] > 0).any()),
      "fl_foot_z": self.cap.foot_z[-1],
      "base_x": self.cap.base_x[-1],
      "body_vx": float(env.scene["robot"].data.root_link_lin_vel_b[0, 0]),
      "wire_substep_contacts": int(sum(s["robot_wire_contact_count"] for s in new)),
      "wire_force_max_n": max((s["robot_wire_force_mag_n"] for s in new), default=0.0),
      "hyb_thigh_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
      "hyb_calf_cmd": float(act.hybrid_calf_torque_ctrl[0]),
      "fl_thigh_pos_gain": float(m.actuator_gainprm[0, act._pos_thigh_ctrl_id, 0]),
      "fl_calf_pos_gain": float(m.actuator_gainprm[0, act._pos_calf_ctrl_id, 0]),
      "aux_thigh_kd": float(m.actuator_gainprm[0, act._aux_thigh_ctrl_id, 0]),
      "newly_entangled": bool(act.newly_entangled[0]),
      "newly_recovered": bool(act.newly_recovered[0]),
      "crossing": bool(act.crossing_achieved[0]),
      "release": bool(act.release_achieved[0]),
      "timeout": bool(act._recovery_failed_timeout[0]),
    }
    for i, v in enumerate(self.cap.obs[-1]):
      row[f"obs{i}"] = float(v)
    for tag, arr in (
      ("raw", self.cap.raw[-1]),
      ("tgt", self.cap.target[-1]),
      ("ctrl", self.cap.ctrl[-1]),
    ):
      for i, v in enumerate(arr):
        row[f"{tag}{i}"] = float(v)
    self.rows.append(row)

  def close(self) -> None:
    self.env.close()


def run_headless(cfg: D557Config, condition: str, seed: int) -> Run:
  r = Run(cfg, condition, seed)
  r.env.reset(seed=seed)
  zero = torch.zeros(1, r.env.action_manager.total_action_dim, device=r.env.device)
  for step in range(cfg.max_steps):
    r.begin(step)
    r.env.step(zero)
    r.record(step)
  r.close()
  return r


def run_viewer(cfg: D557Config, condition: str, seed: int) -> Run:
  from mjlab.viewer import NativeMujocoViewer

  r = Run(cfg, condition, seed)
  r.env.reset(seed=seed)
  pending: dict[str, Any] = {"step": None}

  def viewer_policy(_obs):  # noqa: ANN001
    if pending["step"] is not None:
      r.record(pending["step"])
      if pending["step"] % 25 == 0:
        print(f"[D5.57 viewer] {condition} seed{seed} step={pending['step']}")
      if pending["step"] + 1 >= cfg.max_steps:
        raise SystemExit(0)
    step = 0 if pending["step"] is None else pending["step"] + 1
    r.begin(step)
    pending["step"] = step
    return torch.zeros(r.env.action_manager.action.shape, device=r.env.device)

  try:
    NativeMujocoViewer(cast(Any, r.env), cast(Any, viewer_policy)).run()
  except SystemExit:
    pass
  r.close()
  return r


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def rising(flags: list[bool], start: int) -> list[int]:
  return [i for i in range(max(1, start), len(flags)) if flags[i] and not flags[i - 1]]


def analyse(
  r: Run,
  window_t0: int | None = None,
  seg_a: tuple[int, int] | None = None,
  seg_b: tuple[int, int] | None = None,
) -> dict[str, Any]:
  rows = r.rows
  fl = [x["fl_contact"] for x in rows]
  z = np.array([x["fl_foot_z"] for x in rows])
  obs = np.array(r.cap.obs)
  raw = np.array(r.cap.raw)
  sc = r.cap.scale
  out: dict[str, Any] = {
    "wire_substep_contacts_total": sum(x["wire_substep_contacts"] for x in rows)
  }
  entry = next((x["step"] for x in rows if x["newly_entangled"]), None)
  out["entry_step"] = entry
  out["phase_all_normal"] = all(x["phase"] == 0 for x in rows)
  exit_ = next((x["step"] for x in rows if x["newly_recovered"]), None)
  out["fsm_exit_step"] = exit_
  out["crossing_step"] = next((x["step"] for x in rows if x["crossing"]), None)
  out["exit_timeout"] = bool(rows[exit_]["timeout"]) if exit_ is not None else None
  out["ever_succeeded"] = r.ever_succeeded
  if r.br is not None and entry is not None:
    rel = next((e["step"] for e in r.br.events if e["msg"].startswith("RELEASE")), None)
    out["release_step"] = rel
    out["resume_events"] = sum(1 for e in r.br.events if e["msg"].startswith("RESUME"))
    out["first_airborne_row"] = r.forced.first_airborne_step if r.forced else None
    t0 = next(iter(rising(fl, rel)), None) if rel is not None else None
  else:
    t0 = None
  if window_t0 is not None:
    t0 = window_t0
  out["touchdown_step"] = t0
  if t0 is not None and t0 + 36 <= len(rows):
    lo, hi = t0 + 12, t0 + 37
    out["plateau"] = (lo, hi - 1)
    out["fl_foot_z_max_plateau"] = float(z[lo:hi].max())
    out["fl_foot_z_max_t0_to_t0_36"] = float(z[t0 : t0 + 37].max())
    after = [i for i in range(t0, min(len(rows), t0 + 60)) if z[i] > SWING_Z_M]
    out["first_step_foot_z_above_0p03_after_touchdown"] = (
      (after[0] - t0) if after else None
    )
    out["imu_vx_plateau_mean"] = float(obs[lo:hi, 0].mean())
    out["raw_p2p_mean_plateau"] = float(p2p(raw[lo:hi]).mean())
    err = [sc[1] * raw[t][1] - obs[t][10] for t in range(lo, hi)]
    out["fl_thigh_target_minus_actual_plateau"] = float(np.mean(err))
    out["fl_thigh_pos_rel_default_plateau"] = float(obs[lo:hi, 10].mean())
    out["base_x_change_plateau"] = float(rows[hi - 1]["base_x"] - rows[lo]["base_x"])
    out["fl_contact_gaps_ge2_in_plateau"] = int(
      sum(
        1
        for i in range(lo, hi - 1)
        if (not fl[i]) and (not fl[i + 1]) and (i == lo or fl[i - 1])
      )
    )
  # --- post-hoc segmentation (added after run 1 showed the literal plateau
  # contains resumed hybrid cycles): segment A = policy-only rows after the
  # first release (until the wrapper resumes the hybrid); segment B = the 25
  # rows after the FSM exit. Both are reported next to the literal plateau.
  status = [x["bounded_status"] for x in rows]
  if seg_a is None and t0 is not None:
    rel_ = out.get("release_step")
    resume = None
    if rel_ is not None:
      resume = next(
        (i for i in range(rel_ + 1, len(rows)) if status[i] == "ACTIVE"), None
      )
    out["resume_row"] = resume
    seg_a = (t0, resume if resume is not None else min(len(rows), t0 + 37))
  if seg_a is not None:
    lo, hi = seg_a
    out["_segA"] = seg_a
    out["segA_steps"] = hi - lo
    out["segA_fl_foot_z_max"] = float(z[lo:hi].max())
    lift = next((i for i in range(lo, hi - 1) if (not fl[i]) and (not fl[i + 1])), None)
    out["segA_fl_stance_steps_before_liftoff"] = (
      (lift - lo) if lift is not None else None
    )
    out["segA_body_vx_mean"] = float(np.mean([x["body_vx"] for x in rows[lo:hi]]))
    out["segA_active_rows"] = sum(1 for i in range(lo, hi) if status[i] == "ACTIVE")
    if hi - lo >= 3:
      out["segA_raw_p2p_mean"] = float(p2p(raw[lo:hi]).mean())
      out["segA_thigh_err"] = float(
        np.mean([sc[1] * raw[t][1] - obs[t][10] for t in range(lo, hi)])
      )
  if seg_b is None and exit_ is not None:
    seg_b = (exit_ + 1, min(len(rows), exit_ + 26))
  if seg_b is not None and seg_b[1] > seg_b[0]:
    lo, hi = seg_b
    out["_segB"] = seg_b
    out["segB_steps"] = hi - lo
    out["segB_body_vx_mean"] = float(np.mean([x["body_vx"] for x in rows[lo:hi]]))
    out["segB_base_x_change"] = float(rows[hi - 1]["base_x"] - rows[lo]["base_x"])
    out["segB_fl_foot_z_max"] = float(z[lo:hi].max())
    out["segB_fl_swings"] = sum(
      1 for i in range(lo, hi - 1) if fl[i - 1] and (not fl[i]) and (not fl[i + 1])
    )
    out["segB_active_rows"] = sum(1 for i in range(lo, hi) if status[i] == "ACTIVE")
  out["plateau_active_rows"] = (
    sum(1 for i in range(t0 + 12, t0 + 37) if status[i] == "ACTIVE")
    if t0 is not None and t0 + 36 <= len(rows)
    else None
  )
  if len(rows) >= 250:
    lo, hi = LATE_LO, LATE_HI
    out["segC_body_vx_mean"] = float(np.mean([x["body_vx"] for x in rows[lo:hi]]))
    out["segC_base_x_change"] = float(rows[hi - 1]["base_x"] - rows[lo]["base_x"])
    out["segC_fl_gap_swings"] = sum(
      1 for i in range(lo, hi - 1) if fl[i - 1] and (not fl[i]) and (not fl[i + 1])
    )
    out["segC_fl_z_gt_0p03_episodes"] = sum(
      1 for i in range(lo, hi) if z[i - 1] <= SWING_Z_M < z[i]
    )
    out["segC_fl_foot_z_max"] = float(z[lo:hi].max())
    out["segC_active_rows"] = sum(1 for i in range(lo, hi) if status[i] == "ACTIVE")
  return out


def d555_wire_on(seed: int) -> dict[str, Any]:
  """Same quantities from the existing D5.55 wire-on trace (no re-run)."""
  rows = list(
    csv.DictReader(open(f"logs/d5_55/trace_bounded_retraction_seed{seed}.csv"))
  )
  fl = [abs(float(x["fl_ground_force_z_n"])) > 0 for x in rows]
  vx = [float(x["base_lin_vel_b_x_mps"]) for x in rows]
  bx = [float(x["base_x_m"]) for x in rows]
  t0 = next(i for i in range(70, len(rows)) if fl[i] and not fl[i - 1])
  lo, hi = t0, t0 + 37
  lift = next((i for i in range(lo, hi - 1) if (not fl[i]) and (not fl[i + 1])), None)
  ex = next(int(x["step"]) for x in rows if x["newly_recovered"] == "True")
  blo, bhi = ex + 1, ex + 26
  return {
    "t0": t0,
    "segA_steps": hi - lo,
    "stance": (lift - lo) if lift is not None else None,
    "vx": float(np.mean(vx[lo:hi])),
    "exit": ex,
    "segB_vx": float(np.mean(vx[blo:bhi])),
    "segB_dx": bx[bhi - 1] - bx[blo],
    "segB_swings": sum(
      1 for i in range(blo, bhi - 1) if fl[i - 1] and (not fl[i]) and (not fl[i + 1])
    ),
    "segC_vx": float(np.mean(vx[LATE_LO:LATE_HI])),
    "segC_dx": bx[LATE_HI - 1] - bx[LATE_LO],
    "segC_swings": sum(
      1
      for i in range(LATE_LO, LATE_HI - 1)
      if fl[i - 1] and (not fl[i]) and (not fl[i + 1])
    ),
  }


def d556_wire_on(metric: str, seed: int) -> str:
  for row in csv.DictReader(open("logs/d5_56/summary.csv")):
    if row["seed"] == str(seed) and row["metric"] == metric:
      return row["stall_plateau_84_108"]
  return "n/a"


def main() -> None:
  import tyro

  cfg = tyro.cli(D557Config)
  trace_dir = Path(cfg.trace_dir) if cfg.trace_dir else None

  if cfg.debug:
    assert cfg.condition != "all" and cfg.seed is not None
    assert trace_dir is not None, "--debug requires --trace-dir (scratch)"
    r = run_viewer(cfg, cfg.condition, cfg.seed)
    write_trace(trace_dir / f"viewer_{cfg.condition}_seed{cfg.seed}.csv", r.rows)
    print(f"[D5.57 viewer] wrote {len(r.rows)} rows")
    return

  conditions = CONDITIONS if cfg.condition == "all" else (cfg.condition,)
  seeds = SEEDS if cfg.seed is None else (cfg.seed,)
  runs: dict[tuple[str, int], Run] = {}
  for seed in seeds:
    for cond in conditions:
      runs[(cond, seed)] = run_headless(cfg, cond, seed)
      if trace_dir is not None:
        write_trace(
          trace_dir / f"headless_{cond}_seed{seed}.csv", runs[(cond, seed)].rows
        )
  if cfg.condition != "all" or cfg.seed is not None:
    return

  checks: list[tuple[str, str, bool]] = []
  summary: list[dict[str, Any]] = []

  def add(seed, metric, wire_on, bounded, control, note=""):
    def fmt(v):
      if v is None:
        return "none"
      return (
        v
        if isinstance(v, str)
        else (round(float(v), 3) if not isinstance(v, bool) else str(v))
      )

    summary.append(
      {
        "seed": seed,
        "metric": metric,
        "d5_56_wire_on_plateau": fmt(wire_on),
        "wire_off_bounded_retraction": fmt(bounded),
        "wire_off_control_same_steps": fmt(control),
        "note": note,
      }
    )

  for seed in SEEDS:
    ctrl_run, br_run = (
      runs[("wire_off_control", seed)],
      runs[("wire_off_bounded_retraction", seed)],
    )
    b = analyse(br_run)
    c = analyse(ctrl_run, window_t0=b["touchdown_step"])
    # --- validation ---
    checks.append(
      (
        f"zero_robot_wire_contacts_seed{seed}",
        f"control={c['wire_substep_contacts_total']} bounded={b['wire_substep_contacts_total']} (all substeps)",
        c["wire_substep_contacts_total"] == 0 and b["wire_substep_contacts_total"] == 0,
      )
    )
    ctrl_gains_ok = all(
      x["fl_thigh_pos_gain"] > 0 and x["hyb_thigh_cmd"] == 0 and x["hyb_calf_cmd"] == 0
      for x in ctrl_run.rows
    )
    checks.append(
      (
        f"control_stays_on_walking_path_seed{seed}",
        f"all NORMAL_WALK={c['phase_all_normal']}, FL gains active and hybrid cmds 0={ctrl_gains_ok}",
        bool(c["phase_all_normal"] and ctrl_gains_ok),
      )
    )
    assert br_run.br is not None and br_run.forced is not None
    ev = [(e["step"], e["msg"]) for e in br_run.br.events] if br_run.br else []
    enter_msg = next((m for _, m in ev if m.startswith("ENTER")), "")
    phase_ok = (
      f"swing_est={D555_SWING_EST}" in enter_msg
      and f"airborne_already={D555_AIRBORNE}" in enter_msg
      and f"budget={D555_BUDGET}" in enter_msg
    )
    # Lift-off row = first FL no-contact row after the stance that follows
    # MIN_STANCE_STEP, read from the recorded trace (independent of the hook,
    # whose step counter is the FSM-call step and sees the previous step's
    # state). The D5.55 reference is measured the same way from its own trace.
    fl_rows = [x["fl_contact"] for x in br_run.rows]
    lift_row = next(
      (
        i
        for i in range(MIN_STANCE_STEP + 1, len(fl_rows))
        if not fl_rows[i] and fl_rows[i - 1]
      ),
      None,
    )
    d555_rows = list(
      csv.DictReader(open(f"logs/d5_55/trace_bounded_retraction_seed{seed}.csv"))
    )
    d555_force = [abs(float(x["fl_ground_force_z_n"])) > 0 for x in d555_rows]
    d555_lift = next(
      i
      for i in range(MIN_STANCE_STEP + 3, len(d555_force))
      if not d555_force[i] and d555_force[i - 1]
    )
    d555_entry = next(
      int(x["step"]) for x in d555_rows if x["hybrid_mode_active"] == "True"
    )
    lift_ok = (
      b["entry_step"] is not None
      and lift_row is not None
      and b["entry_step"] - lift_row == AIRBORNE_STEPS_AT_ENTRY
      and d555_entry - d555_lift == AIRBORNE_STEPS_AT_ENTRY
    )
    checks.append(
      (
        f"entry_phase_matches_d5_55_seed{seed}",
        f"wire-off: lift-off row {lift_row}, entry {b['entry_step']}; "
        f"D5.55 wire-on: lift-off row {d555_lift}, hybrid active {d555_entry}; "
        f"{enter_msg.strip()}",
        bool(phase_ok and lift_ok),
      )
    )
    er = br_run.rows[b["entry_step"]] if b["entry_step"] is not None else {}
    cmds_ok = (
      er.get("hyb_thigh_cmd") == 2.0
      and er.get("hyb_calf_cmd") == -5.0
      and er.get("fl_thigh_pos_gain") == 0.0
      and er.get("fl_calf_pos_gain") == 0.0
    )
    checks.append(
      (
        f"hybrid_commands_at_entry_seed{seed}",
        f"thigh={er.get('hyb_thigh_cmd')} calf={er.get('hyb_calf_cmd')} pos gains {er.get('fl_thigh_pos_gain')}/{er.get('fl_calf_pos_gain')}",
        bool(cmds_ok),
      )
    )
    rel = b.get("release_step")
    rel_ok = rel is not None and rel - b["entry_step"] == D555_BUDGET - 1
    after = (
      br_run.rows[rel + 1] if rel is not None and rel + 1 < len(br_run.rows) else {}
    )
    gains_ok = (
      after.get("fl_thigh_pos_gain") == 20.0
      and after.get("fl_calf_pos_gain") == 40.0
      and after.get("hyb_thigh_cmd") == 0.0
      and after.get("hyb_calf_cmd") == 0.0
      and after.get("aux_thigh_kd") == 0.0
    )
    preempt = (
      b["fsm_exit_step"] is not None and rel is not None and b["fsm_exit_step"] <= rel
    )
    checks.append(
      (
        f"release_matches_d5_55_seed{seed}",
        f"release={rel} (entry+{None if rel is None else rel - b['entry_step']}, D5.55: entry+8); "
        f"FL gains after release {after.get('fl_thigh_pos_gain')}/{after.get('fl_calf_pos_gain')}; "
        f"FSM exit {b['fsm_exit_step']} (preempts release: {preempt})",
        bool(rel_ok and gains_ok and not preempt),
      )
    )
    # --- summary rows ---
    add(
      seed,
      "validation_entry_step",
      "62",
      b["entry_step"],
      "none",
      "D5.55 wire-on entry = 62 (FL lift-off row 60)",
    )
    add(
      seed,
      "validation_release_step",
      "70",
      b.get("release_step"),
      "none",
      "entry + 8 in both",
    )
    add(
      seed,
      "validation_touchdown_step_T0",
      "72",
      b["touchdown_step"],
      b["touchdown_step"],
      "first FL contact at/after release; plateau = T0+12..T0+36",
    )
    add(
      seed,
      "validation_fsm_exit_step",
      "109 (cap timeout)",
      b["fsm_exit_step"],
      "none",
      "FSM exit; D5.55 wire-on timed out at the 48-step cap",
    )
    add(
      seed,
      "validation_robot_wire_substep_contacts",
      "continuous",
      b["wire_substep_contacts_total"],
      c["wire_substep_contacts_total"],
      "all substeps, whole run",
    )
    for key, metric, w_on_metric in (
      ("fl_foot_z_max_plateau", "FL_foot_z_max_m_plateau", "FL_foot_z_max_m"),
      (
        "imu_vx_plateau_mean",
        "imu_lin_vel_x_mps_plateau_mean",
        "imu_lin_vel_x_mps_mean",
      ),
      (
        "raw_p2p_mean_plateau",
        "raw_action_p2p_mean_12_joints",
        "raw_action_p2p_mean_over_12_joints",
      ),
      (
        "fl_thigh_target_minus_actual_plateau",
        "FL_thigh_target_minus_actual_rad",
        "FL_thigh_target_minus_actual_rad_mean",
      ),
      (
        "fl_thigh_pos_rel_default_plateau",
        "FL_thigh_pos_rel_default_rad",
        "FL_thigh_joint_pos_rel_default_rad",
      ),
      ("base_x_change_plateau", "base_x_change_m_plateau", "base_x_change_84_to_108_m"),
    ):
      add(seed, metric, d556_wire_on(w_on_metric, seed), b.get(key), c.get(key))
    add(
      seed,
      "FL_foot_z_max_m_T0_to_T0+36",
      "0.022 (plateau)",
      b.get("fl_foot_z_max_t0_to_t0_36"),
      c.get("fl_foot_z_max_t0_to_t0_36"),
      "swing reference threshold 0.03 m (reporting only)",
    )
    add(
      seed,
      "steps_after_touchdown_until_foot_z_above_0.03",
      "never",
      b.get("first_step_foot_z_above_0p03_after_touchdown"),
      c.get("first_step_foot_z_above_0p03_after_touchdown"),
      "within 60 steps of T0; none = no swing",
    )
    add(
      seed,
      "FL_contact_gaps_ge2_steps_in_plateau",
      "0",
      b.get("fl_contact_gaps_ge2_in_plateau"),
      c.get("fl_contact_gaps_ge2_in_plateau"),
      "2 consecutive no-contact steps = the D5.55 debounce",
    )

  for seed in SEEDS:
    ctrl_run, br_run = (
      runs[("wire_off_control", seed)],
      runs[("wire_off_bounded_retraction", seed)],
    )
    b = analyse(br_run)
    c = analyse(
      ctrl_run,
      window_t0=b["touchdown_step"],
      seg_a=b.get("_segA"),
      seg_b=b.get("_segB"),
    )
    w = d555_wire_on(seed)
    add(
      seed,
      "plateau_rows_with_hybrid_ACTIVE",
      "0",
      b["plateau_active_rows"],
      "none",
      "literal T0+12..T0+36 window; >0 means resumed hybrid retract inside it",
    )
    add(
      seed,
      "fsm_exit_step_and_reason",
      f"{w['exit']} cap timeout",
      f"{b['fsm_exit_step']} {'cap timeout' if b['exit_timeout'] else 'crossing'}",
      "none",
      "crossing step bounded: " + str(b["crossing_step"]),
    )
    add(
      seed,
      "segA_policy_only_after_first_release_steps",
      w["segA_steps"],
      b["segA_steps"],
      b["segA_steps"],
      "rows from first FL touchdown until hybrid resumes (wire-on: never resumed, T0..T0+36)",
    )
    add(
      seed,
      "segA_FL_stance_steps_before_liftoff",
      w["stance"] if w["stance"] is not None else "no lift-off",
      b["segA_fl_stance_steps_before_liftoff"],
      c["segA_fl_stance_steps_before_liftoff"],
      "first 2 consecutive no-contact rows (D5.55 debounce); normal stance in control = 13",
    )
    add(
      seed,
      "segA_FL_foot_z_max_m",
      "0.022",
      b["segA_fl_foot_z_max"],
      c["segA_fl_foot_z_max"],
      "policy-only rows (hybrid not active)",
    )
    add(
      seed,
      "segA_body_vx_mps_mean",
      w["vx"],
      b["segA_body_vx_mean"],
      c["segA_body_vx_mean"],
      "body-frame forward speed",
    )
    add(
      seed,
      "segA_raw_action_p2p_mean",
      d556_wire_on("raw_action_p2p_mean_over_12_joints", seed),
      b.get("segA_raw_p2p_mean"),
      c.get("segA_raw_p2p_mean"),
      "wire-on value = D5.56 plateau (84-108)",
    )
    add(
      seed,
      "segA_FL_thigh_target_minus_actual_rad",
      d556_wire_on("FL_thigh_target_minus_actual_rad_mean", seed),
      b.get("segA_thigh_err"),
      c.get("segA_thigh_err"),
      "wire-on value = D5.56 plateau",
    )
    add(
      seed,
      "segB_after_fsm_exit_body_vx_mps_mean",
      w["segB_vx"],
      b.get("segB_body_vx_mean"),
      c.get("segB_body_vx_mean"),
      "25 rows after exit; control = same absolute rows as bounded",
    )
    add(
      seed,
      "segB_after_fsm_exit_base_x_change_m",
      w["segB_dx"],
      b.get("segB_base_x_change"),
      c.get("segB_base_x_change"),
      "",
    )
    add(
      seed,
      "segB_after_fsm_exit_FL_swings",
      w["segB_swings"],
      b.get("segB_fl_swings"),
      c.get("segB_fl_swings"),
      "FL leaves ground for >=2 consecutive rows",
    )
    add(
      seed,
      "segB_after_fsm_exit_FL_foot_z_max_m",
      "n/a",
      b.get("segB_fl_foot_z_max"),
      c.get("segB_fl_foot_z_max"),
      "",
    )
    add(
      seed,
      "segB_hybrid_ACTIVE_rows",
      "0",
      b.get("segB_active_rows"),
      "none",
      "must be 0: policy only",
    )
    add(
      seed,
      "segC_rows_140_249_body_vx_mps_mean",
      w["segC_vx"],
      b.get("segC_body_vx_mean"),
      c.get("segC_body_vx_mean"),
      "late window after every FSM exit; wire-on from the existing D5.55 trace",
    )
    add(
      seed,
      "segC_rows_140_249_base_x_change_m",
      w["segC_dx"],
      b.get("segC_base_x_change"),
      c.get("segC_base_x_change"),
      "",
    )
    add(
      seed,
      "segC_rows_140_249_FL_swings_by_contact_gap",
      w["segC_swings"],
      b.get("segC_fl_gap_swings"),
      c.get("segC_fl_gap_swings"),
      "FL off the ground >=2 consecutive rows",
    )
    add(
      seed,
      "segC_rows_140_249_FL_z_above_0.03_episodes",
      "n/a",
      b.get("segC_fl_z_gt_0p03_episodes"),
      c.get("segC_fl_z_gt_0p03_episodes"),
      "foot-height swings (threshold is reporting only)",
    )
    add(
      seed,
      "segC_rows_140_249_FL_foot_z_max_m",
      "n/a",
      b.get("segC_fl_foot_z_max"),
      c.get("segC_fl_foot_z_max"),
      "",
    )

  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  with (out / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
  print(f"[D5.57] wrote {out / 'summary.csv'} ({len(summary)} rows)")
  for name, value, ok in checks:
    print(f"[D5.57] {'PASS' if ok else 'FAIL'} {name}: {value}")
  for s in summary:
    print(
      s["seed"],
      s["metric"],
      "|",
      s["d5_56_wire_on_plateau"],
      "|",
      s["wire_off_bounded_retraction"],
      "|",
      s["wire_off_control_same_steps"],
    )
  if not all(ok for _, _, ok in checks):
    print("[D5.57] VALIDATION FAILURE -- do not interpret as a matched comparison")
    sys.exit(2)


if __name__ == "__main__":
  main()
