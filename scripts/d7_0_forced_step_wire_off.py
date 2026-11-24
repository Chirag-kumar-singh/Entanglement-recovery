"""D7.0 -- Does the frozen walking policy hold the body up when a front leg steps?

Question (see D6 summary): front-pair failures show the body sagging to
0.14-0.18 m. Is that caused by the wire, or does the walking policy simply not
carry weight on three legs while a front leg runs the scripted step?

Per seed, three runs of the D6.1 front-pair setup (same controller, same start x):
  wire_on     the real wire; records the step at which each front leg enters
              recovery (first wire contact).
  wire_off    the robot is moved 1.5 m sideways so it never touches the wire, and
              each leg's FSM entry is forced at the step recorded in wire_on, so
              the scripted steps run at the same gait phase with no wire at all.
  control     wire off, no forced entry (walking policy only).

Metrics over the window [first entry, first entry + 80 steps]: minimum base
height, peak roll, peak scripted-step foot height, fell (z < 0.15 m or roll > 45).

    uv run python scripts/d7_0_forced_step_wire_off.py --seeds 0-39 --jobs 2
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent))

from d6_1_front_pair_wire import (  # noqa: E402
  D61Config,
  _parse_seeds,
  build_env,
)
from per_leg_wire import LEG_NAMES  # noqa: E402

from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY  # noqa: E402
from mjlab.tasks.recovery.mdp.multi_leg import MultiLegRecoveryAction  # noqa: E402

WINDOW = 80
FELL_Z = 0.15
FELL_ROLL_DEG = 45.0


@dataclass
class D70Config:
  seeds: str = "0-39"
  jobs: int = 2
  max_steps: int = 220
  start_x_range: tuple[float, float] = (-0.485, -0.35)
  off_wire_y: float = -1.5
  out: str = "logs/d7_0_forced_step/summary.csv"
  spans_from: str = ""
  """Summary CSV from a baseline run. If set, only the wire-off run is done, with
  each seed's forced-contact spans read from it (no wire-on or control run)."""
  lift_thigh: float = 0.0
  lift_calf: float = 0.0
  """Body support offsets (rad) on the standing legs, see D61Config."""
  label: str = "wire_off"


def _install_forced_entry(leg_ctrl, forced: tuple[int, int], sensors) -> dict:
  """Force one leg's FSM contact input true for policy steps [first, last), the
  span over which the real wire kept that leg in recovery."""
  first, last = forced
  state = {"calls": 0}
  original = leg_ctrl._update_fsm
  name = leg_ctrl.cfg.wire_contact_sensor_name

  def forced() -> None:
    step = state["calls"]
    state["calls"] += 1
    if not first <= step < last:
      original()
      return
    real = sensors[name]
    found = torch.ones_like(real.data.found)
    sensors[name] = SimpleNamespace(data=SimpleNamespace(found=found))  # ty: ignore
    try:
      original()
    finally:
      sensors[name] = real

  leg_ctrl._update_fsm = forced
  return state


def simulate(
  cfg: D61Config, seed: int, max_steps: int, forced: dict[str, tuple[int, int]] | None
):
  env = build_env(cfg, seed, "cpu")
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, MultiLegRecoveryAction)
  ctrl = dict(zip(("FL", "FR"), term.legs, strict=True))
  entity = term._entity
  site_names = list(entity.site_names)
  env.reset(seed=seed)
  if forced:
    for leg, step in forced.items():
      _install_forced_entry(ctrl[leg], step, env.scene.sensors)
  zero = torch.zeros(env.action_manager.action.shape, device="cpu")
  rows, entered, left = [], {}, {}
  for step in range(max_steps):
    env.step(zero)
    for leg, c in ctrl.items():
      if leg not in entered and bool((c._phase == PHASE_RECOVERY)[0]):
        entered[leg] = step
      elif (
        leg in entered and leg not in left and not bool((c._phase == PHASE_RECOVERY)[0])
      ):
        left[leg] = step
    g = entity.data.projected_gravity_b[0]
    rows.append(
      {
        "step": step,
        "z": float(entity.data.root_link_pos_w[0, 2]),
        "roll": math.degrees(math.atan2(float(g[1]), -float(g[2]))),
        "FL_foot_z": float(entity.data.site_pos_w[0, site_names.index("FL"), 2]),
        "FR_foot_z": float(entity.data.site_pos_w[0, site_names.index("FR"), 2]),
        "stepping": {leg: bool(c._step_started[0]) for leg, c in ctrl.items()},
      }
    )
  env.close()
  return rows, entered, left


def metrics(rows: list[dict], start: int) -> dict:
  win = [r for r in rows if start <= r["step"] < start + WINDOW]
  fell = any(r["z"] < FELL_Z or abs(r["roll"]) > FELL_ROLL_DEG for r in win)
  peak = {}
  for leg in ("FL", "FR"):
    z = [r[f"{leg}_foot_z"] for r in win if r["stepping"][leg]]
    peak[leg] = max(z) if z else float("nan")
  return {
    "min_z": min(r["z"] for r in win),
    "max_roll": max(abs(r["roll"]) for r in win),
    "FL_peak": peak["FL"],
    "FR_peak": peak["FR"],
    "fell": fell,
  }


def _load_spans(path: str) -> dict[int, dict[str, tuple[int, int]]]:
  out = {}
  for r in csv.DictReader(open(path)):
    if r["cond"] == "wire_off":
      out[int(r["seed"])] = ast.literal_eval(r["entered"])
  return out


def run_seed(args: tuple[D70Config, int]) -> list[dict]:
  cfg70, seed = args
  base = D61Config(
    pair="front",
    start_x_range=cfg70.start_x_range,
    support_lift_thigh=cfg70.lift_thigh,
    support_lift_calf=cfg70.lift_calf,
  )
  if cfg70.spans_from:
    off = dataclasses.replace(base, spawn_y_offset=cfg70.off_wire_y)
    spans = _load_spans(cfg70.spans_from)[seed]
    rows_f, _, _ = simulate(off, seed, cfg70.max_steps, spans)
    r = {
      "seed": seed,
      "cond": cfg70.label,
      **metrics(rows_f, min(a for a, _ in spans.values())),
    }
    r["entered"] = str(spans)
    return [r]
  rows_on, entered, left = simulate(base, seed, cfg70.max_steps, None)
  out = []
  if not entered:
    return [{"seed": seed, "cond": "wire_on", "note": "no wire contact"}]
  start = min(entered.values())
  out.append({"seed": seed, "cond": "wire_on", **metrics(rows_on, start)})
  off = dataclasses.replace(base, spawn_y_offset=cfg70.off_wire_y)
  spans = {g: (entered[g], left.get(g, cfg70.max_steps)) for g in entered}
  rows_f, _, _ = simulate(off, seed, cfg70.max_steps, spans)
  out.append({"seed": seed, "cond": "wire_off", **metrics(rows_f, start)})
  rows_c, _, _ = simulate(off, seed, cfg70.max_steps, None)
  out.append({"seed": seed, "cond": "control", **metrics(rows_c, start)})
  for r in out:
    r["entered"] = str(spans)
  return out


def main() -> None:
  import multiprocessing as mp

  import tyro

  cfg = tyro.cli(D70Config)
  ctx = mp.get_context("spawn")
  results: list[dict] = []
  with ctx.Pool(cfg.jobs, maxtasksperchild=1) as pool:
    for rows in pool.imap(run_seed, [(cfg, s) for s in _parse_seeds(cfg.seeds)]):
      for r in rows:
        print(r, flush=True)
      results.extend(rows)
  Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
  keys = sorted({k for r in results for k in r})
  with open(cfg.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(results)
  print("wrote", cfg.out, "legs", LEG_NAMES)


if __name__ == "__main__":
  main()
