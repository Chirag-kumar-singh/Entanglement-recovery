"""D7 stability metrics: one row per (run, seed) from the per-seed trajectories.

For every run directory under ``logs/d7_*`` that holds ``seed*/trajectory.csv`` and a
sibling ``<run>.txt`` sweep log, compute body height, peak roll/pitch and angular
velocity over steps 40 to the end (``*_full``) and 40 to 200 (``*_w200``), and write
them with the seed's verdict to one CSV. This replaces keeping every per-seed
trajectory in the repo.

    uv run python scripts/d7_stability.py --out logs/d7_summary/per_seed_stability.csv
"""

from __future__ import annotations

import csv
import math
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path

START = 40
SHORT_END = 200
VERDICT = re.compile(r"seed=(\d+) (PASS|FAIL\w*|FELL|SKIP)")


@dataclass
class Args:
  root: str = "logs"
  out: str = "logs/d7_summary/per_seed_stability.csv"


def _verdicts(path: Path) -> dict[int, str]:
  return {
    int(m[1]): m[2] for line in path.read_text().splitlines() if (m := VERDICT.match(line))
  }


def _window(rows: list[dict[str, float]], end: int) -> dict[str, float]:
  win = rows[START : min(len(rows), end)]
  z = [r["base_z"] for r in win]
  roll = [
    abs(math.degrees(math.atan2(r["grav_y"], -r["grav_z"]))) for r in win
  ]
  pitch = [
    abs(math.degrees(math.atan2(r["grav_x"], -r["grav_z"]))) for r in win
  ]
  wxy = [r["angvel_x"] ** 2 + r["angvel_y"] ** 2 for r in win]
  return {
    "z_mean": st.mean(z),
    "z_min": min(z),
    "z_range": max(z) - min(z),
    "roll_max": max(roll),
    "pitch_max": max(pitch),
    "angvel_rms": math.sqrt(st.mean(wxy)),
  }


def _load(path: Path) -> list[dict[str, float]]:
  keys = ("base_z", "grav_x", "grav_y", "grav_z", "angvel_x", "angvel_y")
  with open(path) as f:
    return [{k: float(r[k]) for k in keys} for r in csv.DictReader(f)]


def main() -> None:
  import tyro

  args = tyro.cli(Args)
  out = []
  for traj_dir in sorted(Path(args.root).glob("d7_*/*")):
    log = traj_dir.with_suffix(".txt")
    if not traj_dir.is_dir() or not log.exists():
      continue
    verdicts = _verdicts(log)
    for seed_dir in sorted(traj_dir.glob("seed*")):
      csv_path = seed_dir / "trajectory.csv"
      seed = int(seed_dir.name.removeprefix("seed"))
      if not csv_path.exists() or seed not in verdicts:
        continue
      rows = _load(csv_path)
      row = {"run": str(traj_dir.relative_to(args.root)), "seed": seed}
      row["verdict"] = verdicts[seed]
      for tag, end in (("full", len(rows)), ("w200", SHORT_END)):
        for k, v in _window(rows, end).items():
          row[f"{k}_{tag}"] = round(v, 4)
      out.append(row)
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
  print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
  main()
