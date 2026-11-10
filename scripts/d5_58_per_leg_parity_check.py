"""D5.58b -- Parity check: D5.58 (--leg FL) vs. D5.3.

Verifies that the generalized per-leg script
(`scripts/d5_58_per_leg_recovery_window.py`) reproduces
`scripts/d5_3_extended_recovery_window.py` EXACTLY when targeting the
front-left leg -- i.e. that generalizing to FR/RL/RR introduced no
silent change to the validated FL setup.

Both scripts are run here with the same seed, device and step budget;
the shared numeric/boolean trajectory columns are then compared
element-wise (D5.58 adds a few columns D5.3 does not have -- `leg`,
`base_x`, the per-other-leg wire-contact flags -- and renames D5.3's
FL-specific columns to leg-agnostic ones; the mapping below is explicit).

Exit code 0 = parity holds, 1 = mismatch (details printed).
"""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from d5_3_extended_recovery_window import D53Config  # noqa: E402
from d5_3_extended_recovery_window import run as run_d53  # noqa: E402
from d5_58_per_leg_recovery_window import D558Config  # noqa: E402
from d5_58_per_leg_recovery_window import run as run_d558  # noqa: E402

# D5.3 column name -> D5.58 column name. Identical names omitted.
RENAMED = {
  "fl_ground_contact": "target_ground_contact",
  "fl_hip_wire_contact_raw": "target_hip_wire_contact_raw",
  "fl_calf_wire_contact_raw": "target_calf_wire_contact_raw",
}
# D5.3's FR-only wire diagnostic; D5.58 logs one per non-target leg.
RENAMED_SENSOR = {"fr_wire_contact": "FR_wire_contact"}


@dataclass
class ParityConfig:
  seed: int = 0
  device: str | None = None
  max_steps: int = 120
  tol: float = 1e-9


def _read(path: Path) -> list[dict]:
  with open(path, newline="") as f:
    return list(csv.DictReader(f))


def _coerce(value: str):
  if value in ("True", "False"):
    return value == "True"
  try:
    return float(value)
  except ValueError:
    return value


def run(cfg: ParityConfig) -> bool:
  device = cfg.device or "cpu"
  tmp = Path(tempfile.mkdtemp(prefix="d5_58_parity_"))

  d53_csv = run_d53(
    D53Config(
      seed=cfg.seed,
      device=device,
      max_steps=cfg.max_steps,
      out_dir=str(tmp / "d53"),
    )
  )
  d558_csv = run_d558(
    D558Config(
      leg="FL",
      seed=cfg.seed,
      device=device,
      max_steps=cfg.max_steps,
      out_dir=str(tmp / "d558"),
    )
  )

  a_rows, b_rows = _read(d53_csv), _read(d558_csv)
  if len(a_rows) != len(b_rows):
    print(f"[PARITY FAIL] row count {len(a_rows)} vs {len(b_rows)}")
    return False

  mapping = {}
  for col in a_rows[0]:
    target = RENAMED.get(col, RENAMED_SENSOR.get(col, col))
    if target in b_rows[0]:
      mapping[col] = target
    else:
      print(f"[PARITY WARN] D5.3 column {col!r} has no D5.58 counterpart")

  mismatches = []
  for i, (a, b) in enumerate(zip(a_rows, b_rows, strict=True)):
    for src, dst in mapping.items():
      av, bv = _coerce(a[src]), _coerce(b[dst])
      if isinstance(av, float) and isinstance(bv, float):
        if abs(av - bv) > cfg.tol:
          mismatches.append((i, src, av, bv))
      elif av != bv:
        mismatches.append((i, src, av, bv))

  extra = set(b_rows[0]) - set(mapping.values())
  print(f"compared {len(mapping)} columns over {len(a_rows)} steps")
  print(f"D5.58-only columns (not compared): {sorted(extra)}")
  if mismatches:
    print(f"[PARITY FAIL] {len(mismatches)} mismatching cells, first 10:")
    for row_i, col, av, bv in mismatches[:10]:
      print(f"  step={row_i} col={col}: d5.3={av!r} d5.58={bv!r}")
    return False
  print("[PARITY OK] D5.58 --leg FL is identical to D5.3 on all shared columns")
  return True


def main() -> None:
  import tyro

  sys.exit(0 if run(tyro.cli(ParityConfig)) else 1)


if __name__ == "__main__":
  main()
