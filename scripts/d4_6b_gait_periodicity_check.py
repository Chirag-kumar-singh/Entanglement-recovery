"""D4.6b -- gait-cycle periodicity check (follow-up to D4.6's recommendation).

Determines whether the FR/RL ground-contact loss observed at step 47
(immediately after recovery entry at step 44) is a NORMAL, periodic
artifact of the walking gait's own stance/swing cycle (i.e. would have
happened anyway, recovery or not), or a NOVEL disruption coincident with
the recovery maneuver.

Read-only. Reuses the exact same unmodified `d4_contact_constraint_diagnostic.py`
(no controller/FSM/wire change), re-run ONLY to widen its existing
`--window-start/--window-end` text-listing coverage back to step 0 (the
previously analyzed window, steps 38-62, could not show gait periodicity
because it didn't include enough pre-recovery gait cycles). This is the
same deterministic seed-0/seed-1 rollout already analyzed in D4.2-D4.6
(confirmed identical ENTER/EXIT RECOVERY log lines).

Outputs: per-leg boolean ground-contact sequence for steps 0-62, the
measured stance/swing period from the pre-recovery portion (steps 0-43),
and whether the step-47 FR/RL loss falls within or outside the normal
periodic pattern.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse_contact_listing(path: Path) -> dict[int, list[tuple[str, str]]]:
  text = path.read_text()
  blocks: dict[int, list[tuple[str, str]]] = {}
  current_step = None
  for line in text.splitlines():
    m = re.match(r"step=(\d+)\s", line)
    if m:
      current_step = int(m.group(1))
      blocks[current_step] = []
      continue
    m2 = re.match(r"\s+(\S+)\s+<->\s+(\S+)", line)
    if m2 and current_step is not None:
      blocks[current_step].append((m2.group(1), m2.group(2)))
  return blocks


def leg_ground_contact(pairs: list[tuple[str, str]], leg: str) -> bool:
  for a, b in pairs:
    if "terrain" in (a, b) and any(f"{leg}_foot_collision" in n for n in (a, b)):
      return True
  return False


def runs(seq: list[bool]) -> list[tuple[int, int, bool]]:
  """Collapse a boolean sequence into (start_idx, end_idx, value) runs."""
  out = []
  start = 0
  for i in range(1, len(seq) + 1):
    if i == len(seq) or seq[i] != seq[start]:
      out.append((start, i - 1, seq[start]))
      start = i
  return out


def analyze_seed(seed: int) -> None:
  listing_path = Path(
    f"logs/d4_6_support_descent_timing/seed{seed}_full/contact_listing.txt"
  )
  pairs_by_step = parse_contact_listing(listing_path)
  steps = sorted(pairs_by_step.keys())

  print(f"\n==== seed {seed} ====")
  for leg in ("FR", "RL"):
    seq = [leg_ground_contact(pairs_by_step[s], leg) for s in steps]
    leg_runs = runs(seq)

    # Pre-recovery portion: steps 0-43 (indices where step < 44).
    pre_idx_end = next((i for i, s in enumerate(steps) if s >= 44), len(steps))
    pre_runs = [r for r in leg_runs if r[1] < pre_idx_end]

    stance_lens = [(steps[r[1]] - steps[r[0]] + 1) for r in pre_runs if r[2] is True]
    swing_lens = [(steps[r[1]] - steps[r[0]] + 1) for r in pre_runs if r[2] is False]

    print(f"\n  -- {leg} --")
    print(f"  Pre-recovery (steps 0-43) stance/swing runs (step range : state):")
    for r in pre_runs:
      state = "STANCE" if r[2] else "SWING"
      print(f"    [{steps[r[0]]:3d}-{steps[r[1]]:3d}] {state}")
    if stance_lens:
      print(
        f"  Mean pre-recovery stance duration: {sum(stance_lens) / len(stance_lens):.1f} steps"
      )
    if swing_lens:
      print(
        f"  Mean pre-recovery swing duration: {sum(swing_lens) / len(swing_lens):.1f} steps"
      )

    # Post-entry behavior: find the run containing/following step 47.
    post_runs = [r for r in leg_runs if steps[r[1]] >= 44]
    print(f"  Post-entry (>=step 44) runs:")
    for r in post_runs[:6]:
      state = "STANCE" if r[2] else "SWING"
      print(f"    [{steps[r[0]]:3d}-{steps[r[1]]:3d}] {state}")

    # Identify the swing run that includes step 47 (if any) and compare
    # its length to the typical pre-recovery swing length.
    swing_at_47 = next(
      (r for r in post_runs if not r[2] and steps[r[0]] <= 47 <= steps[r[1]]), None
    )
    if swing_at_47 is not None:
      length = steps[swing_at_47[1]] - steps[swing_at_47[0]] + 1
      is_open_ended = steps[swing_at_47[1]] == steps[-1]
      typical = f"{sum(swing_lens) / len(swing_lens):.1f}" if swing_lens else "N/A"
      print(
        f"  Swing run covering step 47: [{steps[swing_at_47[0]]}-{steps[swing_at_47[1]]}]"
        f" (len={length}{'+ (still ongoing at window end)' if is_open_ended else ''}, "
        f"typical pre-recovery swing={typical} steps)"
      )
    else:
      print("  No swing run covers step 47 (leg was in stance at step 47).")


def main() -> None:
  for seed in (0, 1):
    analyze_seed(seed)


if __name__ == "__main__":
  main()
