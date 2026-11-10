"""D4.6c -- corrected gait-periodicity baseline (fixes D4.6b's step-44-crossing
interval contamination).

D4.6b reported a pre-recovery FR/RL "stance" run of steps 32-46, which
ILLEGITIMATELY crosses recovery entry (step 44) -- so step 46 (and the
whole tail of that run) is already inside the recovery window, not a
clean pre-entry baseline. This script re-parses the SAME already-existing
full contact listings from D4.6b (`logs/d4_6_support_descent_timing/
seed{0,1}_full/contact_listing.txt`, steps 0-62, unmodified controller)
with a strict rule: ANY stance/swing interval that includes step 44 or
later is EXCLUDED from the baseline-cycle statistics. Only fully-contained
step<44 cycles count as "pre-entry" baseline measurements.

No new simulation. Purely re-analysis of already-generated, already-
preserved data.
"""

from __future__ import annotations

import re
from pathlib import Path

RECOVERY_ENTRY_STEP = 44


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


def runs(steps: list[int], seq: list[bool]) -> list[tuple[int, int, bool]]:
  """Collapse into (start_step, end_step, value) runs, using actual step
  numbers (not list indices), so gaps/boundaries are unambiguous."""
  out = []
  start_i = 0
  for i in range(1, len(seq) + 1):
    if i == len(seq) or seq[i] != seq[start_i]:
      out.append((steps[start_i], steps[i - 1], seq[start_i]))
      start_i = i
  return out


def hip_wire_contact(pairs: list[tuple[str, str]]) -> bool:
  for a, b in pairs:
    if any("FL_hip_collision" in n for n in (a, b)) and any(
      "wire_seg" in n for n in (a, b)
    ):
      return True
  return False


def analyze_seed(seed: int, lines: list[str]) -> None:
  listing_path = Path(
    f"logs/d4_6_support_descent_timing/seed{seed}_full/contact_listing.txt"
  )
  pairs_by_step = parse_contact_listing(listing_path)
  steps = sorted(pairs_by_step.keys())

  lines.append(f"\n==== seed {seed} ====")

  for leg in ("FR", "RL"):
    seq = [leg_ground_contact(pairs_by_step[s], leg) for s in steps]
    leg_runs = runs(steps, seq)

    # STRICT pre-entry baseline: only runs whose END step is < 44 (so the
    # run never touches or crosses recovery entry at all).
    pre_entry_runs = [r for r in leg_runs if r[1] < RECOVERY_ENTRY_STEP]
    # Runs that start before 44 but end at/after 44 are CONTAMINATED and
    # explicitly excluded (this is exactly the step 32-46 bug being fixed).
    contaminated_runs = [r for r in leg_runs if r[0] < RECOVERY_ENTRY_STEP <= r[1]]

    lines.append(f"\n  -- {leg} --")
    lines.append("  ALL runs (step range : state) [* = excluded, crosses step 44]:")
    for r in leg_runs:
      state = "STANCE" if r[2] else "SWING"
      flag = " *CONTAMINATED/EXCLUDED*" if r in contaminated_runs else ""
      lines.append(f"    [{r[0]:3d}-{r[1]:3d}] {state}{flag}")

    # Complete pre-entry CYCLES: a cycle is a (stance, swing) or
    # (swing, stance) pair that is fully resolved (both runs end < 44).
    complete_pre_entry_runs = pre_entry_runs
    lines.append(
      f"\n  Strictly pre-entry (end < step {RECOVERY_ENTRY_STEP}) COMPLETE runs only:"
    )
    for r in complete_pre_entry_runs:
      state = "STANCE" if r[2] else "SWING"
      dur = r[1] - r[0] + 1
      lines.append(f"    [{r[0]:3d}-{r[1]:3d}] {state}  duration={dur} steps")

    stance_durs = [r[1] - r[0] + 1 for r in complete_pre_entry_runs if r[2] is True]
    swing_durs = [r[1] - r[0] + 1 for r in complete_pre_entry_runs if r[2] is False]
    n_complete_swing_cycles = len(swing_durs)
    lines.append(
      f"  Pre-entry COMPLETE stance cycles: {len(stance_durs)}  durations={stance_durs}"
    )
    lines.append(
      f"  Pre-entry COMPLETE swing cycles: {n_complete_swing_cycles}  durations={swing_durs}"
    )
    if swing_durs:
      lines.append(
        f"  Mean pre-entry swing duration: {sum(swing_durs) / len(swing_durs):.2f} steps "
        f"(n={n_complete_swing_cycles} cycle{'s' if n_complete_swing_cycles != 1 else ''})"
      )
    else:
      lines.append("  No complete pre-entry swing cycle available.")

    if n_complete_swing_cycles < 2:
      lines.append(
        f"  SAMPLE SIZE WARNING: only {n_complete_swing_cycles} complete pre-entry "
        f"swing cycle(s) available -- insufficient to reliably estimate normal "
        f"swing-duration variance or predict an expected touchdown step. Any "
        f"single-cycle comparison below is reported as a single data point, "
        f"not a statistically validated baseline."
      )

    # Post-entry: the run that contains/starts at or after step 44.
    post_entry_runs = [r for r in leg_runs if r[1] >= RECOVERY_ENTRY_STEP]
    lines.append(f"\n  Post-entry (includes step >= {RECOVERY_ENTRY_STEP}) runs:")
    for r in post_entry_runs:
      state = "STANCE" if r[2] else "SWING"
      lines.append(f"    [{r[0]:3d}-{r[1]:3d}] {state}")

    swing_at_47 = next(
      (r for r in post_entry_runs if not r[2] and r[0] <= 47 <= r[1]), None
    )
    if swing_at_47 is not None:
      length = swing_at_47[1] - swing_at_47[0] + 1
      lines.append(
        f"\n  Swing run covering step 47: [{swing_at_47[0]}-{swing_at_47[1]}] "
        f"(length={length} steps)"
      )
      if swing_durs:
        mean_swing = sum(swing_durs) / len(swing_durs)
        lines.append(
          f"  Compared to pre-entry mean swing duration {mean_swing:.2f} steps "
          f"(n={n_complete_swing_cycles}): "
          f"{'LONGER by ' + f'{length - mean_swing:.1f} steps' if length > mean_swing else 'not longer'}"
        )
      else:
        lines.append(
          "  No valid pre-entry swing baseline exists for comparison -- cannot "
          "classify this as abnormal or normal relative to a clean baseline."
        )
    else:
      lines.append("\n  No swing run covers step 47 (leg was in stance at step 47).")

    # Expected FR touchdown prediction feasibility.
    if leg == "FR":
      if n_complete_swing_cycles >= 2:
        mean_swing = sum(swing_durs) / len(swing_durs)
        lines.append(
          f"\n  Touchdown prediction feasibility: {n_complete_swing_cycles} complete "
          f"pre-entry swing cycles available (mean={mean_swing:.2f} steps); "
          f"a gait-phase-based touchdown prediction COULD be attempted but with low "
          f"confidence given the small sample."
        )
      else:
        lines.append(
          f"\n  Touchdown prediction feasibility: only {n_complete_swing_cycles} complete "
          f"pre-entry swing cycle(s) available. The gait baseline is TOO SHORT to "
          f"reliably predict an expected FR touchdown step. Not attempting a "
          f"prediction -- stating this as an explicit limitation."
        )

  # Full step-by-step dump for steps 38-62, cross-referenced with other events.
  lines.append(f"\n  -- Full FR/RL/hip-wire contact states, steps 38-62 --")
  lines.append(f"  {'step':>5} {'FR':>6} {'RL':>6} {'hip_wire':>9}")
  for s in steps:
    if 38 <= s <= 62:
      fr = leg_ground_contact(pairs_by_step[s], "FR")
      rl = leg_ground_contact(pairs_by_step[s], "RL")
      hw = hip_wire_contact(pairs_by_step[s])
      marker = ""
      if s == RECOVERY_ENTRY_STEP:
        marker = "  <- recovery entry"
      elif s == 47:
        marker = "  <- base descent onset (D4.6)"
      elif s == 51:
        marker = "  <- hip-wire contact (D4.2-D4.6)"
      elif s == 52:
        marker = "  <- FL foot collapse (D4.2-D4.6)"
      lines.append(f"  {s:5d} {str(fr):>6} {str(rl):>6} {str(hw):>9}{marker}")


def main() -> None:
  lines: list[str] = []
  lines.append("D4.6c -- Corrected gait-periodicity baseline (strict pre-entry only)")
  lines.append("=" * 72)
  lines.append(
    "Fixes D4.6b: the previously reported 'stance [32-46]' interval ILLEGITIMATELY "
    "crossed recovery entry (step 44). This report re-derives pre-entry baseline "
    "cycles using ONLY runs whose end step is strictly < 44."
  )

  for seed in (0, 1):
    analyze_seed(seed, lines)

  out_path = Path(
    "logs/d4_6_support_descent_timing/gait_periodicity_corrected_report.txt"
  )
  out_path.write_text("\n".join(lines) + "\n")
  print("\n".join(lines))
  print(f"\n[D4.6c] report written: {out_path}")


if __name__ == "__main__":
  main()
