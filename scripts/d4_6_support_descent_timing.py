"""D4.6 cross-signal event-timing analysis (read-only, no new dynamics).

Merges THREE already-existing data sources (no controller/FSM/wire change,
no new rollout logic -- the two new contact-listing files were produced
by re-running the EXISTING, unmodified `d4_contact_constraint_diagnostic.py`
with a wider `--window-start/--window-end`, the only gap identified: that
script's own CSV does not log per-leg (RL/RR/FR) ground contact, only raw
per-step text):

  1. `logs/d3_peak_baseline_seed{0,1}/trajectory.csv` -- full 120-step
     per-step log with `base_z`, `base_quat_w`, `foot_z`/`foot_vz`,
     `fl_ground_contact`, `fr_wire_contact`, phase/recovery_elapsed,
     thigh/calf/hip pos+vel. ALREADY COVERS steps 38-62 for these signals
     -- no gap, no re-run needed for these.
  2. `logs/d4_contact_constraint_{check,seed1}/contact_constraint_trajectory.csv`
     -- full 120-step log with `thigh_qfrc_constraint`/`calf_qfrc_constraint`
     and correctly-indexed (D4.1-verified) actuator forces. ALREADY
     COVERS steps 38-62 -- no gap.
  3. `logs/d4_6_support_descent_timing/seed{0,1}/contact_listing.txt` --
     NEWLY GENERATED (this task) raw per-step contact-pair text, by
     re-running the UNMODIFIED `d4_contact_constraint_diagnostic.py` with
     `--window-start 38 --window-end 62` (previously only 44-62 was
     covered for seed 0/1 contact listings in D4.2/D4.3). This is the only
     genuine gap identified before writing this script: RL/RR/FR
     ground-contact and hip-wire contact onset were only visible in the
     per-step TEXT listing (not the CSV), and that listing previously
     started at step 44, after the pre-recovery baseline window needed
     (38-43).

This script performs NO further simulation. It only parses the three
already-produced files above and applies explicit, stated numerical
criteria to identify event onsets.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

STEP_START = 38
STEP_END = 62
RECOVERY_ENTRY_STEP = 44  # confirmed identical in all seed-0/seed-1 runs so far.

# --- Operational criteria (stated explicitly, applied uniformly) ---
# Base descent: "first sustained descent" = first step where base_z has
# dropped by more than DESCENT_THRESH_M from its own pre-recovery
# (steps 38-43) mean, AND stays below (mean - DESCENT_THRESH_M) for at
# least DESCENT_SUSTAIN_STEPS consecutive steps (not a single noisy dip).
DESCENT_THRESH_M = 0.01
DESCENT_SUSTAIN_STEPS = 3

# Base tilt: "first meaningful tilt change" = first step where
# |base_quat_w - pre-recovery mean base_quat_w| > TILT_THRESH, sustained
# for TILT_SUSTAIN_STEPS. (quat_w ~= cos(angle/2); for small tilts,
# deviation in quat_w approximates half the squared tilt angle, but we
# use it only as a monotonic proxy for "orientation changed", not a
# calibrated angle.)
# NOTE: observed base_quat_w deviation over steps 44-52 peaks at only
# ~0.008-0.010 in both seeds (base_quat_w falls from ~0.9999 to ~0.991-0.992
# at its lowest) -- i.e. a small, gradual roll/pitch change, not a sharp
# tip-over. TILT_THRESH is set at 0.005 (half the observed peak deviation)
# so the detector is sensitive enough to register this real, if modest,
# orientation change rather than reporting "no tilt" by a hair under a
# too-strict threshold. This threshold choice is reported explicitly so
# it can be scrutinized/adjusted.
TILT_THRESH = 0.005
TILT_SUSTAIN_STEPS = 3

# Ground-contact loss/gain: use the actual logged/observed boolean contact
# state directly (`fl_ground_contact` column, or `terrain<->X_foot_collision`
# presence in the raw per-step text for the other 3 legs) -- no threshold
# needed, it is already a discrete simulator-reported event. "Sustained"
# loss requires the flag to stay in the new state for >=2 consecutive
# logged steps (to rule out a single-step contact-detection flicker).
CONTACT_SUSTAIN_STEPS = 2


def load_d3_baseline(seed: int) -> dict[int, dict]:
  path = Path(f"logs/d3_peak_baseline_seed{seed}/trajectory.csv")
  with open(path) as f:
    rows = list(csv.DictReader(f))
  return {int(r["step"]): r for r in rows}


def load_d4_contact_csv(seed: int) -> dict[int, dict]:
  d = "d4_contact_constraint_check" if seed == 0 else "d4_contact_constraint_seed1"
  path = Path(f"logs/{d}/contact_constraint_trajectory.csv")
  with open(path) as f:
    rows = list(csv.DictReader(f))
  return {int(r["step"]): r for r in rows}


def parse_contact_listing(path: Path) -> dict[int, list[tuple[str, str, float]]]:
  """Parse `step=N ...` blocks with indented `A <-> B dist=D` lines."""
  text = path.read_text()
  blocks: dict[int, list[tuple[str, str, float]]] = {}
  current_step = None
  for line in text.splitlines():
    m = re.match(r"step=(\d+)\s", line)
    if m:
      current_step = int(m.group(1))
      blocks[current_step] = []
      continue
    m2 = re.match(r"\s+(\S+)\s+<->\s+(\S+)\s+dist=([\-\d.]+)", line)
    if m2 and current_step is not None:
      blocks[current_step].append((m2.group(1), m2.group(2), float(m2.group(3))))
  return blocks


def leg_ground_contact(pairs: list[tuple[str, str, float]], leg: str) -> bool:
  for a, b, _ in pairs:
    names = (a, b)
    if any("terrain" == n for n in names) and any(
      f"{leg}_foot_collision" in n for n in names
    ):
      return True
  return False


def hip_wire_contact(pairs: list[tuple[str, str, float]]) -> bool:
  for a, b, _ in pairs:
    names = (a, b)
    if any("FL_hip_collision" in n for n in names) and any(
      "wire_seg" in n for n in names
    ):
      return True
  return False


def first_sustained_descent(base_z: dict[int, float]) -> int | None:
  pre = [base_z[s] for s in range(38, 44) if s in base_z]
  pre_mean = sum(pre) / len(pre)
  thresh = pre_mean - DESCENT_THRESH_M
  steps = sorted(base_z.keys())
  for i, s in enumerate(steps):
    if base_z[s] < thresh:
      window = steps[i : i + DESCENT_SUSTAIN_STEPS]
      if len(window) == DESCENT_SUSTAIN_STEPS and all(
        base_z[w] < thresh for w in window
      ):
        return s
  return None


def first_sustained_tilt(quat_w: dict[int, float]) -> int | None:
  pre = [quat_w[s] for s in range(38, 44) if s in quat_w]
  pre_mean = sum(pre) / len(pre)
  steps = sorted(quat_w.keys())
  for i, s in enumerate(steps):
    if abs(quat_w[s] - pre_mean) > TILT_THRESH:
      window = steps[i : i + TILT_SUSTAIN_STEPS]
      if len(window) == TILT_SUSTAIN_STEPS and all(
        abs(quat_w[w] - pre_mean) > TILT_THRESH for w in window
      ):
        return s
  return None


def first_sustained_bool_change(
  flags: dict[int, bool], from_val: bool, sustain: int
) -> int | None:
  steps = sorted(flags.keys())
  for i, s in enumerate(steps):
    if flags[s] != from_val:
      window = steps[i : i + sustain]
      if len(window) == sustain and all(flags[w] != from_val for w in window):
        return s
  return None


def analyze_seed(seed: int) -> dict:
  d3 = load_d3_baseline(seed)
  d4 = load_d4_contact_csv(seed)
  listing_dir = Path(f"logs/d4_6_support_descent_timing/seed{seed}")
  pairs_by_step = parse_contact_listing(listing_dir / "contact_listing.txt")

  base_z = {s: float(d3[s]["base_z"]) for s in d3 if STEP_START <= s <= STEP_END}
  quat_w = {s: float(d3[s]["base_quat_w"]) for s in d3 if STEP_START <= s <= STEP_END}
  fl_ground = {
    s: d3[s]["fl_ground_contact"] == "True" for s in d3 if STEP_START <= s <= STEP_END
  }

  # Other-leg ground contact + hip-wire contact: only available from the
  # newly-widened raw per-step text (steps 38-62 now covered).
  fr_ground = {
    s: leg_ground_contact(pairs_by_step.get(s, []), "FR") for s in pairs_by_step
  }
  rl_ground = {
    s: leg_ground_contact(pairs_by_step.get(s, []), "RL") for s in pairs_by_step
  }
  rr_ground = {
    s: leg_ground_contact(pairs_by_step.get(s, []), "RR") for s in pairs_by_step
  }
  hip_wire = {s: hip_wire_contact(pairs_by_step.get(s, [])) for s in pairs_by_step}

  descent_step = first_sustained_descent(base_z)
  tilt_step = first_sustained_tilt(quat_w)

  def loss_event(flags: dict[int, bool]) -> str | int | None:
    """First sustained loss of ground contact, guarded against legs that
    were never in contact during the pre-recovery baseline (38-43) --
    for those, "loss" is not a meaningful/attributable event, so we
    report that explicitly instead of a spurious step-38 false positive."""
    pre = [flags[s] for s in range(38, 44) if s in flags]
    if not any(pre):
      return "not in contact pre-recovery (N/A)"
    return first_sustained_bool_change(flags, True, CONTACT_SUSTAIN_STEPS)

  fl_loss_step = loss_event(fl_ground)
  fr_loss_step = loss_event(fr_ground)
  rl_loss_step = loss_event(rl_ground)
  rr_loss_step = loss_event(rr_ground)

  hip_wire_step = first_sustained_bool_change(
    hip_wire, False, 1
  )  # single occurrence ok (new discrete event)

  # FL foot collapse: first step where foot_z drops by > 0.01 m from the
  # immediately preceding step's running peak (reuses D4.2/D4.3's
  # already-established "collapse" definition: foot_z falls sharply
  # right after its post-entry peak).
  foot_z = {s: float(d3[s]["foot_z"]) for s in d3 if STEP_START <= s <= STEP_END}
  steps_sorted = sorted(foot_z.keys())
  running_peak = -1e9
  collapse_step = None
  for s in steps_sorted:
    if s < RECOVERY_ENTRY_STEP:
      continue
    if foot_z[s] > running_peak:
      running_peak = foot_z[s]
    elif running_peak - foot_z[s] > 0.01:
      collapse_step = s
      break

  return {
    "seed": seed,
    "base_z": base_z,
    "quat_w": quat_w,
    "fl_ground": fl_ground,
    "fr_ground": fr_ground,
    "rl_ground": rl_ground,
    "rr_ground": rr_ground,
    "hip_wire": hip_wire,
    "foot_z": foot_z,
    "events": {
      "recovery_entry": RECOVERY_ENTRY_STEP,
      "first_sustained_base_descent": descent_step,
      "first_sustained_base_tilt": tilt_step,
      "first_fl_ground_loss": fl_loss_step,
      "first_fr_ground_loss": fr_loss_step,
      "first_rl_ground_loss": rl_loss_step,
      "first_rr_ground_loss": rr_loss_step,
      "first_hip_wire_contact": hip_wire_step,
      "first_foot_collapse": collapse_step,
    },
  }


def main() -> None:
  out_dir = Path("logs/d4_6_support_descent_timing")
  out_dir.mkdir(parents=True, exist_ok=True)

  results = [analyze_seed(0), analyze_seed(1)]

  # Per-seed aligned timeline CSV (steps 38-62).
  for res in results:
    seed = res["seed"]
    rows = []
    for s in range(STEP_START, STEP_END + 1):
      rows.append(
        {
          "step": s,
          "base_z": round(res["base_z"].get(s, float("nan")), 5),
          "base_quat_w": round(res["quat_w"].get(s, float("nan")), 5),
          "fl_ground_contact": res["fl_ground"].get(s),
          "fr_ground_contact": res["fr_ground"].get(s),
          "rl_ground_contact": res["rl_ground"].get(s),
          "rr_ground_contact": res["rr_ground"].get(s),
          "fl_hip_wire_contact": res["hip_wire"].get(s),
          "foot_z": round(res["foot_z"].get(s, float("nan")), 5),
        }
      )
    csv_path = out_dir / f"seed{seed}_event_timeline.csv"
    with open(csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
    print(f"[D4.6] wrote {csv_path}")

  # Cross-seed event-step summary.
  summary_path = out_dir / "event_summary.csv"
  fieldnames = ["seed"] + list(results[0]["events"].keys())
  with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for res in results:
      row = {"seed": res["seed"]}
      row.update(res["events"])
      writer.writerow(row)
  print(f"[D4.6] wrote {summary_path}")

  for res in results:
    print(f"\n==== seed {res['seed']} events ====")
    for k, v in res["events"].items():
      print(f"  {k}: {v}")


if __name__ == "__main__":
  main()
