# D5.57 — Wire-Off Replay of the Bounded Release

**Question:** does the post-release stall seen in D5.55/D5.56 need the wire, or does it also
happen without it?

**Answer:** the wire is needed for the *persistent* stall, but not for the post-release
disturbance. Without the wire the same release still produces a braking transient and a long FL
stance, and then the policy walks off on its own. With the wire it never does. So the evidence
supports a mixed explanation (an abnormal-state transient the policy handles by itself, which
the wire turns into a stall), not "wire restraint alone" and not "policy alone". Two caveats
matter: the two seeds are effectively one replicate, and the wire-on/wire-off runs differ in
more than the wire (section 5).

## 1. What was run

[scripts/d5_57_wire_off_replay.py](../../scripts/d5_57_wire_off_replay.py), seeds 0 and 11, 250
policy steps, same checkpoint, config and reset as D5.55, using the D5.52 `MjSpec.add_exclude`
wire exclusion (all 91 robot × wire-segment body pairs). Existing code reused unchanged:
D5.52 `build_condition_env` and `SubstepWireProbe`, D5.55 `BoundedRetraction`, D5.56 `Capture`.
No production file, checkpoint, hybrid command, budget, actuator parameter or observation
processing was touched.

- **`wire_off_control`:** normal frozen policy, no hybrid entry.
- **`wire_off_bounded_retraction`:** the D5.55 wrapper, with the existing FSM entry forced once
  by making its contact input true for one step: the first step at which FL has been off the
  ground for two consecutive steps after a touchdown at/after step 44. This is the one new
  piece of logic. No touchdown guard is needed (entry already occurs after lift-off).

## 2. Validation — all pass, both seeds

| Check | Result |
|---|---|
| Robot–wire contacts, every substep, whole run | 0 in both conditions on both seeds |
| Control on the walking path | all 250 rows NORMAL_WALK, FL position gains active, hybrid commands 0 |
| Entry phase matches D5.55 | wire-off: FL lift-off row 57, entry 59 (+2). D5.55 wire-on: lift-off row 60, hybrid active 62 (+2). Wrapper log: `swing_est=11 airborne_already=2 budget=9`, identical to D5.55 |
| Hybrid commands at entry | thigh +2.0 rad/s, calf −5.0 N·m, FL thigh/calf position gains 0 |
| Release | entry + 8 (row 67), as D5.55 (62 → 70); FL gains restored to 20/40, aux commands and damping 0 |
| FSM exit does not preempt the release | exit at row 106, after release |

The gait is the same as the wire-on runs but shifted 3 rows earlier (lift-off 57 vs 60), presumably
because the wire perturbs the wire-on stance (not tested). One validation check initially failed: it compared entry
against the hook's own step counter (the FSM-call step, which sees the previous step's state), an
off-by-one in my bookkeeping. I fixed the check to read the recorded trace rows and to measure
D5.55's phase from its own trace, rather than loosening the tolerance.

## 3. Results ([summary.csv](summary.csv))

The literal comparison asked for was the 25 steps after FL touchdown (T0+12…T0+36, matching the
D5.56 plateau, T0 = first FL contact after release: 72 wire-on, 68 wire-off). It is in the CSV but
is **confounded**: in the wire-off run the wrapper resumes the hybrid after FL lifts off again, so
6 of those 25 rows are hybrid-driven (the "FL foot z max 0.157 m" there is the hybrid's second
retract, not a policy swing). Wire-on never resumed. So I added three segments that avoid this.
These were defined after seeing the first run, so treat them as post-hoc.

**Segment A — policy only, from first FL touchdown until the hybrid resumes** (wire-on: rows
72–108, never resumed; wire-off: rows 68–74):

| | wire-on (D5.55/56) | wire-off bounded | wire-off control (same rows) |
|---|---|---|---|
| Rows | 37 | 7 | 7 |
| FL stance before lift-off | none in 37 rows | **5** | n/a |
| FL foot z max (m) | 0.022 | **0.037** | 0.021 |
| Body forward speed (m/s) | −0.02 / −0.00 | **0.32** | 0.25 |

(wire-on shown as seed 0 / seed 11 where they differ).

**Segment B — 25 rows after the FSM exit** (wire-on exit 109, wire-off 106):

| | wire-on | wire-off bounded | control (same rows) |
|---|---|---|---|
| Body forward speed (m/s) | −0.01 / −0.03 | 0.18 | 0.27 |
| Base x change (m) | −0.00 / −0.02 | 0.09 | 0.13 |

**Segment C — rows 140–249, after every exit:**

| | wire-on | wire-off bounded | control |
|---|---|---|---|
| Body forward speed (m/s) | 0.013 / 0.027 | **0.267** | 0.268 |
| Base x change (m) | 0.02 / 0.05 | **0.584** | 0.587 |
| FL swings (≥2 rows off ground) | 0 | 4 | 5 |
| FL foot-z swings above 0.03 m | n/a | 5 | 5 |
| FL foot z max (m) | n/a | 0.036 | 0.035 |

**Between the second release and the recovery in the wire-off run:** the second release is at
row 86; FL is then on the ground for about 25 rows (86–110) apart from brief gaps, and body
speed dips to **−0.32 m/s at row 91** (IMU speed about +0.66 m/s just before the release). Body
speed is back to ≥ 0.15 m/s sustained by row 113–114 (the FSM exits at 106), and the gait is
control-like by rows 140–249. Status transitions (both seeds): ACTIVE 59, release 67, resume 75,
release 86, FSM exit 106.

## 4. Answers

1. **Entry and release phase reproduced:** yes, on both seeds (section 2).
2. **Genuine FL swing and forward trotting resumed:** yes, but not immediately. FL lifts again
   5 rows after the first touchdown (policy-driven, apex 0.037 m, about the normal 0.035 m), and
   after the second release the gait takes about 25 rows to recover. By rows 140–249 it is
   indistinguishable from control on forward speed, distance, and FL swing count and height.
3. **Difference from control and from the wire-on plateau:** vs control, the wire-off run has a
   transient after each release (braking, a negative-speed dip, one long FL stance) that lasts
   until about row 113, then matches. Vs the wire-on run, it differs in kind: wire-on shows no FL lift-off for 37+ rows, near-zero speed, and is still
   stalled at rows 140–249 (0.01–0.03 m/s).
4. **What the results support:**
   - *Wire contact is necessary for the persistent stall:* supported. The same release (same
     phase, budget, commands, gains) recovers by itself without the wire, in both seeds.
   - *An abnormal-state / policy explanation is also supported* for the transient: braking and a
     long FL stance occur without any wire. That matches the D5.56 finding that observations
     sit far from the normal gait right after release, and it says the policy can recover from
     that state unaided.
   - *Not shown:* that wire force alone sustains the stall. The wire-on state at release already
     differs from the wire-off state (FL calf pressed against the wire for the whole stance), so
     "the wire holds the leg after release" and "the wire-on release state is already
     pathological" are not separated.
5. **Viewer commands:** section 6.

## 5. Caveats

- **Effectively one replicate.** The wire-off bounded runs for seeds 0 and 11 differ by at most
  0.02 m/s in speed and 0.002 m in foot height, so the seed contributes no independent check.
- **The comparison is not a pure wire toggle.** Wire-off forced entry at row 59 vs wire-on
  guarded entry at 62; the wire perturbs the earlier stance; and in the wire-off run the wrapper
  completes a second hybrid cycle (rows 75–85) that never occurred wire-on. Wire-off also hit the
  48-step cap (no crossing, exit 106) just as wire-on did (109), so neither run is a genuine
  recovery in the FSM's terms.
- **Segments are post-hoc** (defined after run 1). The pre-planned plateau rows are in the CSV
  and agree on direction (wire-off speed 0.01, wire-on −0.03) but cannot be interpreted as
  policy-only because of the resumed hybrid.
- Swing counts by contact gap include light-load flicker; the foot-z counts (0.03 m threshold,
  reporting only) are the cleaner measure. Wire-on late-window values come from the existing
  D5.55 trace (body-frame speed, contact gaps only; no foot height).

## 6. Viewer and parser verification

All four commands were parser-checked (`tyro.cli(D557Config, args=…)` parses each), then run in
viewer mode to completion: 250 policy steps, covering entry/release (59–67), the second cycle,
the FSM exit (106) and the recovery to rows 140–249. Each exited 0 and **each viewer-mode trace
is byte-identical (`cmp`) to the corresponding headless trace** (251 lines each, i.e. the same
250 rows, which include the robot–wire substep contact count: 0 throughout). I did not visually
inspect the viewer window; byte equality shows it stepped the same physics. `--trace-dir`
writes the comparison trace to scratch space and is required with `--debug`.

```sh
S=/tmp/claude-1000/-home-arc01-mjlab/65c5cc81-0e49-4564-9a5e-326329dcd802/scratchpad/d557/viewer
uv run python scripts/d5_57_wire_off_replay.py --condition wire_off_control            --seed 0  --debug --max-steps 250 --trace-dir $S
uv run python scripts/d5_57_wire_off_replay.py --condition wire_off_bounded_retraction --seed 0  --debug --max-steps 250 --trace-dir $S
uv run python scripts/d5_57_wire_off_replay.py --condition wire_off_control            --seed 11 --debug --max-steps 250 --trace-dir $S
uv run python scripts/d5_57_wire_off_replay.py --condition wire_off_bounded_retraction --seed 11 --debug --max-steps 250 --trace-dir $S
```

Headless (all conditions and seeds, validation, `summary.csv`; about 15 min on CPU):
`uv run python scripts/d5_57_wire_off_replay.py --trace-dir <scratch>` (omit `--trace-dir` to
write nothing but the summary).

## 7. Recommended next experiment (not implemented)

**State transplant at release.** Run the wire-on bounded schedule to the release row (step 70,
identical to D5.55), snapshot the full simulator state, and continue from that exact state in the
wire-off scene with the same policy. The D5.49 restore reproduced `qpos`/`qvel` exactly, and the
wire-off scene should have the same state layout (the excludes change no joints; to be verified). D5.48 showed hidden state matters, so the policy's last
action (`_last_walk_raw_action`), the FSM phase/counters and the wrapper state must be restored too.

- Stall persists after the transplant: the release state itself is the cause, and the wire's role
  was to produce that state, not to hold the robot afterwards.
- It recovers like the wire-off run: wire contact *after* release sustains the stall.

This separates the two readings that section 4 could not. If the restore cannot reproduce the
hidden state, report that rather than comparing. A behavior-changing task must again provide
parser-checked, run, byte-compared viewer commands for the baseline and the variant on seeds 0
and 11.

## 8. Retained files

`logs/d5_57/report.md`, `logs/d5_57/summary.csv` (64 rows), `scripts/d5_57_wire_off_replay.py`.
Per-run traces were written only to the session scratch directory for the viewer comparison,
outside the repo.
