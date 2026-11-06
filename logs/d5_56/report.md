# D5.56 — Why the Walking Policy Stalls After Bounded Retraction

Read-only diagnostic. No controller, checkpoint, hybrid command, trajectory, or observation
was changed; no viewer was run (nothing was ambiguous enough to need one).

**Answer in one line:** the controller is *not* overriding the policy. The policy is acting on
observations that sit far outside the normal gait, while the robot is held back by continuous
wire contact on the FL calf. The two causes cannot be separated with the data that exists.

## What was measured

[scripts/d5_56_policy_stall_diag.py](../../scripts/d5_56_policy_stall_diag.py) re-runs two
things and only records: the D5.55 `bounded_retraction` run (wire on, seeds 0 and 11) and a
reference run (wire off, no hybrid, same seeds and checkpoint; FL touchdowns at steps 44, 66,
89 give three ordinary gait cycles). Per step it captures the 48-d observation the policy
actually received, the raw 12-d policy output, the processed targets, the applied actuator
controls, FL foot height, and per-substep wire force. Numbers are in
[summary.csv](summary.csv) (28 rows). Comparison window: **plateau = steps 84–108**, after the
post-release transient (steps ~72–83). Both are separated because the transient dominates
any 20-step window: FL raw thigh/calf outputs swing −2.2…+2.1 for ~8 steps after touchdown.

Reference sample is small: one gait, ~140 steps, four FL cycles (the policy's own start from
rest is included).

## Questions

**1. Are post-release observations within the normal range?** Term by term, mostly yes; as a
whole vector, no.
- Per-dimension, the plateau stays within 1.7 (seed 0) / 2.1 (seed 11) policy-normalizer units
  at most, versus 1.33 in normal walking: not extreme.
- As a vector it is far from anything in the reference: nearest-neighbour distance (normalized
  space, over the 48 dims) is **3.06 / 2.75**, versus **0.114 / 0.116** between normal gait
  cycles. The nearest reference steps are mostly the start from rest (steps 7–9) and step 34–35.
- Largest term differences: IMU forward velocity −0.038 / −0.026 m/s versus +0.269 in walking
  (command is 0.3 in both), FL thigh joint 0.22 / 0.20 rad off default versus 0.045 in normal
  stance, plus small gravity-vector and yaw-rate shifts.
- Zero velocity with a +0.3 command is itself a state the policy handles: it walked off from
  rest at the start of the reference run (first swing at step 7).

**2. Does the output request forward motion, or look like standing?** Neither can be shown.
Observed: the raw output has much less oscillation than a trot (mean peak-to-peak over the 12
joints **0.36 / 0.60** versus **0.98**), and the mean posture differs (FR/RL thigh outputs
+0.3 to +0.9 versus 0.0 to 0.23 in walking). Seed 0 settles with all four feet loaded; seed 11
is more active (RR steps in and out, forward speed −0.16 → +0.05 m/s after step 96). A low
body speed is not evidence of what the policy is asking for, and I do not infer it.

**3. Do applied targets match the raw output (does the controller interfere)?** Yes, exactly.
Over the plateau on both seeds: applied actuator control minus the policy's processed target is
**0.0** for all 12 joints; target equals default + scale × raw output to float precision; the
aux thigh-velocity control, aux calf-motor control and aux damping are all 0.0 (D5.55 trace,
steps 74–108); FL position gains are restored (thigh 20, calf 40). Nothing masks or replaces
the policy output in this window.

**4. Does the FL target command a new swing after touchdown?** The targets are numerically in
the reference's swing-onset range, not its stance range, yet no swing happens.
- FL raw calf output averages **−0.32 / −0.28** over the plateau. In the reference it is +0.35
  to +1.4 during stance and −0.09 to −0.17 at swing onset (k = 14–16 after touchdown). FL
  raw thigh averages 0.02 / 0.01, equal to the reference swing-onset 0.01.
- FL foot height never exceeds **0.0220 m** (resting height) on either seed; reference swings
  reach 0.035 m. The foot never leaves the ground in 25 steps.
- FL does not follow its target: the thigh sits **0.22 / 0.20 rad** from it (retracted
  direction; 0.10 rad in normal stance).
- Wire contact is continuous: FL calf2 touches the wire on **100%** of plateau steps, with a
  mean force on the robot of **−20.3 / −20.4 N in x** (opposing the +x command) and −13.4 /
  −13.3 N in z. The base drifts backward (**−0.015 / −0.008 m** over 25 steps versus +0.129
  forward in the wire-free reference).
- Rough consistency check, labelled as such: 20 N on the calf with a ≈0.21 m lever is ≈4 N·m,
  and the FL thigh position actuator is applying ≈20 × 0.22 ≈ 4 N·m toward its target without
  moving. The numbers fit a wire-held leg; this ignores geometry and is not proof.

**5. Same behavior on both seeds?** Qualitatively yes (same FL foot ceiling, same continuous
wire force, same observation distance, same applied-equals-target). Seed 11 shows more
activity (action amplitude 0.60 versus 0.36, RR stepping). Quantities are in `summary.csv`.

## Corrections to the D5.55 report (left unedited)

- "FL never lifts off again" holds by foot height (max 0.0220 m), but the ground-contact flag
  flickers in single-step gaps at 10–25 N load. The debounce (2 consecutive steps) therefore
  never fired, so `RESUME` not occurring is not by itself evidence of no lift-off.
- "All four feet stay loaded" is exact for seed 0 only; on seed 11 RR steps in and out.

## Classification

- **Overridden by the controller:** no (question 3).
- **Standing by choice:** not supported. The FL targets look like swing onset, and the output
  is low-amplitude but not static (question 4).
- **Reacting to abnormal observations:** supported as a fact (observation vector far from the
  reference gait, FL thigh offset, near-zero measured velocity) but not shown to be the cause.
- **Physically restrained by the wire:** supported (continuous contact, −20 N against the
  command, backward drift, thigh stuck off-target) but not shown to be sufficient on its own.
- The wire could also be the *reason* the observations are abnormal, so these overlap. Present
  data cannot separate "policy stalls because the state is atypical" from "robot cannot move
  because the wire holds it".

## Uncertainty

Small reference sample; the stall plateau starts after a ~10-step transient whose effect on
posture I did not model; the sign of the wire force comes from my D5.52 contact decode (force
on the robot), corroborated by the backward drift but not independently verified; `ty` and
`ruff` pass on the script, but no unit test was added.

## Single next experiment (not implemented)

**Wire-off replay of the same bounded release.** Reuse the D5.52 wire-exclusion build and the
D5.55 `BoundedRetraction` wrapper unchanged, force hybrid entry at the matching FL gait phase
(two steps after FL lift-off, as in the wire-on runs), same gait-derived budget, and compare the
25 steps after FL touchdown against the plateau numbers above.
- Trot resumes (FL foot rises above ~0.03 m, forward speed back near 0.25 m/s): the wire
  restraint, not the policy, causes the stall; the next experiment must change how FL meets
  the wire, not the policy inputs.
- Stall persists with no wire: the stall is the policy's response to the post-retraction state,
  and the next experiment is an observation-level test (FL joint terms), not a wire one.
- If forced entry in the wire-off case cannot reproduce the FL phase or the retract
  (different dynamics), report that rather than adjusting the timing.

Required measurements are all available in the current logger (this script's capture plus the
D5.52 row fields). A behavior-changing task must include exact, parser-verified viewer
commands for baseline and variant on seeds 0 and 11, run long enough to cover the stall window,
with a byte-identical headless-trace comparison, as in D5.53–D5.55.

## Retained files

`logs/d5_56/report.md`, `logs/d5_56/summary.csv`, `scripts/d5_56_policy_stall_diag.py`. All
other traces and the temporary per-step dump were kept in memory or in the session scratch
directory and are not retained.
