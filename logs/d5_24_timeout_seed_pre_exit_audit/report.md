# D5.24 — Timeout-Seed Pre-Exit Observational Audit

## Objective

Extend D5.23's validated, step-indexed observational logging to the
four cap-timeout seeds (5, 6, 7, 9) to test whether the pre-exit
instability pattern found in the six naturally-completing seeds
generalizes to recovery failures. Purely observational — no cap, FSM,
gain, or timing change.

## Step 1 — Inspection and verification before running

Re-read D5.23's report and its final (bug-fixed) logging
implementation. Re-checked D5.19–D5.22 existing logs/run_logs for
seeds 5, 6, 7, 9 and confirmed, from the `calf_clearance_achieved`/
`foot_crossed_achieved` columns directly (not inferred):

| seed | stage1 (calf clearance) latched? | stage2 (foot crossed) latched? | exit reason | elapsed@exit |
|---|---|---|---|---|
| 5 | **yes** (step 82) | **no** | cap_timeout | 48 |
| 6 | **yes** (step 88) | **no** | cap_timeout | 48 |
| 7 | **no** | **no** | cap_timeout | 48 |
| 9 | **yes** (step 87) | **yes** (step 88) | cap_timeout | 48 |

This confirms, per the task's explicit caution, that **not all
timeout seeds reach stage 2**, and seed 7 reaches **neither** stage —
its final RECOVERY steps are not phase-equivalent to D5.23's
stage2-anchored natural-exit windows. Only seed 9 reaches stage 2 (and
therefore has a structurally comparable, if ultimately
release-incomplete, window); seeds 5 and 6 reach only stage 1; seed 7
reaches neither. These distinctions are preserved throughout the
analysis below rather than treating all 4 as interchangeable.

## Step 2 — Observational logging extension

Reused `scripts/d5_23_intra_window_observational_audit.py`
**completely unmodified** (same validated sensor-force aggregation fix,
same coordinate-frame conventions, same column set) — no new script
was written, since D5.23's implementation is already seed-agnostic and
requires no changes to log the 4 timeout seeds. `max_recovery_steps_override`
left at its existing default (48), identical to all of D5.19–D5.23's
configuration for these seeds.

## Step 3 — Runs and trajectory-parity validation

Ran all 4 timeout seeds headless:

```
uv run python scripts/d5_23_intra_window_observational_audit.py --seed <N> --max-steps 200 \
  --out-dir logs/d5_24_timeout_seed_pre_exit_audit/seed<N>
```

for N in {5, 6, 7, 9}. All 4 reproduce D5.19/D5.21/D5.22's exact
existing summary values (`trigger_step`, `stage1/2_first_latch_step`,
`EXIT RECOVERY step=... elapsed=48 timeout=True`) — confirming this is
a clean rerun of the same existing, unmodified baseline, not a new
configuration.

**Trajectory parity check** (required by the task before any
analysis, since new logging was added): diffed each new CSV against
the existing D5.19 cap-48 baseline on all shared columns (step, phase,
recovery_elapsed, thigh_pos, calf_pos, ctrl targets, base_z,
fl_ground_contact, genuine_recovery_complete,
recovery_failed_timeout_flag) — **zero mismatches, all 4 seeds, 200/200
rows each.** Parity is established; the new logging columns
(`fl_ground_contact_force_mag`, saturation flags) are purely additive
and do not affect simulated behavior, consistent with D5.23's finding
for the 6 natural-exit seeds. Analysis below proceeds on this
validated basis.

## Step 4 — Per-seed event-ordering results

All step indices are direct simulation `step` values; Δ = step − exit
step.

| seed | stage1 | stage2 | **exit** | thigh-err pattern pre-exit | base_vz downward onset | FL foot_vx sign-flip | FR/RL/RR contact loss timing | body-height drop (from D5.19/21/22) | nadir offset |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 82 | **none** | **86** | roughly flat ≈ −2.6 to −2.7 rad for ~5 steps before exit (no late growth spike) | **not before exit** — bvz stays positive through exit+7, first goes negative at Δ+8 | Δ+1 (−1.99→+0.84) | RL lost at Δ0 (exit); FR lost at Δ+1; RR already not in contact before this window (pre-existing, not newly lost) | 0.0122 | Δ+17 (far later/shallower than natural-exit seeds) |
| 6 | 88 | **none** | **90** | continuous monotonic growth from −1.98 (8 steps before exit) to −2.85 at exit | Δ−2 (stage1 step) | Δ+1 (−2.41→+0.22) | RL lost at Δ−2 (=stage1 step); FR lost at Δ−1; RR already not in contact pre-existing | 0.0306 | Δ+5 |
| 7 | **none** | **none** | **91** | already large (~−2.6 to −2.8 rad) for the entire visible window; no stage1/2 latch ever anchors a "growth" reference point | Δ−5, sustained mild negative (~−0.02 to −0.13) for several steps before exit — not a sharp onset | Δ+1 (−1.98→+1.13) | FR/RL remain in contact through exit, lost at Δ+2/Δ+4 respectively (**later** than exit, unlike all other seeds); RR gains contact exactly at exit (Δ0) | 0.0000 | floor (no drop) |
| 9 | 87 | **88** | **91** | continuous monotonic growth from −2.01 (stage1, Δ−4) through −2.42 (stage2, Δ−3) to −3.08 at exit | Δ−1 | Δ+1 (−2.20→+0.50) | FR/RL both lost at stage2 step (Δ−3); RR already not in contact pre-existing, regains at Δ+2 | 0.0261 | Δ+4 |

## Analysis

**(1) Crossing-stage latches reached:** Only seed 9 reaches stage 2;
seeds 5 and 6 reach stage 1 only; seed 7 reaches neither. This is
reported as a hard constraint on comparability, not glossed over.

**(2) Contact-loss timing relative to exit:** Seeds 6 and 9 show
FR/RL contact loss **before** exit (Δ−2/Δ−1 for seed 6; Δ−3/Δ−3 for
seed 9) — directly reproducing D5.23's finding for the natural-exit,
non-floor seeds (where other-leg contact loss also preceded exit by
2-4 steps). Seed 5 shows a mixed pattern (RL lost exactly at exit,
FR lost one step after). **Seed 7 is a clear exception**: FR and RL
both remain in contact *through and past* the exit step, only losing
contact at Δ+2 and Δ+4 respectively — the opposite timing from seeds
6 and 9, and structurally similar to D5.23's floor-effect seeds (3, 4),
which also showed FR/RL support preserved through handoff.

**(3) Tracking-error growth and base-vz onset relative to exit:**
Seeds 6 and 9 show continuous monotonic tracking-error growth right up
to the exit step, and a downward base-vz onset 1-2 steps before exit —
both directly reproducing the pre-exit pattern D5.23 found in its
natural-exit non-floor seeds (0, 1, 2, 8). Seed 5's tracking error is
roughly flat (not growing) in its final pre-exit steps, and its base-vz
does not turn negative until well after exit (Δ+8) — more similar to
D5.23's floor-effect pattern despite being a timeout failure. Seed 7's
tracking error is already large and roughly flat throughout the
visible window (consistent with having never latched any crossing
stage to anchor a "growth" reference against), and its base-vz is
mildly negative for an unusually long, sustained period before exit
rather than showing a sharp pre-exit onset — a genuinely different
qualitative pattern from all other seeds (natural or timeout).

**(4) FL foot-velocity behavior around exit:** **All 4 timeout seeds
show the identical exit+1 sign-flip** (approach-velocity →
retreat/leveling) found in all 6 natural-exit seeds in D5.23, with no
exception. This is now confirmed across all 10 of the existing D5.5/
D5.19 seeds without a single counterexample — the strongest, most
seed-invariant finding across the entire D5.19–D5.24 series. It occurs
regardless of whether the seed ultimately succeeds, times out with
stage2 latched (9), times out with only stage1 latched (5, 6), or
times out without any latch at all (7).

**(5) Body-height drop and timing:** Seeds 6 and 9 show both
magnitude (0.0306, 0.0261 m) and nadir-timing (Δ+5, Δ+4) closely
matching D5.23's natural-exit non-floor seeds (which showed Δ+4/Δ+5
nadirs of 0.023-0.041 m). Seed 7 shows zero measurable drop, matching
D5.23's floor-effect pattern. Seed 5 is an intermediate/ambiguous
case: a small, nonzero drop (0.0122 m) but with an unusually delayed
nadir (Δ+17, roughly 3-4x later than every other seed in this entire
series) — this does not cleanly fit either the "natural non-floor"
or "floor-effect" pattern established in D5.23, and is reported as its
own distinct case rather than forced into either bucket.

**(6) Comparability with D5.23's natural-exit windows:**
- **Seed 9** is the most directly comparable case: it reaches the
  same stage-2 latch event as all 6 D5.23 natural seeds, and shows the
  same qualitative pre-exit signature (monotonic error growth, early
  contact loss, pre-exit downward base-vz, Δ+1 foot-velocity flip,
  Δ+4 nadir) as D5.23's natural non-floor seeds — the only structural
  difference is that it fails to complete the 5-step release counter
  before the cap forces exit. This strongly suggests seed 9's
  post-exit dip mechanism is not qualitatively different from the
  natural-exit non-floor seeds' mechanism, independent of whether
  genuine release is ultimately achieved.
- **Seeds 5 and 6** are partially comparable — both reach stage 1 but
  not stage 2, so any stage-2-anchored offset comparison from D5.23
  cannot be applied to them directly; an exit-relative (not stage-2-
  relative) comparison was used instead. Seed 6's exit-relative pattern
  closely tracks the natural non-floor seeds; seed 5's does not
  (flat error, delayed negative base-vz, very late/shallow nadir).
- **Seed 7 is explicitly non-comparable** to D5.23's stage-2-anchored
  windows: it never reaches either crossing stage, so there is no
  shared event to anchor a phase-matched comparison, and its pre-exit
  trajectory is already in a qualitatively different (persistently
  large, non-growing error; sustained rather than onset-like negative
  base-vz) regime from every other seed in the dataset. It is reported
  as its own case, not combined with the others' correlational
  patterns.

## Full ten-seed summary table

| seed | outcome | stage2 reached? | exit step | drop (m) | nadir offset | foot_vx flip offset | comparable to D5.23 natural non-floor pattern? |
|---|---|---|---|---|---|---|---|
| 0 | genuine | yes | 88 | 0.0362 | Δ+4 | Δ+1 | yes (reference) |
| 1 | genuine | yes | 90 | 0.0409 | Δ+5 | Δ+1 | yes (reference) |
| 2 | genuine | yes | 90 | 0.0226 | Δ+5 | Δ+1 | yes (reference) |
| 3 | genuine | yes | 95 | 0.0000 | Δ0 (floor) | Δ+1 | floor-effect (reference) |
| 4 | genuine | yes | 88 | 0.0000 | Δ0 (floor) | Δ+1 | floor-effect (reference) |
| 8 | genuine | yes | 87 | 0.0336 | Δ+4 | Δ+1 | yes (reference) |
| 5 | **timeout** | no (stage1 only) | 86 | 0.0122 | **Δ+17 (anomalous)** | Δ+1 | **no — distinct pattern** |
| 6 | **timeout** | no (stage1 only) | 90 | 0.0306 | Δ+5 | Δ+1 | **yes** |
| 7 | **timeout** | **no (neither stage)** | 91 | 0.0000 | Δ0 (floor) | Δ+1 | floor-effect-like, but **non-comparable** (no shared anchor event) |
| 9 | **timeout** | **yes** | 91 | 0.0261 | Δ+4 | Δ+1 | **yes (most comparable timeout seed)** |

## Observation vs. inference

- **Observed, not inferred:** all step indices, event flags, and
  computed offsets in the tables above, read directly from
  step-indexed CSV columns.
- **Inference, not proof:** that seed 9's structural similarity to the
  natural non-floor seeds implies a shared causal mechanism for the
  post-exit dip; that seed 7's non-growing, already-large tracking
  error and sustained (rather than onset) negative base-vz reflect a
  genuinely different underlying dynamical regime rather than simply
  a later or different sampling point in a continuous process; that
  seed 5's anomalously delayed nadir (Δ+17) reflects a distinct
  recovery dynamic rather than noise/a one-off trajectory quirk. None
  of these are established by this purely observational data; they
  are reported as open questions.

## Limitations

- n=4 timeout seeds, further subdivided into 3 incomparable
  sub-groups (1 stage-2-reaching, 2 stage-1-only, 1 reaching neither)
  — no correlation coefficient is computed, consistent with the task's
  instruction not to combine unlike recovery phases into a single
  correlation.
- Seed 5's anomalous nadir timing (Δ+17) is based on a single seed and
  is not explained by this analysis; it is flagged, not resolved.
- Event-detection thresholds are identical to D5.23's (0.3 rad
  tracking-error-growth threshold not formally reapplied here since
  no timeout seed shows a clean "stage2 + offset" reference point
  except seed 9; threshold-sensitivity caveats from D5.23 apply
  equally here.
- This remains a purely observational design; no intervention was
  performed to test whether any of the seed-conditional differences
  (e.g., seed 7's distinct error/base-vz pattern) are caused by
  differences in trigger timing, initial conditions, or some other
  factor not measured here.

## Recommended next experiment (exactly one)

Given that seed 9 (the one timeout seed that reaches stage 2) closely
reproduces the natural non-floor seeds' full pre-exit signature, while
seed 7 (reaching neither stage) is qualitatively distinct, and seed 5
has an unexplained outlier nadir timing, the next experiment should be
a **purely observational, still-non-interventional check**: extend
this same logging to the existing D5.5 seeds 10-19 (not yet
instrumented with D5.23/D5.24's columns, but already run at the
existing default cap=48 configuration in prior work), to determine
whether seed 5's anomalous Δ+17 nadir and seed 7's distinct
no-stage-latch pattern are rare one-off cases or recur at a
meaningful rate across a larger existing cohort — using only the
already-validated, unmodified D5.23 logging script, no new cap value,
and no FSM/controller change. This is **not implemented** in this
task, per instructions to stop after reporting and recommend rather
than execute the next experiment.
