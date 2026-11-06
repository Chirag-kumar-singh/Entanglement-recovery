# D5.22 — Event-Relative Handoff Validation

## Objective

Test whether the tracking-error/post-exit-dip relationship (D5.19-D5.21)
persists when recovery exits are compared at an **equivalent phase
relative to the crossing-latch event**, rather than at a fixed absolute
`max_recovery_steps` cap value — directly addressing the touchdown-timing
confound D5.21 identified (cap=44 lands at very different relative
phases across seeds).

## Step 1 — Inspection and per-seed event timeline

Re-read D5.19, D5.20, D5.21 reports, and re-confirmed from
`_update_fsm` source (as in D5.20/D5.21) that:
- `recovery_elapsed` increments once per step from the trigger step
  onward, independent of cap value.
- The two-stage crossing latch (`calf_clearance_achieved` then
  `foot_crossed_achieved`) is purely sensor/geometry-driven and occurs
  at a fixed `recovery_elapsed` value for a given seed, **regardless**
  of the cap (confirmed empirically below: every seed's `stage2`
  elapsed-step is identical between its cap-48 baseline and any
  cap-restricted intervention, since the cap only changes the step at
  which `exiting_recovery` is forced, never the FSM logic that produces
  the latch events themselves).
- `exiting_recovery = still_recovery_after_entry & (genuine_recovery_complete | cap_reached)`,
  evaluated every `_update_fsm` call (once per step), so the "offset"
  `recovery_elapsed_at_exit − recovery_elapsed_at_stage2_latch` is an
  exact, well-defined, step-counted quantity, not an approximation.

Per-seed event timeline extracted from existing cap-48 logs
(D5.19/D5.5 cohort), using `recovery_elapsed` at each event:

| seed | trigger step | stage2 latch step | elapsed@stage2 | natural exit step | elapsed@exit | **offset (exit − stage2)** | genuine@cap48 |
|---|---|---|---|---|---|---|---|
| 0 | 44 | 84 | 41 | 88 | 45 | **4** | True |
| 1 | 43 | 86 | 44 | 90 | 48 | **4** | True |
| 2 | 43 | 86 | 44 | 90 | 48 | **4** | True |
| 3 | 51 | 91 | 41 | 95 | 45 | **4** | True |
| 4 | 42 | 84 | 43 | 88 | 47 | **4** | True |
| 8 | 43 | 83 | 41 | 87 | 45 | **4** | True |
| 9 | 44 | 88 | 45 | 91 | 48 | **3** | False (already cap-limited) |
| 5, 6, 7 | — | None/None | — | — | — | — | False, no stage2 latch reached |

**Key finding:** every genuinely-completing seed's natural exit happens
at **exactly offset = 4** steps after its own stage-2 latch (i.e.
`release_required_steps=5` manifests as a 4-step gap in this
`recovery_elapsed`-indexed accounting — consistent with the
counter needing 5 consecutive non-contact steps but the exit check
itself firing on the step the counter reaches the threshold). This is
not an approximation; it is exact across all 6 genuine seeds.

Seed 9 is a special case: it reaches stage2 latch (elapsed 45) but the
cap (48) forces exit only 3 steps later — i.e. **seed 9's own cap-48
run already sits at offset=3**, one step short of genuine release.
Seeds 5, 6, 7 never reach stage2 at all (even at cap=48) and are
therefore **ineligible** for any stage2-relative comparison.

## Step 2 — Defining the phase-matched comparison

D5.21 flagged seeds 0 and 8 as the "clean" pair under the uniform
cap=44 condition. Checking why: both seeds have `elapsed@stage2 = 41`
(identical), so cap=44 produces an identical offset
(`44 − 41 = 3`) for both — this is exactly why they were phase-matched
by coincidence under a single fixed cap value, while seeds 1, 2, 6, 9
(whose stage2 elapsed-steps differ: 44, 44, None, 45) were not.

**Proposed per-seed relative cap:** `cap = elapsed@stage2 + 3`, chosen
to reproduce the exact offset=3 that seeds 0/8 already exhibited under
cap=44, applied individually per seed so that every eligible seed is
compared at the **same event-relative phase** (3 steps past its own
crossing latch) rather than the same absolute elapsed count.

Verification against Step 2's four requirements, before running:
- **Does not force exit before crossing latch:** offset=3 > 0 for all
  eligible seeds by construction (cap = stage2 + 3 > stage2).
- **Does not mix pre/post-crossing regimes:** all 6 eligible seeds
  reach stage2 latch before their target cap is reached (stage2 < cap
  by exactly 3 steps in every case) — single, consistent regime.
- **Unambiguous, consistently-defined offset:** `recovery_elapsed`
  (source-confirmed, FSM-internal, cap-independent counter) at exit
  minus `recovery_elapsed` at stage2 latch = 3, exactly, for all 6
  interventions (confirmed post-hoc below).
- **Preserves genuine-vs-timeout distinction:** offset=3 < the natural
  offset=4 required for `release_required_steps=5` to complete, so
  every one of these interventions is **guaranteed** (by construction,
  from source) to be a `cap_timeout` failure, never a coincidental
  genuine completion — this is intentional and consistent with the
  task's instruction to never count a forced exit as success.

Per-seed target caps (seeds 0, 3, 8 already available from D5.21's
existing cap=44 runs, reused without rerunning since
`elapsed@stage2 + 3` happens to equal 44 for all three):

| seed | elapsed@stage2 | target cap (= stage2+3) | source |
|---|---|---|---|
| 0 | 41 | 44 | reused: D5.21 `seed0_cap44` |
| 3 | 41 | 44 | reused: D5.21 `seed3_cap44` |
| 8 | 41 | 44 | reused: D5.21 `seed8_cap44` |
| 1 | 44 | 47 | **new run** |
| 2 | 44 | 47 | **new run** |
| 4 | 43 | 46 | **new run** |
| 9 | 45 | 48 | reused: D5.19 `seed9` (cap48) — already natively at offset=3, no rerun needed |
| 5, 6, 7 | — | — | **ineligible** — no stage2 latch to anchor an offset; excluded, not substituted with a different experiment |

No eligible-seed case required abandoning the experiment: a common,
interpretable, event-relative phase (offset=3) was identified and
achieved for 7 of the 10 seeds using only the existing
`max_recovery_steps_override` mechanism.

## Step 3 — Runs executed

New commands (seeds 1, 2, 4 only; all others reused unmodified from
D5.19/D5.21):

```
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 1 --max-steps 200 \
  --max-recovery-steps-override 47 --out-dir logs/d5_22_event_relative_handoff_validation/seed1_cap47
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 2 --max-steps 200 \
  --max-recovery-steps-override 47 --out-dir logs/d5_22_event_relative_handoff_validation/seed2_cap47
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 4 --max-steps 200 \
  --max-recovery-steps-override 46 --out-dir logs/d5_22_event_relative_handoff_validation/seed4_cap46
```

All headless; no viewer run was needed (same justification as
D5.20/D5.21 — pure parameter-value reuse of already-validated logging
code; headless output remains authoritative). Checkpoint, wire
geometry, hip-wire setting, initial conditions, and sim config are
unchanged from all prior D5.x runs (only `max_recovery_steps_override`
differs between conditions, confirmed by reusing the identical,
unmodified `scripts/d5_18_post_recovery_diagnostic.py`).

## Step 4 — Per-seed paired results

| seed | baseline (cap48) exit reason | offset@baseline | intervention exit reason | offset@intervention | genuine@baseline | genuine@intervention | \|thigh_err\| base→interv | \|calf_err\| base→interv | sat (both) | drop base→interv (m) | foot_vx base→interv (m/s) | FL contact @exit (both) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | crossing+release | 4 | cap_timeout | 3 | True | False | 2.990→2.746 | 1.406→1.431 | yes/yes | 0.0362→0.0276 | −2.226→−2.166 | False/False |
| 1 | crossing+release | 4 | cap_timeout | 3 | True | False | 2.854→2.588 | 1.335→1.372 | yes/yes | 0.0409→0.0307 | −2.084→−1.990 | False/False |
| 2 | crossing+release | 4 | cap_timeout | 3 | True | False | 2.778→2.478 | 1.320→1.251 | yes/yes | 0.0226→0.0133 | −2.307→−1.874 | False/False |
| 3 | crossing+release | 4 | cap_timeout | 3 | True | False | 2.440→2.466 | 0.675→0.590 | yes/yes | 0.0000→0.0000 | −1.720→−1.287 | False/False |
| 4 | crossing+release | 4 | cap_timeout | 3 | True | False | 2.422→2.444 | 0.662→0.580 | yes/yes | 0.0000→0.0000 | −1.816→−1.403 | False/False |
| 8 | crossing+release | 4 | cap_timeout | 3 | True | False | 3.152→2.948 | 1.320→1.348 | yes/yes | 0.0336→0.0233 | −2.287→−2.183 | False/False |
| 9 | cap_timeout (own natural cap) | 3 | *(identical — no separate higher-offset point exists for this seed)* | 3 | False | False | 3.080 (unchanged) | 1.263 (unchanged) | yes | 0.0261 (unchanged) | −2.204 (unchanged) | False |

Seed 9's row is **not a paired comparison** — its own cap-48 run
already sits at offset=3 with no genuine completion ever achieved at
any larger offset in this dataset, so there is no "baseline at
offset=4" available for it. It is reported as a single unpaired
reference point, not forced into the paired analysis.

**All 6 interventions (0, 1, 2, 3, 4, 8) are confirmed
`cap_timeout` failures** (`recovery_failed_timeout_flag=True`,
`genuine_recovery_complete=False`), exactly as predicted by
construction. None of these — including the 4 that show a measurably
smaller drop — are counted as successful or "improved" recoveries; all
represent the FL leg failing to complete sustained release before
actuator restoration is forced.

FL ground contact at the exact exit step is `False` in all 12 rows
(6 seeds × 2 conditions) — consistent with every prior D5.x finding
that the FL foot has not yet landed at the moment of handoff in this
recovery regime, in both the longer and shorter conditions alike.

## Analysis

**(1) Phase-matched correlation, n=6:** Pearson
r(Δ|thigh_err|, Δdrop) = **0.961** across all 6 phase-matched seeds.
This is numerically even stronger than D5.21's unmatched cross-seed
result (0.912), which is noteworthy in itself.

**(2) But this is substantially a floor-effect artifact.** Two of the
six seeds (3, 4) show **zero** drop at *both* offset=4 and offset=3 —
the tracking error actually changes negligibly or even increases
slightly in absolute value for these two (seed 3: 2.440→2.466, i.e.
Δ = −0.026; seed 4: 2.422→2.444, Δ = −0.022), yet both are "matched" in
the correlation purely because both drop values are pinned at the
floor (0.0000 in each condition). Restricting to the **4 seeds with a
genuinely graded (non-floor) drop in both conditions** (0, 1, 2, 8):

Pearson r(Δ|thigh_err|, Δdrop) = **−0.325** (n=4) — weak and in the
**opposite** direction to the hypothesis. Calf-error deltas for this
same subset are mixed in sign (−0.025, −0.037, +0.069, −0.028) and show
no consistent relationship with drop change either.

**This is the central finding of D5.22:** once the touchdown-timing
confound from D5.21 is explicitly controlled for (by matching the
event-relative offset rather than the absolute cap), the previously
strong tracking-error/drop correlation **does not hold** in the subset
of seeds where a graded (non-floor) comparison is actually possible.
The apparent strength of the phase-matched correlation at n=6 is
driven almost entirely by two seeds landing at a shared floor value,
not by a real graded relationship between tracking-error magnitude and
dip magnitude.

**(3) What does change consistently between offset=4 and offset=3,
across all 6 seeds:** `foot_vx_base` (FL foot forward velocity,
base-relative) decreases in magnitude in every single seed (e.g. seed
2: −2.307→−1.874; seed 3: −1.720→−1.287) — i.e. the foot is
consistently moving slower in its forward/landing-approach direction
one step earlier in its trajectory (offset=3 vs. 4), which is a purely
kinematic consequence of being one step earlier in a continuous swing
trajectory, not a tracking-error effect. The fact that drop also tends
to decrease (in the 4 non-floor seeds, though not in a way that
correlates with tracking error specifically) is at least as
parsimoniously explained by "one step less of accumulated swing motion
at a consistently decelerating approach phase" as by "smaller
tracking-error magnitude."

**(4) Seed 9 (unpaired):** sits at offset=3 nativly, already a
`cap_timeout` failure with no counterfactual at offset=4 to compare
against (its own cap-48 run never reaches genuine completion at any
elapsed value in this dataset) — it cannot contribute to the
paired analysis and is reported only as an existing, already-logged
reference point.

**(5) Causation:** This event-relative design is a stronger control
than D5.20/D5.21's fixed-absolute-cap design, and it now shows that
the earlier strong correlations were **substantially inflated by the
touchdown-timing confound**, exactly as D5.21 warned. The remaining,
properly phase-matched evidence (n=4 graded seeds) does **not**
support tracking-error magnitude as a reliable predictor of dip
magnitude once timing is controlled; if anything it points toward
foot-velocity/swing-phase-timing as at least as plausible an
explanation, which this design still cannot fully separate from
tracking error (since both continue to change together whenever
`recovery_elapsed` changes by a fixed step count). No causal claim is
supported by this data in either direction.

## Sample sizes and limitations

- 6 paired seeds total (0, 1, 2, 3, 4, 8); only 4 of these (0, 1, 2, 8)
  avoid the zero-drop floor and support a genuinely graded comparison.
- 1 unpaired reference seed (9); 3 ineligible seeds (5, 6, 7) excluded
  per Step 2's explicit criteria, not substituted.
- With n=4 for the floor-free subset, a Pearson r should be treated as
  illustrative only — this is an even smaller, more fragile sample
  than D5.19-D5.21's already-small cohorts, and the negative sign
  should not itself be over-interpreted as proof that tracking error
  is irrelevant; it is reported honestly as "no supporting evidence in
  this corrected design," not as a refutation.
- Only two offset values (3 and 4) were compared, both very close
  together numerically (a 1-step difference in `recovery_elapsed`);
  this experiment cannot characterize the relationship's shape over a
  wider span of offsets without risking re-introducing the touchdown-
  timing confound this design was built to avoid (going much further
  back from stage2 latch would approach the pre-crossing regime; going
  much further forward starts to re-approach genuine completion).

## Recommended next experiment (exactly one)

Because the touchdown-timing confound and the tracking-error signal
are both driven by the same single scalar (`recovery_elapsed`), no
further manipulation of `max_recovery_steps` alone can separate them.
The next experiment should instead use **purely observational,
higher-resolution logging within the existing, unmodified offset=4
genuine-completion window** (no new cap values) — recording
`thigh_err`/`calf_err`/`foot_vx_base` at every single intervening step
between stage2 latch and genuine exit (not just at the final handoff
step) for the 6 eligible seeds, to test whether tracking-error
magnitude and foot-velocity magnitude decouple from each other at any
point within that 4-step natural window (e.g., if tracking error peaks
earlier or later than foot-velocity deceleration). If they remain
perfectly collinear across the full window as well, that would be
strong evidence the two quantities cannot be disentangled with this
FSM/cap mechanism at all, and the appropriate conclusion would be to
stop pursuing cap-based interventions for this question entirely. This
is **not implemented** in this task, per instructions to stop after
the experiment and report.
