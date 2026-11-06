# D5.20 — Recovery-Duration-Cap Sweep and Handoff Stability

## Objective

Test, via a controlled intervention on the existing
`max_recovery_steps_override` parameter (not a new mechanism), whether
changing RECOVERY-phase duration changes the FL handoff tracking error
and the post-exit base-height dip, and whether any resulting change is
in the direction predicted by D5.19's correlational finding. This is a
diagnostic intervention, not an attempt to optimize the controller.

## Before running

### 1. Inspection of prior work
Reviewed D5.19's report/seed-level CSVs, and re-read D5.3/D5.16/D5.18
source and logs before designing this sweep (see citations below).

### 2. Verifying the cap mechanism from source

`src/mjlab/tasks/recovery/mdp/actions.py`, `_update_fsm`
(lines ~1066-1270):

- `_recovery_elapsed` increments by 1 every step while
  `still_recovery_after_entry` is true (i.e. from the trigger step
  onward), independent of the cap value.
- `cap_reached = self._recovery_elapsed >= self.cfg.max_recovery_steps`.
- `exiting_recovery = still_recovery_after_entry & (genuine_recovery_complete | cap_reached)`
  — exit fires on **whichever condition is met first**; the cap can
  only force an *earlier* exit than the natural crossing+release
  completion, never a later one, and never changes when crossing
  itself is detected (the two-stage `calf_clearance_achieved` /
  `foot_crossed_achieved` latches are purely geometric/sensor-driven,
  not elapsed/cap-driven).
- `genuine_recovery_complete = self._crossing_achieved & (self._release_counter >= self.cfg.release_required_steps)`
  (`release_required_steps=5`, confirmed from `ResidualRecoveryActionCfg`,
  unchanged).
- `self._recovery_failed_timeout` is set `True` iff
  `exiting_recovery & ~genuine_recovery_complete` — i.e. the run is
  classified as a *failure/timeout* whenever the cap, not genuine
  crossing+release, is what caused the exit, regardless of how close
  the leg was to completing. This confirms the task's warning: a
  cap-forced exit must be scored as a failed recovery, never as a
  "smaller dip because it exited sooner."
- Actuator restoration (`_apply_hybrid_actuators`, gated on
  `act.newly_recovered`, i.e. on `exiting_recovery`) fires **identically**
  on a cap-forced exit and a genuine exit — there is no separate
  "abort" path. This is what makes the cap a meaningful, non-invasive
  lever: it changes *when* the existing restoration logic fires, not
  *what* that logic does.
- `go2_wire_recovery_residual_env_cfg(..., max_recovery_steps=...)`
  (`env_cfg_residual.py` line 262) passes the value straight into
  `ResidualRecoveryActionCfg.max_recovery_steps` with no other
  side effects — confirmed no other cfg field is touched by this
  parameter.

### 3. Predeclared cap values and rationale

Seed-0 baseline (cap=48, from D5.18/D5.19, re-confirmed below)
naturally exits via genuine crossing+release at
`recovery_elapsed=45`, with stage-1/stage-2 crossing latched at
`recovery_elapsed=39` (step 84, trigger at step 45). The cap (48) is
never binding in the baseline — 45 < 48.

Predeclared set (chosen **before** running anything, based only on
these elapsed/latch numbers): **{40, 44, 48, 56, 64}**.

| cap | relation to natural thresholds (latch=39, exit=45) | predicted outcome |
|---|---|---|
| 40 | below the stage-1/2 latch step (39) by only 1 step | forces exit before crossing is even geometrically detected — severe truncation, timeout, qualitatively different kinematic state at handoff (mid-swing, pre-crossing) |
| 44 | above latch (39), below natural exit (45) | forces exit after crossing has latched but before the 5-step release-counter requirement completes — a "near-miss" timeout, same qualitative state as success but less release time |
| 48 | = natural exit requirement | **control**: reproduces the existing baseline exactly |
| 56 | above natural exit requirement | predicted **no-op**: cap never binds, so trajectory should be byte-identical to cap=48 |
| 64 | further above natural exit requirement | second, independent **no-op control**, confirming the no-op result isn't coincidental to one specific margin |

This set deliberately spans all three qualitatively distinct regimes
identified from source (pre-latch truncation, post-latch/pre-release
truncation, and the no-op region) with the minimum number of runs
needed to characterize each regime once, plus one redundant check
(56 vs 64) to confirm the no-op is robust — not a broad, undirected
seed/value sweep.

### 4. Timeout risk check
Per the mechanism above, caps 40 and 44 are **predicted and confirmed**
to force `cap_timeout` (see Results). This is expected and intentional
— these two conditions are the "lower cap" arm of the sweep precisely
because they are expected to prevent genuine crossing. Per instructions,
these are recorded as **failed recoveries**, and any smaller measured
post-exit dip in these conditions is explicitly **not** interpreted as
a stability improvement (see Analysis, point 5).

### 5. Baseline reconfirmation
Re-ran cap=48 (the control) as part of this sweep and confirmed it
exactly reproduces D5.18/D5.19's seed-0 numbers:
`trigger_step=45`, `EXIT RECOVERY step=88 elapsed=45 timeout=False`,
`stage1/2_first_latch_step=84`, `first_actuator_restore_step=88`,
`first_sustained_fl_ground_step=89` — unchanged from both prior
reports, confirming no drift and that this script continues to reuse
the unmodified D5.3 build path, frozen checkpoint, wire geometry, and
all actuator parameters (only `max_recovery_steps` differs between
conditions).

## Experimental design actually run

Reused `scripts/d5_18_post_recovery_diagnostic.py` **completely
unmodified** (it already exposes `--max-recovery-steps-override` and
logs every required column — `pos_thigh_ctrl_target`/
`pos_calf_ctrl_target`, all four legs' ground contact, foot
position/velocity in base frame, base kinematics, actuator forces).
Only the CLI argument value differs between conditions, not any code.

Exact commands run (seed 0, headless, each to its own new directory):

```
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 --max-steps 200 --max-recovery-steps-override 40 --out-dir logs/d5_20_recovery_cap_sweep/cap40
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 --max-steps 200 --max-recovery-steps-override 44 --out-dir logs/d5_20_recovery_cap_sweep/cap44
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 --max-steps 200 --max-recovery-steps-override 48 --out-dir logs/d5_20_recovery_cap_sweep/cap48
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 --max-steps 200 --max-recovery-steps-override 56 --out-dir logs/d5_20_recovery_cap_sweep/cap56
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 --max-steps 200 --max-recovery-steps-override 64 --out-dir logs/d5_20_recovery_cap_sweep/cap64
```

No viewer run was needed: this is a pure parameter-value sweep using
an already-validated (D5.15/D5.18) headless stepping path; no new
visualization question is being asked, so a `--debug` run was not
performed for this task (consistent with "use the corrected viewer
pattern only if visualization is needed").

## Results

| cap | exit step | elapsed | exit reason | stage1/2 latch step | thigh err (rad) | calf err (rad) | thigh force before→after (N·m) | base_z @exit | base_z min (step) | drop (m) | 4-leg contact loss in window |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 83 | 40 | **cap_timeout** | None / None (never latched) | −1.761 | +0.516 | 0.00 → −16.09 | 0.2799 | 0.2799 (83) | **0.0000** | all 4 |
| 44 | 87 | 44 | **cap_timeout** | 84 / 84 | −2.746 | +1.431 | 0.00 → −23.5 (saturated) | 0.2666 | 0.2390 (91) | 0.0276 | all 4 |
| 48 (control) | 88 | 45 | crossing+release | 84 / 84 | −2.990 | +1.406 | 0.00 → −23.5 (saturated) | 0.2544 | 0.2182 (92) | 0.0362 | all 4 |
| 56 | 88 | 45 | crossing+release | 84 / 84 | −2.990 | +1.406 | 0.00 → −23.5 (saturated) | 0.2544 | 0.2182 (92) | 0.0362 | all 4 |
| 64 | 88 | 45 | crossing+release | 84 / 84 | −2.990 | +1.406 | 0.00 → −23.5 (saturated) | 0.2544 | 0.2182 (92) | 0.0362 | all 4 |

Confirmed via direct file diff: the cap=48, cap=56, and cap=64
`trajectory.csv` files are **byte-for-byte identical**, every row, all
columns (`diff` returns no output for both comparisons). This is the
cleanest possible confirmation that, once the cap exceeds the
naturally-required elapsed time, it has **zero** effect on the
simulation — not just on the summary statistics but on every logged
quantity at every single step, which is exactly what the FSM source
inspection predicted (the physics and FSM transitions are fully
determined by sensor/geometry conditions, never by the cap, once the
cap is non-binding).

Foot-crossing geometry at handoff (base-relative), illustrating how
qualitatively different the cap=40 state is from the others:
`foot_x_base`=0.0937 (cap 40) vs. 0.0751-0.0808 (caps 44/48+); more
notably `foot_vx_base`=+1.52 m/s (cap 40, still moving *forward/up* in
its swing) vs. −2.17 to −2.23 m/s (caps 44, 48+, already reversing
into its post-recovery stance placement) — this sign flip indicates
cap 40 catches the leg mid-swing in a fundamentally different part of
its trajectory, not simply "the same trajectory, truncated."

## Analysis

**(1) Recovery duration and actual exit timing:** As designed, caps 40
and 44 shorten the RECOVERY phase (exit at elapsed 40 and 44
respectively, both earlier than the natural 45); caps 48/56/64 all
exit at elapsed 45 — the cap has no effect once ≥45.

**(2) Handoff tracking error:** Monotonic with elapsed, as predicted by
D5.19's hypothesis, across the ordered truncation levels: |thigh_err|
= 1.761 (elapsed 40) → 2.746 (elapsed 44) → 2.990 (elapsed 45,
caps 48+). |calf_err| follows the same ordering: 0.516 → 1.431 → 1.406
(44 and 48/56/64 are close but not perfectly monotonic between 44 and
48 for calf — 1.431 vs 1.406, essentially flat/slightly non-monotonic
at the top end, within what could be normal per-step geometric noise
near the crossing-latch event). The large jump from cap 40 to cap 44
(1.761→2.746 thigh) coincides with crossing the stage-1/2 latch
threshold (39 elapsed) — i.e. part of the error-magnitude increase
reflects a qualitative state change (post- vs. pre-latch), not a
smooth function of elapsed time alone.

**(3) Immediate actuator-force transient:** Thigh actuator force at the
exit step is 0.00 N·m immediately before restoration in all 5
conditions (actuator disabled throughout RECOVERY, as expected), then
jumps to −16.09 N·m (cap 40, **not** saturated), −23.5 N·m (cap 44,
saturated), and −23.5 N·m (caps 48/56/64, saturated) immediately after.
This is the first case in the whole D5.x series where the thigh
actuator does **not** saturate at restoration — confirming that
saturation (observed universally across all 10 D5.19 seeds) is not an
absolute, cap-independent constant but is itself a function of
handoff-error magnitude: at the smallest error (cap 40), the restoring
torque demand is below the −23.5 N·m limit.

**(4) Post-exit body-height drop and contact loss:** Drop is 0.0000 m
(cap 40 — no measurable dip in the 20-step window after this
particular truncated exit), 0.0276 m (cap 44), 0.0362 m (caps
48/56/64) — monotonically increasing with elapsed/error magnitude,
consistent with D5.19's correlational finding. All 5 conditions show
contact loss on all 4 feet somewhere in the post-exit window
(universal, as in D5.19 — does not discriminate between conditions).

**(5) Genuine crossing success or failure:** Only cap=48/56/64 achieve
`genuine_recovery_complete=True`. Caps 40 and 44 are both
`cap_timeout` failures (`recovery_failed_timeout_flag=True`) and are
reported as such — their smaller (cap 44) or zero (cap 40) measured
dip is **not** interpreted as a stability improvement; it is a failed
recovery attempt that happens to also have a smaller error/dip, which
is exactly the predicted (not coincidental) relationship under the
tracking-error hypothesis, but it does not make the shorter run "more
stable" in any outcome sense that matters (the robot's FL leg did not
clear the wire).

### Does increasing/decreasing the cap change tracking error and dip in the predicted direction?

**Yes, within the regime where the cap actually binds** (40, 44, 45):
shorter cap → smaller elapsed-at-handoff → smaller |thigh_err| and
|calf_err| → smaller actuator-force transient (non-saturating at
cap 40) → smaller post-exit drop. This is the predicted monotonic
relationship, and the fact that it was *produced by an experimental
manipulation* (not merely observed across uncontrolled seeds as in
D5.19) is somewhat stronger evidence that tracking-error magnitude is
mechanistically upstream of the drop magnitude, within this specific
intervention.

**However, interpretation is complicated by confounds this experiment
cannot separate:**
- Cap 40 does not just reduce error magnitude — it changes the
  *qualitative* leg state at handoff (pre- vs. post-crossing-latch,
  and `foot_vx_base` changes sign, indicating a different phase of the
  swing trajectory entirely, not merely "the same trajectory, less far
  along"). The zero measured drop in cap 40 could be a function of this
  qualitative difference (e.g., the leg is not yet in a
  ground-approaching configuration at all) rather than purely a
  function of the smaller numeric tracking error.
- Caps 44 and 48 share the same stage-1/2 latch step (84) and are much
  more directly comparable (same qualitative regime, genuinely
  differing mainly in release-counter completion and the extra step of
  thigh retraction) — this pair is the cleanest evidence for the
  monotonic relationship, since the confound above does not apply to
  it.
- This remains an n=1-seed intervention (seed 0 only, per the task's
  "do not launch a broad seed sweep until seed-0 results have been
  reviewed"). A single seed's monotonic trend across 3 binding-cap
  conditions is suggestive but not statistically conclusive on its
  own; D5.19's separate, independent 10-seed correlational result is
  the stronger evidence base, and this experiment's role is to test
  whether an active intervention points the same direction (it does).

**Causation:** This experiment still cannot fully establish that
tracking error *causes* the dip rather than both being downstream of
elapsed recovery time (which also determines how far the whole
robot's gait phase has drifted from the frozen policy's expected
walking cadence, a separate potential channel to instability
independent of the specific FL joint errors). The cap manipulates
elapsed time directly; error magnitude and dip magnitude are both its
consequences, so this is a cleaner manipulation than D5.19's seed
cross-section but does not, by itself, isolate tracking error as the
exclusive causal path (e.g. it's also possible a shorter elapsed time
simply gives the whole-body dynamics less opportunity to destabilize,
independent of the specific joint-tracking-error channel).

## Limitations

- Single seed (0) only, per instructions — this experiment is not a
  cross-seed sweep and its conclusions are conditioned on this one
  seed's specific trigger timing and wire-crossing geometry.
- Only 3 of the 5 conditions (40, 44, 48) differ in outcome; 56 and 64
  are confirmed no-ops and contribute only confirmatory evidence (the
  cap mechanism works exactly as read from source), not additional
  data points for the trend.
- The cap=40 condition is not directly comparable to 44/48 due to the
  qualitative pre-/post-crossing-latch state difference noted above;
  treating all 3 binding points as one clean monotonic series
  overstates the comparability of cap 40 to the other two.
- No viewer/visual validation was performed for this sweep (judged
  unnecessary — all quantities are read directly from already-validated
  logging code, and the no-op confirmation via byte-identical CSVs is
  itself a strong procedural validation).
- No statistical test is reported (n=1 seed × 5 conditions, 3 of which
  are literally identical runs) — differences are described
  qualitatively/by magnitude only.

## Recommended next experiment (exactly one)

Run this same predeclared-cap-sweep design (caps in the 44-48 range,
the pair confirmed to share the same qualitative post-crossing-latch
state) across the existing D5.5/D5.19 seed cohort (seeds 0-9), to test
whether the monotonic error/dip relationship observed here for seed 0
holds across seeds with different trigger timings and natural exit
elapsed values — this would let the tracking-error-magnitude
hypothesis be tested with a genuine multi-seed intervention (not just
a multi-seed correlation as in D5.19, nor a single-seed intervention as
here), while still using only the existing `max_recovery_steps`
override and no new controller mechanism. This is **not implemented**
in this task, per instructions to stop after the seed-0 sweep and
report.
