# D5.23 — Intra-Window Recovery/Handoff Observational Audit

## Objective

Purely observational investigation of the temporal ordering of
tracking error, FL foot velocity/contact, whole-body contact state, and
base-height dynamics across the interval from the stage-2 crossing
latch through the natural (genuine) recovery exit and its aftermath,
for the 6 naturally-completing seeds identified in D5.22 (0, 1, 2, 3,
4, 8). No cap value, FSM logic, actuator gain, or timing was changed —
this task adds logging only.

## Step 1 — Definitions confirmed from source and existing logs

Re-read D5.19–D5.22 reports and re-verified from
`src/mjlab/tasks/recovery/mdp/actions.py`:

- **Recovery elapsed**: `_recovery_elapsed` increments by 1 every
  `_update_fsm` call from the trigger step onward; purely a step
  counter, independent of any cap.
- **Stage-2 latch**: `foot_crossed_achieved` — a one-shot latch set
  the step the toe/foot crosses `foot_crossing_margin` on the far
  side, **only evaluated after** `calf_clearance_achieved` (stage 1)
  has already latched. Identified per seed directly from the
  `foot_crossed_achieved` column (first `True` row), not inferred from
  any plot.
- **Genuine exit**: `exiting_recovery = still_recovery_after_entry & (genuine_recovery_complete | cap_reached)`,
  found as the exact `step` index where the `phase` column transitions
  1→2 in the CSV.
- **Tracking error**: `pos_thigh_ctrl_target − thigh_pos` (and calf
  analog), both already in physical joint-radian units (confirmed in
  D5.19/D5.20 from the `JointPositionAction` scale/offset pipeline —
  `env.sim.data.ctrl` is read post-pipeline).
- **Actuator force / saturation**: `env.sim.data.actuator_force` at the
  position-actuator's control id; saturation flag defined as
  `|force| >= 23.49` N·m (within 0.01 of `GO2_ACTUATOR_THIGH`/`_CALF`'s
  `effort_limit = 23.5`, confirmed in `go2_constants.py`).
- **Foot velocity/contact**: `foot_vx_base`/`foot_vz_base` (base-frame,
  via `quat_apply_inverse`, same convention as D5.16–D5.22);
  `fl/fr/rl/rr_ground_contact` booleans from the existing per-leg
  `ContactSensorCfg` (`fields=("found","force")`, `reduce="netforce"`).
- **Body-height drop**: `base_z` at the exit step minus its minimum
  over the following 20 steps, same convention as D5.19–D5.22.

All event steps below are read directly from simulation-step-indexed
CSV columns, not inferred from any plotted curve.

## Step 2 — Observational logging added (no intervention)

Created `scripts/d5_23_intra_window_observational_audit.py`, which
reuses D5.18's build path, sensor set, and FSM/actuator pipeline
**unmodified**, adding only two new log columns not present in
D5.18–D5.22's scripts:

- `fl_ground_contact_force_mag`: magnitude of the FL leg's net ground
  contact-force vector, read from the already-configured
  `fl_ground_contact_diag` sensor's `force` field (the sensor already
  computed this field every step via `fields=("found","force")`; it
  was simply never logged before). **Bug found and fixed during this
  task:** the FL contact-geom regex (`FL_.*_collision\d*$`) matches
  multiple primaries (hip/thigh/calf collision geoms), giving `force`
  shape `[B, 5, 3]`, not `[B, 1, 3]`; the first implementation read
  only index 0 (a single, usually-uninvolved primary), always
  returning 0.0 regardless of true contact. Fixed by summing the
  per-primary net-force vectors before taking the magnitude. Confirmed
  the fix via direct sensor inspection (values up to ~108 N observed on
  a genuine touchdown step) and by re-running all 6 seeds and
  reconfirming zero mismatches against the existing baseline logs
  (below) — i.e. the bug was purely in the new logging arithmetic, not
  in anything touching simulation state.
- `pos_thigh_force_saturated` / `pos_calf_force_saturated`: boolean
  flags derived from the already-logged actuator-force values.

`max_recovery_steps_override` was left at its existing default (48),
non-binding for all 6 of these seeds (confirmed: every seed's natural
exit elapsed value is ≤ 48, reproducing D5.19/D5.22's exact exit
steps). No FSM, actuator gain, or timing change was made.

## Step 3 — Runs and validation

Ran all 6 seeds headless:

```
uv run python scripts/d5_23_intra_window_observational_audit.py --seed <N> --max-steps 200 \
  --out-dir logs/d5_23_intra_window_observational_audit/seed<N>
```

for N in {0, 1, 2, 3, 4, 8}. Same checkpoint, wire geometry, hip-wire
setting, initial conditions, and sim config as all prior D5.x runs
(only new log columns added; `max_recovery_steps_override` left at
its default).

**Validation:** cross-checked every seed's new trajectory against the
existing D5.19 cap-48 baseline logs on all shared columns (step, phase,
recovery_elapsed, thigh_pos, calf_pos, ctrl targets, base_z,
fl_ground_contact, genuine_recovery_complete) — **zero mismatches
across all 6 seeds**, confirming this purely-observational script
introduces no change to simulated behavior (trigger/stage1/stage2/exit
steps are identical to D5.19/D5.21/D5.22's existing values for every
seed). No viewer run was needed; headless output is authoritative, as
instructed.

## Step 4 — Event-ordering table

All step indices below are the actual simulation `step` column values.
"Δ(exit)" = step − exit step (negative = before exit, 0 = at exit,
positive = after exit).

| Seed | stage2 latch | **exit** | thigh-force saturates | \|thigh_err\| grows ≥0.3 rad past stage2 value | foot_vx_base sign-flips (approach→retreat) | FR/RL/RR first lose contact | base_vz turns downward | base_z begins measurable drop | base_z nadir |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 84 | **88** | 88 (Δ0) | 85 (Δ−3) | 89 (Δ+1) | 84/84/84 (Δ−4) | 87 (Δ−1) | 89 (Δ+1) | 92 (Δ+4) |
| 1 | 86 | **90** | 90 (Δ0) | 88 (Δ−2) | 91 (Δ+1) | 87/87/86 (Δ−3/−3/−4) | 89 (Δ−1) | 91 (Δ+1) | 95 (Δ+5) |
| 2 | 86 | **90** | 90 (Δ0) | 88 (Δ−2) | 91 (Δ+1) | 87/88/86 (Δ−3/−2/−4) | 89 (Δ−1) | 91 (Δ+1) | 95 (Δ+5) |
| 3 | 91 | **95** | 95 (Δ0) | *(never grows ≥0.3 rad)* | 96 (Δ+1) | 97/95/91 (Δ+2/0/−4) | 99 (Δ+4) | *(no measurable drop)* | 95 (Δ0, floor) |
| 4 | 84 | **88** | 88 (Δ0) | *(never grows ≥0.3 rad)* | 89 (Δ+1) | 90/88/84 (Δ+2/0/−4) | 91 (Δ+3) | *(no measurable drop)* | 88 (Δ0, floor) |
| 8 | 83 | **87** | 87 (Δ0) | 84 (Δ−3) | 88 (Δ+1) | 84/83/83 (Δ−3/−4/−4) | 86 (Δ−1) | 88 (Δ+1) | 91 (Δ+4) |

## Analysis

**(1) Tracking-error growth timing:** In the 4 seeds that go on to show
a measurable post-exit drop (0, 1, 2, 8), |thigh_err| grows materially
**2-3 steps before** the exit step itself — i.e. error accumulation is
already well underway during the final few steps of RECOVERY, before
actuator restoration. In the 2 floor-effect seeds (3, 4), tracking
error never grows by as much as 0.3 rad past its stage-2 value at all
in this window — consistent with D5.20-D5.22's observation that these
two seeds have systematically smaller handoff errors.

**(2) Foot-velocity sign flip:** In **all 6 seeds without exception**,
`foot_vx_base` flips from a negative (forward-approach) to a
non-negative (retreating/leveling) sign at **exactly exit+1**, one
step after the phase transition, regardless of whether the seed later
shows a large or zero drop. This is the single most
temporally-consistent event in the entire table — it occurs the step
immediately after actuator restoration in every seed, which is
consistent with (but does not prove) it being a direct, immediate
mechanical consequence of the hard actuator switch itself (the foot's
kinematic trajectory abruptly changes direction once position control
resumes), rather than being downstream of the magnitude of tracking
error (seeds 3 and 4, with near-zero tracking-error growth, show the
identical one-step-delayed sign flip as seeds 0, 1, 2, 8).

**(3) FR/RL/RR contact loss precedes exit, not follows it:** In every
one of the 4 non-floor seeds (0, 1, 2, 8), at least one of FR/RL/RR
**already** loses ground contact 2-4 steps **before** the exit step —
well before actuator restoration. This directly confirms (not just
correlates) that the other-leg contact-loss pattern first flagged in
D5.18/D5.19 is **not** a consequence of the FL handoff event — it is
already underway as part of the normal in-progress gait/recovery
motion while FL is still in RECOVERY. In the 2 floor-effect seeds (3,
4), by contrast, RR loses contact early (Δ−4, same as the other
seeds) but FR and RL do **not** lose contact until **after** the exit
step (Δ+2, Δ0/Δ+2) — i.e. in the seeds with the smallest drop, the
other legs' support is better preserved through the handoff itself,
which is a genuine, seed-conditional difference in whole-body contact
state coincident with (not necessarily caused by) the smaller
tracking error.

**(4) Base vertical velocity:** In the 4 non-floor seeds, `base_vz`
turns downward **1 step before** the exit step — i.e. the base is
already beginning to descend slightly before the actuator handoff
occurs, not purely as a result of it. In the 2 floor-effect seeds, the
downward turn is delayed to 3-4 steps **after** exit and is far milder
(these seeds show no measurable `base_z` drop at all by the ≥0.003 m
threshold used here).

**(5) Drop onset and nadir:** For the 4 non-floor seeds, the
measurable drop begins at exit+1 (the same step as the foot-velocity
sign flip) and reaches nadir at exit+4 or exit+5, consistent across
all 4 — a very tight, repeatable 1-step-onset/4-5-step-nadir pattern.
The 2 floor-effect seeds show no measurable onset at all within this
window.

**(6) Ordering relative to the natural exit event:** Saturation is, by
construction of the actuator-restoration mechanism, always exactly
coincident with the exit step (Δ0) in all 6 seeds — this is not new
information (consistent with D5.18-D5.22) but confirms the saturation
event itself cannot be used to distinguish before/after since it
*defines* part of the exit-step actuator transition. The two events
that are **consistently seen before** the exit step across all 4
non-floor seeds are: (a) tracking-error growth (Δ−2 to Δ−3) and (b)
downward base-vz onset (Δ−1); the two events **consistently seen
after** are: (c) foot-velocity sign flip (Δ+1, universal across all 6
seeds including the floor-effect ones) and (d) measurable base_z-drop
onset (Δ+1, non-floor seeds only).

## Seed-to-seed consistency and exceptions

- The foot-velocity sign-flip-at-exit+1 pattern is the **most
  consistent** finding in this dataset: identical offset in all 6
  seeds regardless of eventual drop magnitude. This argues it is a
  near-deterministic kinematic consequence of the actuator-restoration
  event itself (every seed's leg physically reverses direction the
  step after the hard switch), rather than a graded function of
  tracking-error magnitude.
- The clearest **exception/divergence** between seed groups is in
  FR/RL/RR contact timing: the 4 non-floor seeds lose at least one
  other leg's contact well *before* exit (part of ordinary recovery
  dynamics), while the 2 floor-effect seeds largely retain FR/RL
  contact through and past the exit step. This is the single clearest
  seed-conditional signal distinguishing the two outcome groups in
  this window, independent of tracking-error magnitude per se.
- Tracking-error growth timing (2-3 steps before exit) and base-vz
  downward onset (1 step before exit) are present in all 4 non-floor
  seeds with consistent small offsets, but are absent/delayed in the 2
  floor-effect seeds — consistent with (not proof of) these two
  quantities being linked to each other and to the eventual drop
  magnitude.

## Observation vs. inference

All step indices reported above are direct observations from
simulation-step-indexed CSV columns. The following are explicitly
**inferences**, not observations, and are not proven by this data:
- That the foot-velocity sign flip is *caused by* the actuator
  restoration event (temporal adjacency at Δ+1 in all 6 seeds is
  strong circumstantial evidence, but no counterfactual was run in
  this task to isolate the actuator switch from other co-occurring
  changes).
- That the pre-exit tracking-error growth and pre-exit base-vz onset
  in the 4 non-floor seeds are mechanistically linked to each other
  (they are temporally coincident within this dataset but this task
  did not intervene on either independently).
- That better-preserved FR/RL contact through handoff in the 2
  floor-effect seeds is a *cause* of their smaller drop, rather than
  both being downstream of some other shared factor (e.g., overall
  gait-phase/stride-timing differences between seeds that also happen
  to produce smaller tracking error). Whole-body contact dynamics and
  actuator handoff interact in ways this purely observational design
  cannot decompose.

## Limitations

- n=6 seeds (4 non-floor, 2 floor-effect); no correlation coefficient
  is computed in this report since the task calls for event-ordering
  evidence, not a repeat of D5.19-D5.22's correlational analyses.
- Event-detection thresholds (0.3 rad for "material" error growth,
  0.003 m for "measurable" drop, 0.1 m/s deadband for sign
  classification, −0.05 m/s for "downward" base_vz) are reasonable but
  arbitrary choices; a different threshold could shift some reported
  offsets by 1 step, though the qualitative before/after-exit ordering
  is robust to small threshold changes given the clear step-level gaps
  observed.
- This design cannot separate the actuator-restoration effect from the
  whole-body-contact-dynamics effect, since both occur in the same
  1-2 step window around exit in every seed; no intervention was
  performed to decouple them (per the stop condition for this task).

## Recommended next experiment (exactly one)

Given that the foot-velocity sign-flip is the most seed-consistent,
tightly-timed event (always exactly exit+1) and is not graded by
tracking-error magnitude, while FR/RL contact-preservation-through-
handoff is the clearest signal distinguishing small-drop from
large-drop seeds, the next experiment should be a **purely
observational extension of this same logging to the full existing
D5.19 10-seed cohort** (adding seeds 5, 6, 7, 9 — the cap-timeout
seeds — for the pre-exit window only, since they never reach a
genuine stage-2-to-exit interval to analyze in full) to test whether
the FR/RL-contact-preservation pattern observed here in 2 of 6 seeds
is itself associated with smaller drops across a larger sample, using
only the existing, unmodified recovery FSM and no new cap or timing
intervention. This is **not implemented** in this task, per
instructions to stop after the observational analysis and recommend
rather than execute the next experiment.
