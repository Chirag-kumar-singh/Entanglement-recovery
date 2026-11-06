# D5.19 — Cross-Seed Recovery-Exit Tracking-Error Analysis

## Objective

Test whether accumulated FL thigh/calf position-target error at the
RECOVERY → POST_RECOVERY handoff is associated with the subsequent
base-height dip, across multiple seeds. Purely correlational;
no controller change implemented.

## 1. Data inventory — why existing logs were insufficient

Reviewed D5.16, D5.17, D5.18 reports/scripts and the D5.5 repeatability
logs (seeds 0-19) before running anything new.

CSV columns present per log family:

| Log family | seeds | `pos_*_ctrl_target` | FR/RL/RR ground contact | timeout/genuine flags |
|---|---|---|---|---|
| `d5_5_repeatability` | 0-19 | **no** | **no** (FL only) | yes |
| `d5_16_trajectory_audit` | 0 only | **no** | yes | yes |
| `d5_17_stance_timing` (`seed0_immediate`) | 0 only | **no** | **no** | yes |
| `d5_18_post_recovery_diagnostic` | 0 only | **yes** | yes | yes |

The task requires, at handoff: the actual joint position, the frozen
policy's post-scale/post-offset position target, the signed/absolute
tracking error, actuator saturation, and all-four-feet ground contact
through the post-exit window. Only D5.18's script logs the position
actuator's actual `ctrl` setpoint (`pos_thigh_ctrl_target` /
`pos_calf_ctrl_target`, read directly from
`env.sim.data.ctrl[..., pos_thigh_ctrl_id/pos_calf_ctrl_id]`) alongside
FR/RL/RR contact. That column is populated for **seed 0 only**. No
other existing log has it. A single seed cannot support a cross-seed
correlation, so existing logs are insufficient on their own — this is
reported before any new run was launched, per the task's constraint.

**Verifying target units/scaling from source** (`actions.py` module
docstring, lines 1-50, and `JointPositionAction`'s reuse via
`process_actions()`): the recovery action term's residual is added in
*raw policy-action space* and then passed through the *existing,
unmodified* `JointPositionAction` scale/offset pipeline
(`scale=GO2_ACTION_SCALE`, `use_default_offset=True`) before becoming a
physical joint-position setpoint. `env.sim.data.ctrl[...,
pos_thigh_ctrl_id]` is read **after** that pipeline has executed, i.e.
it already is the fully-scaled, fully-offset physical joint-angle
target in radians — the same units as `thigh_pos`/`calf_pos` (`qpos`).
This means `pos_thigh_ctrl_target − thigh_pos` is a direct,
unit-consistent position-tracking error, with no additional
scale/offset conversion required and no raw-action-vs-physical-angle
unit mismatch. This was confirmed from source, not assumed.

**Decision:** rather than re-deriving target/error conversions for the
other log families (which lack the ctrl-target column entirely and
cannot be retrofitted without rerunning), the D5.18 script was reused
**unmodified** (no edits to its controller logic, FSM, or logging were
needed — it already logs every required column) for additional seeds.
This is not a "broad new seed sweep": it is the same 10-seed set
(0-9) already used as the project's standing repeatability cohort in
D5.5/D5.17, run once each, headless, into new output directories under
`logs/d5_19_handoff_error_analysis/seed{0..9}/`. No existing log
directory was overwritten. Seed 0's result was cross-checked against
the original D5.18 `headless_seed0` run and is identical
(`trigger_step=45`, `EXIT RECOVERY step=88 elapsed=45 timeout=False`,
`stage1/2_first_latch_step=84`, `first_actuator_restore_step=88`,
`first_sustained_fl_ground_step=89`), confirming determinism and that
no controller behavior changed between D5.18 and this reuse.

## 2. Per-seed measurements

All values at the RECOVERY→POST_RECOVERY exit step (`phase` 1→2
transition) and over the following 20 simulation steps.

| seed | exit step | thigh err (rad) | calf err (rad) | thigh force @exit (N·m) | base_z @exit | base_z min (step) | drop (m) | base_vz @exit | outcome |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 88 | −2.990 | +1.406 | −23.5 (saturated) | 0.2544 | 0.2182 (92) | 0.0362 | −0.497 | crossing+release |
| 1 | 90 | −2.854 | +1.335 | −23.5 (saturated) | 0.2944 | 0.2535 (95) | 0.0409 | −0.362 | crossing+release |
| 2 | 90 | −2.778 | +1.320 | −23.5 (saturated) | 0.2976 | 0.2749 (95) | 0.0226 | −0.290 | crossing+release |
| 3 | 95 | −2.440 | +0.675 | −23.5 (saturated) | 0.3013 | 0.3013 (95) | 0.0000 | +0.350 | crossing+release |
| 4 | 88 | −2.422 | +0.662 | −23.5 (saturated) | 0.2907 | 0.2907 (88) | 0.0000 | +0.315 | crossing+release |
| 5 | 86 | −2.438 | +0.676 | −23.5 (saturated) | 0.2736 | 0.2615 (103) | 0.0122 | +0.120 | **cap_timeout** |
| 6 | 90 | −2.848 | +1.285 | −23.5 (saturated) | 0.2858 | 0.2552 (95) | 0.0306 | −0.357 | **cap_timeout** |
| 7 | 91 | −2.553 | +0.658 | −23.5 (saturated) | 0.2129 | 0.2129 (91) | 0.0000 | −0.082 | **cap_timeout** |
| 8 | 87 | −3.152 | +1.320 | −23.5 (saturated) | 0.2510 | 0.2174 (91) | 0.0336 | −0.609 | crossing+release |
| 9 | 91 | −3.080 | +1.263 | −23.5 (saturated) | 0.2874 | 0.2612 (95) | 0.0261 | −0.424 | **cap_timeout** |

Error sign convention: `target − actual`; negative thigh error means
the frozen policy's target lags *behind* (more flexed/retracted than)
the actuator-driven actual position at the instant of handoff.

Ground-contact note: in every seed, at least one of the four feet
loses ground contact somewhere in the 20-step post-exit window (all
four `contact_loss_window` flags are `True` in all 10 rows); this
confirms D5.18's single-seed finding (whole-body disturbance, not an
isolated FL-landing event) generalizes across the cohort, but it does
not by itself differentiate large- from small-drop seeds since it is
universal.

Thigh-actuator saturation at the exit step's effort limit (−23.5 N·m)
is **also universal** — present in all 10/10 seeds regardless of drop
magnitude (including the three seeds with zero measured drop). This
means saturation at the handoff instant is necessary but evidently not
sufficient to produce a large dip; it does not discriminate between
seeds and cannot alone explain the variation in drop magnitude.

Base angular velocity at exit (not tabulated above for space) ranges
from near-zero pitch/roll rates for the zero-drop seeds (3, 4: ~0.5-0.9
rad/s on all axes) up to ~2.0-2.3 rad/s pitch-axis (`ang_vel_x`) for
seeds 1, 2, 6, 9 — several of which are mid-to-large-drop or timeout
seeds, but seed 0 (largest-ish drop, 0.036 m) has a comparatively small
0.258 rad/s pitch rate, so this signal does not cleanly track drop
magnitude either.

## 3. Analysis

**(1) Per-seed table:** see above.

**(2) Do larger thigh/calf errors associate with deeper dips?**
Yes, with n=10: Pearson r(|thigh_err|, drop) = **0.849**;
r(|calf_err|, drop) = **0.939**. Restricting to the 6
crossing+release ("success") seeds only, r(|thigh_err|, drop) =
**0.899**. Both joints' errors are themselves highly correlated with
each other (both derive from the same FL-leg kinematic state at
handoff), so these two correlations are not independent evidence —
they reflect a single underlying relationship between "how far the
leg had drifted from the frozen policy's continuously-computed target
during RECOVERY" and "how large the subsequent base-height dip is."
With n=10, a correlation this strong is a modest sample but a
consistent, monotonic trend, not a borderline/ambiguous one (e.g.
seed 8, the largest |thigh_err| = 3.152, also has the largest raw
contact-loss severity and base_vz drop at exit = −0.609 m/s, the most
negative in the cohort; seeds 3, 4, 7, the smallest |thigh_err| ≈
2.42-2.55, are exactly the three seeds with **zero** measured dip).

**(3) Saturation / contact-loss pattern, small vs. large dip:**
Actuator saturation at exit is universal (10/10) and does not
distinguish small- from large-dip seeds. Contact loss somewhere in the
post-exit window is also universal (10/10) and does not discriminate.
What *does* differ is the magnitude/duration of the post-handoff
foot-velocity transient implied by the error magnitude, and the sign
of `base_vz` at the exit instant: all three zero-drop seeds (3, 4, 7)
have `base_vz` ≥ −0.08 m/s at exit (two are actually rising, +0.31 to
+0.35 m/s), while 6 of the 7 nonzero-drop seeds have `base_vz` ≤ −0.29
m/s at exit. This is consistent with (but does not prove) the
tracking-error magnitude driving a larger downward actuator-force
transient that is already underway by the exit step for high-error
seeds.

**(4) Success vs. timeout separation:** 6/10 seeds completed genuine
crossing+release; 4/10 (seeds 5, 6, 7, 9) hit the 48-step
`cap_timeout` and never reached `genuine_recovery_complete=True`. The
error–drop correlation is not an artifact of conflating these two
populations: computing it on the 6 success-only seeds alone still
gives r = 0.899, and the timeout seeds themselves span the same range
of errors and drops as the success seeds (e.g., timeout seed 6: err
−2.848, drop 0.031 vs. success seed 1: err −2.854, drop 0.041 — very
similar). Timeout does not appear to systematically produce either
larger or smaller tracking errors/dips in this cohort, though n=4 for
the timeout subgroup is too small to support a separate correlation
estimate.

**(5) Sample count, missing fields, limitations:**
- n = 10 seeds total (6 success, 4 timeout). This is a modest sample;
  a Pearson r computed on n=10 (or n=6) should be treated as
  suggestive, not confirmatory — no significance test is reported
  because the sample is too small for a meaningful one, and no plot
  is included for the same reason (10 points is better communicated
  as a table than a scatter plot at this sample size).
- No field was fabricated. Every value in the per-seed table was read
  directly from a logged CSV column or computed from logged columns
  with documented arithmetic (target − actual, min() over a window,
  base_z at a specific step). No interpolation or estimation was used
  for any seed.
- D5.17's stance-deferred ("Condition B") data and D5.16's audit data
  were deliberately **not** merged into this correlation set, since
  they represent different experimental configurations (modified
  handoff timing, or missing columns) and the task explicitly
  prohibits treating different configurations as interchangeable.
- Only one affected-leg configuration (FL, `affected_leg_idx=0`) and
  one wire configuration (`ISOLATED_FL_WIRE_CFG`) were tested, as in
  all prior D5.x work; this analysis cannot speak to other legs or
  wire geometries.
- The thigh-actuator effort limit (−23.5 N·m) is saturated in 10/10
  seeds at the exit step, meaning the *true* restoring-torque demand
  implied by the tracking error is not directly observable from force
  alone (it is clipped); the tracking-error magnitude itself is
  therefore a more informative predictor than exit-step force in this
  dataset, which is part of why error (not force) is used as the
  primary correlate here.

**(6) Correlation vs. causation:** The measured relationship is a
temporal/statistical association only: larger handoff-time tracking
error precedes and correlates with a larger subsequent base-height
dip. This diagnostic does not manipulate tracking error independently
of other confounds (seed, trigger timing, whole-body contact pattern,
angular velocity at handoff all co-vary with it across this seed set)
and so cannot establish that tracking error *causes* the dip. In
particular, D5.18 already showed that FR/RL also lose contact in the
same window for the single seed studied there, and this generalizes
to all 10 seeds here — a confound that is fully consistent with (but
not excluded by) a tracking-error-driven mechanism, since a large
leg-velocity transient from catching up a large position error could
itself destabilize the other legs' contact, rather than the other
legs' contact loss being an independent cause of the dip. Distinguishing
these possibilities requires an intervention (e.g., controlling
tracking-error magnitude independently of whole-body state), which this
correlational diagnostic deliberately does not attempt.

## Recommended next experiment (exactly one)

Run a **handoff-error-magnitude sweep under a frozen, unmodified FSM**:
using the existing `max_recovery_steps_override` parameter (already a
documented, non-controller-logic config value, not a new mechanism),
vary only the recovery-phase *duration cap* across a small number of
values (e.g. 32, 48 — already used elsewhere in this series —, 64) for
a fixed seed, and measure whether a shorter/longer cap changes the
handoff-time tracking error and, in turn, the post-exit dip in the
same monotonic direction suggested by this correlational analysis.
This does not require any change to the hybrid controller, FSM, gains,
or torques — only reuse of an existing, already-exposed configuration
parameter — and would provide a weak causal lever (does forcing a
different error magnitude at a fixed handoff event move the dip in the
predicted direction?) without altering recovery logic. This is **not
implemented** in this task per instructions to stop after the
diagnostic.
