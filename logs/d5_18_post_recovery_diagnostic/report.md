# D5.18 — Post-Recovery Handoff and Landing Diagnostic

Diagnostic only. No production controller file was edited; no actuator
blending, delayed handoff, extra hold, or modified recovery command was
introduced. This script
(`scripts/d5_18_post_recovery_diagnostic.py`) reuses D5.3's exact
`go2_wire_recovery_residual_env_cfg(..., recovery_mode="hybrid", ...)`
build path (same isolated FL wire, same 48-step recovery cap, same
frozen checkpoint) with **no monkeypatch of any kind** on the action
term — it only adds purely observational sensors/columns (all four
legs' ground contact, and the actual per-step position-actuator CTRL
target) on top of the unmodified, immediate-entry baseline.

## Files

- New script: [scripts/d5_18_post_recovery_diagnostic.py](../../scripts/d5_18_post_recovery_diagnostic.py)
- Headless baseline: [logs/d5_18_post_recovery_diagnostic/headless_seed0/trajectory.csv](headless_seed0/trajectory.csv)
- Viewer run: [logs/d5_18_post_recovery_diagnostic/viewer_seed0/trajectory.csv](viewer_seed0/trajectory.csv)

## Before running

Inspected [logs/d5_16_trajectory_audit/report.md](../d5_16_trajectory_audit/report.md)
and [logs/d5_17_stance_timing/report.md](../d5_17_stance_timing/report.md)
plus their source scripts. D5.16 established the existence of this
secondary dip (0.064 m, step 92, 4 steps after actuator restoration at
step 88); D5.17 showed a deeper version of the same dip under
stance-deferred handoff timing, but per this task's explicit
instruction, this diagnostic uses **only** the immediate-entry baseline.

Confirmed via direct source reading of `_apply_hybrid_actuators` and
`_update_fsm` (`src/mjlab/tasks/recovery/mdp/actions.py`) the exact
restoration mechanism: on `self.newly_recovered` (set by `_update_fsm`
the instant `genuine_recovery_complete | cap_reached`), the position
actuators' `actuator_gainprm`/`actuator_biasprm` are restored to their
ORIGINAL (pre-recovery) captured values in the SAME step the phase
transitions to `PHASE_POST_RECOVERY`, and the auxiliary actuators are
simultaneously disabled (gain zeroed) — this is a single, hard,
same-step switch with no blending, confirmed identical in the new CSV's
`pos_actuators_active` column (False through step 87, True from step 88
onward) and cross-checked directly against `hybrid_thigh_pos_gain_active`.

## Measured dense window (seed 0, steps 78–119; recovery exit at step 88)

Full data in the CSV; key columns summarized:

| step | phase | pos_act | fl/fr/rl/rr ground | base_z | base_vz (lin_vel_z) | pos_thigh_force | foot_vz_base |
|---|---|---|---|---|---|---|---|
| 83 | RECOVERY | False | F,T,T,F | 0.2790 | 0.291 | 0.00 | 1.291 |
| 84 | RECOVERY | False | F,F,F,F | 0.2819 (local peak) | 0.141 | 0.00 | 0.745 |
| 87 | RECOVERY | False | F,F,F,T | 0.2657 | -0.486 | 0.00 | -0.562 |
| **88** | **POST_RECOVERY (exit)** | **True (restored this step)** | F,F,F,T | 0.2544 | **-0.497** | **-23.50 (saturated)** | **-4.342** |
| 89 | POST_RECOVERY | True | T,F,F,T | 0.2424 | -0.542 | -5.52 | -5.904 |
| 90 | POST_RECOVERY | True | T,F,F,T | 0.2311 | -0.461 | -17.38 | -0.021 |
| 91 | POST_RECOVERY | True | T,T,T,F | 0.2217 | -0.313 | -23.50 | 0.734 |
| **92** | POST_RECOVERY | True | T,T,T,F | **0.2182 (nadir)** | **0.008** | -23.50 | 0.658 |
| 93 | POST_RECOVERY | True | T,T,T,F | 0.2208 | 0.279 | -23.40 | 0.778 |
| 94 | POST_RECOVERY | True | F,T,T,F | 0.2279 | 0.439 | -8.65 | 0.260 |
| 97 | POST_RECOVERY | True | F,T,T,F | 0.2564 | 0.455 | 6.01 | -0.268 |
| 98 | POST_RECOVERY | True | T,T,T,T (all four, first time) | 0.2643 | 0.377 | 3.37 | -0.105 |
| 107 | POST_RECOVERY | True | — | 0.2997 (within 0.01 of pre-entry baseline) | 0.127 | -0.51 | -0.001 |

Event markers (from this run's `run_log.txt`):
```
trigger_step=45
EXIT RECOVERY step=88 elapsed=45 timeout=False
stage1_first_latch_step=84
stage2_first_latch_step=84
first_actuator_restore_step=88
first_sustained_fl_ground_step=89
```

## Analysis

**1. Secondary dip — magnitude, timing, duration.**
- Base height at recovery exit (step 88): 0.2544 m.
- Post-exit nadir: **0.2182 m at step 92** — a **0.0362 m** drop
  relative to the exit-step height (distinct from, and smaller than,
  the PRIMARY RECOVERY-entry sag already characterized in D5.16; this
  is specifically the secondary, post-exit dip).
- Duration: the dip bottoms 4 steps after exit (step 92) and the base
  height does not return to within 0.01 m of the pre-entry baseline
  (~0.3084 m) until **step 107** — i.e. roughly 19 steps (≈0.38 s)
  elapse between RECOVERY exit and full height recovery, though the
  sharpest part of the dip (exit → nadir → partial recovery) resolves
  within the first ~10 of those steps.

**2. Does actuator restoration precede, follow, or merely coincide with the dip?**
**Precedes.** `pos_actuators_active` becomes True at step 88 — the SAME
step RECOVERY exits (confirmed above, a hard same-step switch per
source). The dip's nadir occurs 4 steps later, at step 92. The dip
therefore cannot be said to CAUSE the restoration (temporally
impossible — restoration happens first), but restoration PRECEDING the
dip is consistent with, though does not by itself prove, a causal role.
A more specific, directly measured fact: at the exact restoration step
(88), `pos_thigh_actuator_force_global` jumps from 0.00 (disabled, step
87) to **-23.5 N·m — the thigh actuator's effort limit, i.e.
immediately saturated** — and `foot_vz_base` (foot velocity relative to
the base) spikes to **-4.34 m/s** at step 88 and **-5.90 m/s** at step
89, versus a smooth ~0.6–1.3 m/s range in the surrounding RECOVERY-phase
steps. This is a genuine, measured DISCONTINUITY in actuator force and
foot velocity coincident with the restoration step — not inferred, but
read directly from the CSV. The underlying mechanism (confirmed by
comparing `pos_thigh_ctrl_target` against actual `thigh_pos`): the
frozen walking policy's own position TARGET for the thigh joint had
continued to be computed throughout RECOVERY (visible in the
`pos_thigh_ctrl_target` column, which changes gradually and
continuously across the whole window) but was never tracked while the
position actuator was disabled; by step 87 the ACTUAL thigh joint angle
(2.328 rad, driven by the auxiliary velocity command) had diverged
dramatically from the policy's target (-0.678 rad) — restoring the
position gain at step 88 instantly exposes this ~3 rad accumulated
tracking error to the (unchanged) PD gains, producing the force
saturation and velocity spike. This is a measured fact about WHAT
happens at handoff, not a claim about why the dip specifically bottoms
4 steps later.

**3. Does FL touchdown precede the dip, and do other legs lose support simultaneously?**
FL ground contact is first observed at step 89 (one step after
restoration, three steps before the nadir at step 92) — so FL touchdown
DOES precede the dip's nadir, consistent with a touchdown-related
component. However, the four-leg contact table shows a **broader,
whole-body contact disruption**, not an isolated FL event: FR and RL
BOTH lose ground contact between steps 84–90 (FR: False from step 84
through 90; RL: False from step 82/84 through 90) — i.e. during the last
few RECOVERY steps AND through the handoff, the robot has very little or
no ground support on any of the three non-affected... wait, RR retains
contact through most of this window (True at steps 87, 88, 89, 90) while
FR/RL are the ones that drop out. All four legs are not simultaneously
grounded again until step 98 — 10 steps after exit. This measured
pattern (FR/RL losing contact in the SAME window FL is regaining it) is
consistent with a genuine whole-body pitch/roll disturbance around the
handoff, not simply "FL lands, impact, recovery" — but establishing the
root mechanical cause of why FR/RL lose contact in this window would
require additional instrumentation (e.g. full-body torque/COM analysis)
beyond this diagnostic's scope.

**4. Comparison with D5.17's existing immediate-condition logs.**
Per the task's explicit instruction, **stance-deferred recovery was NOT
rerun**. Instead, this run's full trajectory was directly compared,
column-by-column, against the ALREADY-EXISTING
`logs/d5_17_stance_timing/seed0_immediate/trajectory.csv` (D5.17's
Condition A, immediate-entry baseline): comparing `step`, `phase`,
`pos_actuators_active`, `fl_ground_contact`, `fr_wire_contact`,
`thigh_pos`, `calf_pos`, `base_z`, and `base_lin_vel_x` across all 200
rows yields **zero mismatches** — confirming this diagnostic's baseline
is the exact same physical trajectory D5.17 already used as its
Condition A reference, not a subtly different build. (D5.16's secondary
dip of 0.0638 m at step 92, computed relative to the D5.16 EXIT-step-88
row (0.2819 was actually the local in-recovery peak at step 84, not the
exit value) differs slightly from this report's 0.0362 m because D5.16
measured the drop from its in-recovery local peak rather than from the
exit-step height specifically — both are measured facts about the same
underlying trajectory, just referenced to different baseline points;
this report's number is the one directly relevant to "the dip relative
to the height AT recovery exit," per this task's specific wording.)

**5. Measured correlation vs. causal claim (explicit distinction).**
**Measured, directly observed facts:** actuator restoration and the
instantaneous force/velocity discontinuity both occur at step 88, the
exact same step RECOVERY exits; the base-height nadir occurs 4 steps
later (step 92); FL first touches ground at step 89, 3 steps before the
nadir; FR and RL both lose ground contact in the window spanning roughly
steps 82–90, overlapping with the handoff. **NOT claimed as proven by
this diagnostic alone:** that actuator restoration CAUSES the secondary
dip. The temporal ordering (restoration before dip) is necessary but not
sufficient for a causal claim, and the simultaneous FR/RL contact loss
indicates other possible contributing factors (e.g. whole-body pitch
momentum from the preceding RECOVERY-phase foot motion, independent of
the specific handoff mechanism) that this single-seed, single-condition
diagnostic cannot rule in or out. Establishing causation would require
a controlled counterfactual (e.g. comparing against a hypothetical
smooth/ramped handoff) — explicitly out of scope here per the
constraints ("no actuator blending... or modified recovery commands").

## Viewer validation

Commands used:
```
uv run python scripts/d5_18_post_recovery_diagnostic.py --help
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 \
  --out-dir logs/d5_18_post_recovery_diagnostic/headless_seed0
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 0 \
  --out-dir logs/d5_18_post_recovery_diagnostic/viewer_seed0 --debug
```

The viewer run used the D5.15/D5.16/D5.17-corrected stepping pattern
(callback never calls `env.step()` itself). Its `trajectory.csv` is
**byte-for-byte identical** to the headless run's CSV (`diff` returns no
output, 200/200 rows) — confirming correct one-step-per-tick behavior.
**The viewer process crashed on teardown with exit code 139 (SIGSEGV)**,
AFTER completing all 200 steps and writing a complete, correct CSV —
reported here separately from the quantitative findings, consistent
with the known, pre-existing, intermittent native-viewer teardown issue
already documented in D5.15/D5.16/D5.17 (unrelated to this script's own
stepping logic).

## Recommendation — exactly one next experiment

**Measure whether the accumulated thigh-target tracking error at handoff
(the ~3 rad gap between the frozen policy's continuously-updated
`pos_thigh_ctrl_target` and the actual auxiliary-actuator-driven
`thigh_pos`, both already logged by this script) is the dominant driver
of the force/velocity discontinuity at step 88, by instrumenting (NOT
modifying) the existing `_apply_hybrid_actuators`/`process_actions` path
to additionally log, every RECOVERY step, what the frozen policy's
target WOULD be if it were being tracked live (already available as
`pos_thigh_ctrl_target`) alongside its continuously-growing divergence
from `thigh_pos` — then, across the existing D5.5 repeatability seed set
(0–9), check whether seeds with a LARGER accumulated divergence at exit
show a correspondingly LARGER secondary dip, and seeds with a SMALLER
divergence show a smaller or absent dip.** This is a pure
cross-sectional correlational analysis of already-loggable quantities
(no new control law, no blending, no handoff-timing change) that would
either support or weaken the "accumulated tracking-error exposure causes
the dip" hypothesis this diagnostic's measurements are consistent with
but do not themselves prove, before considering any controller-level
mitigation (e.g. a smooth re-engagement blend) that the task's
constraints explicitly forbid implementing in this round.

This report concludes the D5.18 diagnostic. Per the task instructions,
no controller change has been implemented — stopping here.
