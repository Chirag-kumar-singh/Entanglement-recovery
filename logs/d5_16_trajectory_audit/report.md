# D5.16 — Baseline Recovery Trajectory Audit

Diagnostic only. No controller code, gain, threshold, FSM transition,
checkpoint, or wire geometry was changed. This script
(`scripts/d5_16_trajectory_audit.py`) reuses D5.3's exact env-build path
(`go2_wire_recovery_residual_env_cfg(..., recovery_mode="hybrid", ...)`,
same isolated FL wire, same 48-step recovery cap, same frozen checkpoint)
and the D5.15-corrected viewer stepping pattern, adding diagnostic-only
logging columns (FR/RL/RR ground contact, FL joint velocities,
base-relative foot position, base linear/angular velocity, full base
orientation quaternion, and the `pos_actuators_active` flag already
tracked internally as `hybrid_thigh_pos_gain_active`).

## Files

- New script: [scripts/d5_16_trajectory_audit.py](../../scripts/d5_16_trajectory_audit.py)
  (reuses, does not modify, `scripts/d5_3_extended_recovery_window.py`,
  `scripts/d3_ground_contact_diagnostic.py`, and
  `src/mjlab/tasks/recovery/env_cfg_residual.py`).
- Headless baseline: [logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv](headless_seed0/trajectory.csv)
- Viewer run (verification only): [logs/d5_16_trajectory_audit/viewer_seed0/trajectory.csv](viewer_seed0/trajectory.csv)

## Commands used

```
uv run python scripts/d5_16_trajectory_audit.py --help
uv run python scripts/d5_16_trajectory_audit.py --seed 0 --max-recovery-steps-override 48 \
  --out-dir logs/d5_16_trajectory_audit/headless_seed0
uv run python scripts/d5_16_trajectory_audit.py --seed 0 --max-recovery-steps-override 48 \
  --out-dir logs/d5_16_trajectory_audit/viewer_seed0 --debug
```

## Viewer verification (D5.15 fix carried forward)

The corrected viewer's `trajectory.csv` is **byte-for-byte identical**
(`diff` returns no output, 201 rows each) to the headless run's CSV —
confirming the viewer still steps exactly once per tick for this new
script. The run exited cleanly this time (exit code 0, no teardown
crash observed on this particular run); this is reported separately
from the quantitative findings below since it is a tooling
observation, not a controller result. (Prior D5.15 runs of the same
`NativeMujocoViewer` with other scripts did crash on teardown with
exit codes 134/139/124 after writing complete, correct output — this
run simply did not reproduce that crash, consistent with it being an
intermittent native/display teardown issue unrelated to the scripts'
own stepping logic.)

## Measured phase timing (seed 0)

| Event | Step | Notes |
|---|---|---|
| NORMAL_WALK → RECOVERY (wire contact) | 44→45 (entry row logged at step 44/45 boundary) | `ENTER RECOVERY step=45` |
| Position actuators disabled (`pos_actuators_active`→False) | 44 | Coincides exactly with RECOVERY entry |
| `calf_side_far` latches True (Stage 1, calf clearance) | 84 | `stage1_first_latch_step=84` |
| `foot_forward_dist` crosses 0 (foot reaches far side) | 84 (value 0.0349) | Same step as Stage 1 |
| `foot_crossed_achieved` / `crossing_achieved` latch (Stage 2) | 84 | `stage2_first_latch_step=84` (both stages latch the same step here) |
| `foot_height_above_wire` reaches its maximum | 86 (0.0933 m) | **After** the Stage 1/2 crossing latch at 84 |
| RECOVERY → POST_RECOVERY exit | 88 | `EXIT RECOVERY step=88 elapsed=45 timeout=False`, reason = crossing + release (not cap timeout) |
| Position actuators re-enabled (`pos_actuators_active`→True) | 88 | Exactly at the phase transition |
| First FL ground contact after re-enable | 89 | One step after handoff |
| Global `base_z` minimum (first, larger dip) | 54 (0.2040 m) | ~34% height loss from the pre-entry ~0.309 m baseline |
| Second, smaller `base_z` dip (post-handoff) | 92 (0.2182 m) | Coincides with FL ground touchdown + actuator handoff settling |
| `base_z` returns to pre-entry baseline (~0.307–0.309 m) | ~107–113 | Stays in that band through the rest of the 200-step window |
| Forward velocity (`base_lin_vel_x`) recovers to a walking-like oscillation (~0.2–0.3 m/s) | ~98 onward | After dipping to 0.16 m/s around step 95–96 |

## Analysis

**1. Does the leg retract relative to the robot base after recovery entry?**
Yes, measurably. `foot_x_base` (foot position along the base's own
forward axis, not world/joint-angle-derived) decreases monotonically
from 0.1852 m at step 44 to a minimum of 0.0783 m at step 81 — a genuine
~0.107 m retraction relative to the body, confirming a real retract
motion using foot-relative-to-base displacement (not inferred from
joint-angle sign alone, per the task's explicit requirement). After
step 81 `foot_x_base` increases again as the foot swings forward through
the remainder of the window.

**2. Interval of maximum foot height; is the foot still on the near side
at that point?**
`foot_height_above_wire` peaks at step 86 (0.0933 m above the wire's live
surface). The near→far crossing (`calf_side_far`, `foot_forward_dist>0`)
already latched two steps earlier, at step 84. **The foot is therefore
already on the far side of the wire at its point of maximum height** —
the peak height does not occur before crossing, it occurs just after.

**3. When does the foot cross the wire plane relative to its height peak?**
Crossing (`foot_forward_dist` crossing 0, and `calf_forward_dist`
crossing 0) both occur at step 84, i.e. 2 steps **before** the height
peak (step 86). The crossing is not what triggers the peak; both the
calf and the toe cross the wire plane while the foot is still rising,
and the rise continues for 2 more steps before turning over.

**4. Descent after crossing — does the foot reach the far side before
recovery exits?**
Yes: the foot is already geometrically past the wire (both stage-1
calf-clearance and stage-2 foot-crossing conditions latched) at step 84,
4 steps before the phase actually exits RECOVERY at step 88. The
"descend" portion of the measured trajectory is brief: `foot_height_
above_wire` falls from its 0.0933 m peak (step 86) to 0.0797 m by the
exit step (88) — only 2–3 steps of visible controlled descent inside
RECOVERY before the phase transition hands control back to the position
actuators. There is no separate, FSM-visible "verify clearance" dwell —
the two-stage latch condition (`crossing_achieved` + `release_required_
steps` of sustained no-contact) is what gates the exit, and in this run
`release_counter` reaches its threshold immediately because wire contact
was already false by step 84 (own recovery trajectory, not logged above
but confirmed via `wire_in_contact` column in the CSV).

**5. Body-height drop and timing of FL ground-support-loss relative to
actuator handoff.**
Two distinct `base_z` dips are observed:
  - **Primary, larger dip**: `base_z` collapses from the pre-entry
    baseline (~0.309 m, steps 0–43) to a minimum of 0.2040 m at step 54
    — a ~0.105 m (34%) drop — occurring in the first 10 steps of
    RECOVERY (steps 44→54), i.e. **immediately after and coincident
    with** `pos_actuators_active` flipping to False at step 44. Critically,
    `fl_ground_contact` is **already False** at steps 40–43 (before
    entry) and remains False throughout this entire dip — the FL foot
    was already airborne/mid-swing at the moment of wire contact, so
    this primary sag is NOT attributable to a sudden loss of FL ground
    support; it is timed to the position-actuator disabling itself,
    consistent with D5.10/D5.12's prior finding that the disabling step
    (not the auxiliary commands, and not a ground-contact-loss event)
    is the proximate cause.
  - **Secondary, smaller dip**: `base_z` falls again from its
    in-recovery local peak (0.2819 m, step 84) to a second minimum of
    0.2182 m at step 92 — a ~0.064 m drop — this one occurring 4 steps
    after `pos_actuators_active` returns to True (step 88) and 3 steps
    after FL first touches the ground again post-handoff (step 89).
    This smaller dip looks like a touchdown/settling transient under
    the restored position actuators, not a new support-loss event.

**6. When does normal walking control resume, and does forward motion
recover?**
`pos_actuators_active` returns to True at step 88 (the FSM exit step),
restoring normal position control to the affected leg immediately at
the phase transition — there is no separate blending or ramp. `base_z`
returns to and stabilizes in the pre-entry baseline band (~0.307–0.309 m)
by roughly step 107–113 and remains there through the rest of the
200-step logged window. `base_lin_vel_x` dips to a minimum of ~0.16 m/s
around steps 95–96 (during the settling transient) and then recovers to
an oscillating ~0.2–0.3 m/s band from step ~98 onward, consistent with
normal walking-gait velocity oscillation rather than a stalled or
diverging base velocity. Net forward progress over the full window
(step 0 → step 199, `base_x`) is positive and continues to increase
after recovery (e.g. `base_x` increases from 0.250 at step 130 to 0.307
at step 199), i.e. forward locomotion does resume.

## Reference comparison — Quad-SDK obstructed-swing control law

Inspected read-only (no edits): `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/controllers/underbrush_inverse_dynamics.cpp`
(confirmed to exist in the actual checked-out quad-sdk source tree, on
the `underbrush-study` branch — not merely referenced in a reconstruction
document).

Verified facts from the real source (`computeLegCommandArray`, "Obstructed
swing mode" branch, ~lines 403–462):
- All three joints' `kp` are zeroed by the initializing loop (lines
  406–417), then joint 0 (ab/ad/hip) has its `pos_setpoint`/`vel_setpoint`/
  `kp` explicitly restored from the swing-phase reference trajectory
  (lines 419–426) — i.e. the ab/ad joint continues normal position-style
  tracking, unmodified by the obstruction logic.
- Joint 1 (hip-pitch/"thigh" in mjlab's naming) gets a constant
  `vel_setpoint = -retract_vel_` (line 430), `kd = swing_kd_[1]`
  (set by the earlier loop), `kp = 0`, `torque_ff = 0` — a pure
  damped-velocity command with a single constant target, for the
  entire duration the leg remains in `force_mode_` (obstructed).
- Joint 2 (knee/"calf") gets `vel_setpoint = 0`, `kd = 0` (line 446,
  explicitly overriding the initializing loop's `kd`), and
  `torque_ff = -tau_push_` (line 449) — a pure open-loop constant
  feedforward torque, no feedback at all.
- This matches mjlab's `_apply_hybrid_actuators` implementation exactly
  in structure: ab/ad continues on the frozen walking policy's own
  position command; thigh is a pure damped-velocity actuator with one
  constant target; calf is a pure open-loop torque actuator — all
  **genuinely verified against the actual quad-sdk C++ source**, not
  assumed from the `QUAD_SDK_UNDERBRUSH_TECHNICAL_RECONSTRUCTION.md`
  document alone (that document's cited line numbers/snippets were
  cross-checked against the real file and match).

**Important divergence — no scripted RETRACT→ADVANCE HIGH→DESCEND→VERIFY
CLEARANCE→RESUME phase sequence exists in the quad-sdk source.** The
"Obstructed swing mode" branch is a single, constant-command mode
(binary with "Usual swing mode", switched by `force_mode_` based on a
force/torque threshold and minimum-switch-time timers — lines 345–381),
held unchanged until either sufficient time elapses, insufficient swing
time remains, or the leg transitions to stance. There is no multiple-stage
internal state machine with distinct RETRACT/ADVANCE/DESCEND/VERIFY
states anywhere in this file. mjlab's own `recovery_mode="hybrid"`
implementation mirrors this directly: a single constant thigh-velocity
and calf-torque command held for the whole RECOVERY phase (confirmed via
`_apply_hybrid_actuators`'s `thigh_vel_cmd`/`calf_torque_cmd` being
constants gated only by `in_recovery`, not by any internal stage), with
the module's own docstring explicitly stating "intentionally NOT a
multi-stage scripted trajectory." The RETRACT→ADVANCE HIGH→DESCEND→VERIFY
CLEARANCE→RESUME sequence named in this task's objective is therefore
**not a phase structure present in either the quad-sdk reference or the
current mjlab controller** — it should be understood as a *conceptual/
descriptive* framing of what emerges from the constant commands plus
gait-phase coupling (confirmed in the Measured phase timing table
above: a retract interval, a rise interval, a crossing event, a brief
post-crossing descent, and a resume), not as literal FSM states in
either codebase. Stating this explicitly per the task's instruction
rather than inventing an explicit phase match.

## Distinguishing measured facts from interpretation

- **Facts** (directly read from the CSV / FSM state, not inferred):
  all step numbers and values in the Measured phase timing table;
  `foot_x_base`'s minimum and monotonic decrease steps 44→81;
  `pos_actuators_active`'s exact True/False transition steps;
  `fl_ground_contact`'s False state throughout steps 40–88; the two
  distinct `base_z` local minima and their step indices.
- **Interpretations** (explicitly flagged as such above): that the
  primary sag is "timed to" the actuator-disabling step rather than a
  ground-support-loss event (supported by, but not purely definable
  from, the fact that `fl_ground_contact` was already False before and
  during the sag — consistent with, not newly proving beyond, D5.10/
  D5.12's prior causal finding); that the secondary dip is a "touchdown/
  settling transient" (inferred from its timing relative to ground
  contact and actuator handoff, not independently isolated/ablated in
  this audit); and the entire "Reference comparison" discussion's framing
  of the named 5-stage sequence as "conceptual, not literal FSM states."

## Recommendation — exactly one next controller experiment

**Isolate whether the primary (larger) base-height sag at RECOVERY entry
(steps 44→54, ~0.105 m drop) is caused by the position-actuator disabling
step itself acting on a leg that is already mid-swing (airborne,
`fl_ground_contact=False`) at the moment of entanglement, by comparing
this baseline against an otherwise-identical run where RECOVERY entry is
deferred (not disabled, just the FSM transition delayed) until the FIRST
step at which the frozen walking policy's own gait would have put the FL
foot in stance (`fl_ground_contact=True`) — i.e. test whether disabling
the position actuators while the leg has ground support (rather than
mid-air) measurably reduces or eliminates the primary sag.** This is a
natural, minimal follow-up to D5.10/D5.12/D5.13's already-established
"the disabling step itself, not the auxiliary commands, causes the sag"
finding, and to this audit's new fact that the leg was already airborne
at the moment of disabling in this baseline run — it tests a specific,
falsifiable hypothesis (stance-vs-swing-phase disabling) that this audit
surfaced but did not itself test, using only a trigger-timing change
(no gain/threshold/blending change).

This report concludes the D5.16 audit. Per the task instructions, the
above experiment has **not** been implemented — stopping here.
