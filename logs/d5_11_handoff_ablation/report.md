# D5.11 — Diagnose the hybrid recovery activation transient

## Setup

New, isolated script: [scripts/d5_11_handoff_ablation.py](scripts/d5_11_handoff_ablation.py). Verified via
`--help` before running (flags: `--seed`, `--device`, `--max-steps`,
`--condition {baseline,ramped}`, `--ramp-steps`, `--out-dir`,
`--debug`/`--no-debug`). Reuses `build_d510_env` from D5.10 unmodified,
so Condition A here is identical to D5.10's Condition A (same
checkpoint, wire geometry, sim settings, `auto_reset=False` stop
safeguard, 4-leg ground-contact sensors).

**Condition A (baseline):** the real, unmodified `_apply_hybrid_actuators`
runs exactly as in D5.10/every prior D5.x script — position actuators
disabled and full-magnitude auxiliary commands (`thigh_vel_cmd=+2.0`,
`calf_torque_cmd=-5.0`) applied on the very first RECOVERY step.

**Condition B (ramped):** the real `_apply_hybrid_actuators` is called
first and unmodified (identical position-actuator disabling/enabling,
gain toggling, peak-detector bookkeeping) — then a thin wrapper scales
the two already-applied auxiliary commands by
`ramp_factor = clip((recovery_elapsed + 1) / ramp_steps, 0, 1)` and
re-issues `set_joint_velocity_target`/`set_joint_effort_target` with the
scaled values. `ramp_steps=5` was chosen as the experimental parameter
(≈0.1 s at the project's 0.02 s control step) — explicitly documented in
the script as unvalidated, not a tuned value.

Verified commanded ramp shape (seed 0, entry at step 45): thigh command
rose 0.8 → 1.2 → 1.6 → 2.0 over steps 44-47 (full magnitude reached 3
steps after entry), calf command rose −2.0 → −3.0 → −4.0 → −5.0 over the
same steps — confirming the ramp is applied as intended.

## Exact commands used

Headless comparison (run and verified first):
```
uv run python scripts/d5_11_handoff_ablation.py --seed 0 --max-steps 150 --condition baseline --out-dir logs/d5_11_handoff_ablation/seed0_condA_baseline
uv run python scripts/d5_11_handoff_ablation.py --seed 0 --max-steps 150 --condition ramped --ramp-steps 5 --out-dir logs/d5_11_handoff_ablation/seed0_condB_ramped5
```

Viewer run of the ramped condition (launched after headless logs were verified):
```
uv run python scripts/d5_11_handoff_ablation.py --seed 0 --max-steps 150 --condition ramped --ramp-steps 5 --out-dir logs/d5_11_handoff_ablation/_viz_seed0_ramped5 --debug
```

## Headless results (seed 0)

| | entry `base_z` | min `base_z` (steps 45-75) | step of min | steps to min | total `fl_ground_contact=False` steps (45-75) |
|---|---|---|---|---|---|
| A baseline | 0.3077 | 0.2040 | 54 | 9 | 31 of 31 steps |
| B ramped(5) | 0.3084 | 0.2046 | 54 | 9 | 31 of 31 steps |

The base-height drop (≈0.104 m), the step at which the minimum occurs,
the time-to-minimum, and the complete loss of FL ground contact for the
entire 45-75 window are **essentially identical** between conditions —
ramping the auxiliary commands over 5 steps produced no measurable
change to the initial sag in this seed.

Downstream FSM outcome did differ slightly: Condition A reached
Stage-1/Stage-2 latch at step 84 and exited via genuine
crossing+release at step 88; Condition B (ramped) reached latch
one step later (step 85/86) and exited one step later via
crossing+release at step 90 — a small delay consistent with the
slower torque ramp-up delaying when the leg geometry crosses the
latch thresholds, not a qualitative change in outcome (recovery still
succeeded in both).

## Viewer run — reported separately, not used as evidence

The viewer run of the ramped condition completed all 150 steps and
wrote a valid `trajectory.csv`, but:

- **It diverged numerically from the headless run of the same
  condition/seed/ramp**: RECOVERY exited at step 45 with Stage-1 latch
  at step 43 in the viewer run, versus RECOVERY exiting at step 90 with
  Stage-1 latch at step 85 in the headless run. The "ENTER RECOVERY"
  marker was also never logged in the viewer run, suggesting the
  viewer's own stepping/reset handling does not align with this
  script's step-counting the same way the headless loop does.
- **The process crashed on exit** after writing output, with exit code
  134 (SIGABRT) and the message `malloc(): unsorted double linked list
  corrupted` — a native-level memory-corruption abort during viewer
  teardown, not a Python exception.

Per the task's instructions, this divergence and crash are reported as
their own finding and are **not** used as a substitute for the headless
numerical comparison above. The viewer could not be used for reliable
visual inspection of the onset behavior in this environment; the
headless CSVs remain the sole evidential basis for this report's
conclusions.

## Interpretation

Ramping the auxiliary thigh-velocity/calf-torque commands from zero
over 5 steps did **not** reduce the initial base-height sag, did not
change when or whether ground support was lost, and did not improve FL
clearance — on this seed, outcomes were essentially indistinguishable
between an instantaneous and a ramped onset of the auxiliary commands.

This is a single-seed, single-ramp-duration result and does not rule out
a longer ramp mattering, but it weighs against "abrupt auxiliary-command
onset" being the dominant contributor to D5.10's sag finding.

**Distinguishing the two still-conflated effects:** D5.10 compared
"override fully active" vs. "override fully disabled" (which also
removes the position-actuator disabling). D5.11 isolates *only* the
auxiliary-command onset profile, holding the position-actuator
disabling fixed in both conditions — and finds no effect from that
isolated factor. Combined, this narrows the likely cause of D5.10's sag
toward the **position-actuator disabling step itself** (the FL
leg losing its normal stance-stabilizing gain/bias at recovery entry),
rather than the magnitude or onset speed of the auxiliary commands that
replace it. This is suggested by elimination, not directly measured by
this experiment — the position-actuator-disabling effect remains
unresolved/untested here.

## Recommendation

The next diagnostic should isolate the position-actuator-disabling step
itself (e.g., comparing "disable position actuators but apply zero
auxiliary commands" vs. the current baseline, keeping the auxiliary
onset profile fixed) to test directly whether losing the FL leg's normal
stance support — independent of the auxiliary commands — reproduces the
sag. This is a natural follow-up ablation using the same isolation
pattern established in D5.10/D5.11, but is explicitly not performed here
per the task's "stop after the comparison" instruction.

Separately, the viewer crash/divergence observed here should be
reported as an environment issue (likely related to `NativeMujocoViewer`
teardown and/or display availability in this environment — the prior
D5.3 `--debug` run also exited abnormally with code 139); this is noted
for awareness and is out of scope to debug further here.
