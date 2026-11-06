# D5.17 — Recovery Entry Timing: Stance vs. Immediate Entry

Diagnostic-only, non-invasive experiment. No production file was edited.
The frozen walking checkpoint, recovery torque/thigh-velocity magnitudes,
actuator gains, crossing thresholds, rewards, and wire geometry are all
unchanged from D5.3/D5.16. Implementation uses the SAME instance-level
monkeypatch pattern already established and documented in
`scripts/d5_13_entry_hold_ablation.py`'s `install_entry_hold` — no new
precedent was set.

## Files

- New script: [scripts/d5_17_stance_timing.py](../../scripts/d5_17_stance_timing.py)
  (reuses, does not modify, `scripts/d5_3_extended_recovery_window.py`'s
  `build_ablation_env`, `scripts/d3_ground_contact_diagnostic.py`'s
  `D3GroundConfig`, and `src/mjlab/tasks/recovery/mdp/actions.py`).
- Condition A (immediate, unmodified baseline): [logs/d5_17_stance_timing/seed0_immediate/trajectory.csv](seed0_immediate/trajectory.csv)
- Condition B (stance-deferred): [logs/d5_17_stance_timing/seed0_stance_deferred/trajectory.csv](seed0_stance_deferred/trajectory.csv)
- Viewer validation (Condition B): [logs/d5_17_stance_timing/seed0_stance_deferred_viewer/trajectory.csv](seed0_stance_deferred_viewer/trajectory.csv)

## Before-implementation analysis (performed before any run, per the task's requirement)

**1. Are the FSM trigger and actuator handoff currently the same event?**
Confirmed by direct source reading (`ResidualRecoveryAction.process_actions`,
[src/mjlab/tasks/recovery/mdp/actions.py](../../src/mjlab/tasks/recovery/mdp/actions.py)):
`process_actions` calls `self._update_fsm()` (sets `self._phase=RECOVERY`
and `self.newly_entangled` the instant wire contact is detected), then —
in `recovery_mode="hybrid"` — immediately calls
`self._apply_hybrid_actuators()`, which disables the position actuators
and enables the auxiliary actuators **keyed off the exact same
`newly_entangled` flag**, in the same function call, same environment
step. **Yes, they are currently the same event** — this experiment's
entire premise is to separate them.

**2. Reliable FL stance signal.**
Reused the existing `fl_ground_contact_diag` `ContactSensorCfg` already
added by D5.3's `build_ablation_env` (geom-vs-terrain contact,
`reduce="netforce"`) — the same sensor D5.16's audit already used to
establish FL is airborne throughout the baseline RECOVERY window. Not
inferred from joint/foot velocity.

**3. Bounded wait and fallback (declared before running).**
`stance_max_wait_steps = 20` RECOVERY steps. Rationale: reuses the
existing `max_recovery_steps` docstring's own stated reference point
("roughly 2x the ~9-step natural FL swing duration measured for plain
walking") and its "~2x" margin convention, rather than inventing a new
number — a leg left under the frozen walking policy's own gait-phase
command should complete its natural swing-to-stance transition in
roughly 9–10 steps; doubling to 20 is a generous, non-arbitrary bound.
**Fallback:** if FL ground contact is not observed within 20 RECOVERY-
elapsed steps, the identical handoff (disable position actuators, enable
auxiliary actuators) that fires immediately in Condition A is forced at
that bound step — Condition B can never get permanently stuck.

**4. What happens during the wait.**
Deliberately **nothing new**: the monkeypatched `_apply_hybrid_actuators`
simply skips the disable/enable branch while `holding=True`. No ctrl
override, no fixed target, no blending. Because `process_actions`'s
hybrid branch already composes `final_raw` as the frozen policy's own
output with a zero residual on the affected leg, the mere absence of the
disable call means the leg's position actuators continue tracking
whatever the frozen policy would command if the leg were not in RECOVERY
at all. This is explicitly **not** a revival of D5.9's rejected "Option
C" (which tracked the policy's live target for the *entire* recovery
window and saturated within 4–5 steps) — here the window is bounded and,
empirically (see Results), turned out to be extremely short (1 physical
step) for seed 0.

**5. Can crossing/release FSM function correctly when trigger and handoff differ?**
Confirmed by direct source reading of `_update_fsm`: the two-stage
crossing condition and release counter are computed every step purely
from current wire-relative geometry and the wire-contact sensor, with no
reference to which actuators are active. Deferring the handoff cannot
break this logic by construction. The only real interaction is budget
consumption: time spent waiting is still time elapsed against
`max_recovery_steps` (48, unchanged) — measured explicitly below.

**Conclusion before running:** stance deferral as designed requires only
a one-instance-level monkeypatch of `_apply_hybrid_actuators` (exactly
the D5.13 `install_entry_hold` pattern) — no invasive production FSM
change was required, so the experiment proceeded without stopping for
pre-approval.

## Measured results — seed 0

| Metric | Condition A (immediate) | Condition B (stance_deferred) |
|---|---|---|
| Wire-contact trigger step (label convention, see below) | 45 | 45 |
| Actual actuator-handoff step (same convention) | 45 (same physical step) | 46 (handoff deferred by exactly 1 physical step — see below) |
| Handoff reason | `immediate` | `stance_detected` |
| FL ground contact at trigger / 1 step later | False / True (step 44 row: True; see CSV) | same — FL was in or near ground contact essentially immediately after entry |
| Pre-entry mean `base_z` | 0.3084 m | 0.3084 m |
| `base_z` global nadir | 0.2040 m @ step 54 (drop 0.1044 m) | **0.1700 m @ step 93 (drop 0.1384 m)** |
| Peak `foot_height_above_wire` | 0.0933 m @ step 86 | 0.0722 m @ step 87 |
| First `crossing_achieved` | step 84 | step 86 |
| RECOVERY → POST_RECOVERY exit step | 88 | 90 |
| Exit reason | crossing+release | crossing+release |
| `genuine_recovery_complete` reached | True | True |

**Trigger/handoff labeling convention note:** both `trigger_step` and
`handoff_step` are reported using the SAME one-step-delayed labeling
convention already established in D5.3/D5.13/D5.16's "ENTER RECOVERY
step=N" messages (a physical `env.step()` call's index + 1). Under this
convention, equal numbers mean the same physical step; `handoff_step -
trigger_step` directly gives the number of physical steps the handoff
was deferred by. For seed 0, this difference is exactly **1 step** — FL
was already at or very near ground contact at the instant of wire
contact, so the bounded 20-step wait was essentially never exercised;
the fallback (`bound_reached`) never fired.

## Analysis

**Crossing capability was NOT sacrificed.** Condition B still reaches
`genuine_recovery_complete` (crossing + sustained release), exits via
the genuine crossing+release path (not the cap-timeout fallback), and
both stages latch only slightly later (step 86/90 vs. 84/88) — consuming
2 extra steps of the 48-step budget, well within margin.

**Stability did NOT improve — it measurably worsened.** The primary
RECOVERY-phase `base_z` trajectory (steps 45–89) is nearly identical in
shape between the two conditions (both dip to ~0.20–0.21 m around
step 53–54 and recover to ~0.27 m by the low-80s), consistent with the
handoff being deferred by only 1 physical step having little effect on
that portion. The substantive difference is a **second, much deeper
dip in POST_RECOVERY**: Condition A's post-handoff settling dip bottoms
at 0.2182 m (step 92, per D5.16's prior audit); Condition B's
corresponding dip bottoms at **0.1700 m (step 93)** — a markedly larger
and lower excursion, occurring 3 steps after FL's exit-phase ground
touchdown (step 90), i.e. in the window where the position actuators
have just been restored to full gain under the policy's own (now
resynchronized) command.

**Interpretation (explicitly flagged as interpretation, not directly
proven by this single-seed run):** a 1-step shift in exactly when the
position actuators are disabled appears to shift the leg's subsequent
trajectory enough that, by the time crossing completes and control is
handed back at POST_RECOVERY, the leg/base are in a different dynamical
state that lands less favorably on touchdown — i.e., this bounded
hybrid-nonlinear system appears sensitive to handoff timing in a way
that is not simply "later is better" or "stance beats swing." This
single seed does not establish WHY the post-recovery dip got worse; it
only establishes, as a measured fact, that it did.

## Per-task success criterion

> The experiment succeeds only if it improves stability without
> silently sacrificing wire-crossing capability.

**Result: FAILS the stability bar for seed 0.** Crossing capability was
preserved, but stability (as measured by the global `base_z` drop,
which is the primary metric this experiment set out to test) got
**worse**, not better (0.1384 m drop vs. 0.1044 m baseline). Per the
task's explicit instruction ("If the result is promising, repeat with a
small, predeclared set of additional seeds, including a seed that
previously failed to cross"), additional seeds were **not** run, since
the seed-0 result does not meet the "promising" bar that gates that
step. For transparency, the predeclared additional-seed set that would
have been used (chosen before knowing seed-0's outcome, from the
existing D5.5 repeatability sweep) was **seed 1** (a representative
successful baseline with different timing) and **seed 7** (the one seed
in that sweep that previously never even reached Stage 1 clearance,
i.e. "failed to cross" most severely) — these are documented here but
intentionally not executed, per the stop-on-negative-result instruction.

## Viewer validation

Commands used:
```
uv run python scripts/d5_17_stance_timing.py --help
uv run python scripts/d5_17_stance_timing.py --seed 0 --condition immediate \
  --out-dir logs/d5_17_stance_timing/seed0_immediate
uv run python scripts/d5_17_stance_timing.py --seed 0 --condition stance_deferred \
  --out-dir logs/d5_17_stance_timing/seed0_stance_deferred
uv run python scripts/d5_17_stance_timing.py --seed 0 --condition stance_deferred \
  --out-dir logs/d5_17_stance_timing/seed0_stance_deferred_viewer --debug
```

The viewer run for Condition B (seed 0) used the D5.15/D5.16-corrected
stepping pattern (callback never calls `env.step()` itself). Its
`trajectory.csv` is **byte-for-byte identical** to the headless run's
CSV (`diff` returns no output). The viewer process also exited cleanly
(exit code 0, no teardown crash on this run) — reported separately from
the quantitative findings, consistent with the intermittent nature of
the native-viewer teardown issue already documented in D5.15/D5.16.

## Recommendation — exactly one next experiment

**Do not adopt stance-deferred handoff as implemented.** Before
attempting any further timing-based mitigation, run a single, minimal
diagnostic: repeat Condition B for seed 0 with `stance_max_wait_steps`
swept over a small predeclared set that FORCES a larger deferral even
though stance is detected early (e.g., by also requiring N consecutive
steps of continuous ground contact, not just the first True reading,
before handing off) — this tests whether the 1-step deferral itself
(rather than "stance vs. swing" per se) is responsible for the worse
post-recovery dip, by seeing whether a longer, deliberately-delayed
handoff (still bounded, still using only the existing ground-contact
signal, no new control law) changes the POST_RECOVERY dip's depth
monotonically, non-monotonically, or not at all. This isolates whether
the regression found here is a genuine "even tiny handoff-timing shifts
destabilize the post-recovery landing" phenomenon (which would caution
strongly against timing-based mitigations generally) or an artifact
specific to the ~1-step shift actually realized in this run.

This report concludes the D5.17 experiment. Per the task instructions,
no further tuning or redesign has been attempted — stopping here.
