# D5.13 — Testing a brief fixed-target PD hold at recovery entry

## Objective

Test D5.12's recommended next step: does a brief, fixed-target PD hold
at the FL leg's recovery-entry pose (before handing off to the existing
position-actuator-disabling + auxiliary-command override) reduce or
prevent the base-height sag isolated in D5.10-D5.12?

## Setup

New, isolated script: [scripts/d5_13_entry_hold_ablation.py](scripts/d5_13_entry_hold_ablation.py), verified
via `--help` before running. Reuses `build_d510_env` unmodified (same
checkpoint/wire geometry/sim settings as D5.10-D5.12).

**Condition A (baseline):** unmodified, identical to all prior D5.1x
baselines.

**Condition D (entry_hold):** on recovery entry, the FL thigh/calf
position actuators' gain/bias are left completely untouched (original
stiffness/damping) for `hold_steps=5` RECOVERY steps. During the hold,
the position-actuator ctrl target is forcibly overridden each step
(after the normal walking-policy write) to the FIXED joint position
captured at the instant of entry (zero initial tracking error, unlike
D5.9's rejected Option C, which tracked the live, diverging policy
target for the whole window). No auxiliary command is applied during
the hold. At `recovery_elapsed == hold_steps`, the normal
disabling+auxiliary-command engagement happens at full magnitude
(no ramp — D5.11 showed ramping has no effect), exactly matching
baseline's transition logic, merely time-shifted by `hold_steps`.

This is explicitly distinct from D5.9's rejected Option C (fixed vs.
live target, short vs. indefinite duration) and does not reopen that
rejection.

Commands used:
```
uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 --condition baseline --out-dir logs/d5_13_entry_hold_ablation/seed0_condA_baseline
uv run python scripts/d5_13_entry_hold_ablation.py --seed 0 --max-steps 150 --condition entry_hold --hold-steps 5 --out-dir logs/d5_13_entry_hold_ablation/seed0_condD_hold5
```

## Results (seed 0, window steps 45-75)

| | entry `base_z` | min `base_z` | step of min | steps to min | `fl_ground_contact=False` count |
|---|---|---|---|---|---|
| A baseline | 0.3077 | 0.2040 | 54 | 9 | 31/31 (100%) |
| D entry_hold(5) | 0.3092 | 0.2045 | 56 | 11 | 28/31 (90%) |

During the hold itself (steps 44-47), the entry_hold condition works
exactly as designed: `base_z` stays flat at ≈0.309-0.310 (no sag at
all), the FL foot has ground contact (`True`), and the position-actuator
force is small and bounded (−0.7 to −3.2 N·m, consistent with a benign,
non-saturating PD correction toward a nearby fixed target — in sharp
contrast to D5.9's Option C, which saturated at ±45 Nm within 4-5 steps
tracking a live, diverging target).

However, **the instant the hold ends (step 48) and the normal
disabling+auxiliary-command override engages, the same sag begins and
reaches nearly the same magnitude and duration as baseline** — drop
0.1046 m vs. baseline's 0.1036 m, minimum reached 11 steps after entry
vs. baseline's 9 (i.e., ~2 steps later, consistent with the hold's
4-step delay partially but not proportionally shifting the dynamics),
and FL ground contact is lost for 28 of the 31 post-entry steps versus
31/31 in baseline — a negligible improvement.

## Interpretation

**The brief fixed-target hold does not reduce or prevent the sag — it
only delays its onset by approximately the hold duration.** The hold
itself is successful and safe on its own terms (zero sag, bounded
forces, no saturation while active), directly confirming D5.12's
inference that the sag is driven by the position-actuator disabling
rather than the auxiliary commands' timing or magnitude — the robot
is stable for as long as that disabling is deferred. But once the
disabling does happen (at any point), essentially the same dynamics
unfold regardless of when: the sag is a direct, time-invariant
consequence of removing the FL leg's stance-stabilizing gain/bias
while under load, not an artifact of the exact moment this happens
relative to wire contact.

This means a short, fixed-duration hold — by itself — is not a viable
mitigation. The disabling has to happen at some point for the existing
auxiliary-command-based recovery maneuver to execute (the auxiliary
actuators cannot operate while the position actuators are still
enabled, since both would fight for control of the same joints), so
merely delaying it defers rather than solves the problem.

## Recommendation

The sag appears to be an essentially unavoidable consequence of the
current control handoff design (disable position hold → engage
open-loop auxiliary commands) under the frozen policy's current gait
state at wire contact, not a timing or magnitude artifact that can be
fixed by adjusting *when* or *how fast* the handoff happens. Further
mitigation attempts should either: (a) accept the sag as an inherent
cost of the current recovery maneuver design and focus on whether it is
survivable/acceptable (termination risk, recovery success rate) rather
than eliminating it, or (b) consider a fundamentally different
handoff — e.g., a smooth position-to-auxiliary blend/co-activation
rather than an instantaneous or delayed switch, or support from a
different leg/posture adjustment during the transition — which would
be a more substantial controller redesign, explicitly out of scope for
this isolated diagnostic sequence and not attempted here.

Per the task's instructions, this comparison stops here — no further
controller change is proposed or made automatically.
