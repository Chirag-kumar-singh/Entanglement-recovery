# D5.10 — Causal ablation: is the initial post-wire-contact stumble caused by the hybrid recovery override, or already present under the frozen walking policy alone?

## Pre-run isolation check (performed before any simulation)

Before running anything, `src/mjlab/tasks/recovery/mdp/actions.py` was
re-read in full to confirm Condition B (override disabled) could be
isolated safely:

- `_update_fsm()` is called unconditionally at the start of
  `process_actions()` and only computes phase transitions, the
  wire-contact-triggered entry/exit event, and all geometric gate
  state. It does not depend on `recovery_mode`.
- `_apply_hybrid_actuators()` — the ONLY method that zeros the FL
  thigh/calf position actuators' gain/bias and issues the auxiliary
  thigh-velocity/calf-torque commands — is called from exactly one
  place, gated on `cfg.recovery_mode == "hybrid"`, inside
  `process_actions()`, strictly *after* `_update_fsm()`.
- In hybrid mode, `delta_a` is always zero, so the composed action fed
  into the underlying `JointPositionAction` is identical to the raw
  walking-policy output in both conditions regardless of whether
  `_apply_hybrid_actuators()` runs.

This confirmed the two conditions are cleanly separable: Condition B
monkeypatches `act._apply_hybrid_actuators = lambda: None` on the
already-constructed action-term instance (no production file edited),
leaving FSM phase-tracking/entry-timing byte-for-byte identical to
Condition A, while the FL thigh/calf position actuators are never
disabled and remain under `pi_walk`'s own control throughout. Isolation
was confirmed achievable, so the run proceeded.

## Setup

- Script: [scripts/d5_10_stumble_ablation.py](scripts/d5_10_stumble_ablation.py) (new, isolated; reuses the
  existing env-construction pattern from `d5_3_extended_recovery_window.
  build_ablation_env`, same checkpoint/wire geometry/sim settings).
- `env_cfg.auto_reset = False` with an explicit `env.reset_buf` check
  after every step, to guarantee the rollout stops immediately on a real
  fall/time-out rather than silently auto-resetting and continuing.
- Two previously-characterized seeds: seed 0 (baseline genuine success)
  and seed 7 (baseline "Stage 1 never latches" failure), both run for a
  150-step window, both conditions (A: baseline hybrid, B: override
  disabled) per seed.
- Four runs total, outputs in `logs/d5_10_initial_stumble_ablation/seed{0,7}_cond{A_baseline,B_disabled}/`.

## Results

Wire contact and RECOVERY-phase entry occur at the identical step (43/44→45)
in all four runs, confirming alignment is exact as designed.

| seed | condition | recovery exit | stage1 latch | stage2 latch | fall/time-out (150-step window) |
|---|---|---|---|---|---|
| 0 | A baseline (hybrid) | step 88, crossing+release (success) | step 84 | step 84 | none |
| 0 | B disabled | step 91, cap_timeout | never | never | none |
| 7 | A baseline (hybrid) | step 91, cap_timeout | never | never | none |
| 7 | B disabled | step 91, cap_timeout | never | never | none |

Base height (`base_z`) around wire-contact entry (step ≈43-45) through step 69:

- **Condition A (hybrid override active), both seeds:** base height sags
  sharply starting the step after entry, dropping from ≈0.309 m to a
  trough of ≈0.20 m (seed 0, step 54) / ≈0.177 m (seed 7, step 55) —
  roughly a 35-43% reduction in base height — before partially
  recovering over the following ~15 steps. `fl_ground_contact` goes
  `False` for the entire window shown in both conditions A runs (FL foot
  lifted/off the ground throughout the sag).
- **Condition B (override disabled), both seeds:** base height stays
  essentially flat, oscillating gently in the 0.296–0.310 m band for the
  entire window — no comparable sag. `fl_ground_contact` is mostly
  `True` starting one step after entry (the walking policy keeps placing
  the FL foot down normally).

This pattern is consistent across both the baseline-success seed (0) and
the baseline-failure seed (7): in both cases, the body-height dip is
present in Condition A and absent in Condition B.

No termination (fall or time-out event registered by `env.reset_buf`)
occurred in any of the four 150-step runs — the sag seen under Condition
A is a significant but non-terminal destabilization within this window,
not a hard fall.

## Interpretation

**Observation** (directly supported by this single-seed-pair, 150-step
comparison): the pronounced initial base-height sag that follows wire
contact is present when the hybrid actuator override is active, and is
absent (or far smaller) when the override is suppressed and the frozen
walking policy continues to control the FL leg normally — in both the
seed that eventually recovers under the baseline controller (seed 0) and
the seed that does not (seed 7).

**Caution on causal interpretation**: this is a 2-seed, single-window
ablation. It demonstrates that *in this experiment*, disabling the
override removes the sag; it does not by itself prove the override is
the sag's mechanism, nor rule out seed-specific or coincidental effects.
It also does not mean Condition B is "better" overall — recall that with
the override disabled, seed 0 never completes recovery (Stage 1 never
latches, exits via 48-step timeout) even though its baseline body
dynamics look calmer; the FL foot and leg never clear/cross the wire
without the override's assistance. The two conditions are answering
different questions (stability of the body vs. whether the leg actually
gets free of the wire).

This reverses the hypothesis this experiment was designed to test: D5.6
root-caused the dominant *downstream* Stage-1-never-latches failure mode
to the frozen policy's own velocity-tracking stall. D5.10 shows the
*initial* base-height stumble immediately following wire contact is, in
this comparison, attributable to the hybrid override's actuator-level
intervention itself (most likely the combination of disabling the FL
position actuators' normal stabilizing gain and/or the open-loop
thigh-velocity/calf-torque commands temporarily removing the FL leg's
ground support), not something already present under walking-only
control.

## Implication for next steps

The hybrid override's actuator-level intervention — not the frozen
policy's inherent disturbance response — appears to be the proximate
cause of the initial post-wire-contact body sag in this experiment. This
suggests any future investigation of the velocity-tracking collapse
(D5.6) and the calf-waiting-state question (D5.7-D5.9) should account
for the possibility that the override's own onset transient is
contributing to the destabilized state the policy is then asked to
recover from. This does not resolve or reopen the D5.9 "reject Option C"
conclusion, and does not imply any specific fix — it is an observation
that narrows where the next diagnostic should look (the override's
onset behavior/actuator-disabling transient) rather than only the
policy's response to it.

Per the task scope, this experiment stops here — no gain/FSM tuning,
velocity-gate work, or calf-waiting-state work was performed, and none
is proposed as an automatic next step.
