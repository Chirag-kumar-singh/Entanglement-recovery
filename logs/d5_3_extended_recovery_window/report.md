# D5.3 -- Extended Recovery Window Diagnostic (does calf_forward_dist resume
# advancing, plateau, or keep regressing beyond the original 16-step cap?)

This is the experiment identified (but not run) as the recommended next step
in D5.2. It directly follows the user's "okay go with it" approval.

## Method

New script `scripts/d5_3_extended_recovery_window.py`, built as a close copy
of D5.1's harness (same `build_ablation_env`, same frozen checkpoint,
deterministic zero-residual rollout, same seeds 0/1, same baseline/ablated
conditions, same full gate-variable instrumentation), with ONE additional
parameter: `max_recovery_steps_override`, passed through to
`go2_wire_recovery_residual_env_cfg`'s existing, already-plumbed
`max_recovery_steps` parameter (default class-level value in
`ResidualRecoveryActionCfg` is untouched; only this script's own isolated
env_cfg instance uses the override). No other gain, threshold, FSM rule,
checkpoint, or wire geometry was changed.

**Verification before use:** ran the new harness with
`max_recovery_steps_override=16` (i.e. the original cap) for seed 0
baseline and confirmed it is bit-identical to the existing D5.1
seed0_baseline trajectory (`foot_z` and `calf_forward_dist` max diff = 0.0
across all 120 steps) -- confirming the extension introduces no side
effects when the cap is left at its original value.

Then ran all four combinations (seed 0/1 x baseline/ablated) with
`max_recovery_steps_override=48` (3x the original budget), `max_steps=200`.

## Results

| | seed0 baseline | seed0 ablated | seed1 baseline | seed1 ablated |
|---|---|---|---|---|
| Exit reason | **crossing+release** | cap_timeout | **crossing+release** | cap_timeout |
| `recovery_elapsed` at exit | 45 | 48 (cap) | 48 (cap) | 48 (cap) |
| Stage 1 first latch | step 84 | never | step 86 | never |
| Stage 2 first latch | step 84 | never | step 86 | never |
| `RECOVERY SUCCESS` (post-recovery stability) | **Yes** | No | **Yes** | No |
| `calf_forward_dist` first step >0 | step 84 | never (ends -0.091) | step 86 | never (ends -0.099) |
| `calf_height_above_wire` at run end | 0.155 | 0.116 | 0.176 | 0.146 |

**Both baseline (hip-wire-contact NOT removed) runs succeed** when given
3x the original time budget -- genuine crossing+release is achieved, and
the post-recovery stability window is reached (full `RECOVERY SUCCESS`).
In both, `calf_height_above_wire` had already climbed well past the 0.02 m
margin many steps before the crossing (e.g. seed 0: height reaches 0.059 m
by step 80, still 4 steps before the forward crossing at step 84) -- the
height condition was satisfied first in both successful runs, and the
limiting factor was specifically the horizontal (`calf_forward_dist`)
component, which stayed negative until a late, rapid sweep (steps 83-87 in
both successful runs, `calf_forward_dist` jumping from roughly -0.01 to
+0.06-0.08 within 3-4 steps).

**Both ablated (hip-wire-contact removed) runs still time out at 48
steps**, i.e. extending the budget alone does NOT rescue the ablated
condition in either seed. Notably, `calf_height_above_wire` is rising
similarly in the ablated runs (reaching 0.12-0.15 m by the end, comparable
to or even exceeding the successful baseline runs) -- so the height
condition is not the obstacle in the ablated case either. The sole
obstacle in both ablated runs is `calf_forward_dist`, which is
**monotonically worsening** throughout the entire extended window (-0.029
-> -0.091 for seed 0; -0.028 -> -0.099 for seed 1), with no late forward
sweep ever occurring, unlike both baseline runs.

The open-loop commands (`thigh_vel_cmd=+2.0`, `calf_torque_cmd=-5.0`) are
confirmed constant throughout the entire window in all four runs (no
internal modulation), and `thigh_pos` increases monotonically in all four
(the thigh keeps retracting under the constant velocity command
regardless of condition) -- so the divergence between baseline and
ablated outcomes is not explained by a different commanded input, only by
how the leg's resulting swing interacts with the now-missing hip-wire
contact.

## Interpretation

This resolves D5.2's open question for the **baseline** condition:
the regression seen at the 16-step mark in D5.2 was a transient dip
within a longer, eventually-successful trajectory -- "insufficient time,"
not "wrong direction," was the correct diagnosis for the unmodified
(non-ablated) recovery under these two seeds. The original
`max_recovery_steps=16` cap appears to cut off both baseline runs well
before a later, successful forward sweep that both would otherwise
complete by step ~84-87 (about 3x the original budget).

For the **ablated** condition, this is a new and important finding not
previously available: removing FL hip-wire contact prevents the abrupt
calf-velocity-reversal collapse (D4.9) and still allows calf height to
climb past the wire (comparable to the successful baseline runs), but it
removes whatever interaction was producing the late forward sweep that
both baseline runs eventually exhibit. With the hip-wire contact excluded,
`calf_forward_dist` never recovers and keeps regressing for the entire
48-step window in both seeds -- strongly suggesting that the hip-wire
contact, despite causing the originally-studied collapse, was also
contributing a forward-directed interaction (e.g., a contact-reaction
force or a sliding/pivoting effect against the wire) that is necessary for
the leg to eventually swing forward past the wire. This directly
qualifies the D4.8/D4.9 ablation finding: removing hip-wire contact is not
an unambiguous improvement -- it trades the abrupt collapse for a
different, still-unsuccessful outcome in which the leg lifts clear
vertically but never advances horizontally, within 3x the original time
budget.

## Remaining uncertainty

- Only 2 seeds were used (per task constraint, no new seed sweep was
  started). This pattern (baseline succeeds with more time, ablated does
  not) should not yet be generalized.
- `max_recovery_steps_override=48` is itself an arbitrary extension factor
  (3x); it was sufficient for both baseline seeds to succeed but it was
  not determined whether the ablated runs would eventually also succeed
  given an even longer budget, or whether `calf_forward_dist`'s regression
  in the ablated condition is permanent. The ablated trend was monotonic
  and worsening through the entire 48-step window with no sign of
  reversal, which is suggestive but not conclusive for arbitrarily longer
  horizons.
- This experiment used the deterministic zero-residual-action hybrid
  controller exactly as in all prior D3-D5 diagnostics; it says nothing
  about how a trained PPO-residual policy would behave in either
  condition.

## Recommendation

Proceed to the 10-20-seed repeatability sweep using the **unmodified**
controller (hip-wire contact NOT ablated) with the extended
`max_recovery_steps_override=48` budget (diagnostic-only override, as
established and verified safe in this task), to determine how consistently
the late forward-sweep-driven success seen in both seed 0 and seed 1
baseline runs generalizes, before considering any production change to
`max_recovery_steps` itself.
