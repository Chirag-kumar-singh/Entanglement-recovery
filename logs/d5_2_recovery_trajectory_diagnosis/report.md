# D5.2 -- Diagnose Calf Motion During the Active Recovery Window

Analysis-only task. No simulations were run. All data is read from the
existing D5.1 per-step CSVs under `logs/d5_1_clearance_gate_diagnostics/`
(seed0_baseline, seed0_ablated, seed1_baseline, seed1_ablated),
restricted strictly to rows where `phase == 1` (RECOVERY), per the task's
explicit instruction not to use post-timeout walking-policy motion as
evidence. Full per-step tables (all logged columns, per relative
recovery-elapsed step) are in `trajectory_tables.txt` in this directory.

## Sign-convention check (done before interpreting motion, per task
instruction not to infer world-space motion from joint angles alone)

Verified directly from `actions.py`:
- `hybrid_thigh_vel_ctrl = -cfg.recovery_thigh_velocity` (cfg value is
  `-2.0`, so the actually-applied command is `+2.0`). Confirmed in the
  logs: `thigh_vel_cmd` is `+2.0` throughout every recovery window in
  all four runs, and `thigh_pos` increases monotonically alongside it
  (actuator tracks its command correctly; no sign bug in the applied
  control).
- `hybrid_calf_torque_ctrl = -cfg.recovery_calf_torque` (cfg `5.0` ->
  applied `-5.0`), confirmed via `aux_calf_actuator_force_global`
  staying pinned at exactly `-5.0` for the entire recovery window in
  all four runs -- this is a constant, non-modulated, open-loop torque
  command (no feedback adjustment during the window).
- `calf_forward_dist`/`calf_height_above_wire` are not joint angles --
  they are the FSM's own already-computed world-frame projections (see
  D5.0/D5.1), so no additional forward-kinematics inference was needed
  or performed here; the analysis uses these values exactly as logged.

## Per-run recovery-window summary (15 RECOVERY-phase steps each)

| | seed0 baseline | seed0 ablated | seed1 baseline | seed1 ablated |
|---|---|---|---|---|
| `calf_forward_dist` start -> end | -0.0286 -> -0.0343 | -0.0286 -> -0.0427 | -0.0282 -> -0.0249 | -0.0282 -> -0.0433 |
| `calf_forward_dist` min / max | -0.0351 / -0.0266 | -0.0427 / -0.0286 | -0.0356 / -0.0216 | -0.0433 / -0.0282 |
| `calf_height_above_wire` start -> end | -0.0224 -> -0.0207 | -0.0224 -> -0.0176 | -0.0231 -> -0.0272 | -0.0231 -> -0.0150 |
| `calf_height_above_wire` min / max | -0.0267 / -0.0196 | -0.0263 / -0.0176 | -0.0299 / -0.0163 | -0.0264 / -0.0150 |
| Net `calf_forward_dist` change | -0.0057 (worse) | -0.0141 (worse) | +0.0033 (slightly better, still negative) | -0.0151 (worse) |
| Net `calf_height_above_wire` change | +0.0017 (slightly better) | +0.0048 (better) | -0.0041 (worse) | +0.0081 (better) |
| `calf_forward_dist` deltas over last 4 steps | -0.0017,-0.0021,-0.0022,-0.0017 (all regressing) | -0.0020,-0.0022,-0.0021,-0.0016 (all regressing) | +0.0004,-0.0005,-0.0012,-0.0016 (regressing) | -0.0030,-0.0037,-0.0047,-0.0029 (all regressing, accelerating) |

## Monotonicity analysis

`calf_forward_dist` is **not monotonic** in any of the four runs. In all
four, there is a brief partial-forward-progress window coinciding with
global steps ~50-53 (recovery-elapsed steps ~6-9) -- the same window
previously identified (D4.2) as the calf-velocity-reversal/collapse
event -- during which `calf_forward_dist` improves (becomes less
negative) by roughly 0.002-0.007 m. In every run, this partial advance
is **fully reversed** over the remaining steps before the 16-step
timeout: the final 4 steps of every single run show `calf_forward_dist`
decreasing (moving further from the near->far crossing), not plateauing
or continuing to approach zero. None of the four runs end anywhere close
to crossing (`calf_forward_dist` stays negative throughout; closest
approach is -0.0216 m in seed1 baseline, still substantially short of
zero).

`calf_height_above_wire` is also non-monotonic in all four runs, with a
similar dip during the same ~step 50-53 window, but in 3 of 4 runs (both
ablated conditions and seed0 baseline) it recovers and ends net higher
than it started; seed1 baseline is the exception, ending net *lower*.

## Effect of hip-wire-contact ablation on the approach to stage 1

Both baseline and ablated trajectories are bit-identical through the
shared pre-divergence window (steps 44-50, confirmed from D4.9), so any
difference is attributable to the post-divergence dynamics only. After
divergence:
- **Height** (`calf_height_above_wire`): ablation modestly *helps* --
  both ablated runs end with a better (less negative) height than their
  baseline counterparts (seed0: -0.0176 vs -0.0207; seed1: -0.0150 vs
  -0.0272).
- **Forward distance** (`calf_forward_dist`): ablation *does not help*
  and in both seeds ends **worse** than baseline (seed0: -0.0427 vs
  -0.0343; seed1: -0.0433 vs -0.0249) -- in the ablated runs the
  regression in the final steps is also the largest observed of all
  four runs (up to -0.0047 m/step, accelerating).

This means removing the FL hip-wire contact prevents the abrupt
collapse and modestly improves vertical clearance, but it does **not**
address -- and in these two seeds actually worsens -- the forward
displacement needed to cross the wire. This is consistent with (but
does not single-handedly prove, given only 2 seeds) the possibility
that hip-wire contact was providing some forward reaction force/lever
effect that, while triggering the collapse, also incidentally
contributed forward progress; removing it removes both the collapse
and that incidental forward push.

## First stage-1 failure mechanism supported by the data

The data support an **ineffective net motion direction**, not merely
insufficient time, as the dominant mechanism: in all four runs, the
calf's forward distance is actively regressing (moving away from the
crossing) over the final 4 logged recovery steps, immediately before
the 16-step cap is reached. A trajectory that is regressing at the
moment the time budget expires provides no evidence that extending the
budget under the same open-loop actuator commands (`thigh_vel_cmd=+2.0`
constant, `calf_torque_cmd=-5.0` constant, unmodulated for the entire
window) would reverse that regression and complete the crossing --
if anything, the trend at timeout is in the wrong direction.

## Unresolved uncertainty

This conclusion is based on only the single fixed 16-step window actually
executed in each of the 4 runs; the controller's open-loop commands are
constant for the whole window, so there is no direct evidence of what
the SAME constant commands would produce if allowed to continue past
step 16, because `max_recovery_steps=16` is a hard cap enforced inside
`_update_fsm` and the harness cannot observe what the (unmodified)
actuators would do afterward without actually extending that cap --
which is explicitly out of scope here (no controller/threshold changes
permitted). The observed late-window regression is suggestive but not
dispositive on its own: a constant-force/constant-velocity open-loop
command acting against a periodically-varying contact/gait state could
in principle re-enter a forward-progressing phase if given more cycles,
and 2 seeds x 15 steps each is too small a sample to rule that out
definitively.

## Recommended next step

Run a single, strictly diagnostic-only extension of the SAME two seeds
(0 and 1) with `max_recovery_steps` temporarily raised ONLY inside an
isolated copy of the action config used by a new diagnostic script (never
touching the production `ResidualRecoveryActionCfg` default or any
checkpoint/training config), purely to observe, for a few additional
steps beyond 16, whether `calf_forward_dist` continues to regress,
plateaus, or resumes advancing under the identical constant open-loop
commands -- this is the smallest experiment that directly distinguishes
"insufficient time" from "wrong direction" without altering any
production behavior. (Per this task's constraints, this experiment was
NOT run here; it is only identified as the next step.)
