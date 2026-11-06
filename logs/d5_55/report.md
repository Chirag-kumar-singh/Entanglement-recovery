# D5.55 — Bounded FL Retraction Before Support Transfer

Tests the D5.54 diagnosis: the second collapse happens because FL stays hybrid-retracted
through the FR/RL support transfer and nothing takes its place. No hybrid command, gain,
threshold, checkpoint, wire model, or the D5.53 touchdown guard was changed.

**Result: the second collapse is prevented on both seeds, but neither seed crosses the
wire — both run out the 48-step recovery cap.** Report these separately, as the decision
rules ask.

## 1. Files

- **New:** [scripts/d5_55_bounded_retraction.py](../../scripts/d5_55_bounded_retraction.py).
  Imports `TouchdownGuard`, `D553Config`, `_setup`, `Recorder` from
  `scripts/d5_53_touchdown_guard.py` unchanged, and `_first`/`_handoff_step`/`_write_csv`
  from `scripts/d5_52_wire_removed_forced_entry.py` unchanged. Neither file was edited.
- No production file touched (`actions.py`, `env_cfg_residual.py`, `d5_53_touchdown_guard.py`
  unchanged). Nothing committed.

## 2. Bounded-retraction design

A new wrapper, `BoundedRetraction`, hooks `act._apply_hybrid_actuators` (separate hook point
from the D5.53 `TouchdownGuard`, which hooks `act._update_fsm`; both run together for the
`bounded_retraction` variant). The wrapper reimplements `_apply_hybrid_actuators`'s exact
enter/exit gain bookkeeping (same gain/bias restore calls, same `_pos_thigh_ctrl_id` /
`_orig_thigh_gainprm` etc. the production code uses) because the original cannot be called
unmodified — it redrives the full retract command from `in_recovery` alone every step, which
would undo any pause. This is the same full-reimplementation approach D5.49's
`install_fixed_delay_handoff` already used for a comparable one-off. The FSM's own
crossing/release/exit logic in `_update_fsm` is not touched and runs unmodified throughout.

State machine, only active once `_update_fsm` sets `phase == RECOVERY`:

| State | Transition |
|---|---|
| `ACTIVE` | hybrid retract running (unchanged command: thigh +2.0 rad/s, calf −5.0 N·m). After `budget` steps → restore FL position gains, `RELEASE_WAIT_TOUCHDOWN` |
| `RELEASE_WAIT_TOUCHDOWN` | FL ground contact ≥2 consecutive steps → `RELEASE_STANCE` |
| `RELEASE_STANCE` | FL ground contact absent ≥2 consecutive steps (a new lift-off) → re-enable hybrid gains, `ACTIVE` (next cycle) |

This loops until the FSM's unmodified exit fires (crossing+release or the 48-step cap),
independent of this sub-state.

**Gait-derived budget (not a fixed step count):** the wrapper tracks FL's ground-contact
transitions every step, phase-gated so the *swing-length estimate* only updates from
completed NORMAL_WALK swings (not from perturbed, mid-recovery steps): `last_complete_swing_
len`. It also tracks `airborne_since_liftoff`, continuously. At hybrid entry (and at each
resume),

    budget = max(1, last_complete_swing_len − airborne_since_liftoff)

i.e. "run the hybrid for the rest of the swing FL is already in," not a flat delay from
entry. Measured on both seeds: `last_complete_swing_len = 11`, `airborne_since_liftoff = 2`
at entry (step 62, FL lifted off at 60) → `budget = 9` → release at step 70, i.e. step
71-equivalent, matching the D5.54 report's own "~step 71" reasoning from first principles,
not by copying it. No fallback was needed (11+ steps of NORMAL_WALK precede the first
contact on both seeds).

## 3. Validation ([validation.csv](validation.csv), 8/8 pass)

- `baseline`/`guard_enabled` on both seeds, re-run in memory through the unmodified
  `d5_53_touchdown_guard._setup`, reproduce the existing D5.53 trace files exactly
  (max diff 0.0 on base z, pitch rate, FL calf, hybrid-active flag, phase; no new trace
  files written, per the lean-artifact instruction — D5.53's own CSVs are the record).
- Wire collisions stay enabled for `bounded_retraction` on both seeds (excludes unchanged
  at 6; 1937/2125 robot–wire substep contacts recorded).
- FL is released to walking control before the FSM's own exit on both seeds.

## 4. Results ([comparison_summary.csv](comparison_summary.csv), 6 rows: baseline/guard_enabled
reused from D5.53 for comparison, plus the two new bounded_retraction runs)

| | seed 0 baseline | seed 0 guard | seed 0 **bounded** | seed 11 baseline | seed 11 guard | seed 11 **bounded** |
|---|---|---|---|---|---|---|
| Hybrid handoff | 44 | 62 | 62 | 51 | 62 | 62 |
| Peak \|pitch rate\| k0–9 (rad/s) | 1.80 | 0.24 | **0.35** | 2.38 | 0.24 | **0.31** |
| Min support k0–9 (N; weight 149) | 0 | 166 | **166** | 0 | 166 | **86** |
| Peak \|pitch rate\|, whole run (rad/s) | 1.80 | 3.45 | **0.70** | 2.38 | 2.80 | **0.64** |
| Min base z, whole run (m) | 0.204 | 0.171 | **0.282** | 0.196 | 0.173 | **0.282** |
| Steps with total support <10 N | 9 | 15 | **1** | 7 | 27 | **1** |
| Crossing step | 84 | 91 | **none** | none | 90 | **none** |
| Recovery exit | 88, crossing | 95, crossing | **109, cap timeout** | 98, cap timeout | 94, crossing | **109, cap timeout** |
| Genuine recovery complete | True | True | **False** | False | True | **False** |

**The second collapse does not happen:** whole-run peak pitch rate drops from 3.45/2.80 rad/s
(guarded) to 0.70/0.64 rad/s (bounded); whole-run minimum height rises from 0.171/0.173 m to
0.282 m on both seeds (above the baseline's own 0.204/0.196 m); steps with support under 10 N
fall from 15/27 to 1. This is a clear, specific improvement over the guard-only variant, not
just "no worse than baseline."

**Neither seed crosses.** Both reach the 48-step recovery cap (`recovery_exit_timeout=True`,
`genuine_recovery_complete=False`). Per the task's own caution, an FSM crossing event is not
being conflated with completion here — there is no crossing event at all in either run.

## 5. Why FL never gets a second swing (both seeds, from the trace)

- Step 62–70 (`ACTIVE`): hybrid retract, identical to the guard-only run for this window.
- Step 70: released to walking control. FL foot is still short of the wire
  (`fl_foot_forward_dist_m` ≈ −0.006, `fl_foot_height_above_wire_m` ≈ −0.10).
- Step 74: FL touches down (confirmed by debounce) — directly under the wire, not past it.
- **Steps 74–109: the robot comes to rest.** All four feet load and stay loaded (ground
  force magnitudes 10–80 N each, no zero-force leg after ~82); `base_lin_vel_b_x` oscillates
  within ±0.1 m/s of zero (not the ~0.2–0.4 m/s walking band seen pre-entry); `base_z` and
  `projected_gravity_b_z ≈ −1.0` stay essentially constant (level, ~0.28–0.31 m); FL
  `foot_forward_dist` creeps from −0.006 to only +0.02 over 36 steps. **FL never lifts off
  again before the cap at step 109** — `RELEASE_STANCE → ACTIVE` never fires on either seed
  (confirmed: no `RESUME` event logged in either run).
- This is a new, specific finding: once released back to position control in this
  configuration, the frozen walking policy settles into a near-static, locally stable stance
  rather than resuming its trot, so the bounded retraction gets exactly one 9-step attempt
  and no second cycle. The swing-based re-arm logic itself works as designed (debounced
  touchdown detection fired correctly); it just never gets a lift-off to detect.

## 6. Answers to the decision rules

Outcome matches: **"the collapse disappears but the robot fails to cross the wire."**
Reported separately, as required — this is a stability result, not a recovery result.

## 7. Recommended single next experiment

Diagnose *why* the policy does not resume walking after this one release, before touching
timing or commands again. Minimal, read-only: log the walking policy's raw 12-d action
(`last_a_walk_raw`, already available) and the full 48-d observation it receives for ~20
steps after release (74–94), compared against the same signals from an unperturbed walking
segment. This directly distinguishes two explanations without changing anything: (a) the
observation is in-distribution and the policy is choosing to stand — possibly a locally
stable fixed point of its own forward-velocity tracking — or (b) some observation term
(e.g. FL joint velocity/position bias, or the IMU-based velocity estimate) is still
off-nominal post-release and suppressing the gait. If (a), the next real experiment is a
second, independent bounded-retraction cycle forced after a timeout rather than waiting on
natural lift-off (a different release/re-arm condition, not a command change). If (b), fix
the observation reconstruction, not the hybrid controller.

## 8. Viewer verification

`--debug` opens the D5.15-pattern viewer (never calls `env.step()` itself). All runs below
were executed to completion (250 policy steps), exit code 0, and each `bounded_retraction`
viewer trace is **byte-identical** to its headless trace. Wire collisions are on in all of
them (validated in section 3). I did not re-run `baseline`/`guard_enabled` here: those exact
commands were already run to completion and verified byte-identical against headless in
D5.53 (unchanged script, unchanged behavior — re-verified indirectly in section 3's exact
trace-reproduction check).

```sh
# New variant (run and verified here)
uv run python scripts/d5_55_bounded_retraction.py --variant bounded_retraction --seed 0  --debug --max-steps 250
uv run python scripts/d5_55_bounded_retraction.py --variant bounded_retraction --seed 11 --debug --max-steps 250

# Existing variants (commands unchanged from D5.53; already run and verified there)
uv run python scripts/d5_53_touchdown_guard.py --variant baseline      --seed 0  --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant guard_enabled --seed 0  --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant baseline      --seed 11 --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant guard_enabled --seed 11 --debug --max-steps 250
```

Headless, both seeds, all validation and comparison CSVs here:

```sh
uv run python scripts/d5_55_bounded_retraction.py
```

Parser check: `--help` lists `--variant {all,bounded_retraction}`, `--seed`, `--device`,
`--max-steps`, `--out-dir`, `--debug`.

## 9. Limits

- Two seeds, one `budget` formula, not swept.
- The RELEASE_STANCE→ACTIVE re-arm logic was exercised in neither run (no lift-off
  occurred), so it is validated only by construction (identical to the entry-guard logic
  already proven in D5.53), not by an observed resume.
- I did not instrument the policy observation/action in this run; section 7's experiment is
  the natural follow-up, not yet done.

## Files

Created: `scripts/d5_55_bounded_retraction.py`, `logs/d5_55/report.md`,
`logs/d5_55/comparison_summary.csv`, `logs/d5_55/validation.csv`,
`logs/d5_55/trace_bounded_retraction_seed{0,11}.csv` (written for inspection and as the
viewer-comparison reference; the `baseline`/`guard_enabled` validation runs were not
persisted, per the reuse-existing-data instruction), `logs/d5_55/viewer_bounded_retraction_
seed{0,11}/trace.csv`, `logs/d5_55/viewer_seed{0,11}.log`. Nothing else modified.
