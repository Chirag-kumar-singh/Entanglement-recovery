# D5.53 — Touchdown-Guard Recovery Experiment

Tests the D5.52 hypothesis: activating the hybrid retract on FL one step before its normal
touchdown removes the support the trot support transfer needs. Hybrid commands, gains,
checkpoint, wire model and baseline defaults are unchanged.

**Note on the task text:** the prompt I received ended mid-sentence in section 5
("Log at every policy step: … FL"). Sections 5–end (the full measurement list, analysis
requirements, artifacts, decision options) were not delivered. I used the D5.52 logger fields
plus guard state/transition fields, wrote the artifacts in the D5.52 layout, and give the
result against the D5.52 pre-registered outcomes. If the missing sections ask for anything
else, say so.

## 1. Files

- **New:** [scripts/d5_53_touchdown_guard.py](../../scripts/d5_53_touchdown_guard.py). It imports the
  D5.52 helpers (`Recorder`, `SubstepWireProbe`, `_first`, `_handoff_step`, `_write_csv`) and
  does not modify them.
- **No production file was edited** (`actions.py`, `env_cfg_residual.py`, `hybrid_actuators.py`
  untouched; their existing uncommitted changes are as before). Because nothing shared
  changed, no unit test was added. The disabled-guard regression is instead checked by
  exact trace comparison against the baseline on both seeds (section 3).
- D5.52 script and results untouched. Nothing was committed.

## 2. Guard design

The guard wraps `act._update_fsm`. While deferring, it replaces only the `found` flag of the
`wire_contact` sensor that the FSM sees with zeros for that one call. The FSM is in
NORMAL_WALK then, where that signal is used only for `entering_recovery`, so the walking
position path continues for FL and no hybrid command is applied or partially applied. When
the guard allows entry, the original code runs unchanged.

Inputs are end-of-previous-step values, the same information the FSM has:
FL foot height = `site_pos_w` z (terrain is a plane at z=0; asserted at build), FL foot
vertical velocity = `site_lin_vel_w` z, and the existing `fl_ground_contact_diag` terrain
sensor (found > 0) for touchdown/lift-off.

| State | Transition |
|---|---|
| `IDLE` | real contact **and** h < h_enter **and** vz < 0 → `DEFER_TOUCHDOWN` (entry masked). Real contact otherwise → `PASSED_THROUGH` (entry allowed). |
| `DEFER_TOUCHDOWN` | FL ground contact for ≥ 2 consecutive steps → `DEFER_STANCE` |
| `DEFER_STANCE` | FL ground contact absent for ≥ 2 consecutive steps → `ARMED` (lift-off complete) |
| `ARMED` | real contact passes through: natural entry, hybrid unchanged |
| any `DEFER_*` | > 75 steps → `ARMED`, logged as TIMEOUT (never silent; not triggered in these runs) |

This is gait-conditioned, not a fixed delay: seed 0 deferred 18 steps and seed 11
deferred 11, each ended by observed touchdown and lift-off.

**Threshold:** `h_enter = 0.030 m` (single value, not tuned). Basis: foot sphere radius 0.022 m,
resting site height 0.021–0.022 m, measured swing apex only 0.039 m (D5.16 seed 0,
steps 0–43), so the entire swing lies within 18 mm of contact height. A height gate alone
cannot identify "about to land", so the descent condition (vz < 0) carries the gait
information, and 0.030 m limits the gate to roughly the second half of the descent.
Seed 0's D5.52 value (h = 0.024 m, vz = −0.075 m/s) lies inside the gate.
**Hysteresis:** the height/velocity test is evaluated once per entry candidate (in `IDLE`).
Everything after is an explicit state transition on the terrain-contact sensor with a
2-step debounce, so nothing can toggle around the threshold.

## 3. Validation ([validation.csv](validation.csv), 15/15 pass)

- Seed-0 baseline matches the D5.52 baseline: max abs diff 0.0 over 9 columns × 180+ steps
  (tolerance 1e-5), which is itself equal to D5.49.
- Guard-disabled equals baseline exactly on every shared numeric column and every
  non-numeric column (seeds 0 and 11).
- Wire collisions enabled in all six wire-on runs (excludes unchanged at 6 = adjacent
  segments only; robot–wire substep contacts 1455–1818 per run).
- No hybrid activation during deferral: 18 (seed 0) and 11 (seed 11) deferred steps, with
  phase NORMAL_WALK, FL position gains > 0, thigh/calf recovery controls = 0 on every one.
- Guard sequence defer → touchdown → lift-off → re-arm observed, in order, on both seeds.
  Logged transitions: seed 0 defer step 44, touchdown 46, re-arm 62. Seed 11 defer step 51,
  touchdown 53, re-arm 62 ([guard_events_*.csv](.)).
- Seed 11 triggers naturally (first contact step 50, entry 51). Entry was not forced for
  any run.

## 4. Results ([comparison_summary.csv](comparison_summary.csv), [aligned_trajectory.csv](aligned_trajectory.csv))

250 policy steps per run, same checkpoint/env/reset, wire on. Window k = steps from each
run's own hybrid handoff.

| | seed 0 baseline | seed 0 guarded | seed 11 baseline | seed 11 guarded |
|---|---|---|---|---|
| First wire contact | 43 | 43 | 50 | 50 |
| Hybrid handoff (hybrid active) | 44 | **62** | 51 | **62** |
| Peak \|pitch rate\| k0–9 (rad/s) | 1.80 | **0.24** | 2.38 | **0.24** |
| Base-z drop k0–9 (m) | 0.105 | **0.007** | 0.112 | **0.007** |
| Min world vz k0–9 (m/s) | −1.23 | **−0.08** | −1.37 | **−0.08** |
| Min total ground support k0–9 (N; weight 149) | 0 | **166** | 0 | **166** |
| Peak \|roll rate\| k0–9 (rad/s) | 2.03 | 0.49 | 2.20 | 0.41 |
| All four feet off ground (k) | 4 | 14 | 6 | 12 |
| Crossing / release step | 84 / 88 | 91 / 95 | never | 90 / 94 |
| Recovery exit | 88, crossing | 95, crossing | 98, **cap timeout** | 94, **crossing** |
| Min base z, whole run (m) | 0.204 | **0.171** | 0.196 | **0.173** |
| Peak \|pitch rate\|, whole run (rad/s) | 1.80 | **3.45 @83** | 2.38 | **2.80 @82** |
| Max \|roll\|, whole run (°) | 15.1 | **41.9** | 16.8 | **33.3** |
| Max robot–wire force, 1 substep (N) | 2890 | 1931 | 3414 | 2401 |
| Fell over (70° limit) | no | no | no | no |
| Post-exit z range, +25 steps on (m) | 0.297–0.311 | 0.298–0.321 | 0.289–0.317 | 0.283–0.339 |

Guarded timeline, seed 0 (rows are state after step):
- Steps 44–46: FL touches down under normal walking with the wire against its calf. FL
  ground force rises to 80–90 N, wire force 10–35 N. Pitch rate stays below 0.1 rad/s,
  base z 0.309.
- Step 60: FL lifts off. Step 62: hybrid activates with FL already airborne and FR/RL
  carrying ~90 N each (total ≥ 166 N). Calf velocity stays at −1.9 rad/s or less in
  magnitude at first, base z falls only 0.309 → 0.295 over k0–9.
- **Steps 74–82: second runaway.** FR/RL unload at 76; all four feet are off the ground
  from step 76 to 81; vz reaches −1.19 m/s, pitch rate 0.58 → 3.45 rad/s at 83, base z
  0.171 at 84, and a 1931 N substep wire peak (base/calf/foot) at step 83. A third dip
  at 91–94 (pitch rate −2.7 rad/s, FL calf folded to −2.8 rad, 389 N on the FL foot at step 93 as it lands)
  occurs as the calf is driven past the wire, just before exit at 95.
- Seed 11 shows the same pattern: first runaway absent, a second at steps ~76–82
  (peak 2.80 rad/s at 82, 2401 N wire peak on the base at 82), exit by crossing at 94.

## 5. Interpretation (labelled)

**Measured:**
- M1. Deferring activation until FL has touched down, stood, and lifted off **removes the
  first runaway on both seeds**: peak pitch rate falls to 13% (seed 0) and 10% (seed 11) of
  baseline, base-z drop to 7% and 6%, vz to 6%, with total support ≥ 166 N over k0–9.
- M2. **Seed 11 changes outcome:** baseline hits the 48-step cap with no crossing; the
  guarded run crosses and releases at step 90/94. Seed 0 still crosses, 7 steps later.
- M3. **A second, delayed runaway appears in both guarded runs** (steps ~76–84), with all
  four feet off the ground and a worse whole-run minimum height, peak pitch rate and roll
  than baseline (0.171 vs 0.204 m, 3.45 vs 1.80 rad/s, 41.9° vs 15.1° on seed 0). The
  first k0–9 window does not contain it; it starts 14 steps after handoff.
- M4. Wire forces at the single worst substep are lower with the guard (1931 vs 2890 N,
  2401 vs 3414 N). Neither guarded run falls over, and both settle to walking height after
  exit.

**Supported but not proven (no intervention separates these):**
- H1. The first runaway is caused by retracting FL when the gait needs it to take weight.
  M1 is the direct test and it is consistent, on two seeds with different entry timing.
  Seed 11's guard fired at vz = −0.006 m/s and h = 0.0206 m (a foot essentially at rest at
  the ground), so for that seed the "descent" qualifier barely held. The result there
  mostly shows "defer until after the next stance" works.
- H2. The second runaway is the same mechanism recurring: the hybrid keeps FL folded for
  the rest of recovery (the guard defers *when* it starts, not how long), so at the next
  FR/RL → FL/RR support transfer FL cannot take load. Timing agrees (≈ one half gait cycle,
  22-step period in the D5.52 control: FL/RR stance 66–78, FR/RL 55–70 and 78–92), but
  nothing here shows what the other legs did differently, and RR also was not on the
  ground at 77–81.

**Not supported:** that the guard fixes recovery. It trades one early failure for a later
one and improves seed 11 success, but the whole-run stability metrics are worse.

## 6. Conclusion against the D5.52 plan

D5.52's pre-registered outcomes were: (a) no runaway at first contact *and* none when the
hybrid later activates mid-swing → support-phase hypothesis confirmed; (b) runaway recurs
whenever the hybrid activates → commands destabilize regardless of phase.

Result: the **activation-phase** part of the hypothesis is confirmed (first runaway gone,
on both seeds). The "none later" part **fails** (second runaway at the next support
transfer while FL remains retracted). The evidence supports: *hybrid retraction of a leg
the gait expects to bear load causes loss of support*, whether that happens at activation
or one cycle later. It does not support "the commands destabilize at any phase", because
the 10 steps after a mid-swing activation were stable on both seeds.

## 7. Recommended next single experiment

**Bound the retract duration to the swing: release FL back to position control when the
hybrid has had FL airborne for the remainder of the current swing, and keep the touchdown
guard on re-entry.** Concretely, reuse the guard's gait states: hybrid active only between
FL lift-off and the point where FL stance is expected (measured, not fixed: the interval
FL was airborne in the previous cycle, or an FL-foot-descent detector as used here), then
return FL to the walking path, and re-enter only after the next touchdown/lift-off.
Hybrid commands unchanged. Discriminating outcomes:
- Second runaway disappears and crossing still completes → H2 confirmed: duration, not
  phase at start, is the remaining cause.
- Second runaway persists even with FL back on position control at the transfer →
  support loss has another source (e.g. RR/policy reaction to the perturbed FL state, which
  D5.52 showed changes the frozen policy's outputs for FR_calf) and the next step is to
  log/mask the FL observation entries.
- Crossing fails because FL returns to the walking path before clearing the wire → report
  as a crossing/stability conflict; the guard's seed 0 crossing already needed 30 steps.

Required measurements: the D5.53 logger plus per-step FL-airborne duration and the
hybrid-active/guard state; no new sensors.

## 8. Visualization / playback — verified commands

`--debug` opens the D5.15-pattern native viewer; the callback never calls `env.step()`. All
four were run to completion (250 policy steps = 5.0 s simulated, covering contact, guard
deferral, both runaways, crossing, exit and settling), exit code 0. Each viewer CSV is
**byte-identical** to the corresponding headless trace. I cannot judge the window
visually from here; trace identity is the evidence that it stepped the same physics.

```sh
uv run python scripts/d5_53_touchdown_guard.py --variant baseline      --seed 0  --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant guard_enabled --seed 0  --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant baseline      --seed 11 --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant guard_enabled --seed 11 --debug --max-steps 250
```

Headless, all variants and seeds, validation and all CSVs here (about 20 min on CPU):

```sh
uv run python scripts/d5_53_touchdown_guard.py
```

Parser check: `--help` lists `--variant {all,baseline,guard_disabled,guard_enabled}`,
`--seed`, `--device`, `--max-steps`, `--max-recovery-steps-override`, `--out-dir`,
`--h-enter-m`, `--debounce-steps`, `--max-defer-steps`, `--debug`.

## 9. Limits

- Two seeds; one wire placement; one gait. `h_enter` was set once and not swept.
- Seed 11's defer decision is marginal on the descent test (vz −0.006 m/s).
- Ground forces are leg-vs-terrain netforce magnitudes; sign convention not verified.
- The 20 ms sampling of body/ground signals applies; wire force is per substep.
- I did not examine why RR stays off the ground at steps 77–81 in the guarded seed-0 run.
- A guarded recovery that reaches `POST_RECOVERY` is not the same as the repo's
  "genuine recovery complete" flag; that flag was not evaluated here.

## Files

[report.md](report.md), [comparison_summary.csv](comparison_summary.csv),
[validation.csv](validation.csv), [aligned_trajectory.csv](aligned_trajectory.csv)
(baseline-handoff-relative window, all variants/seeds, with substep wire force),
`trace_<variant>_seed<N>.csv`, `substeps_<variant>_seed<N>.csv`,
`guard_events_guard_enabled_seed<N>.csv`, `viewer_<variant>_seed<N>/trace.csv` and `.log`,
`headless_run.log`.
