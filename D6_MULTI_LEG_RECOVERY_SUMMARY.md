# D6: Multi-leg wire-entanglement recovery (Go2) — summary

Status as of 2026-11-23. Nothing here is committed yet. Single-leg recovery
(D5.58, about 128/136 seeds) is left as it is; this document covers extending it to
two legs (front pair, rear pair) and all four legs (front first, then rear).

## What was built

- `src/mjlab/tasks/recovery/mdp/multi_leg.py`: `MultiLegRecoveryAction`, a wrapper
  that runs one `ResidualRecoveryAction` per leg. Each leg has its own contact
  sensor (`wire_contact_{leg}`); contacts of legs outside the tested set are
  excluded. Steps can be serialized so one leg's step does not start while another
  is stepping.
- `scripts/d6_1_front_pair_wire.py`: runner and verdicts (`--pair front|rear|all`).
  `--seeds 0-39 --jobs 3` spawns one process per seed (keep `--jobs` at 3 or lower;
  7 parallel sweeps crashed the laptop). `--debug --seed N --out-dir ...` opens the
  viewer.
- Opt-in additions to `ResidualRecoveryAction` (`actions.py`): `step_block`
  hook, retry (`step_max_attempts`), own-touchdown window (`step_td_age_min/max`),
  `crossing_fallback_steps`. Opt-in additions to the wrapper: `serialize_gap_steps`,
  `support_gate_z`, `support_diagonal`, `support_mask`.
- All defaults are unchanged; nothing new is on unless a flag is passed.

## Recovery mechanism (same in every setup)

`scripted_step`: when the wire touches a leg, that leg's foot follows an IK path
back and up, forward over the wire, then down (4/5/4 steps). The walking policy
keeps running for the other legs. Only the **start rules** (when each leg's step
begins) differ between the setups below.

## Setups compared

**Earlier setup** (the one used at HEAD): fixed start delays tuned at one start
pose (FL waits 23 steps, FR 0, RL 0, RR gated on the gait), 2 attempts per leg,
FR step reach 0.18 m, steps serialized.

**New setup** (front legs only; rear legs keep the earlier rules): each leg starts
when its own foot has been on the ground 17-19 steps (`--td-window 17 19`), the
second leg waits 12 steps after the first leg's step ends
(`--serialize-gap-steps 12`), a leg waits while the other front foot is in the air
(`--support-gate-z 0.04`), a leg starts only if the two support feet landed within
6 steps (`--step-gate FL 6 FR 6`), and a crossing latches once the foot is past the
wire for 10 steps with no contact (`--crossing-fallback 10`). Use
`--new-legs FL FR` to apply it to the front legs only.

## Results (pass = both legs across, no fall, stable, walks on)

Two kinds of test. **Default start**: every run starts at the same place. **Varied
approaches**: 40 start positions across one gait cycle
(`--start-x-range -0.485 -0.35`). They are not comparable with each other.

| Case | Test | Earlier setup | New setup |
|---|---|---|---|
| Rear 2 legs | default start | 60/60 (100%) | — |
| Rear 2 legs | varied, seeds 0-39 | 39/40 (98%) | 13/40 if applied to the rear legs (do not) |
| Front 2 legs | default start | 47/60 (78%) | not run |
| Front 2 legs | varied, 80 seeds | 29/80 (36%) | 46/80 (57.5%) |
| All 4 legs | default start | 36/60 (60%) | 14/60 (23%) |
| All 4 legs | varied, 80 seeds | 23/80 (29%) | 31/80 (39%) |

Small samples: expect several points of uncertainty, more on the varied tests.

Front-pair steps in the new setup: touchdown rule alone 38% (0-39); + gap 12 45%
(80); + support gate 54%; + crossing fallback 46%; + gait gate 57.5%.

## What helped and what did not

- **Retry (2 attempts per leg)**: the one big gain (front 8/20 to 13/20).
- **FR step reach 0.18 m** (instead of 0.30): falls 12 to 2 over 60 seeds.
- **Touchdown rule, gap, support gate, gait gate**: each added a few points on the
  varied test and together lose badly at the default start.
- **No effect or worse**: body hold (cmd/freeze), partner hold, roll gate, FL-first
  order, blanket RR delay, own-foot-planted gate, contact-based retry,
  `reactive_step` (removed, 0/16 parity), `support_diagonal` (same as the simple
  support gate).

## Why the front pair still fails (findings from the logs)

1. A step succeeds when the foot peaks at about 0.15 m or higher; failures peak at
   0.05-0.13 m while the body sags to 0.14-0.18 m.
2. The second front leg's step succeeds when the support feet stay down. FR as the
   second leg: 24/24 crossed when the support feet stayed down, 8/30 when the
   walking gait lifted them during the step (FL as second leg: 9/13 vs 1/13).
3. FL as the second leg starts with the wire above its calf (foot about 4 cm past
   the wire, calf about 6 cm below wire height), so retract-back then lift goes up
   into the wire. This is common to passing and failing steps.
4. Seeds 62 and 54 cross visually but the strict latch (calf must rise above the
   wire) never fires, so the leg keeps recovering. `crossing_fallback_steps`
   fixes seed 62.
5. Timing shifts with start position: at x = -0.43 / -0.41 / -0.39 the passing
   delay combinations differ, with no single pair of delays that works everywhere.

## Decisions and open questions

- Keep the earlier setup as the default (60% for all four legs at the default
  pose); the new setup stays opt-in. It is better only on varied approaches.
- Rear after the front stage is weaker (about 69-76% pass) than rear alone (98%).
  Not investigated.
- Untried: a geometry-based front step (use foot and wire position instead of
  timing), and fixing the second leg's retract when the wire is already above it.
- Reference-paper rule: borrow control logic from Yim et al. 2023 and Quad-SDK
  (`underbrush_inverse_dynamics.cpp`), never their numeric values (different robot).

## Housekeeping before commit

- Run `make check` (not yet run). Add the wrapper test and the changelog entry
  under "Upcoming version" in `docs/source/changelog.rst`.
- A blanket `ruff --fix` modified six unrelated files (`recovery/__init__.py`,
  `env_cfg.py`, `env_cfg_wire.py`, `mdp/__init__.py`, `mdp/metrics.py`,
  `mdp/rewards.py`, plus `mdp/terminations.py`); revert them with
  `git checkout --` on those paths.
- Per-seed run folders were archived to `~/d6_1_untracked_runs_2026-11-22.tar.gz`
  (73 MB) and removed. Verdict summaries remain in
  `logs/d6_1_front_pair_wire/sweeps/*.txt`; summaries of the last runs (gap fix,
  validation, support gate, fallback, mixed per-leg) are not saved there yet.

> **Sweep vs viewer.** Any command with `--seeds A-B --jobs N` is a headless sweep: it
> prints one verdict per seed and never opens a window. Only commands with `--seed N
> --debug` open the viewer. Viewer commands are in the "Visual commands" sections at the
> end of this file, one per result.

## Commands behind the results

Added 2026-11-23. Every command is run from the repo root. For a visual run of any
sweep line below, replace `--seeds A-B --jobs N` with `--seed N --debug` (opens the
viewer; one seed). Per-run settings were recovered from the `run_log.txt` files in
`~/d6_1_untracked_runs_2026-11-22.tar.gz`. From now on `run_log.txt` ends with a
`FULL_CONFIG={...}` line holding every flag, so a run can always be reproduced.

Common prefix for the front-pair benchmark ("varied approaches", seeds 0-39):

    uv run python scripts/d6_1_front_pair_wire.py --pair front \
      --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 3 \
      --max-attempts FL 2 FR 2 --max-recovery-steps 96

| Result | Settings (recovered from run logs) | Confidence |
|---|---|---|
| Front, varied, 13/40 (`bench_front`) | the prefix above, nothing else | exact |
| Front, varied, 13/40 (`bench_front_ownp`) | prefix + `--own-planted FL True` | exact |
| Front, varied, 15/40 (`bench_front_td`) | prefix + `--td-window 17 19` | exact |
| Front, varied, 19/40 (`bench_front_td_gapfix12`) | prefix + `--td-window 17 19 --serialize-gap-steps 12` | partly: the header did not record gate or fallback flags |
| Front, 57.5% (46/80), the best front-pair result | `uv run python scripts/d6_1_front_pair_wire.py --pair front --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR` | VERIFIED 2026-11-23: re-run gave exactly 46/80, 0 falls (`logs/d7_2_best_baseline/best.txt`) |
| Front, seeds 20-59, 32/40 (`val_fradv018_s20_59`) | `--step-advance FR 0.18 --max-attempts FL 2 FR 2 --max-recovery-steps 96` | start position not recorded |
| Rear pair, 39/40 | `--pair rear --start-x-range -0.485 -0.35 --seeds 0-39` (earlier setup) | not re-checked |
| All four, earlier setup, 10/40 (`bench_all`) | `--pair all` with the front-pair prefix, 2 attempts per leg | exact for attempts and cap |

## D7 (2026-11-23): why the front pair sags, and three fixes that did not help

All runs below used the bare defaults (ONE attempt per leg, recovery cap 48, no new-setup
flags), so they are NOT comparable with the table above. The baseline is 7/20 on seeds
0-19, not the best front-pair setting. Re-run with the best settings before trusting them.

| Result | Command | Output |
|---|---|---|
| Wire removed, steps forced at the wire-on entry steps: body min height 0.205 m vs 0.208 m with the wire, 0.307 m walking only (40 seeds) | `uv run python scripts/d7_0_forced_step_wire_off.py --seeds 0-39 --jobs 2` | `logs/d7_0_forced_step/summary.csv` |
| Body lift on standing legs, wire off, seeds 0-11: +2.5 cm (a) and +5 cm (b); opposite sign sags more | `... d7_0_forced_step_wire_off.py --seeds 0-11 --jobs 2 --spans-from logs/d7_0_forced_step/summary.csv --lift-thigh -0.2 --lift-calf 0.3 --label lift_a` (b: `-0.4 0.6`) | `logs/d7_0_forced_step/lift_*.csv` |
| Real wire, seeds 0-19: baseline 7/20, lift_a 6/20, lift_b 9/20 (5 falls) | `uv run python scripts/d6_1_front_pair_wire.py --pair front --start-x-range -0.485 -0.35 --seeds 0-19 --jobs 2 [--support-lift-thigh=-0.2 --support-lift-calf=0.3]` | `logs/d7_0_forced_step/real/` |
| Lift-and-slide, real wire, seeds 0-19: 0/20 for `step_back` 0 and for -0.05 | same as above with `--step-back FL 0.0 FR 0.0` (and `-0.05`) | `logs/d7_1_lift_slide/` |

## D7.2 (2026-11-23): re-test on the best front-pair setup

Baseline = the verified best command above. Seeds 0-39 of that run: 20/40. All runs add
only the flag shown. Seeds 0-39 unless noted, paired with the baseline per seed.

| Change | Pass | Falls | Paired vs baseline | Output |
|---|---|---|---|---|
| none (baseline, seeds 0-39) | 20/40 | 0 | - | `logs/d7_2_best_baseline/best.txt` |
| `--support-lift-thigh=-0.4 --support-lift-calf=0.6` (body lift on standing legs) | 32/40 | 0 | +14 seeds gained, 2 lost | `logs/d7_2_best_baseline/lift_b.txt` |
| `--step-back FL 0.0 FR 0.0` (lift-and-slide) | 2/40 | 0 | 1 gained, 19 lost | `logs/d7_2_best_baseline/back0.txt` |

Both commands are the verified best command plus the flag in the first column.
The earlier D7 conclusion that the body lift does not help was wrong: it came from the
weak one-attempt baseline above.

Held-out check, seeds 40-79 (same verified best command plus `--support-lift-thigh=-0.4
--support-lift-calf=0.6`, `--seeds 40-79`): 33/40 vs baseline 26/40, 0 falls, 9 gained and
2 lost. Output: `logs/d7_2_best_baseline/lift_b_s40_79.txt`. Over all 80 seeds the body
lift gives 65/80 (81%) against 46/80 (57.5%) for the best setup without it.

## Visual commands (open the viewer for one seed)

Same flags as the sweep, with `--seed N --debug` instead of `--seeds A-B --jobs N`.
`--out-dir` keeps the viewer run from overwriting an older run folder. Run from the repo
root. All use the verified best front-pair setup; the second block adds the body lift
(the current best, 65/80 on seeds 0-79).

Best setup, no body lift (seed 1 fails here, seed 0 passes):

    uv run python scripts/d6_1_front_pair_wire.py --pair front \
      --start-x-range -0.485 -0.35 --seed 1 --debug --out-dir logs/d7_viewer/nolift_seed1 \
      --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR

Best setup plus body lift (seed 1 passes; seed 2 still fails with FL not across):

    uv run python scripts/d6_1_front_pair_wire.py --pair front \
      --start-x-range -0.485 -0.35 --seed 1 --debug --out-dir logs/d7_viewer/lift_seed1 \
      --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR \
      --support-lift-thigh=-0.4 --support-lift-calf=0.6

Change `--seed 1` and the folder name for other seeds. With the lift, seeds 0-39 fail on
2, 5, 10, 14, 16, 23, 26, 34 and seeds 40-79 on 42, 52, 57, 58, 65, 73, 78. Seeds 4, 8
and 9 also fail without the lift and pass with it.

## Visual commands for the earlier runs

Settings come from the `run_log.txt` files in the archive (see "Commands behind the
results"). Each block is the sweep command with `--seed N --debug`; use any seed. "Default
start" runs have no `--start-x-range`. The varied runs assume `--start-x-range -0.485
-0.35`, the range in the summary; the log records only the per-seed start x, so check
that if a seed behaves differently from its sweep.

Single leg (D5.58, about 128/136 seeds; any of FL, FR, RL, RR):

    uv run python scripts/d5_58_per_leg_recovery_window.py --leg FL --seed 0 \
      --recovery-mode scripted_step --debug

Rear pair, earlier setup, 39/40 varied (`bench_rear`):

    uv run python scripts/d6_1_front_pair_wire.py --pair rear \
      --start-x-range -0.485 -0.35 --seed 0 --debug --out-dir logs/d7_viewer/rear_seed0 \
      --max-attempts RL 2 RR 2 --max-recovery-steps 96

Rear pair with the touchdown rule only, 33/40 varied (`bench_rear_td`): same as above
plus `--td-window 17 19`. The 13/40 "new setup on the rear legs" row in the results table
is a different run with more flags; its folder is not in the archive.

All four legs, earlier setup, 10/40 varied (`bench_all`):

    uv run python scripts/d6_1_front_pair_wire.py --pair all \
      --start-x-range -0.485 -0.35 --seed 0 --debug --out-dir logs/d7_viewer/all_seed0 \
      --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96

All four legs, front touchdown rule, 7/40 varied (`bench_all_td`): same as above plus
`--td-window 17 19`.

All four legs, default start, FR reach 0.18, 36/60 (`all_fradv018`):

    uv run python scripts/d6_1_front_pair_wire.py --pair all --seed 0 --debug \
      --out-dir logs/d7_viewer/all_default_seed0 \
      --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96

(FR reach 0.18 is already the default.) The `all_rr30` variant (49/73 on seeds 0-59,
default start) adds `--start-delay RR 30`.

Not recovered: the exact commands behind "rear 60/60 default start" and "front 47/60
default start" (no matching run folder in the archive). The `FULL_CONFIG` line in new
run logs removes this problem from now on.

## D7.3 (2026-11-23): four legs with the body lift

Seeds 0-39, `--pair all`. Common prefix (the verified best front-pair flags, 2 attempts on
all four legs):

    uv run python scripts/d6_1_front_pair_wire.py --pair all \
      --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 \
      --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR

| Added to the prefix | Pass | Front-stage fails | Rear-stage fails | Output |
|---|---|---|---|---|
| nothing | 13/40 | 20 | 7 | `logs/d7_3_four_leg/nolift.txt` |
| `--support-lift-thigh=-0.4 --support-lift-calf=0.6` | 12/40 | 6 | 22 | `logs/d7_3_four_leg/lift_b.txt` |

The lift fixes the front stage (20 -> 6 failures) but the rear stage gets much worse
(7 -> 22: sidestep drift 0.15-0.33 m, and falls after the front stage cleared). Next
test: only front-leg steps trigger the lift (`--support-lift-trigger FL FR`), so the
rear legs' own steps run without it; result in `logs/d7_3_four_leg/lift_b_fronttrig.txt`.

Result of the front-only trigger (added to the prefix: `--support-lift-thigh=-0.4
--support-lift-calf=0.6 --support-lift-trigger FL FR`): **28/40 (70%)**, front-stage fails 6,
rear-stage fails 6, paired against the no-lift run 16 gained and 1 lost. So the rear-stage
damage came from lifting during the rear legs' own steps. Output:
`logs/d7_3_four_leg/lift_b_fronttrig.txt`. Remaining rear failures: 7 "not across", 3
"dragged back", 1 collapse, 1 sidestep. Held-out seeds 40-79 (same command,
`--seeds 40-79`): **24/40 (60%)**, front-stage fails 7, rear-stage fails 9
(`logs/d7_3_four_leg/lift_b_fronttrig_s40_79.txt`). Over seeds 0-79: 52/80 (65%). The
earlier four-leg figure for the new setup without the lift is 31/80 (39%) in the table
at the top; no same-seed no-lift run exists for seeds 40-79 yet.

Viewer commands for these runs: see "Visual commands for D7.3" below.

## Visual commands for D7.3 (four legs; these open the viewer)

Close the viewer window to end the run. Run from the repo root. Each command uses one
seed; change `--seed` and the folder name for others. Seeds are from the 0-39 run.

Current best, four legs with the front-only body lift (28/40 on seeds 0-39, 24/40 on
40-79). Seed 3 passes; seed 1 fails in the rear stage (RR not across, FR dragged back);
seed 8 fails (RL not across); seeds 4, 9, 12 pass:

    uv run python scripts/d6_1_front_pair_wire.py --pair all \
      --start-x-range -0.485 -0.35 --seed 3 --debug --out-dir logs/d7_viewer/all_lift_seed3 \
      --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR \
      --support-lift-thigh=-0.4 --support-lift-calf=0.6 --support-lift-trigger FL FR

Same seed with no lift (13/40 on seeds 0-39), to compare:

    uv run python scripts/d6_1_front_pair_wire.py --pair all \
      --start-x-range -0.485 -0.35 --seed 3 --debug --out-dir logs/d7_viewer/all_nolift_seed3 \
      --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR

Lift on every leg's step (12/40; the version that hurt the rear stage), seed 3:
the first command with `--support-lift-trigger FL FR` removed.

I checked that the first command starts and keeps running on this machine (display
available); I have not looked at the window itself.

## D7.4 (2026-11-24): why the rear stage fails after the front stage

The 15 rear-stage failures of the best four-leg setup (seeds 0-79: 28/40 + 24/40, command
in D7.3). Analysis of the per-seed `trajectory.csv` files plus a re-run of the 15 failing
and 6 passing seeds that logs the support-lift level every step (it also reproduced
the same 15 failures and 6 passes exactly):

    uv run python scripts/d6_1_front_pair_wire.py --pair all \
      --start-x-range -0.485 -0.35 --seeds 1,8,18,19,35,38,43,45,47,51,60,62,67,76,77,3,4,9,12,13,15 \
      --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR \
      --support-lift-thigh=-0.4 --support-lift-calf=0.6 --support-lift-trigger FL FR \
      --out-dir logs/d7_4_rear_stage/lift       # output: logs/d7_4_rear_stage/lift.txt

Visual run of one: same flags with `--seed 8 --debug` instead of `--seeds ... --jobs 2`.
(Seeds 1 and 8 fail in the rear stage; 3 passes.)

Findings:
- **Start state at rear contact is the same** in passes and failures: base z 0.31 m,
  roll and pitch about 0, forward speed 0.25 m/s, all feet planted in 63 of 67 runs.
- **The front-leg lift does not linger:** the logged lift level is 0.00 at first rear
  contact and for every later step in all 21 re-run seeds.
- **The failing rear step is too low:** the rear foot peaks about 0.11-0.12 m in the 25
  steps after contact for legs that never cross, against 0.16-0.17 m for legs that do
  (finding 1 above: a step needs about 0.15 m). In 15 failures the rear leg stepped
  (step starts one step after contact) but its calf never cleared the wire: 10 with RL,
  10 with RR, some both.
- **RL gait phase (weak evidence, small counts):** RL crossed 44 of 45 times when its foot
  was in the air or just landed at contact; RL planted 3-7 steps crossed 0 of 4. RR showed
  no phase difference.
- **Rear contact comes later in failures** (median step 196 vs 177).
- **Not mainly a lift side effect:** rear-only passes 5 of the 6 rear failures in seeds
  0-39, and 5 of those 6 failed in the front stage without the lift. Only seed 18 passed
  without the lift and fails with it.

## D7.5 Does the body lift make the robot unstable? Smaller lifts

Observation from the viewer: with the full lift the robot stands stretched and wobbles
even in passing runs. The pass/fail rule is geometric and cannot see that, so stability
was measured from the logged trajectories (steps 40-200: mean body height, peak roll and
pitch, RMS roll/pitch angular velocity). Standing height in this setup is 0.308 m.

Front pair, best setup, seeds 0-39. Stability columns are the mean over the 10 seeds
that pass in every configuration, so they are rough.

| Lift (thigh / calf, rad) | Passes /40 | Body z mean (m) | Peak roll (deg) | Peak pitch (deg) | Ang. vel RMS (rad/s) |
|---|---|---|---|---|---|
| none (baseline) | 20 | 0.301 | 4.8 | 6.6 | 0.43 |
| -0.1 / +0.15 | 23 | 0.310 | 4.7 | 10.5 | 0.52 |
| -0.2 / +0.3 (half) | 29 | 0.321 | 6.2 | 10.7 | 0.57 |
| 0 / +0.3 (calf only) | 30 (2 falls) | 0.315 | 8.3 | 10.2 | 0.62 |
| -0.4 / +0.6 (full) | 32 | 0.332 | 8.9 | 14.7 | 0.73 |

Over the 18 seeds that pass in both baseline and full lift: roll 4.6 -> 8.1 deg, pitch
5.8 -> 15.5 deg, angular velocity 0.41 -> 0.75 rad/s, body 0.301 -> 0.330 m (about
2 cm above standing height). The viewer impression is confirmed.

Held-out seeds 40-79 with the half lift: **33/40** (baseline 26/40; 9 gained, 2 lost,
0 falls). Over all 80 seeds: baseline 46, half lift 62, full lift 65.

Conclusion: the half lift keeps most of the gain with less instability and is the
working setting. It is still less stable than no lift in pitch and angular velocity, so
it is a trade-off, not a fix; the lift stands in for the level, momentum-driven gait the
paper's robot has.

Commands (best setup plus lift flags, `--out-dir` under `logs/d7_5_small_lift/`):

```
B="--pair front --start-x-range -0.485 -0.35 --jobs 2 --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR"
uv run python scripts/d6_1_front_pair_wire.py $B --seeds 0-39  --support-lift-thigh=-0.2 --support-lift-calf=0.3 --out-dir logs/d7_5_small_lift/half
uv run python scripts/d6_1_front_pair_wire.py $B --seeds 40-79 --support-lift-thigh=-0.2 --support-lift-calf=0.3 --out-dir logs/d7_5_small_lift/half_s40_79
uv run python scripts/d6_1_front_pair_wire.py $B --seeds 0-39  --support-lift-thigh=-0.1 --support-lift-calf=0.15 --out-dir logs/d7_5_small_lift/tenth
uv run python scripts/d6_1_front_pair_wire.py $B --seeds 0-39  --support-lift-thigh=0 --support-lift-calf=0.3 --out-dir logs/d7_5_small_lift/calf
```

Visualizer for one seed (drop `--seeds` and `--jobs`, add `--seed N --debug`):

```
uv run python scripts/d6_1_front_pair_wire.py --pair front --start-x-range -0.485 -0.35 --seed 5 --debug --out-dir logs/d7_viewer/half_seed5 --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3
```

Results: `logs/d7_5_small_lift/{half,tenth,calf,half_s40_79}.txt`.

### D7.6 Four legs with the half lift

Same four-leg setup as D7.3 (`--pair all`, 4 legs x 2 attempts, lift triggered by front-leg
steps only), with the half lift in place of the full one.

| Lift | Seeds 0-39 | Seeds 40-79 | All 80 |
|---|---|---|---|
| none | about 13 | not run | - |
| half (-0.2 / +0.3) | 19 (9 front fails, 12 rear fails) | 26 (7 front, 7 rear) | **45/80** |
| full (-0.4 / +0.6) | 28 | 24 | 52/80 |

```
uv run python scripts/d6_1_front_pair_wire.py --pair all --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-trigger FL FR --out-dir logs/d7_6_four_leg_half/s0_39
```

(Seeds 40-79: change `--seeds` and `--out-dir` to `s40_79`. Viewer: replace `--seeds ... --jobs 2`
with `--seed N --debug`.) Results: `logs/d7_6_four_leg_half/{s0_39,s40_79}.txt`.

### D7.7 Stability of the four-leg runs (seeds 0-39)

Measured from `trajectory.csv` over steps 40 to the end of the run (covers the rear
stage). Front-trigger lift in both lifted runs. Standing height 0.308 m.

| Lift | Passes /40 | z mean (m) | Peak roll (deg) | Peak pitch (deg) | Ang. vel RMS (rad/s) |
|---|---|---|---|---|---|
| none | 13 | 0.302 | 6.4 | 7.8 | 0.55 |
| half | 19 | 0.313 | 5.8 | 12.0 | 0.59 |
| full | 28 | 0.319 | 9.8 | 15.3 | 0.71 |

(Means over each configuration's own passing seeds. Over the 6 seeds that pass in all
three: roll 5.0 / 5.5 / 9.4, pitch 7.1 / 11.8 / 15.4, ang. vel 0.50 / 0.60 / 0.74.)
Only 6 seeds pass in all three, so treat the paired numbers as rough.

### Visual command: four legs, half lift (D7.6)

Opens the viewer for one seed (change `SEED`; close the window to end the run). Same flags
as the D7.6 sweep, with `--seed N --debug` in place of `--seeds ... --jobs 2`.

```
SEED=3
uv run python scripts/d6_1_front_pair_wire.py --pair all --start-x-range -0.485 -0.35 --seed $SEED --debug --out-dir logs/d7_viewer/fourleg_half_seed$SEED --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-trigger FL FR
```

## D7.8 Slower lift ramp and shorter gate timeout (front pair, half lift, seeds 0-39)

Why: in four-leg seeds 5 and 8 the step started only after the 60-step `step_gate_timeout`
(no foot touchdown ever fell in the 17-19 window because the robot stood pressed against
the wire), and the lift ramped in over only 3 steps, which launched the body up
(0.30 -> 0.38 m in 12 steps in seed 5).

| Setting (on top of half lift) | Passes /40 | Falls | z mean | z range | Peak roll | Peak pitch | Ang. vel RMS |
|---|---|---|---|---|---|---|---|
| baseline, no lift | 20 | 0 | 0.301 | 0.081 | 4.7 | 5.9 | 0.42 |
| half lift (ramp 3, timeout 60) | 29 | 1 | 0.321 | 0.115 | 5.9 | 11.5 | 0.58 |
| `--support-lift-ramp-steps 12` | **35** | 0 | 0.321 | 0.111 | **4.3** | 9.3 | **0.48** |
| `--step-gate-timeout 30` | **37** | 0 | 0.324 | 0.103 | 7.6 | 11.2 | 0.52 |

Stability columns are means over each configuration's own passing seeds. Each setting was
tested alone; the combination is not yet tested, and these are seeds 0-39 only (no
held-out seeds yet).

```
B="--pair front --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 --max-attempts FL 2 FR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3"
uv run python scripts/d6_1_front_pair_wire.py $B --support-lift-ramp-steps 12 --out-dir logs/d7_8_ramp_timeout/ramp12
uv run python scripts/d6_1_front_pair_wire.py $B --step-gate-timeout 30 --out-dir logs/d7_8_ramp_timeout/to30
```

Viewer: replace `--seeds 0-39 --jobs 2` with `--seed N --debug`. Results:
`logs/d7_8_ramp_timeout/{ramp12,to30}.txt`.

## D7.9 Combined: half lift + ramp 12 + gate timeout 30

Setting: `--support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12
--step-gate-timeout 30` on top of the best setup (four-leg runs also `--support-lift-trigger
FL FR`, 2 attempts per leg).

| Setup | Seeds 0-39 | Seeds 40-79 | All 80 | Earlier best (same setup) |
|---|---|---|---|---|
| Front pair | 38 | 36 | **74/80** | 46 (no lift), 62 (half), 65 (full) |
| Four legs | 29 (2 front, 9 rear fails) | 31 (4 front, 5 rear fails) | **60/80** | 52 (full), 45 (half) |

Stability (means over each configuration's own passing seeds; steps 40 to end; standing
height 0.308 m):

| Setup | Passes | z mean (m) | z range (m) | Peak roll (deg) | Peak pitch (deg) | Ang. vel RMS (rad/s) |
|---|---|---|---|---|---|---|
| Front 0-39, half lift | 29 | 0.319 | 0.115 | 5.9 | 11.5 | 0.55 |
| Front 0-39, combined | 38 | 0.324 | 0.096 | 5.4 | 8.8 | 0.40 |
| Front 40-79, half lift | 33 | 0.320 | 0.120 | 5.3 | 11.7 | 0.53 |
| Front 40-79, combined | 36 | 0.324 | 0.099 | 5.4 | 8.8 | 0.40 |
| Four 0-39, no lift | 13 | 0.302 | 0.082 | 6.4 | 7.8 | 0.55 |
| Four 0-39, half lift | 19 | 0.313 | 0.129 | 5.8 | 12.0 | 0.59 |
| Four 0-39, full lift | 28 | 0.319 | 0.142 | 9.8 | 15.3 | 0.71 |
| Four 0-39, combined | 29 | 0.315 | 0.128 | 5.4 | 8.9 | 0.45 |
| Four 40-79, half lift | 26 | 0.313 | 0.127 | 5.8 | 11.6 | 0.55 |
| Four 40-79, full lift | 24 | 0.319 | 0.145 | 9.6 | 15.8 | 0.70 |
| Four 40-79, combined | 31 | 0.315 | 0.137 | 5.4 | 8.7 | 0.46 |

The combined setting has the highest pass count in every group and lower roll, pitch and
angular velocity than the half or full lift. Pitch (8.7-8.9 deg) is still above the
no-lift 7.8 deg in the four-leg case, and z range is still above no lift. Rear stage
remains the main loss in the four-leg runs (14 of the 20 failures).

```
C="--start-x-range -0.485 -0.35 --jobs 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30"
uv run python scripts/d6_1_front_pair_wire.py $C --pair front --max-attempts FL 2 FR 2 --seeds 0-39 --out-dir logs/d7_9_combined/front_0_39
uv run python scripts/d6_1_front_pair_wire.py $C --pair all --max-attempts FL 2 FR 2 RL 2 RR 2 --support-lift-trigger FL FR --seeds 0-39 --out-dir logs/d7_9_combined/four_0_39
```

(Seeds 40-79: change `--seeds` and the `--out-dir` suffix.) Results:
`logs/d7_9_combined/{front_0_39,front_40_79,four_0_39,four_40_79}.txt`.

Visualizer, four legs (change `SEED`; close the window to end):

```
SEED=3
uv run python scripts/d6_1_front_pair_wire.py --pair all --start-x-range -0.485 -0.35 --seed $SEED --debug --out-dir logs/d7_viewer/combined_seed$SEED --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR
```

Visualizer, front pair: use `--pair front --max-attempts FL 2 FR 2`, drop the
`--support-lift-trigger` flag, keep the rest.

## Logs removed from the repo (D7)

The per-seed `seed*/trajectory.csv` and `run_log.txt` files of the D7 runs (about 1,900
files, 340 MB) are not committed. What is kept instead:

- Every sweep's one-file summary, `logs/d7_*/<run>.txt` (verdict, drift, minimum base
  height and event steps per seed) and the `run.sh` files that hold the exact commands.
- `logs/d7_summary/per_seed_stability.csv`: one row per run and seed (1,693 rows, D7.2-D7.15) with the
  verdict and the stability metrics used in D7.5-D7.9 (body height mean/min/range, peak
  roll and pitch, RMS roll/pitch angular velocity), for steps 40 to the end (`*_full`) and
  40 to 200 (`*_w200`). Regenerate it from trajectories with
  `uv run python scripts/d7_stability.py`.
- `logs/d7_summary/samples/`: full trajectories for the four-leg seeds discussed in the
  text (combined setting: 3, 5, 8; half lift: 5, 8).

Also removed earlier, because they were scratch or not comparable: `logs/d7_viewer/`,
`logs/d7_0_forced_step_run.txt` (raw stdout), and the per-seed folders of the weak-baseline
`d7_0_forced_step/real/{base,lift_a,lift_b}` runs, `d7_1_lift_slide/{back0,backm05}` and the
all-legs-lift `d7_3_four_leg/lift_b`. Any per-seed trajectory can be regenerated by running
that run's command from the summary with the same seed.

## D7.10 Rear legs with the new start rules and a lift: negative result

Question: the rear legs still use the old start rule (`--new-legs FL FR`) and the lift is
triggered by front steps only. Do the touchdown window, crossing fallback and a lift during
rear steps fix the 14 rear failures of D7.9?

First test, on 24 seeds picked from D7.9 (the 14 rear failures: 4, 8, 11, 18, 20, 21, 31, 32,
35, 46, 64, 67, 73, 75, plus the first 10 passing seeds). The D7.9 setting passes exactly the
10 passing seeds by selection.

| Variant | Passes /24 | Of 14 failures now passing | Of 10 passes now failing |
|---|---|---|---|
| A: `--new-legs FL FR RL RR` | 12 | 6 | 4 |
| B: lift on every leg's step (no `--support-lift-trigger`) | 13 | 5 | 2 |
| C: A and B | 14 | 8 | 4 |

That sample looked favourable, but it was biased toward seeds that already fail. Full run of
C on all 80 seeds:

| Seeds | D7.9 setting | Variant C | Gained | Lost |
|---|---|---|---|---|
| 0-39 | 29 | 24 | 6 | 11 |
| 40-79 | 31 | 23 | 2 | 10 |
| all 80 | **60** | **47** | 8 | 21 |

Variant C is worse overall (47 against 60), with 6 front-stage and 27 rear-stage failures.
The rear stage is sensitive: different rules fix different seeds and break others, so a
failure-seed sample cannot be used to judge a rear change. D7.9 stays the best four-leg
setting. Not tested alone on all 80 seeds: A and B.

```
C="--pair all --start-x-range -0.485 -0.35 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --new-legs FL FR RL RR"
uv run python scripts/d6_1_front_pair_wire.py $C --seeds 0-39 --out-dir logs/d7_10_rear/C_full_0_39
```

(Variants A and B: see `logs/d7_10_rear/run.sh`. Viewer: replace `--seeds ... --jobs 2` with
`--seed N --debug`.) Results: `logs/d7_10_rear/*.txt`.

## D7.11 Rear settling: longer step gap and rear roll gate (four legs, seeds 0-79)

Motivation (D7.10 follow-up, re-runs of seeds 4, 21, 73, 75, 20, 35, 18, 32 and the saved
seed 8): the first rear step starts 1-3 steps after contact and dips the body 4-8 cm; the
second rear leg starts 1-3 steps after the first finishes, with the body still low
(0.277-0.287 m) and then rolls 15-28 deg. Seed 8's RL entered recovery from a one-step wire
brush (contact sensor true at step 156, foot 2 cm behind the wire).

Base: D7.9 combined setting (60/80). Only the flag shown changes.

| Variant | Passes /80 (0-39 / 40-79) | Gained | Lost | Failures |
|---|---|---|---|---|
| D7.9 (gap 12, no roll gate) | 60 (29 / 31) | - | - | 6 front, 14 rear |
| `--serialize-gap-steps 24` | 36 (17 / 19) | 9 | 33 | 0 front, 44 rear |
| `--max-roll-deg RL 5 RR 5` | 61 (30 / 31) | 1 | 0 | 6 front, 13 rear |
| both | 34 (15 / 19) | 9 | 35 | 0 front, 46 rear |

Conclusions: the roll gate at 5 deg changes almost nothing (+1 seed, no losses), so it rarely
holds a rear step. The 24-step gap is much worse (36/80): `--serialize-gap-steps` is global, so
it also delays the front legs (for example FR's step moved from step 84 to 96 in seed 8) and
the whole sequence shifts; the cause was not isolated. D7.9 remains the best four-leg setting.
A body-height gate for rear steps and a rear-only gap were not tested (the gap has no
per-leg option).

```
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 24 --out-dir logs/d7_11_rear_settle/gap24
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 12 --max-roll-deg RL 5 RR 5 --out-dir logs/d7_11_rear_settle/roll5
```

(Viewer: replace `--seeds 0-79 --jobs 2` with `--seed N --debug`.) Results:
`logs/d7_11_rear_settle/*.txt`.

## D7.12 Body-height gate for rear steps (four legs, seeds 0-79): negative result

New option `--min-body-z RL Z RR Z` (`step_min_body_z` in `MultiLegRecoveryActionCfg`, with
`height_gate_max_steps`, default 30): a rear leg's step start waits while the base is lower
than Z, for at most 30 steps. Base: D7.9 combined setting (60/80).

| Threshold | Passes /80 (0-39 / 40-79) | Gained | Lost | Failures |
|---|---|---|---|---|
| none (D7.9) | 60 (29 / 31) | - | - | 6 front, 14 rear |
| 0.29 m | 54 (27 / 27) | 4 | 10 | 6 front, 20 rear |
| 0.30 m | 48 (23 / 25) | 8 | 20 | 6 front, 26 rear |

Timing (seeds with both rear steps): the first rear step still starts 5.1 steps after contact
in all three runs, so the gate never held it (the body is not low at the first rear contact).
It delayed the second rear step: 17.3 steps after the first in D7.9, 23.8 at 0.29 m, 34.4 at
0.30 m. Passes fell as the second step was delayed more.

Together with D7.11 (gap 24: 36/80) the trend is that making the second rear leg wait longer
lowers the pass rate. Likely reason (not tested): while a rear leg waits, the robot stands
pressed against the wire and the wire pulls the body back. D7.9 stays the best four-leg
setting. Not tested: a shorter gap (gap 12 is global, so it also moves the front legs).

```
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --min-body-z RL 0.29 RR 0.29 --out-dir logs/d7_12_height_gate/z029
```

(Viewer: replace `--seeds 0-79 --jobs 2` with `--seed N --debug`.) Results:
`logs/d7_12_height_gate/*.txt`.

## D7.13 Shorter serialization gap (four legs, seeds 0-79)

Base: D7.9 combined setting (gap 12, 60/80). Only `--serialize-gap-steps` changes.

| Gap (steps) | Passes /80 (0-39 / 40-79) | Gained | Lost | Failures |
|---|---|---|---|---|
| 24 (D7.11) | 36 (17 / 19) | 9 | 33 | 0 front, 44 rear |
| 12 (D7.9) | 60 (29 / 31) | - | - | 6 front, 14 rear |
| **6** | **66 (35 / 31)** | 13 | 7 | 7 front, 7 rear |

Rear failures halved (14 to 7) and both halves are at least as good as D7.9. Caveats: 13 seeds
gained and 7 lost, so a lot of seeds changed outcome; and the mean gap between the two rear
steps barely moved (17.3 steps in D7.9, 18.7 at gap 6), so the improvement is not simply
"the second rear leg goes sooner". Front steps are 25.8 apart in D7.9 and 23.5 at gap 6. The
mechanism is not established. Stability metrics for this run are not measured yet.

```
uv run python scripts/d6_1_front_pair_wire.py --pair all --start-x-range -0.485 -0.35 --seeds 0-79 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR --serialize-gap-steps 6 --out-dir logs/d7_13_short_gap/gap6
```

(Viewer: replace `--seeds 0-79 --jobs 2` with `--seed N --debug`.) Results:
`logs/d7_13_short_gap/gap6.txt`.

D7.13 stability (steps 40 to end, means over passing seeds; the 53 seeds that pass in both
runs give the paired numbers):

| Setting | Passes | z mean (m) | z range (m) | Peak roll (deg) | Peak pitch (deg) | Ang. vel RMS (rad/s) |
|---|---|---|---|---|---|---|
| gap 12 (D7.9), all passes | 60 | 0.315 | 0.133 | 5.4 | 8.8 | 0.45 |
| gap 6, all passes | 66 | 0.313 | 0.129 | 5.9 | 8.8 | 0.48 |
| gap 12, 53 paired passes | 53 | 0.315 | 0.132 | 5.5 | 8.7 | 0.45 |
| gap 6, 53 paired passes | 53 | 0.313 | 0.130 | 6.0 | 8.8 | 0.48 |

Stability is essentially unchanged: pitch and body height match, roll and angular velocity are
a little higher (about 0.5 deg and 0.03 rad/s).

### D7.14 Gap 3 (four legs, seeds 0-79)

| Gap (steps) | Passes /80 (0-39 / 40-79) | Failures |
|---|---|---|
| 24 | 36 (17 / 19) | 0 front, 44 rear |
| 12 (D7.9) | 60 (29 / 31) | 6 front, 14 rear |
| 6 (D7.13) | **66** (35 / 31) | 7 front, 7 rear |
| 3 | 55 (29 / 26) | 10 front, 15 rear |

Gap 3 is worse than both 12 and 6 (gained 15 and lost 20 against gap 12; gained 10 and lost 21
against gap 6), with more failures in both stages. The pass rate peaks around gap 6, but the
seeds change a lot between gaps and the gap-6 advantage over gap 12 (+6 of 80) is about
1.5 binomial standard deviations (about 4 seeds), so it needs seeds outside 0-79 to confirm.
Command: as D7.13 with `--serialize-gap-steps 3 --out-dir logs/d7_14_gap3/gap3`; results
`logs/d7_14_gap3/gap3.txt`.

## D7.15 Fresh-seed check of the gap (four legs, seeds 80-159)

Seeds 80-159 were never used for tuning. D7.9 setting with only `--serialize-gap-steps`
changed.

| Gap | Passes /80 (80-119 / 120-159) | Failures |
|---|---|---|
| 12 (D7.9) | **65** (33 / 32) | 5 front, 10 rear |
| 6 | 60 (31 / 29) | 7 front, 13 rear |

Gap 6 gained 8 seeds and lost 13 against gap 12. The gap-6 advantage seen on seeds 0-79 (66 against
60) did not hold on fresh seeds (60 against 65), so it was seed noise. **Gap 12 stays the default.**
D7.9 itself generalizes: 65/80 here, and 125/160 (78%) over seeds 0-79 and 80-159 together.

```
C="--pair all --start-x-range -0.485 -0.35 --seeds 80-159 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 12 --out-dir logs/d7_15_fresh_seeds/gap12
uv run python scripts/d6_1_front_pair_wire.py $C --serialize-gap-steps 6 --out-dir logs/d7_15_fresh_seeds/gap6
```

(Viewer: replace `--seeds 80-159 --jobs 2` with `--seed N --debug`.) Results:
`logs/d7_15_fresh_seeds/*.txt`.

The per-seed folders of D7.10-D7.15 (`logs/d7_10_rear`, `d7_11_rear_settle`, `d7_12_height_gate`,
`d7_13_short_gap`, `d7_14_gap3`, `d7_15_fresh_seeds`) are removed the same way; their metrics are
in `per_seed_stability.csv` and their sweep summaries and `run.sh` files stay.

## D8 Wire height (four legs, best setting, seeds 0-39)

New option `--wire-height` (m, default 0.10, the value used in every result above) sets both
wire anchors to that height. Everything else is the D7.9 setting. The measured wire height at
the legs is slightly below the anchor height (about 0.088-0.098 m at 0.10, 0.13-0.146 m at 0.14).

| Wire height | Passes /40 | Front fails | Rear fails | No contact (SKIP) |
|---|---|---|---|---|
| 6 cm | 30 | 5 | 5 | 0 |
| 8 cm | 31 | 1 | 8 | 0 |
| 10 cm (D7.9, seeds 0-39) | 29 | 2 | 9 | 0 |
| 12 cm | 18 | 9 | 13 | 0 |
| 14 cm | 0 | 37 | 0 | 3 |

6-10 cm are about equal (29-31 of 40, within seed noise). Pass rate falls at 12 cm and is zero
at 14 cm. At 14 cm the contact sensor fires and the steps start, but the scripted foot peaks at
about 0.13 m (seed 0, FL) while the wire is at 0.13-0.146 m at the leg, so the foot does not
clear the wire (the nominal lift height is 0.20 m but the foot only reaches about 0.13-0.15 m).
Three seeds never touch the wire. Not tested: a higher `step_lift_height` for taller wires.
The sensor only reads the lower shin (`calf2_collision`), so very low or high wires may touch
other parts of the leg without triggering.

```
C="--pair all --start-x-range -0.485 -0.35 --seeds 0-39 --jobs 2 --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 --support-lift-trigger FL FR"
uv run python scripts/d6_1_front_pair_wire.py $C --wire-height 0.14 --out-dir logs/d8_wire_height/h0.14
```

(Viewer: replace `--seeds 0-39 --jobs 2` with `--seed N --debug`.) Results:
`logs/d8_wire_height/h*.txt` (per-seed folders are scratch, same pruning as D7).

## D9 Stage 0: delayed oracle trigger (detector-latency sensitivity, four legs, seeds 0-9)

Goal: replace the simulator contact-sensor trigger with a learned detector. A detector needs
time to see the event, so first measure how much trigger delay the recovery tolerates. New
option `--entry-delay-steps N` (default 0 = original behaviour): each leg enters recovery N
control steps (0.02 s each) after its first wire contact. The delay is latched from the first
contact. Everything else is the D7.9 setting. Delay 0 reproduces the stored D7.9 seeds 0-9
line for line, so existing results are unchanged.

| Delay | Steps | Passes /10 | Seeds failing |
|---|---|---|---|
| 0 ms (D7.9) | 0 | 7 | 4, 6, 8 |
| 40 ms | 2 | 7 | 1, 4, 9 |
| 100 ms | 5 | 6 | 1, 4, 7, 9 |
| 200 ms | 10 | 4 | 1, 4, 6, 7, 8, 9 |

Seeds 0-9 alone could not separate 0-100 ms; the 40-seed run below did (it did not agree
with that reading: 100 ms is already a large drop).

### Seeds 10-49 (40 seeds; baseline = stored D7.9 results for the same seeds)

| Delay | Steps | Passes /40 | Front fails | Rear fails | Gained | Lost |
|---|---|---|---|---|---|---|
| 0 ms (D7.9) | 0 | 30 | 2 | 8 | - | - |
| 100 ms | 5 | 16 | 1 | 23 | 3 | 17 |
| 200 ms | 10 | 19 | 1 | 20 | 9 | 20 |
| 300 ms | 15 | 15 | 6 | 19 | 3 | 18 |

Even 100 ms costs about half the passes, and 200 and 300 ms are no worse beyond noise (about
+-3 seeds), so the pass rate drops at once and then stays at 15-19/40. The extra failures
are almost all rear-stage. Not yet tested: whether the delay hurts through the front legs
(wire pushes the body back before recovery starts) or through the rear legs' own entry,
since the delay is applied to every leg.

```
source logs/d9_entry_delay/common.sh
logs/d9_entry_delay/run_delay.sh <delay_steps> <seeds>   # 0-9 and 10-49 above
```
(`run_delay.sh` runs `scripts/d6_1_front_pair_wire.py $C --seeds 0-9 --jobs 2 --entry-delay-steps N`.)
Results: `logs/d9_entry_delay/delay*_seeds*.txt`.

## D9 Stage 3: the go2-entanglement-tcn ONNX detector on sim logs (seeds 0-7)

The robot's own detector (`go2-entanglement-tcn-main`, ONNX, its own `EntanglementEngine` and
`config.yaml`) run offline on proprioception logged from our sim. Recovery is off (entry delay
set to 10^9 steps), so the walking policy walks into the wire and stays pressed against it.
`scripts/d9_tcn_probe_log.py` records the channels the detector needs at the 400 Hz physics rate
(joint q/dq/torque, base orientation, angular velocity, linear velocity, foot-ground force) plus
the per-leg wire-contact flag; `scripts/d9_tcn_probe_eval.py` resamples to 500 Hz, builds
/lowstate-shaped samples (roll/pitch from the base quaternion, accelerometer as body-frame
specific force from the differentiated base velocity) and feeds the engine. Latency = first
alarm minus first wire contact on any leg. Controls: wire raised to 0.5 m (never touched).

| Variant | Alarmed /8 | Latency, ms (median, min-max) | Alarm leg | Controls (2) |
|---|---|---|---|---|
| shipped config (thr 0.9999, stationary gate on) | 0 | - | - | no alarm |
| thr 0.99, gate on | 1 | 872 | FR | no alarm |
| gate off (thr 0.9999) | 8 | 1817 (875-2467) | FR 7, FL 1 | no alarm |
| thr 0.99, gate off | 8 | 1058 (584-2291) | FR 7, FL 1 | no alarm |
| thr 0.9, gate off | 8 | 730 (550-1705) | FR 7, FL 1 | no alarm |

No false alarm before first contact in any run (0% of pre-contact windows above threshold).
The raw P(entangled) does reach ~1 in every wire run, but 0.5-2.5 s after contact, when the
robot has stalled against the wire; at contact itself it is low (~0.03). With the shipped
config nothing alarms, because the detector's Stationary gate clears its buffer once the joints
are nearly still, which is also what a robot stuck on the wire looks like.

Reading: against the 100-300 ms we measured the recovery to tolerate (D9 Stage 0), this
detector alarms 0.55-2.5 s after contact in the best variants and never with the shipped config. It names a front leg (FR mostly, so the
leg is only partly right: FL touches first or at the same time in most seeds). Caveats: sim
torque, foot force (assumed N ~ raw units) and accelerometer are not calibrated to the real
sensors; only the front legs touched the wire in 320 steps (recovery off); 8 seeds. This is a
zero-shot probe, not a verdict on the TCN on the robot.

```
logs/d9_tcn_probe/run_log.sh                       # 8 wire seeds + 2 controls -> raw/*.npz
uv run python scripts/d9_tcn_probe_eval.py --out logs/d9_tcn_probe/results [--no-stationary-gate] [--threshold 0.99]
```
Results and plots: `logs/d9_tcn_probe/results/` (`summary_*.txt`, `plots/*.png`).

## D9 Loose-to-tight wire: TCN probe and wire-integrity check (seeds 0-2)

Question: what if the wire starts loose and tightens later, and does the Go2 pass through it?
`scripts/d9_loose_wire_probe_log.py` swaps in a wire built with extra length (`--slack-cm`) and a
sag, its end pinned to a mocap anchor. After `--tighten-delay` (0.2 s) from the first contact the
anchor moves along the wire axis over `--tighten-time` until the chain is straight; the tight
state is at 10 cm (the height of every earlier result), the loose state hangs below it (sag 3.1 cm
at 0.5 cm slack, 4.3 cm at 1 cm). Recovery is off. Same logging as Stage 3, plus per-substep
checks computed from geometry, independent of the contact solver: smallest surface gap between 16
robot parts (thigh, calf1, calf2, foot per leg) and any wire segment, largest gap between
consecutive wire segments, and distance from the last segment to its anchor. Existing wire and
recovery code are untouched (the wire is swapped on the env cfg inside the script only).

21 runs: seeds 0-2 x slack {0.5, 1.0} cm x tighten time {0.5, 2, 4} s, plus 3 never-tighten
controls (slack 1.0).

Wire integrity (all 21 runs): no robot part overlapped the wire (largest overlap 0.0 mm, closest
approach about 10 mm, so within the 15 mm contact margin only); consecutive wire segments never
separated (largest gap 0.00 mm, the chain is kinematic); the last segment slips off its soft
anchor by 24-28 mm under the robot's pull, also in the never-tighten controls. Only FL and FR
touched the wire.

TCN probe (same engine and config as Stage 3), latency from first contact, median ms:

| Tighten time | Shipped (gate on) | Gate off, thr 0.9999 | Gate off, thr 0.9 |
|---|---|---|---|
| never (control, 3) | no alarm | 1648 | 742 |
| 0.5 s (6) | no alarm | 1785 | 760 |
| 2 s (6) | no alarm | 1862 | 766 |
| 4 s (6) | no alarm | 1822 | 772 |

The tightening speed does not change when the TCN alarms, and neither does tightening at all
(control alarms at the same time). The wire evolves the same in the control and the tightened
runs: within about 0.3 s of first contact the robot's push lifts the wire from about 5.5 cm to
about 10 cm and takes up the 0.5-1 cm slack, and the soft anchor yields before the moving anchor
can pull it taut (chain stays about 1.7 cm longer than its end-to-end distance). So this probe
does not test slow tightening; with this much slack the robot, not the wire, sets the alarm time.
A real test needs more slack with a stiffer anchor (not done).

```
logs/d9_loose_wire/run_batch.sh
uv run python scripts/d9_tcn_probe_eval.py --raw logs/d9_loose_wire/raw --out logs/d9_loose_wire/results [--no-stationary-gate] [--threshold 0.9]
```
Results and plots: `logs/d9_loose_wire/` (`run_batch.txt`, `results/summary_*.txt`, `results/plots/`).

### Stiffer anchor, more slack (D9 loose wire, part 2; seeds 0-2)

The first version did not test slow tightening (the robot absorbed the slack, see above). Changes:
the end-anchor constraint is 4x stiffer (`--anchor-solref 0.005`, was 0.02), the anchor ends 1 cm
past straight so the tight wire is under tension (`--overtravel-cm 1`), the tight wire is at 15 cm
(`--wire-height 0.15`) so the loose wire hangs near leg height (about 9 cm at the legs), and slack
is 3 and 5 cm (sag 7.5 and 9.7 cm). 24 runs: seeds 0-2 x slack {3, 5} x tighten time {0.5, 2, 4} s
+ never-tighten controls, 500 control steps. Now the tightened wire does become taut and the
controls stay loose.

Wire integrity, from geometry (not the contact solver):

| | Never tighten (6) | Tightened (18) |
|---|---|---|
| Runs where a robot part overlaps the wire | 0 | 17 |
| Worst overlap (mm) | 0 | 4-19, mostly FL/FR calf1 |
| Runs with the wire centreline deep inside a part (> 6 mm) | 0 | 8 (up to 655 substeps = 1.6 s) |
| Largest gap between wire segments | 0.00 mm | 0.00 mm |
| Last segment slip off its anchor (mm) | 1-2 | 15-54 |

So the wire does go into the robot when it is tightened with a stiff, pre-tensioned anchor. It is
the upper shin (calf1 cylinder, r = 12 mm), which the wire-contact sensor does not watch (it
reads calf2 only), so the oracle trigger would not see it either. Worst case: the centreline
sits on the cylinder axis for 1.6 s. Overlap is deepest for the fastest tightening (0.5 s) and
the most slack (5 cm), and it appears seconds after tightening has finished, while the robot is
stalled against the taut wire. Wire segments never separate (ball-jointed chain), so gaps in the
wire cannot happen as built; what fails is the contact between wire and leg. Three runs also
touched a rear leg (RR/RL), i.e. the taut wire was dragged back along the body. The check that
detects it: `gap` in the logs (per substep, 16 robot parts x all wire segments; overlap deeper
than the wire radius = went through). Not yet checked: whether this is the contact solver
(geom solref 0.01 is softer than the 0.005 anchor) or a cylinder-capsule contact problem (the
build warns about it); the original wire has not been run with the stiff anchor.

TCN probe, median ms from first contact; "legs ok" = alarm leg is one of the touched legs (FL, FR):

| Tighten | Shipped | Gate off, thr 0.9999 | Gate off, thr 0.9 |
|---|---|---|---|
| never (6) | 0/6 alarmed | 0/6 | 6/6, 2233 ms, legs ok 6/6 |
| 0.5 s (6) | 1/6 | 5/6, 4758 ms, legs ok 0/5 | 6/6, 3916 ms, legs ok 1/6 |
| 2 s (6) | 1/6 | 5/6, 5356 ms, legs ok 0/5 | 6/6, 3412 ms, legs ok 2/6 |
| 4 s (6) | 1/6 | 6/6, 4972 ms, legs ok 0/6 | 6/6, 3116 ms, legs ok 2/6 |

The tightening changes the TCN's behaviour a lot: later alarms (3-5 s vs about 2 s) and mostly
the wrong legs (RL/RR while FL/FR touched). Not tested why; one candidate is that the body
disturbance from the taut wire looks like a rear-leg event to a model trained on a walking robot.

```
logs/d9_loose_wire/run_batch2.sh
uv run python scripts/d9_tcn_probe_eval.py --raw logs/d9_loose_wire/raw2 --out logs/d9_loose_wire/results2 [--no-stationary-gate] [--threshold 0.9]
```

Viewer for the overlap above (`--viewer` prints `[CONTACT]` and `[OVERLAP]` lines live; in the
viewer window tick Geom groups > 3 to show the collision shapes, the wire is cyan). Worst case
(seed 0, 5 cm slack, 0.5 s tightening, wire centre 0.9-10 mm inside FR/FL calf1 from about 4 s)
and its never-tighten control (no `[OVERLAP]` lines):

```
uv run python scripts/d9_loose_wire_probe_log.py --seed 0 --slack-cm 5 --tighten-time 0.5 --wire-height 0.15 --anchor-solref 0.005 --overtravel-cm 1 --steps 500 --viewer --out logs/d9_loose_wire/viewer
uv run python scripts/d9_loose_wire_probe_log.py --seed 0 --slack-cm 5 --tighten-time 0 --wire-height 0.15 --anchor-solref 0.005 --overtravel-cm 1 --steps 500 --viewer --out logs/d9_loose_wire/viewer
```

### Step 1: a loose-to-tight wire at 10 cm with no overlap (D9 loose wire, part 3)

The 15 cm tight wire above was a poor choice: the verified recovery height is 10 cm (D8: 12 cm
18/40, 14 cm 0/40). Here the tight wire is back at 10 cm (`--wire-height 0.10`) and the loose
wire hangs below it (about 4-5 cm at the legs). Sweep on seed 0, tighten time 0.5 s (fastest, the
worst case for overlap), slack {2, 3} cm x anchor time constant {0.005, 0.01, 0.02} s x overtravel
{0, 1} cm, plus no-slack controls. Overlap = geometric gap check (see above); straightness = wire
end-to-end distance / length 1.5 s after tightening ends.

| Anchor solref | Overtravel | Max overlap | Straightness | Wire height after (cm) |
|---|---|---|---|---|
| 0.005 | 1 cm | **15.9 mm** (slack 2: 1194 deep substeps; slack 3: 66) | 0.995-0.998 | 9.7-9.9 |
| 0.005 | 0 | 0 | 0.995 | 10.1-10.2 |
| 0.01 | 0 | 0 | 0.987-0.989 | 10.9-11.1 |
| 0.01 | 1 cm | 0 | 0.993-0.994 | 10.3-10.4 |
| 0.02 | 0 / 1 cm | 0 | 0.979-0.988 | 11.4-12.3 |
| no slack, 0.005 | 0 / 1 cm | 0 | 0.995-1.000 | 10.3-10.9 |

The overlap only appears with the stiffest anchor plus pre-tension; every other setting has none
and still ends taut at about 10-12 cm. Two settings confirmed on seeds 0-2 x slack {2, 3}
(12 runs each, T = 0.5 s, 500 steps): `--anchor-solref 0.005 --overtravel-cm 0` and
`--anchor-solref 0.01 --overtravel-cm 1`. Both: 0 overlap in every run, straightness 0.993-0.997,
final height 9.8-10.4 cm, only FL and FR touched. The first has the smaller anchor slip (5-8 mm
against 18-20 mm) and no pre-tension, so it is the default for what follows. Not tested: slower
tightening (2 and 4 s) with these settings, and whether the recovery still passes on this wire.

```
logs/d9_loose_wire/run_batch3.sh   # seed 0 sweep (14 runs)
logs/d9_loose_wire/run_batch4.sh   # confirmation, seeds 1-2 (8 runs)
uv run python scripts/d9_loose_wire_probe_log.py --seed 0 --slack-cm 3 --anchor-solref 0.005 --overtravel-cm 0 --tighten-time 0.5 --wire-height 0.10 --steps 500 --viewer --out logs/d9_loose_wire/viewer
```

### Step 3: TCN probe on the clean 10 cm loose-to-tight wire (D9 loose wire, part 4)

Same detector, engine and config as before, on the wire from step 1 (anchor solref 0.005,
overtravel 0, tight height 10 cm). Seeds 0-2 x slack {2, 3} cm x tighten {0.5, 2, 4 s, never}:
24 runs, 500 steps; the 6 runs at 0.5 s are the confirmation runs from step 1. Overlap check:
0 of 24 runs have any robot/wire overlap. Only FL and FR touch the wire. Median ms from first
contact (alarmed/6 in brackets, "legs ok" = alarm names FL or FR, the touched legs):

| Tighten | Shipped (gate on) | Gate off, thr 0.9999 | Gate off, thr 0.9 |
|---|---|---|---|
| never (control) | 2/6, 1527 | 6/6, 1660 | 6/6, 719 |
| 0.5 s | 2/6, 1826 | 6/6, 2621 | 6/6, 1406 |
| 2 s | 3/6, 1546 | 6/6, 1780 | 6/6, 934 |
| 4 s | 3/6, 1273 | 6/6, 1675 | 6/6, 738 |

Legs ok in every alarmed run (shipped 10/10, other variants 24/24). Compared with the 15 cm
wire, where the shipped config alarmed 4/24 and the leg was almost always wrong (RL/RR), so
those late, wrong-leg alarms came from the 15 cm height and the overlap, not from tightening.
On this wire the shipped config alarms in 10 of 24 runs (including 2 of 6 controls that never
tighten; the loose wire hangs at foot height). Tightening slowly (2, 4 s) changes the timing
little compared with never tightening (1675-1780 against 1660 ms at thr 0.9999); the fastest
tightening (0.5 s) is the slowest to alarm (2621 ms). Not tested why. All variants are still
0.7-2.6 s after contact, against the 100-300 ms the recovery tolerates. Only 6 runs per cell.

```
logs/d9_loose_wire/run_batch5.sh
uv run python scripts/d9_tcn_probe_eval.py --raw logs/d9_loose_wire/raw5 --out logs/d9_loose_wire/results5 [--no-stationary-gate] [--threshold 0.9]
```

## D9 Step 2: the four-leg recovery on a loose-to-tight wire (seeds 0-9, oracle trigger)

`scripts/d9_loose_wire_recovery.py` runs `d6_1_front_pair_wire.py` unchanged with the wire swapped
for the loose-to-tight one (anchor solref 0.005, overtravel 0, tight height 10 cm, 3 cm slack,
tightening starts 0.2 s after the first contact) and adds the geometric overlap check from the
probe. D7.9 best recipe, same seeds. Baseline = stored D7.9 results.

| Wire | Passes /10 | Front fails | Rear fails |
|---|---|---|---|
| Original wire (D7.9; re-run with the overlap check, verdicts identical) | 7 | 1 | 2 |
| New build, no slack (control) | 7 | 1 | 2 |
| 3 cm slack, tightening 0.5 s | 5 | 4 | 1 |
| 3 cm slack, tightening 2 s | 3 | 5 | 2 |
| 3 cm slack, never tightening | 5 | 4 | 1 |

Reading, with only 10 seeds (about +-2-3 seeds of noise):
- The control passes 7/10 like the baseline, but 4 of the 10 seeds changed verdict (1 and 7 now fail, 4 and 6 now pass), so the new wire build is not equivalent to the original; per-seed results move with a small change to the wire.
- Loose wires lower the pass rate (3-5/10 against 7/10), almost all of it in the front stage (4-5 front fails against 1). Seeds 0, 2 and 5 fail in the front stage under all three loose conditions and pass in both the baseline and the control, which is more than chance.
- Tightening does not separate from never tightening (5, 3, 5), so the loose wire itself, not its tightening, is the likely cause. Not tested why; one candidate is that the loose wire first touches the foot, not the calf, so the contact trigger and the step geometry see a different situation.

Overlap during recovery (wire centre line inside a robot part): original wire 3.0 mm at most, 0 deep
substeps in all 10 runs. New wire build: 10 of the 40 runs overlap, up to 17.6 mm (control 3 runs,
loose 7), with 0-24 deep substeps each. So the stiff anchor (solref 0.005) that was overlap-free in
the no-recovery probe (step 1) is not overlap-free while the legs are stepping, and the step 2 numbers
above are confounded by it. Not done: the same test with an anchor that matches the original
(solref 0.02, which should reproduce the baseline) or a softer one (0.01).

```
logs/d9_loose_wire/run_batch6.sh   # 4 conditions x 10 seeds on the new wire
logs/d9_loose_wire/run_batch7.sh   # original wire + overlap check
```

### Step 2b: softer anchors (same recipe, seeds 0-9, oracle trigger)

Baseline = original wire, D7.9 (7/10; max overlap 3.0 mm, no deep overlap).

| Wire (slack, anchor solref, overtravel) | Passes /10 | Failures | Verdicts changed vs baseline | Max overlap, runs with overlap, deep substeps |
|---|---|---|---|---|
| A: 0 cm, 0.02, 0 (validity control) | 7 | 1 front, 2 rear | 0/10 | 3.0 mm, 3 runs, 0 |
| B: 3 cm, 0.02, 0 (tightening 2 s) | 6 | 4 front | 4/10 | 8.1 mm, 5 runs, 57 |
| C: 3 cm, 0.01, 1 cm (tightening 2 s) | 6 | 3 front, 1 rear | 5/10 | 17.8 mm, 3 runs, 9 |
| D: 0 cm, 0.01, 1 cm (control for C) | 3 | 1 front, 6 rear | 5/10 | 16.8 mm, 3 runs, 63 |

A reproduces the original wire verdict for verdict (10/10) with the same overlap, so the new wire
build is valid when the anchor matches the original (solref 0.02, no overtravel). The earlier
mismatch (step 2) came from the stiffer anchor. D shows that the pre-tensioned 0.01 anchor is
itself harmful (3/10 on a taut wire, overlap up to 17 mm), so C cannot be read. B is the usable
loose-wire result: 6/10 against 7/10, but front-stage failures go from 1 to 4 (seeds 2, 5, 6, 8) and
rear failures from 2 to 0 (seeds 4 and 8 no longer fail at the rear), and the overlap shows up in 1 run
as a real pass-through (seed 0, FL_calf1, 8.1 mm, 57 deep substeps; that seed still passes). Only 10
seeds, no never-tighten run on this wire. Nothing in the tightening speed has been separated here.

```
logs/d9_loose_wire/run_batch8.sh
```

### Step 2c: condition B on seeds 10-29 (3 cm slack, anchor solref 0.02, tightening 2 s)

Baseline = stored D7.9 results for the same seeds.

| Seeds | Wire | Passes | Front fails | Rear fails |
|---|---|---|---|---|
| 10-29 | original | 15/20 | 1 | 4 |
| 10-29 | loose-to-tight (B) | 13/20 | 5 | 2 |
| 0-29 | original | 22/30 (73%) | 2 | 6 |
| 0-29 | loose-to-tight (B) | 19/30 (63%) | 9 | 2 |

On seeds 10-29 alone 9 of 20 verdicts changed (gained 18, 20, 21; lost 13, 15, 23, 28, 29). Overlap
on seeds 10-29: at most 3.0 mm and no deep overlap in any run, the same as the original wire (the
pass-through seen on seed 0 did not recur). Over 30 seeds the pass rate difference (22 vs 19) is not
significant (Fisher exact p = 0.58, unpaired), but front-stage failures rise from 2 to 9 while rear
failures fall from 6 to 2 (p = 0.04 for the front count, unpaired, not corrected for looking at both
stages; no paired test run). Not tested: never-tighten and 0.5 s tightening on this wire; why the
front stage is affected.

```
logs/d9_loose_wire/run_batch9.sh
```

## D9 Stage 2 (v0): a sim-trained detector, first offline results

Replace the contact-sensor trigger by a detector trained in the sim with the oracle as the labeler.
`scripts/d9_detector_data.py` records, per 50 Hz control step, 60 features pooled over the 8 physics
substeps (joint q, dq, torque, max |torque|, foot force, roll, pitch, gyro, accelerometer) and the
per-leg wire-contact flag; `scripts/d9_detector_train.py` trains a small causal CNN (window 10 steps =
0.2 s, per-leg heads) and evaluates it next to a torque-threshold baseline (z-score of max |torque|
against normal walking, in the paper's spirit, calibrated here). Dataset v0 = 38 episodes, seeds 200-237
(apart from the tuning and evaluation seeds 0-159): 12 walking into a taut wire at 6/8/10/12 cm with recovery
off, 8 into a loose-to-tight wire, 12 full recovery runs (best recipe, 10 cm), 6 with the wire out of reach.
Episodes with a deep wire/leg overlap are dropped. Split by seed (test = seed % 4 == 0): 27 train / 9 test
episodes, 7 test episodes with wire contact. This is a pipeline check, not a result.

Trigger view on the 7 test episodes with contact (first alarm of any leg against the first contact of any
leg; latency in ms; "early" = an alarm before any leg has touched the wire):

| Detector | Alarm rule | Alarm at/after contact: median latency | Early alarms | Missed | Alarm names a touching leg | False-alarm episodes (2 no-wire) |
|---|---|---|---|---|---|---|
| CNN | p >= 0.9, 1 step | 140 | 3/7 (about 60 ms before contact) | 0 | all | 0 |
| CNN | p >= 0.9, 2 steps | 160 | 3/7 (about 40 ms before) | 0 | all | 0 |
| CNN | p >= 0.5, 1 step | 170 | 5/7 | 0 | all | 2/2 |
| Torque baseline | z >= 3, 1 step | 240 | 0/7 | 0 | 5/7 wrong | 0 |
| Torque baseline | z >= 5, 1 step | 360 | 0/7 | 2 | 2/7 wrong | 0 |

So the CNN is faster and names the leg correctly where the threshold detector is slow or misses, but the
latency of 140-160 ms (where there is no early alarm) is already inside the range where the recovery lost
about half its seeds (Stage 0: 100 ms 16/40 against 30/40), and alarms before contact need explaining
(real precursor in the gait, or leakage from the fixed start and approach, e.g. time since the start).
Per-leg numbers (18 events): CNN front-leg median 20-40 ms, rear-leg 30-50 ms, but those include
alarms that were already running because the other front leg had touched. Only 9 test episodes.

Side finding: of the 12 recovery episodes on the ORIGINAL wire (seeds 220-231) two (224, 227, both
FAIL_REAR) have a deep wire-through-leg overlap (up to 16 mm, 44 and 33 deep substeps), so the
pass-through is not specific to the loose or stiff-anchor wires; seeds 0-9 had none.

```
logs/d9_detector/run_data_v0.sh
uv run python scripts/d9_detector_train.py --data logs/d9_detector/data_v0
```

### D9 Stage 2 (v1): larger sim dataset

Data: `logs/d9_detector/run_data_v1.sh` (seeds 300-479, 180 episodes: 60 walk at 6/8/10/12 cm
with two start ranges, 48 loose-to-tight, 40 recovery on the best four-leg recipe, 32 no-wire).
6 recovery episodes dropped for deep wire overlap (s410, 428, 431, 436, 439, 444). Split by
seed (test = seed % 4 == 0): 132 train, 42 test (34 with contact, 8 no-wire).
Train: `uv run python scripts/d9_detector_train.py --data logs/d9_detector/data_v1 --out logs/d9_detector/model_v1.pt`
(log: `logs/d9_detector/train_v1.txt`).

Test, trigger view (first alarm of any leg vs first oracle contact of any leg), 34 contact episodes:

| Detector | Median latency | Early alarms | Missed | Wrong-leg first alarm | False-alarm eps (8 no-wire) |
|---|---|---|---|---|---|
| CNN p>=0.9, 1 step | 40 ms | 18/34 (lead ~60 ms) | 0 | 0 | 0 |
| CNN p>=0.9, 2 steps | 50 ms | 10/34 | 0 | 1 | 0 |
| CNN p>=0.5, 1 step | 20 ms | 25/34 | 0 | 0 | 0 |
| Torque z>=3 | 180 ms | 2/34 | 0 | 24 | 0 |
| Torque z>=5 | 420 ms | 0 | 17 | 10 | 0 |

Per-leg events (82 test events): CNN p>=0.9 misses 4, all rear legs.

Caveats: early alarms are still unexplained. No-wire episodes follow the same start, so pure
timing leakage should fire there too and does not. Hypothesis: the wire touches calf1, thigh
or foot, which the oracle sensor (calf2 only) does not label. Not tested. Walking commands
were not varied. Not yet closed-loop.

### D9 Stage 2 (v1): early-alarm check and first closed loop

> INVALID closed-loop numbers below (12/40): the alarms were applied to the wrong legs (see
> the correction at the end of this file). The early-alarm check is unaffected.

Code: `ResidualRecoveryAction.external_trigger` (opt-in, None = original oracle trigger),
`scripts/d9_detector_closed_loop.py`, `logs/d9_detector/run_closed_loop.sh`.

Early alarms (log only, 15 walk test episodes, seeds 300..356 step 4, recovery off, log
`logs/d9_detector/early_check/walk_log.txt`): 11/15 alarm 1-10 steps before the oracle
contact. At the alarm the closest part to the wire is mostly a foot (calf2 otherwise), within
about 9-16 mm and closing 30-80 mm over 10 steps. The oracle labels calf2 only, so these are
real contacts the label does not cover, not timing leakage. The geometric monitor reads about
14 mm at oracle contact (constant offset, unexplained).

Closed loop, seeds 10-49, best four-leg recipe, original wire 10 cm, model_v1, p>=0.9, 1 step,
alarm replaces the entry trigger (release still reads the oracle contact):
`logs/d9_detector/run_closed_loop.sh 10 49 logs/d9_detector/closed_loop/thr09_m1_s10_49.txt`

| Trigger | PASS | FAIL_FRONT | FAIL_REAR |
|---|---|---|---|
| Oracle contact (D7.9) | 30/40 | | |
| CNN, closed loop | 12/40 | 10 | 18 |

Per-leg across the 40 episodes: 19 legs alarmed with no oracle contact (mostly rear legs while
the front legs recover), 6 legs never alarmed (RL in several), and 48 of 130 contacted legs
alarmed more than 5 steps late (median 3 steps). The detector was trained on walking into the
wire, so recovery-phase motion is outside its data. Not yet fixed.

### D9 Stage 2: closed-loop data (seeds 500-599) and model_v2

> INVALID: collected and evaluated with swapped legs (see the correction at the end of this
> file). Do not train on data_v2cl or use model_v2.

Collection (v1 model as trigger, 100 seeds): 20 PASS, 16 FAIL_FRONT, 64 FAIL_REAR; 21 episodes
dropped for deep overlap. `DATA_DIR=logs/d9_detector/data_v2cl logs/d9_detector/run_closed_loop.sh 500 599 logs/d9_detector/closed_loop/collect_thr09_s500_599.txt`
Retrain on v1 + closed-loop (`logs/d9_detector/data_v1v2` = symlinks, `train_v2.txt`, `model_v2.pt`).
Closed loop, seeds 10-49, model_v2: `logs/d9_detector/run_closed_loop.sh 10 49 logs/d9_detector/closed_loop/v2_thr09_m1_s10_49.txt --model logs/d9_detector/model_v2.pt`

| Trigger | PASS | FAIL_FRONT | FAIL_REAR |
|---|---|---|---|
| Oracle (D7.9) | 30/40 | | |
| CNN model_v1 | 12/40 | 10 | 18 |
| CNN model_v2 | 4/40 | 9 | 27 |

model_v2 per leg: 0 missed, 15 false-alarm legs, 28 of 138 alarms before the oracle contact,
median latency 1 step, 28 more than 5 steps late. The detector is earlier and misses nothing,
yet passes less, so latency is not what limits the recovery now. Open: early entry (before
calf2 contact) and false rear entries.

### D9 Stage 2: correction, leg order bug, and the valid closed-loop result

Bug: the detector orders legs FR FL RR RL, the controller terms FL FR RL RR. The closed-loop
runner applied alarm[i] to term i, so FR drove FL and RR drove RL. Fixed by mapping on the
leg name (`scripts/d9_detector_closed_loop.py`). The swapped logs are kept in
`logs/d9_detector/closed_loop/invalid_swapped_legs/`. The per-leg latency/false/miss statistics
were never affected (labels and alarms share one order); the pass rates were.

Valid closed loop, model_v1, p>=0.9, 1 step, seeds 10-49, best four-leg recipe, 10 cm wire:
`logs/d9_detector/run_closed_loop.sh 10 49 logs/d9_detector/closed_loop/fixed_v1_thr09_m1_s10_49.txt --model logs/d9_detector/model_v1.pt`

| Trigger | PASS | FAIL_FRONT | FAIL_REAR |
|---|---|---|---|
| Oracle (D7.9) | 30/40 | 2 | 8 |
| CNN model_v1 (fixed) | 19/40 | 3 | 18 |

Per leg (alarm step minus oracle contact step, 50 Hz): FR median 1, FL 4, RR 5, RL 2; late
by more than 5 steps FR 9, FL 15, RR 19, RL 14 (of ~40 each); early alarms RR 12, RL 6;
false alarms RL 4, RR 1; missed RL 1. The front stage is close to the oracle; the loss is the
rear stage (12 oracle passes became FAIL_REAR).

### D9 Stage 2: round 2 (valid closed-loop data, model_v2)

Collection with the fixed mapping, model_v1 as trigger, seeds 500-599:
`DATA_DIR=logs/d9_detector/data_v2cl logs/d9_detector/run_closed_loop.sh 500 599 logs/d9_detector/closed_loop/collect_fixed_v1_s500_599.txt --model logs/d9_detector/model_v1.pt`
-> 54 PASS, 6 FAIL_FRONT, 40 FAIL_REAR; 17 dropped for deep overlap.
Retrain on v1 + these (`logs/d9_detector/data_v1v2`, symlinks; `train_v2.txt`; `model_v2.pt`).
Evaluation, seeds 10-49: `logs/d9_detector/run_closed_loop.sh 10 49 logs/d9_detector/closed_loop/fixed_v2_thr09_m1_s10_49.txt --model logs/d9_detector/model_v2.pt`

| Trigger | PASS | FAIL_FRONT | FAIL_REAR |
|---|---|---|---|
| Oracle (D7.9) | 30/40 | 2 | 8 |
| CNN model_v1 | 19/40 | 3 | 18 |
| CNN model_v2 | 24/40 | 3 | 13 |

Per leg, model_v2 (alarm minus oracle contact, steps): FR median 0, FL 1, RR 0, RL 0; late by
more than 5 steps FR 2, FL 8, RR 3, RL 1; early alarms FR 4, FL 0, RR 16, RL 14; false alarms
RR 5, RL 5; missed 0. Offline no-wire false-alarm episodes at p>=0.9: 2/8 (model_v1: 0).
Lateness is solved; the open problems are early and false rear alarms.
