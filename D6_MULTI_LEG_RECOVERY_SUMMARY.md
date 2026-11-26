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
