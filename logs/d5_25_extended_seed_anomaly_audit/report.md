# D5.25 — Extended-Seed Recovery Anomaly Audit

## Objective

Test whether the D5.24 anomalies (seed 5's late body-height nadir;
seed 7's neither-stage-reached / contact-preserved pattern) recur in
additional recovery trajectories, and whether the exit+1 FL
foot-velocity sign flip remains universal, using a larger seed cohort.

## Step 1 — Identifying the additional cohort

Searched `logs/` and `scripts/` for any existing trajectories beyond
seeds 0–9 generated under the same checkpoint/environment/wire
configuration used throughout D5.18–D5.24 (`go2_wire_recovery_residual_env_cfg`,
`recovery_mode="hybrid"`, `ISOLATED_FL_WIRE_CFG`, `max_recovery_steps=48`).
**None exist** — every prior D5.x log directory (`d5_5_repeatability`,
`d5_19`...`d5_24`) only ever used seeds 0–9, and no other script in
`scripts/` uses a seed ≥10 under this configuration. Other seed-numbered
directories in `logs/` (`d3_clean_fl_seed{0-3}`, `d3_peak_baseline_seed{0-3}`,
`d4_contact_constraint_seed1`) use different, earlier-era D3/D4 configs
(non-hybrid recovery mode and/or different sensor sets) and are not
reusable for this purely observational signal set.

Per the task's instruction, since no suitable existing seeds were
available, **new seeds were simulated** using the existing unmodified
observational setup. Ran seeds **10–19** (10 new seeds, same count as
the original cohort) via:

```
uv run python scripts/d5_23_intra_window_observational_audit.py --seed <N> --max-steps 200 \
  --out-dir logs/d5_25_extended_seed_anomaly_audit/seed<N>
```

for N in {10,...,19} — **the identical, unmodified D5.23 script**
(same checkpoint, same `go2_wire_recovery_residual_env_cfg` build path,
same wire geometry, same cap=48, same sensor/logging definitions). The
only varying input is `--seed`, which flows only into
`D3GroundConfig(seed=...)` → `env_cfg.seed` and `env.reset(seed=...)`,
i.e. initial-condition/domain-randomization draw — not controller
behavior. Provenance: all 10 new seeds' `run_log.txt` + `trajectory.csv`
are in `logs/d5_25_extended_seed_anomaly_audit/seed{10..19}/`, with a
combined console log in `run_all.log`. No controller, gain, FSM, cap,
or policy change of any kind was made to produce these runs.

## Step 2 — Logging reuse

No new logging code. Same validated D5.23 FL-force-aggregation fix,
coordinate frames (`quat_apply_inverse`-based base-relative foot
position/velocity), and full column set (FSM phase/stage
latches/exit/elapsed/timeout, thigh/calf tracking error, actuator
force + saturation flags, foot forward/vertical velocity and ground
contact + force magnitude, FR/RL/RR contact, base height/vertical
velocity, crossing/release counters) as used in D5.23/D5.24.

## Step 3 — Recurrence analysis

Outcome summary for the new cohort (seeds 10–19):

| seed | outcome | stage1 | stage2 | exit | timeout |
|---|---|---|---|---|---|
| 10 | timeout | None | None | 91 | True |
| 11 | timeout | None | None | 98 | True |
| 12 | **genuine** | 86 | 87 | 91 | False |
| 13 | timeout | None | None | 98 | True |
| 14 | timeout | None | None | 91 | True |
| 15 | timeout | None | None | 94 | True |
| 16 | timeout | None | None | 94 | True |
| 17 | timeout | None | None | 92 | True |
| 18 | **genuine** | 89 | 89 | 93 | False |
| 19 | **genuine** | 74 | 75 | 79 | False |

3/10 genuine completions, 7/10 cap-timeouts — and critically, **all 7
timeout seeds in the new cohort reach neither stage1 nor stage2**,
unlike the original cohort where timeout seeds 5/6 reached stage1 only
and only seed 7 reached neither stage.

### Full 20-seed event-ordering table (0–9 original + 10–19 new)

| seed | group | outcome | stage reached | FL foot_vx flip offset | base-z drop (m) | nadir offset | base-vz onset offset | FR/RL contact-loss offset (exit-relative) |
|---|---|---|---|---|---|---|---|---|
| 0 | orig | genuine | both | +1 | 0.0362 | +4 | −3 | −4 / −6 |
| 1 | orig | genuine | both | +1 | 0.0409 | +5 | −1 | −3 / −3 |
| 2 | orig | genuine | both | +1 | 0.0226 | +5 | −1 | −3 / −2 |
| 3 | orig | genuine (floor) | both | +1 | 0.0000 | 0 | none | +2 / 0 |
| 4 | orig | genuine (floor) | both | +1 | 0.0000 | 0 | none | +2 / 0 |
| 8 | orig | genuine | both | +1 | 0.0336 | +4 | −3 | −3 / −4 |
| 5 | orig | timeout | stage1 only | +1 | 0.0122 | **+17 (anomalous)** | none | +2 / 0 |
| 6 | orig | timeout | stage1 only | +2 | 0.0306 | +5 | −1 | −1 / −2 |
| 7 | orig | timeout | **neither** | +1 | 0.0000 | 0 | −2 | +2 / +3 |
| 9 | orig | timeout | both | +1 | 0.0261 | +4 | −1 | −3 / −3 |
| 10 | new | timeout | neither | **none detected** | 0.0000 | 0 | −7 | +4 / +7 |
| 11 | new | timeout | neither | +1 | 0.0202 | +4 | −2 | +2 / 0 |
| 12 | new | genuine | both | +1 | 0.0259 | **+17 (anomalous)** | none | +2 / −1 |
| 13 | new | timeout | neither | +1 | 0.0057 | +5 | −2 | +5 / +3 |
| 14 | new | timeout | neither | +2 | 0.0000 | 0 | −6 | +3 / +2 |
| 15 | new | timeout | neither | +1 | 0.0000 | 0 | −5 | +2 / +2 |
| 16 | new | timeout | neither | **none detected** | 0.0000 | 0 | −6 | +4 / +3 |
| 17 | new | timeout | neither | +1 | 0.0314 | +4 | −2 | −1 / −1 |
| 18 | new | genuine | both | +1 | 0.0391 | +4 | −3 | −4 / −4 |
| 19 | new | genuine | both | +2 | 0.0007 | +1 | −6 | −8 / +1 |

### (a) Does the exit+1 foot-velocity sign flip persist?

**Mostly, but no longer universally.** Across n=20: flip detected at
exactly exit+1 in 15/20 (75%); at exit+1 or exit+2 in 18/20 (90%); **not
detected at all within the exit+0..+10 window in 2/20 (10%): seeds 10
and 16** — both are "neither-stage-reached" timeouts whose FL foot
velocity never develops a clear pre-exit approach signal large enough
to register a flip under the same threshold used throughout D5.23/D5.24
(this is a detection-threshold question, not necessarily evidence the
underlying kinematic event doesn't occur — see Limitations). The prior
claim of a fully universal exit+1 flip (established on n=10) does
**not** hold unchanged at n=20; it should be revised to "exit+1 flip is
the dominant but not exceptionless pattern (~75-90% depending on
tolerance)."

### (b) Do seed-5-like late nadirs recur?

**Yes.** Seed 12 (a *genuine* completion, not even a timeout) shows the
identical anomalous nadir offset of **+17** found in seed 5, with a
comparable drop magnitude (0.0259 m vs. seed 5's 0.0122 m). This
directly answers the open question from D5.24: the late-nadir pattern
is not a one-off artifact of seed 5 specifically — it recurs at
roughly 2/20 (10%) in this combined cohort, and notably is **not
confined to timeout/failure seeds** (seed 12 succeeds genuinely despite
the same anomalous late nadir), which argues against a simple
"late nadir = failure indicator" interpretation.

### (c) Does seed-7-like stage-unreached/contact-preserved behavior recur?

**Yes, and far more often than the original cohort suggested.** 8/20
seeds (40%) reach **neither** stage1 nor stage2: seed 7 (original) plus
seeds 10, 11, 13, 14, 15, 16, 17 (new) — compared to 1/10 (10%) in the
original cohort alone. Of these 8, most (6/8: seeds 7, 10, 13, 14, 15,
16) show the seed-7 signature of FR/RL contact **preserved through and
past exit** (positive, i.e. post-exit, contact-loss offsets on both
legs), while 2/8 (seeds 11, 17) show at least one leg losing contact
before exit despite never reaching a crossing latch — a partial,
not full, match to the seed-7 pattern. This confirms the
neither-stage-reached failure mode is common (not an edge case) and
that it is usually, but not always, accompanied by the specific
contact-preservation signature.

### (d) Counts and proportions (denominator = 20 unless noted)

- Exit+1 exact flip: 15/20 (75%)
- Exit+1 or +2 flip: 18/20 (90%)
- No detectable flip: 2/20 (10%) — seeds 10, 16
- Anomalous late nadir (~+17): 2/20 (10%) — seeds 5, 12
- Neither crossing stage reached: 8/20 (40%) — seeds 7, 10, 11, 13, 14, 15, 16, 17
- Of the "neither-stage" group, contact preserved past exit on ≥1 leg: 6/8 (75%)
- Genuine completions overall: 9/20 (45%) — seeds 0,1,2,3,4,8,12,18,19
- Cap-timeout overall: 11/20 (55%)

### (e) Phase-comparable vs. non-comparable classification

- **Stage-2-comparable** (directly comparable to D5.23's stage-2-anchored
  natural-exit windows): seeds 0,1,2,3,4,8,9,12,18,19 — 10/20, all of
  which latch stage2 (6 genuine-original + seed 9 timeout + 3
  new-genuine: 12,18,19).
- **Stage-1-only comparable** (exit-relative comparison only, no
  stage-2 anchor available): seeds 5, 6 — 2/20.
- **Non-comparable** (no crossing-stage anchor at all — exit-relative
  only, and even that is of a qualitatively different pre-exit regime
  per D5.24's seed 7 findings): seeds 7, 10, 11, 13, 14, 15, 16, 17 —
  8/20, now the single largest group. No stage-anchored correlation is
  computed across this group and the stage-2-comparable group; they
  are reported separately, consistent with the task's instruction not
  to combine non-comparable phases into one correlation.

### (f) Does the added cohort change interpretation or next-experiment recommendation?

**Yes, materially.** Three revisions to the D5.24 interpretation:

1. The exit+1 FL-foot-velocity flip, previously reported as
   seed-invariant across all 10 original seeds, is now shown to have
   exceptions (seeds 10, 16) at n=20. It should be described as the
   dominant pattern (~75-90%), not a universal law.
2. The seed-5 late-nadir anomaly recurs (seed 12) and, notably, does so
   in a *genuine success* seed — meaning late nadir timing is not
   simply a failure correlate, weakening any inference that it reflects
   recovery quality specifically rather than an independent dynamical
   quirk.
3. The seed-7 "neither stage reached" pattern, previously treated as a
   single outlier (1/10), is now the **largest single sub-group** in the
   combined cohort (8/20, 40%) — this changes its status from "rare
   edge case" to "common, likely structurally driven failure mode"
   that merits its own dedicated investigation (e.g., what
   initial-condition features predict landing in this sub-group vs. the
   stage-reaching sub-groups), rather than being treated as noise.

## Observation vs. inference

- **Observed:** all step indices, flags, and derived offsets in the
  tables above, computed directly and identically (same thresholds as
  D5.23/D5.24) from step-indexed CSV columns for all 20 seeds.
- **Inference, not established:** that seeds 10/16's absent flip
  reflects a genuinely different kinematic mechanism rather than a
  detection-threshold artifact (their FL foot may simply not develop a
  large enough velocity swing within the logged window); that seed 12's
  late nadir and seed 5's late nadir share a common cause; that the
  40% neither-stage-reached proportion found here generalizes beyond
  this specific 10-seed draw (it is itself a single sample of 10 new
  seeds, not an exhaustive characterization).

## Visualization

A meaningful behavioral difference was found (the neither-stage-reached
failure mode, now the largest sub-group, and the recurring anomalous
late-nadir case). Before recommending a command, the repository's
supported interactive-viewer path was verified directly from source,
not assumed:

- `scripts/d5_23_intra_window_observational_audit.py` declares a
  `--debug` flag in its config dataclass, but **does not actually wire
  it to any viewer code** (no `cfg.debug` branch exists in the script
  body) — confirmed via grep; passing `--debug` to this script would
  have no effect. It is not usable for visualization as-is.
- `scripts/d5_18_post_recovery_diagnostic.py` **does** implement a
  verified `cfg.debug` branch using `NativeMujocoViewer` with the
  D5.15-corrected single-step-per-tick callback (confirmed via its
  `--help` output and source: "If True, use the D5.15/D5.16/D5.17-corrected
  native MuJoCo viewer... CSV/log output is identical either way").
  This is the verified, supported interactive path.

Recommended command to visually inspect one of the newly-found
neither-stage-reached, contact-preserved-through-exit cases (seed 11,
representative of the dominant new failure sub-group) in the native
viewer:

```
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 11 --max-steps 200 \
  --out-dir logs/d5_25_extended_seed_anomaly_audit/_viz_seed11 --debug
```

(On macOS this requires `mjpython` per `scripts/fix_mjpython_macos.sh`;
not applicable here, run in Linux headless-capable terminal with a
display available.) This is a **recommendation only** — it was not
executed as part of this task, since launching an interactive viewer
falls outside the observational/analysis scope of this audit and the
task's stop condition.

## Limitations

- The new cohort is itself only 10 seeds; proportions reported (e.g.,
  40% neither-stage-reached) are not a stable population estimate.
- Flip-detection and nadir/onset thresholds are identical to
  D5.23/D5.24's; no threshold-sensitivity re-analysis was performed
  here specifically for the two no-flip-detected seeds (10, 16) — it
  remains possible a looser threshold would detect a smaller flip in
  both.
- No causal claims are made anywhere in this report; all findings are
  temporal/structural associations within single trajectories.
- Checkpoint and environment configuration provenance were verified by
  inspecting source (`D3GroundConfig`, `go2_wire_recovery_residual_env_cfg`
  call signature) rather than assumed; no part of this task altered the
  frozen PPO checkpoint, wire geometry, FSM logic, gains, or caps.

## Recommended next experiment (exactly one, not executed)

Given that the neither-stage-reached failure mode is now the largest
sub-group (40%) and its membership appears to depend on seed/initial
conditions rather than the crossing-latch mechanics themselves,
the next experiment should be a **purely observational** audit that
logs, for a larger seed sample (e.g. 20–50 additional seeds using this
same unmodified script and configuration), the state at the recovery
**trigger** step (base orientation, FL joint angles/velocities, wire
contact geometry) and tests whether any of these entry-condition
features separate the "neither-stage-reached" sub-group from the
"stage-reaching" sub-group — still with no caps, gains, FSM, or
checkpoint changes. This is **not implemented** here, per the stop
condition.

## Stop-condition compliance

No caps, gains, FSM logic, or frozen-policy changes were made. No
production cap was selected. No training was run. No follow-up
experiment was launched automatically. The one recommended next
experiment above is a recommendation only.
