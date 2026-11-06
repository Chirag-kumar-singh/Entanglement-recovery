# D5.5 — 20-Seed Repeatability Sweep of the 48-Step Extended Recovery Window

## Harness / config verification

- Script: [scripts/d5_3_extended_recovery_window.py](scripts/d5_3_extended_recovery_window.py) (unmodified, same
  harness validated and audited in D5.3/D5.4).
- Config for every run: `--no-ablate-hip-wire` (baseline condition), `--max-recovery-steps-override=48`
  (default), same frozen walking checkpoint
  (`logs/rsl_rl/go2_velocity/2026-09-17_13-00-16/model_9900.pt`), same wire geometry/gains. No parameter was
  tuned at any point during the sweep.
- 20 seeds (0–19) run headless (no `--debug`/viewer), each writing to its own
  `logs/d5_5_repeatability/seed{N}/{trajectory.csv,run_log.txt}` plus a `seed{N}_console.log`.
- Execution integrity: all 20 runs exited with code 0 and produced complete 200-step trajectories (verified
  `step` column runs 0→199 with no gaps). **0 invalid/execution-error runs** — no runs were discarded.

## Headline result

| Metric | Value |
|---|---|
| Genuine recovery successes | 9 / 20 |
| Raw success rate | 45.0% |
| 95% Wilson score CI | [25.8%, 65.8%] |
| Timeouts (`recovery_failed_timeout`) | 11 / 20 |
| Stability failures (fall/termination) | 0 / 20 |
| Invalid runs | 0 / 20 |

The confidence interval is wide (small n) but excludes anything near the 100% implied by the D5.3/D5.4
two-seed validation — the true success rate of this configuration is almost certainly well below 100%.

## Classification (6 categories)

| Category | Seeds | Count |
|---|---|---|
| 1. Comfortable success (margin ≥ 3) | 0 (margin 3), 3 (margin 3), 8 (margin 3), 19 (margin 11) | 4 |
| 2. Low-margin success (margin 1–2) | 4 (margin 1) | 1 |
| 3. Zero-margin success (margin 0) | 1, 2, 12, 18 | 4 |
| 4. Timeout (no genuine completion) | 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17 | 11 |
| 5. Stability failure (fell over) | — | 0 |
| 6. Invalid run (execution error) | — | 0 |

**44% of all successes (4 of 9) are zero-margin "knife-edge" cases** — they complete in exactly the final
step before the 48-step cap. Combined with the low-margin case (seed 4, margin 1), 5 of 9 successes (56%)
clear the bar with 1 step or less to spare.

## Timeout sub-mode breakdown (all 11 timeouts independently verified from per-step gate columns)

| Sub-mode | Seeds | Count | % of timeouts |
|---|---|---|---|
| Stage 1 (`calf_clearance_achieved`) never latches at all | 7, 10, 11, 13, 14, 15, 16, 17 | 8 | 73% |
| Stage 1 latches but Stage 2 (`foot_crossed_achieved`) never does | 5, 6 | 2 | 18% |
| Full crossing achieved but `release_counter` doesn't reach 5 before the cap ("near-miss") | 9 | 1 | 9% |

Seed 9 is the clearest near-miss: stage1=87, stage2=crossing=88, but only 3 remaining steps before the
step-91 cap (`recovery_elapsed`=48), insufficient for the 5-consecutive-step release-sustain requirement
(`release_counter` reached 4/5). All three sub-modes were confirmed directly from the latched gate columns
in each seed's trajectory.csv, not inferred.

**The dominant failure mode (73% of timeouts, 40% of all 20 seeds) is that Stage 1 — the calf ever
achieving side-far position + height clearance over the wire — never happens at all within 48 steps.**
This is the same gate that D5.1 found never latched within the original 16-step cap; extending the cap to
48 steps fixes it for roughly half of seeds but leaves it completely unresolved for the other half.

## Stage-gate timing statistics (successes only, n=9)

- Stage 1 latch step: range 74–90 (median 86)
- Stage 2 latch step: range 75–91 (median 86)
- Crossing latch step: identical to stage 2 (stage 2 implies crossing, as expected from the gating logic)
- `recovery_elapsed` at exit: {37, 45, 45, 45, 47, 48, 48, 48, 48}

This is consistent with, but wider than, the D5.3/D5.4 two-seed sample (which only sampled steps 84 and 86).
The full distribution shows genuine successes latching anywhere from step 74 to step 90 — i.e. very close
to, or exactly at, the 48-step cap in the majority of cases.

## Post-recovery stability assessment

Across all 20 seeds, including all 11 timeouts, no fall or termination event was observed: `base_z` stayed
in the ~0.28–0.31 m standing range and `base_quat_w` stayed above 0.92 through the final logged step (199)
in every seed inspected, including the two lowest-margin stability proxies (seed 5: min quat_w 0.926; seed
19: min quat_w 0.928 — both still well above an upright threshold, and both are otherwise-ordinary runs with
no instability on closer inspection). The environment's actual termination definition
(`mdp.fell_over`, `limit_angle=70°` in [src/mjlab/tasks/recovery/env_cfg_residual.py](src/mjlab/tasks/recovery/env_cfg_residual.py#L516))
was never triggered in any of the 20 runs. **No stability failures occurred in this sweep** — every failure
mode observed is a timeout (FSM gates not satisfied in time), not a fall.

## Comparison to the D5.3/D5.4 baseline (seeds 0, 1)

The original n=2 validation (seed 0: margin 3; seed 1: margin 0) showed 100% genuine success and was the
basis for treating the 48-step cap as "validated." The 20-seed sweep shows this did **not** generalize:

- True success rate is ~45%, not 100%.
- Even seed 1's zero-margin result, flagged in D5.4 as a possible outlier "knife-edge" case, turns out to be
  the *typical* shape of a success in this configuration (4 of 9 successes are zero-margin, matching seed
  1's pattern; only seed 0's comfortable margin-3 result generalizes to a minority of seeds).
- Seeds 0 and 1 happened to sample two of the more favorable seeds; a majority of seeds (11/20) never even
  reach Stage 1 and Stage 2 jointly within the extended window.

## Seed-dependent pattern evidence

- Entry step into recovery (gait phase at wire contact) ranges 39–51 across all seeds and **overlaps
  between successes (42–51) and failures (39–51)** — entry timing alone does not predict outcome.
- No single scalar inspected (entry step, max calf clearance, max foot-forward distance) cleanly separates
  success from failure; the split appears to depend on specific per-seed dynamics of the calf-clearance
  maneuver (whether `calf_side_far` and the 0.02 m height clearance are jointly achieved at all), consistent
  with the task's known sensitivity established back in D4.8/D4.9.

## Recommendation

**Do not yet adopt the 48-step configuration as the next development baseline.** The 20-seed sweep shows
only a ~45% (95% CI 26–66%) raw genuine-success rate, dominated by a single, well-characterized failure
mode: in 8 of 20 seeds (and 73% of all timeouts), Stage 1 (`calf_clearance_achieved`) never latches at all
within 48 steps — the calf never achieves both the side-far transition and the 0.02 m height clearance over
the wire. This is the same root gate failure identified in D5.1 under the original 16-step cap; the extended
window only resolves it in roughly half of seeds, and even among successes, most (5 of 9) clear the bar with
a 0–1 step margin, indicating the 48-step cap is barely adequate even for seeds that do succeed.

**Next step should be a targeted investigation of why Stage 1 fails to latch in the 8/20 "never-latches"
seeds specifically** — e.g., instrumenting `calf_side_far` and `calf_height_above_wire` trajectories for
those 8 seeds to determine whether the calf-clearance maneuver itself is geometrically infeasible for
certain gait-phase-at-entry conditions, before considering any further extension of the time budget or
adoption of this configuration as a baseline.
