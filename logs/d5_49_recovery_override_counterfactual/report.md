# D5.49 — Recovery Override Counterfactual

Diagnostic only. This experiment compares the existing seed-0 baseline against a replay from the exact same verified pre-recovery simulator state, but with the normal walking position actuators kept active for a short fixed window after first contact before the existing hybrid override is allowed to take over.

## Configuration

- Seed: 0
- Device: cpu
- Fixed post-contact hold: 8 recovery steps
- Recovery cap: 48 steps

## Artifacts

- Baseline trace: [baseline_trace.csv](baseline_trace.csv)
- Counterfactual trace: [counterfactual_trace.csv](counterfactual_trace.csv)
- Shared entry snapshot: [entry_snapshot.csv](entry_snapshot.csv)
- Summary metrics: [comparison_summary.csv](comparison_summary.csv)
- Aligned window: [aligned_entry_window.csv](aligned_entry_window.csv)

## What changed in Run B

- The same contact-triggered RECOVERY FSM still starts at first contact.
- The wire, trigger, walking checkpoint, and hybrid commands are unchanged.
- Only the timing of the actuator handoff changes: thigh/calf position actuators stay active for a short fixed window, then the original hybrid override takes over.

## Restore validation

- Pre-entry snapshot step: 44
- Counterfactual restore qpos max abs diff: 0.000e+00
- Counterfactual restore qvel max abs diff: 0.000e+00
- Counterfactual restore wire contact / force: True / 10.2652 N

## Comparison

| Metric | Baseline | Counterfactual |
|---|---:|---:|
| Recovery entry step | 44 | 44 |
| Hybrid handoff step | 44 | 52 |
| Delay after entry | 0 | 8 |
| First base-height excursion step | 47 | 44 |
| First pitch-rate excursion step | 45 | 44 |
| Minimum base height (m) | 0.2040 @ 54 | 0.1816 @ 92 |
| Peak |pitch rate| (rad/s) | 1.8040 @ 49 | 2.0482 @ 57 |
| Max wire force magnitude (N) | 250.8633 @ 51 | 87.7105 @ 98 |
| First calf-wire raw contact step | 44 | 46 |
| First hip-wire raw contact step | 51 | 58 |
| Crossing achieved | True | False |
| Release achieved | True | False |

## Interpretation

- If the counterfactual substantially delays or reduces the first pitch-rate and base-height excursions while the wire-contact history stays comparable, that points to the hybrid handoff itself as a major amplifier of the stumble.
- If both runs destabilize at nearly the same time and magnitude despite the delayed handoff, then wire contact / entanglement mechanics dominate and the override timing is secondary.

## Observed result

- Baseline reaches its first pitch-rate excursion at step 45 and first base-height excursion at step 47.
- Counterfactual reaches its first pitch-rate excursion at step 44 and first base-height excursion at step 44.
- Baseline minimum base height is 0.2040 m versus 0.1816 m in the counterfactual.
- Baseline peak |pitch rate| is 1.8040 rad/s versus 2.0482 rad/s in the counterfactual.
- The counterfactual delays the hybrid handoff by 8 recovery steps.

## Bottom line

- The delayed-handoff counterfactual does not materially reduce the early instability relative to baseline, so wire-contact mechanics remain the stronger explanation than the hybrid handoff itself.
