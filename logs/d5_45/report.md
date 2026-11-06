# D5.45 — Small State-Dependent Forward-Advance Bias

## Verdict
The smallest safe change I tested was a reduced-magnitude forward-advance thigh target after confirmed calf clearance. It preserved seed 0’s successful crossing, but it did not rescue seed 11: the calf still never achieved a positive forward distance in the available window, and the FSM still timed out without stage-1 / stage-2 latches.

## What changed
- The recovery trigger, FSM, thresholds, release logic, wire geometry, rewards, and termination were left unchanged.
- The experimental mode reuses the existing two-phase hybrid gate, but scales the post-clearance thigh advance target by a fixed fraction instead of using the full sign-reversed target.
- The active sign convention is unchanged: the baseline hybrid retract target still maps to a positive thigh velocity command, and the forward-advance phase still maps to a negative command.

## Observed facts
- Seed 0 baseline still succeeds with crossing release.
- Seed 0 forward-bias also succeeds with crossing release.
- Seed 11 baseline still times out without stage-1 / stage-2 latches.
- Seed 11 forward-bias still times out without stage-1 / stage-2 latches.
- Seed 11 does show a smaller max calf-forward-distance deficit under the forward-bias variant, but it remains negative throughout the observed window.
- Seed 11’s max calf height drops substantially under the forward-bias variant, which is evidence that the change altered the motion profile, but that extra lift did not translate into the required near-to-far crossing.

## Directly supported conclusion
The forward-advance magnitude is not the missing ingredient by itself. The seed 11 failure is still dominated by entry-state geometry and timing: the leg lifts, but it does not convert that lift into a successful calf crossing before timeout.

## What the new signal says about the command choice
- The D5.44 comparison showed seed 11 enters recovery with a more negative vertical velocity and less favorable contact persistence.
- The D5.45 experiment reduced the post-clearance advance magnitude and kept seed 0 intact, which is useful confirmation that the experiment is safe enough to preserve the successful case.
- However, seed 11 still fails, so the evidence does not support a stronger claim that “just enough forward bias” is the missing control ingredient.
- In practical terms, the safe forward-bias knob changed the trajectory shape but not the outcome.

## Smallest safe next step
Do not widen the command space further yet. The remaining uncertainty is whether a different state trigger, not a smaller forward magnitude, would matter — for example, a trigger based on sustained contact plus a body-state guard, or a more explicit foot/calf phase split. That would be a new control hypothesis, not a simple magnitude tweak.

## Files written
- [logs/d5_45/forward_bias_experiment/summary.csv](logs/d5_45/forward_bias_experiment/summary.csv)
- [logs/d5_45/forward_bias_experiment/summary.txt](logs/d5_45/forward_bias_experiment/summary.txt)
- [logs/d5_45/report.md](logs/d5_45/report.md)

## Code changed
- [src/mjlab/tasks/recovery/mdp/actions.py](src/mjlab/tasks/recovery/mdp/actions.py)
- [src/mjlab/tasks/recovery/env_cfg_residual.py](src/mjlab/tasks/recovery/env_cfg_residual.py)
- [tests/test_two_phase_hybrid_recovery.py](tests/test_two_phase_hybrid_recovery.py)
- [scripts/d5_29_visualize_recovery.py](scripts/d5_29_visualize_recovery.py)
- [scripts/d5_45_state_dependent_forward_bias_experiment.py](scripts/d5_45_state_dependent_forward_bias_experiment.py)
- [docs/source/changelog.rst](docs/source/changelog.rst)
