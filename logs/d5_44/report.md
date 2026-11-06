# D5.44 — Recovery Entry-State Robustness

## Verdict
Seed 11’s unfavorable entry state does contribute to the failure, but the data do not support a claim that the calf becomes irrecoverable within the first 10–15 recovery steps. The clearest supported conclusion is that the fixed hybrid recovery command is not robust enough to handle both entry states without changing the control strategy.

## Observed facts
- Seed 0 enters recovery at step 44 with wire contact already true and base vertical velocity near -0.0327 m/s.
- Seed 11 enters recovery at step 51 with wire contact false and base vertical velocity near -0.1497 m/s.
- In the first 16 recovery steps, both seeds keep the calf behind the wire plane, but seed 11 stays more retracted and more vertically unstable.
- Seed 0 later reaches positive calf forward distance at step 84 and completes recovery; seed 11 never reaches positive calf forward distance in the available window.
- The D5.36 actuator audit shows the auxiliary hybrid commands themselves are the same across seeds during recovery, so the split is not caused by different command magnitudes.
- The D5.40 gradual thigh transition preserves seed 0 success but still leaves seed 11 at cap timeout with no stage-1 latch.

## Entry-window comparison
- The first 10–15 recovery steps do not show a hard geometric collapse for seed 11; instead, they show a consistently worse version of the same blocked trajectory.
- Seed 11’s body becomes less favorable very quickly after entry: base vertical velocity drops to strongly negative values while calf forward distance remains negative.
- The later persistent-contact window is not obviously safer: by the time contact becomes sustained again, the body is already more downward and the calf is still behind the wire.

## When the trajectory becomes unrecoverable
- The telemetry does not support an “instant failure” at recovery entry.
- The first clearly persistent divergence is the calf-forward-distance gap that becomes stable around recovery offset 39, after which seed 0 continues toward crossing while seed 11 remains negative.
- At that point, seed 0 is already near the wire plane and later crosses; seed 11 is still about -0.03 to -0.04 m behind it and never catches up.

## Contact-persistence interpretation
- Waiting for persistent contact would likely postpone the same unfavorable recovery rather than create a safer entry.
- The later contact window appears after the body has descended further, so the entry state is not improved just by delaying until contact is sustained.
- That makes a pure trigger delay a weak control experiment compared with changing the recovery motion itself.

## Controller robustness conclusion
The current fixed hybrid command does not appear robust enough to handle both entry states without changing the walking/recovery interaction. The seed 11 failure is best explained by the combination of a more negative entry vertical velocity, brief/fragmented contact near entry, and a calf trajectory that never develops the near-to-far crossing needed by the FSM.

## Smallest safe next experiment
Test a minimal state-dependent recovery tweak rather than a trigger delay: keep the wire-contact gate unchanged, but add a short forward-advance bias once the calf has cleared the wire so the hybrid command is no longer purely fixed through the whole recovery. This is the smallest change that directly targets the unsupported forward-crossing behavior without relaxing the contact gate.

## Files written
- [logs/d5_44/recovery_entry_comparison.csv](logs/d5_44/recovery_entry_comparison.csv)
- [logs/d5_44/report.md](logs/d5_44/report.md)

## Reference points from prior logs
- [logs/d5_41/report.md](logs/d5_41/report.md)
- [logs/d5_43/report.md](logs/d5_43/report.md)
- [src/mjlab/tasks/recovery/mdp/actions.py](src/mjlab/tasks/recovery/mdp/actions.py)
