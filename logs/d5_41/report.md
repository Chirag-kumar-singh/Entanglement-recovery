# D5.41 — Recovery Failure Moment

## Verdict
Seed 11 fails because the FL calf never makes the near-to-far transition needed for stage 1 / stage 2 latching. The body does keep moving forward for much of recovery, so the primary failure is the leg’s horizontal trajectory, not an immediate body stall.

## Directly supported
- Seed 0 baseline enters recovery at step 44 with wire contact already true and reaches stage 1 / stage 2 at step 84 / 84.
- Seed 11 baseline enters recovery at step 51 with wire contact false, never latches stage 1 / stage 2, and times out at step 98.
- Seed 11 does lift the calf above the wire at times, but the calf forward distance stays negative for the full recovery window: the diagnosis in the D5.35 CSV is “missing near-to-far transition.”
- Wire contact is not inferred from signed distance alone: seed 11 actually acquires wire contact after recovery starts (first true at step 53), so the issue is not simply “never touched the wire.”
- The hybrid auxiliary commands do not explain the split: D5.36 records the same auxiliary thigh velocity and calf motor commands across seeds during recovery.

## What the aligned rows show
- The first persistent divergence is the calf forward distance: D5.36 reports that the persistent gap begins at recovery offset 39; at that point seed 0 is at about -0.0091 m and seed 11 at about -0.0343 m.
- Seed 11’s body does not immediately stall at recovery entry. The body forward velocity is positive early in recovery and remains positive for many offsets before later weakening and reversing.
- That means the leg’s horizontal trajectory fails while the body is still moving forward, which is the strongest supported explanation.

## Hypotheses still open
- Seed 11 enters recovery with a more negative base vertical velocity and without wire contact, so the entry state is less favorable than seed 0. That may contribute to the failure, but the existing telemetry does not isolate it as the root cause.
- The data does not prove the trigger is too early on its own; it only shows that the seed 11 entry state is different enough that the fixed hybrid motion never carries the calf across.

## Answers to the questions
1. **Does the FL foot approach or contact the wire before achieving sufficient clearance?**
   It does contact the wire, but only after recovery begins and not in a way that produces the needed near-to-far calf transition. So contact is present, but it is not the decisive failure mode.
2. **Does the robot's body stop advancing or move backward before the calf fails to cross?**
   No, not initially. The body keeps advancing through early recovery; the failure happens while the calf remains behind the wire. Later body velocity weakens/reverses, but that looks downstream of the failed leg geometry.
3. **Does the leg's horizontal trajectory fail even while the body continues moving forward?**
   Yes. This is the best-supported explanation.
4. **Is there evidence that the recovery trigger occurs too early relative to the robot's gait phase or entry dynamics?**
   There is suggestive evidence only: seed 11 enters recovery with more negative vertical velocity and a different contact state, so timing may matter, but the current logs do not isolate timing as the cause.
5. **Which explanation is directly supported by measurements, and which remains a hypothesis?**
   Directly supported: the calf fails to advance far enough horizontally to cross the wire, so stage 1 never latches. Hypothesis: the seed 11 entry state / gait phase makes the fixed hybrid motion less effective.

## Recommended next experiment
Run a phase-aligned recovery-entry ablation on seed 11: hold the controller fixed, but delay or phase-lock the recovery trigger to match seed 0’s entry-state conditions (especially the body vertical velocity and wire-contact state), then check whether the calf forward distance becomes positive and stage 1 latches.

Evidence supporting that choice:
- seed 0 enters recovery with wire contact already true and eventually crosses;
- seed 11 enters without contact, then contact appears later while calf forward distance stays negative;
- the hybrid commands themselves are identical across seeds, so the missing variable is the entry-state timing / dynamics rather than command magnitude.

## Files written
- [logs/d5_41/recovery_alignment.csv](logs/d5_41/recovery_alignment.csv)
- [logs/d5_41/report.md](logs/d5_41/report.md)
