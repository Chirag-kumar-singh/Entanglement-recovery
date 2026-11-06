# D5.48 Controlled Recovery Entry-State Swap

## Question
Does seed 11 fail only because it enters recovery from a worse pre-recovery state, or does the outcome remain different even after transplanting seed 0's recovery-entry snapshot into the seed 11 replay?

## Protocol
- Frozen baseline only: no changes to `actions.py`, FSM logic, thresholds, wire geometry, rewards, or checkpoints.
- Captured the last pre-step state before recovery entry for `seed0_baseline` and `seed11_baseline`.
- Replayed seed 11 after restoring the seed 0 pre-recovery snapshot via root pose/velocity and full joint state writes, followed by `sim.forward()`.
- Wrote outputs under `logs/d5_48/`.

## Key results
- `seed0_baseline` succeeded: recovery entry at step 44, stage 1 at 84, stage 2 at 84, full crossing at 84, exit `crossing_release`.
- `seed11_baseline` failed: recovery entry at step 51, never reached stage 1 or stage 2, exit `cap_timeout`.
- `seed11_seed0_entry_swap` partially improved but still failed: recovery entry at step 3, stage 1 at 46, never reached stage 2, exit `cap_timeout`.
- The swap run substantially improved forward geometry relative to seed 11 baseline:
  - `max_calf_forward_dist_m`: `-0.0124 -> 0.4922`
  - `max_foot_forward_dist_m`: `0.0287 -> 0.5154`
- Despite that improvement, the swapped run still did **not** reproduce seed 0's successful crossing.

## Important caveat
This was not a perfectly identical simulator-state transplant.

Restore verification shows:
- root pose diff: `0.0`
- root velocity diff: `2.98e-08`
- joint position diff: `0.0`
- joint velocity diff: `0.0`
- raw `qpos` diff: `0.0294`
- raw `qvel` diff: `0.7093`
- wire contact after restore: `False`, while the source seed 0 snapshot had `wire_contact=True`

So the kinematic restore was exact at the entity API level, but hidden/contact-related simulator state was **not** preserved. That means D5.48 rules out the simplest hypothesis (“only base/joint entry kinematics matter”), but it does **not** fully isolate seed identity from contact-state/history effects.

## Interpretation
- D5.47 already showed the recovery commands are the same across seeds.
- D5.48 shows that giving seed 11 a seed-0-like entry snapshot is **not sufficient** to recover seed 0's full success.
- However, because contact state was not preserved through the restore, the clean conclusion is:
  - failure is **not explained solely by the coarse entry kinematics we restored**;
  - the remaining gap is likely due to missing simulator/contact history state, or another pre-recovery state variable not captured by the current transplant.

## Recommendation
Next step should be a **contact-consistent replay intervention**, not a controller change:
- either restore at an earlier point before active wire contact develops,
- or use a lower-level simulator-state snapshot/restore path that preserves contact-relevant state if the platform exposes one safely,
- then rerun the same seed-0-to-seed-11 swap.

At this point, D5.48 weakens the pure entry-kinematics explanation but does not yet cleanly falsify the broader entry-state-sensitivity hypothesis.