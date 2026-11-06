# D5.42 — Trigger Timing Check

## Conclusion
I did **not** implement a new delayed-trigger variant. The current recovery entry already uses the real wire-contact sensor, so “delay recovery until actual wire contact” is already the baseline behavior rather than a missing timing mode.

## What the code does
- In [src/mjlab/tasks/recovery/mdp/actions.py](src/mjlab/tasks/recovery/mdp/actions.py), recovery entry is `entering_recovery = is_normal & in_contact`, where `in_contact` comes from `self._env.scene.sensors[self.cfg.wire_contact_sensor_name]`.
- In [src/mjlab/tasks/recovery/env_cfg_residual.py](src/mjlab/tasks/recovery/env_cfg_residual.py), the sensor named `wire_contact` is a `ContactSensorCfg` with the wire geoms as primary and the FL calf collision geom as secondary.
- That means the trigger is already keyed to real contact; changing it to “wait for actual contact” would be redundant unless we wanted a stronger gate like sustained contact or a phase-locked delay.

## Evidence from existing telemetry
- Seed 0 baseline shows wire contact at step 43 and the RECOVERY row at step 44, then reaches stage 1 at step 84 and stage 2 at step 84.
- Seed 11 baseline shows wire contact at step 50 and the RECOVERY row at step 51, but still never latches stage 1 or stage 2.
- The D5.41 alignment and D5.35/40 summaries show the primary failure is geometric: seed 11’s FL calf forward distance stays negative through recovery, while seed 0 eventually becomes positive.
- The body does not instantly stall at recovery entry for seed 11; forward body velocity remains positive for several recovery steps before later weakening.

## Can the existing telemetry compare entry state consistently?
Yes.
- Both seed 0 and seed 11 CSVs record the same recovery-step telemetry, including wire contact, phase, body velocity, FL joint position/velocity, calf height, and signed forward distance.
- The only difference is the entry state itself: seed 0 reaches recovery immediately after contact, while seed 11 reaches recovery one row after its first contact event.

## Why I stopped here
Because the trigger is already contact-gated, a “delay until wire contact” experiment would not be a meaningful new controller variant. The smallest valid alternative timing test would be a stronger gate such as **sustained wire contact for N consecutive steps** before entering recovery, but that would be a new controller change and would need separate justification.

## Recommended next experiment
Do **not** change the trigger yet. The single missing measurement is a phase-aligned contact-duration trace showing whether seed 11’s first wire contact is too brief or too late to help the calf cross before the fixed hybrid motion runs out of favorable body pose.

Evidence for that recommendation:
- the trigger is already wire-contact based;
- seed 11 does eventually contact the wire, but not at entry;
- the failure mode is still the missing near-to-far calf transition, not a missing contact signal.

## Files written
- [logs/d5_42/recovery_alignment.csv](logs/d5_42/recovery_alignment.csv)
- [logs/d5_42/report.md](logs/d5_42/report.md)
