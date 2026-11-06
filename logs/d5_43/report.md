# D5.43 — Contact Duration and Gait-Phase Diagnostic

## Signal used for recovery entry
The recovery-entry gate in [src/mjlab/tasks/recovery/mdp/actions.py](src/mjlab/tasks/recovery/mdp/actions.py) is:
`entering_recovery = is_normal & in_contact`, where `in_contact` comes from the `wire_contact` sensor's `found > 0` signal.

The `wire_contact` sensor itself is defined in [src/mjlab/tasks/recovery/env_cfg_residual.py](src/mjlab/tasks/recovery/env_cfg_residual.py) as a `ContactSensorCfg` with the wire geoms as the primary contact set and the FL calf collision geom as the secondary set.

## What the existing telemetry can and cannot tell us
- The D5.35 CSVs already contain wire contact, recovery phase, elapsed recovery steps, base velocity, FL joint position/velocity, and the wire-relative calf/toe geometry.
- No existing CSV in the D5.35–D5.42 logs includes a wire-contact force column, so contact force is not available for this diagnostic.
- The logs do allow a consistent comparison of entry state and gait state between seeds because the same fields are recorded on both runs.

## Key aligned observations
- Seed 0 first wire contact: step 43; first recovery row: step 44.
- Seed 11 first wire contact: step 50; first recovery row: step 51.
- Seed 0 has an initial contact block of 4 steps (43–46), which spans the recovery entry and stays continuous for several steps.
- Seed 11 has only a 1-step initial contact blip (50–50), followed by a gap through its recovery-entry row, then later contact blocks 53–55 and beyond.
- At the recovery-entry row, seed 0 is already in contact, while seed 11 is not. That is the first meaningful difference in the contact history around entry.
- Entry dynamics also differ materially: seed 11 enters recovery with a much more negative base vertical velocity than seed 0 (-0.1464 vs -0.0302 m/s), while base forward velocity is lower as well (+0.2444 vs +0.3224 m/s).
- The calf forward distance is negative at entry for both seeds, but only seed 0 later turns positive (84); seed 11 never does in the available window.

## First divergence before the calf trajectories separate
The first meaningful difference is not a threshold event but the shape of the contact window around recovery entry:
1. seed 0 preserves contact through entry and into the next few steps;
2. seed 11 loses contact at the recovery row itself and only reacquires it later;
3. seed 11's entry state is more downward and less favorable for advancing the calf across the wire.

That makes contact duration and entry dynamics both relevant, but the logs support the entry-dynamics explanation more strongly than a pure phase-only story.

## Answer to the diagnostic question
- **Contact-duration hypothesis:** partially supported. Seed 0's contact is sustained across entry; seed 11's first contact is brief and interrupted.
- **Gait-phase hypothesis:** only weakly supported. The phase label changes when recovery starts, but the logs do not show a phase-only failure; they show different contact persistence and different entry kinematics.
- **Entry-dynamics hypothesis:** best supported. Seed 11 enters recovery with a more negative vertical velocity and a less favorable early recovery state, and the calf never reaches positive signed forward distance.

## Recommended next step
Do **not** change the trigger yet. The evidence does not justify a trigger change, because the current trigger already uses the real wire-contact sensor and the main difference is the contact-duration/entry-state sequence around recovery onset.

The most justified next experiment is a logging-only trace of **contact persistence at and just after recovery entry** (for example, explicitly logging the sustained-contact duration and any contact gaps across the entry boundary). That is the single missing measurement needed to decide whether a later timing change would be meaningful.

## Files written
- [logs/d5_43/contact_duration_window.csv](logs/d5_43/contact_duration_window.csv)
- [logs/d5_43/report.md](logs/d5_43/report.md)
