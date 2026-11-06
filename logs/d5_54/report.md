# D5.54 — FL Recovery Geometry Before Changing the Maneuver

Diagnostic only. No controller, guard, hybrid command, checkpoint or wire model was changed;
no training or sweep. Written for the person who watched the D5.53 viewer runs.

## Data and validation

D5.53 artifacts had the wire-relative FL foot/calf signed distances but not the FR foot
position, true geom-to-wire distances, or which robot geom touched the wire each step. I added
[scripts/d5_54_fl_geometry_diag.py](../../scripts/d5_54_fl_geometry_diag.py), which re-runs the four
D5.53 conditions (baseline and `guard_enabled`, seeds 0 and 11, unmodified D5.53 setup) and logs
only steps 38–111 to [comparison_summary.csv](comparison_summary.csv) (296 rows, one per
seed/variant/step).

- **Validation (all pass):** the re-run reproduces D5.53 exactly (max diff 0.0 on base z, FL calf,
  FL/FR ground force, phase and guard state; 4 runs × 74 steps). Wire collisions on.
- **Distances:** `fl_foot_dx_wire` is the FSM's signed forward distance of the foot from the
  interpolated wire centerline (+ = far side); `fl_foot_dz_wire` is foot z minus wire centerline z
  (wire: z ≈ 0.10, radius 0.006, y 0.30–0.50, x ≈ 0). `d_*_wire` are `mj_geomDistance` values to
  the nearest wire capsule (sphere/capsule exact). **The FL calf2 geom is an ellipsoid; its
  distance sits at a constant 0.015 m whenever it touches, so for calf2 I use the per-step contact
  list `wire_touch`, not `d_calf2_wire`.**
- Only FL geoms and the trunk (`base1`) ever touch the wire. FR/RL/RR never do (FR foot y stays
  outside the wire span (y ≥ 0.30) until step ~91-103, and no FR contact was logged).

## Measured geometry (all four runs)

**At hybrid activation** (baseline: 44 / 51; guarded: 62 / 62):

| | seed 0 base | seed 0 guard | seed 11 base | seed 11 guard |
|---|---|---|---|---|
| FL foot dx / dz to wire (m) | +0.007 / −0.074 | −0.002 / −0.084 | +0.002 / −0.077 | −0.004 / −0.080 |
| FL calf2 dx / dz (m) | −0.029 / −0.016 | −0.025 / −0.021 | −0.027 / −0.016 | −0.028 / −0.017 |
| Trunk x (m, wire at 0) | −0.174 | −0.102 | −0.151 | −0.111 |
| FR foot world x (m) | −0.033 | +0.085 | +0.023 | +0.076 |

In every run the FL foot is on the ground directly beneath the wire (within 7 mm in x, 7–8 cm
below it) and the shank (calf2) is touching the wire's near face 2.5–2.9 cm behind its plane.
The wire is 10 cm up and the Go2 swing apex is 3.9 cm, so a normal step puts the foot under the
wire, not into it. The first wire contact (step 43 / 50) is the calf2 pressing the wire from
below-behind.

**FL after activation, guarded** (seed 11 numbers; seed 0 within 2 steps):

| step | event | foot dx / dz | calf2 | trunk–wire d |
|---|---|---|---|---|
| 62 | hybrid on, FL airborne since 60 | −0.004 / −0.080 | touching | 0.134 |
| 62–85 | retract; calf2 touches wire continuously (24 steps) while rising from dz −0.017 to +0.05 | foot drifts back to dx −0.042 by 75, rises to dz −0.016 | touching | 0.134 → 0.117 |
| 76 | **FL foot first touches wire** (seed 0: 78) | −0.041 / −0.016 | touching | 0.109 |
| 82–84 | trunk lands on wire; calf2, foot and trunk all touch it | −0.042 / +0.018 | touching | 0.012 |
| 89–91 | foot crosses **over** the wire (dx −0.014 → +0.061, dz +0.043 → +0.020) | | clear | |
| 94 | exit; later FL_calf1/thigh touch wire (95–97) | | | |

The foot never passes *under* the wire as a result of recovery: it starts under it, rises behind
it (4 cm back), and crosses above it at step 89–91. The retract is already backward: foot x in
base frame goes 0.130 → 0.007 m (12 cm) by step 75, but the trunk advances ~9 cm over the planted
foot meanwhile, so the net move behind the wire is only ~4 cm.

## Questions

**1. Does FL pass under or get trapped?** It starts under the wire (see above) and ends up
*pinned* in two ways: calf2 rubs the wire's near face for 24+ consecutive steps, and from step
76/78 the foot also touches the wire from behind, then the sinking trunk presses the wire onto
calf2 and foot (steps 82–84, wire peaks 1931 N seed 0 / 2401 N seed 11). So "lifts into or
underneath the wire" is supported as geometry. It does not pass beneath: dx only changes sign
above the wire (dz +0.02 to +0.04).

**2. Wrong side at activation, or moved there?** Already there. The foot is directly beneath the
wire with the calf against it in all four runs, including baseline. The guard doesn't change that
(dx within ±7 mm); it changes where the body is (7–9 cm farther over the wire).

**3. Does FR's displacement change FL's geometry or support?** Not materially. Planted FR foot x
is 1–2 cm *less* far forward than baseline during 56–72 (0.075–0.085 vs 0.092–0.105 m;
body-relative 0.18–0.20 vs 0.23–0.24 m), and FR never touches the wire. What looks like "FR
farther forward" is the body advancing over the wire (trunk x −0.11 → +0.06 vs stalled at −0.13
in baseline seed 11). The conspicuous FR motion is later: after support is lost the FR foot swings
to z 0.146 (step 80) and 0.22 m (step 92) vs a 0.04 m normal apex. That is a reaction, not a cause.
FR carries 80–95 N through steps 62–73, and that is the available support.

**4. Fall before or after FL becomes problematic?** Before. Guarded seed 11: all four feet unloaded
(<10 N) at step 74, vz −0.29 at 75; FL foot touches the wire at 76. Seed 0: support lost 76, FL
foot-wire 78. Calf2 has touched the wire since first contact, so it is not a new event. Trunk-wire
contact comes later still (82). Roll rate grows from step 68 (−0.4 → −1.2 rad/s by 75).

**5. Consistent with FL retracted at the next support transfer?** Yes, but it is not the whole
story. FR and RL (the only supporters since 56) unload at 73–76, the normal trot transfer. FL is
retracted by the hybrid (0 N since 60) and **RR is also not down**: it lifts at 57 as normal but
never lands (the unperturbed control lands it at 66–68; RR calf in guarded seed 0 is 0.16–0.18 rad
more flexed than the control, 0.35 → 0.70 → 2.07 raw policy-action difference over 54–61 /
62–67 / 68–73, vs the control, which is offset by ~2 steps in phase). So steps 62–73 are a two-leg
diagonal (FR–RL) stance; when it unloads, nothing takes over.

**6. Smallest next experiment:** see below.

## What is measured vs hypothesised

- Measured: FL starts beneath the wire; calf2 contact is continuous; collapse precedes FL-foot
  contact by 2 steps; FR is not displaced forward relative to baseline; RR does not land; foot
  crosses over the wire at 89–91.
- Hypothesis: FL's retract-while-planned-to-bear-load causes the collapse (D5.53 H2); RR's failure
  to land is a policy reaction to the perturbed FL state (D5.52 showed other-leg actions change;
  not isolated here). The control comparison is phase-offset by 2 steps.
- Not shown: that a curved or backward path is needed. The retract is already backward, and the
  foot needs ~20 steps (62→82) to reach wire height, about twice a normal swing (11–12 steps), so
  geometry limits any retract confined to one swing.

## Recommended single next experiment

Decision rule hit: multiple causes, so test the first failure in time. The first post-activation
failure is the unavailable support at the transfer, with FL retracted and RR not landing.

**Bounded retraction:** keep the D5.53 touchdown guard and the hybrid commands unchanged, but
end the hybrid for the current swing before the planned transfer and return FL to the walking
position path, then re-arm through the guard's touchdown/lift-off sequence. End point: FL liftoff +
the previous measured FL swing duration (11 steps here), a gait-derived value, not a swept one
(release ≈ step 71, ahead of the FR/RL unloading at 73–76).

Outcomes:
- No collapse at 73–76 and RR lands: unavailable FL at the transfer is the cause. Expected cost:
  foot only reaches dz ≈ −0.05 within the swing, so crossing needs several swing cycles; measure
  per-cycle progress.
- Collapse persists or RR still does not land with FL back on the walking path: the support loss has
  another source (RR/policy reaction, or the diagonal stance itself). Next: mask the FL entries in
  the policy observation.
- FL lands under the wire again each cycle without progress: report it, don't tune.

Measurements: the D5.53 logger plus `comparison_summary.csv` columns (FL foot dx/dz, geom–wire
distances, FR foot, contact list). Nothing new needs adding.

**Not verified before copying Quad-SDK behavior to Go2:** knee sign and whether "−tau_push"
folds or extends the knee on Spirit (D5.51 left this open), Spirit's 15 rad/s vs 2 rad/s here,
and the 0.10 m wire vs Spirit's obstacle height.

## Viewer commands (guarded behavior)

These are the D5.53 commands, unchanged. Each was run to completion in D5.53 (250 steps, exit 0)
and the viewer trace was byte-identical to the headless trace. I cannot see the window from here,
so I make no visual claim: this establishes the viewer stepped the same physics, not what it
looks like.

```sh
uv run python scripts/d5_53_touchdown_guard.py --variant guard_enabled --seed 11 --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant guard_enabled --seed 0  --debug --max-steps 250
uv run python scripts/d5_53_touchdown_guard.py --variant baseline      --seed 11 --debug --max-steps 250
```

Steps to watch, seed 11 guarded: FL touchdown 53, activation 62, FR/RL unload 73–74, FL foot–wire
76, trunk onto wire 82, crossing 90, exit 94. Seed 0: touchdown 46, activation 62, support lost 76,
foot–wire 78, trunk 82, crossing 91, exit 95. The D5.53 viewer traces are in
`logs/d5_53/viewer_*_seed*/trace.csv`.

## Files

Created: `scripts/d5_54_fl_geometry_diag.py`, `logs/d5_54/comparison_summary.csv`,
`logs/d5_54/report.md`. Nothing else modified.
