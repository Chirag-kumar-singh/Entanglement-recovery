# D5.28 — Gait-Phase Classification Audit

## Objective

Investigate whether the trigger-state differences found in D5.26 correlate
with specific gait-cycle phases, and quantify the phase-sensitivity of the
recovery FSM's crossing-latch mechanism.

## Step 1 — Inspection and phase classification approach

**Key finding from inspection**: All 20 seeds enter recovery during the
**swing phase** of the walking gait — the FL leg is airborne (not in ground
contact) and the foot is moving upward (positive foot_vz_base). No seeds
trigger during stance.

This is expected: the wire only engages the FL leg during swing (the leg's
forward motion through the air). During stance, the leg is on the ground and
the wire geometry does not intersect the calf.

**Classification strategy**: Since all seeds are in swing, classify them into
swing *sub-phases* based on:
- **Foot height** (base-relative): how high is the foot in its swing arc
- **Foot vertical velocity**: is it rising fast, slow, or near a peak?
- **Leg joint velocities**: how fast are the thigh and calf joints moving?

These metrics distinguish:
1. **Early swing / push-off**: foot is still low, joints moving upward fast
2. **Mid swing, rising**: foot is high, joints moving upward
3. **Peak swing**: foot is near its highest point, joints near zero velocity
4. **Late swing, descending**: foot is descending, preparing to land

## Step 2 — Gait-phase classification results

All 20 seeds classified at their recovery trigger step:

### Stage 2 reaching (n=10, seeds 0,1,2,3,4,8,9,12,18,19)

**Swing subphase distribution:**
- Early swing (push-off): 6/10 (60%)
- Mid swing (rising): 4/10 (40%)
- Peak swing: 0/10
- Late swing: 0/10

**Leg dynamics:**
- Leg speed (|thigh_vel| + |calf_vel|): mean=4.562, range=[3.112, 6.652]
- Foot vertical velocity: mean=0.1404 m/s, range=[0.0514, 0.2872]
- Foot height: all in range [−0.279, −0.271] m (low swing)

### Stage 1 only (n=2, seeds 5,6)

**Swing subphase distribution:**
- Early swing / push-off: 0/2
- Mid swing (rising): 1/2 (50%)
- Peak swing: 1/2 (50%)
- Late swing: 0/2

**Leg dynamics:**
- Leg speed: mean=3.811, range=[3.255, 4.368] — **lowest of all groups**
- Foot vertical velocity: mean=0.1163 m/s, range=[0.0443, 0.1883]
- Foot height: all in range [−0.278, −0.271] m

**Note:** Stage 1 only group is very small (n=2). Seed 5 is at peak swing
(near-zero foot velocity), seed 6 is in mid-swing rising. Both have notably
slower overall leg speed.

### Neither stage reaching (n=8, seeds 7,10,11,13,14,15,16,17)

**Swing subphase distribution:**
- Early swing (push-off): 4/8 (50%)
- Mid swing (rising): 4/8 (50%)
- Peak swing: 0/8
- Late swing: 0/8

**Leg dynamics:**
- Leg speed: mean=5.801, range=[4.138, 7.866] — **significantly higher than both other groups**
- Foot vertical velocity: mean=0.1370 m/s, range=[0.0925, 0.1870]
- Foot height: all in range [−0.280, −0.269] m

**Critical subgroup within neither-stage:** Seeds 11 and 13 have exceptionally
high leg speed (7.866 and 7.755, respectively), more than 60% faster than
the mean for stage-2-reaching seeds. Both are in mid-swing-rising phase with
high foot upward velocity (0.187 and 0.157 m/s).

## Step 3 — Comparison and interpretation

### What the gait-phase data reveals:

**Leg speed is the strongest phase-based differentiator** between
stage-2-reaching and neither-stage-reaching seeds:

| group | mean leg speed | range | ratio to stage2 |
|---|---|---|---|
| stage2 | 4.562 | 3.112 − 6.652 | 1.0× (reference) |
| stage1 | 3.811 | 3.255 − 4.368 | 0.84× (slowest) |
| neither | 5.801 | 4.138 − 7.866 | 1.27× **(27% faster)** |

This **directly echoes D5.26's finding**: neither-stage-reaching seeds have
faster downward calf velocity, faster base angular velocity, and lower base
height — i.e., they enter recovery during a **high-energy, dynamically
active** phase of swing. The FL leg is moving faster (larger joint velocities),
which means it is in a more energetic part of the swing arc.

### Why fast leg speed predicts crossing-stage failure:

The recovery FSM's crossing-latch thresholds (stage 1: calf near→far transition
**while lifted above wire** by ≥0.02 m; stage 2: foot forward of far-side by
≥0.03 m) are **kinematic snapshots** dependent on the exact leg trajectory and
velocity at contact.

**Hypothesis** (not proven): When the FL leg enters the wire at high speed
(leg_speed ≥ 6.0), the leg is already in a high-momentum swing trajectory
such that:

1. The calf passes through or below the "cleared and lifted" zone too quickly,
   failing to satisfy the stage-1 "calf lifted and far-side" check.
2. OR the calf lands early in recovery (before stage 1 can latch) because the
   high downward velocity (confirmed in D5.26 as faster downward calf_vel)
   overcomes the recovery controller's thigh-lifting commands, collapsing the
   leg into contact with or below the wire.
3. OR the foot never accumulates enough forward-distance to reach the stage-2
   threshold (foot_crossing_margin = 0.03 m) within the 48-step cap, because
   the leg is too busy recovering from the high-speed entry to perform the
   required forward crossing.

### Foot height at trigger is NOT discriminating:

Contrary to a naive expectation, neither-stage seeds do NOT enter recovery
at a notably different foot height than stage-2-reaching seeds:

- Neither-stage mean: −0.2748 m
- Stage-2 mean: −0.2762 m
- Difference: 0.0014 m (less than 2 mm)

This suggests that **timing and dynamics (leg speed) matter more than absolute
position** for crossing-stage success. Two legs at the same geometric position
but with different velocities will follow very different trajectories through
the recovery window.

### Swing subphase distribution:

Both stage-2-reaching and neither-stage-reaching seeds cluster in early-swing
and mid-swing phases — no clean separation by subphase label alone. However,
the critical difference is **leg_speed within each subphase**. Seeds 11 and 13
(neither-stage, very high leg_speed) are in mid-swing-rising, but with
exceptionally fast joint velocities compared to, e.g., seed 8 (stage-2, also
mid-swing-rising, but with leg_speed = 4.878 vs. 7.866 for seed 11).

## Full 20-seed gait-phase table

| seed | outcome | group | swing phase | leg speed | foot_vz | foot_z | stage reached |
|---|---|---|---|---|---|---|---|
| 0 | genuine | stage2 | mid_swing_rising | 4.774 | 0.228 | −0.275 | yes |
| 1 | genuine | stage2 | early_swing_push_off | 3.943 | 0.128 | −0.278 | yes |
| 2 | genuine | stage2 | mid_swing_rising | 4.269 | 0.172 | −0.278 | yes |
| 3 | genuine | stage2 | early_swing_push_off | 6.652 | 0.076 | −0.271 | yes |
| 4 | genuine | stage2 | early_swing_push_off | 3.758 | 0.093 | −0.278 | yes |
| 8 | genuine | stage2 | mid_swing_rising | 4.878 | 0.287 | −0.274 | yes |
| 9 | timeout | stage2 | early_swing_push_off | 4.367 | 0.146 | −0.278 | yes |
| 12 | genuine | stage2 | early_swing_push_off | 3.880 | 0.051 | −0.279 | yes |
| 18 | genuine | stage2 | mid_swing_rising | 5.983 | 0.166 | −0.275 | yes |
| 19 | genuine | stage2 | early_swing_push_off | 3.112 | 0.056 | −0.276 | yes |
| 5 | timeout | stage1 | peak_swing | 3.255 | 0.044 | −0.271 | **no** |
| 6 | timeout | stage1 | mid_swing_rising | 4.368 | 0.188 | −0.278 | **no** |
| 7 | timeout | neither | early_swing_push_off | 4.317 | 0.122 | −0.279 | **no** |
| 10 | timeout | neither | mid_swing_rising | 4.442 | 0.152 | −0.279 | **no** |
| 11 | timeout | neither | mid_swing_rising | **7.866** | 0.187 | −0.269 | **no** |
| 13 | timeout | neither | mid_swing_rising | **7.755** | 0.157 | −0.269 | **no** |
| 14 | timeout | neither | early_swing_push_off | 4.138 | 0.093 | −0.280 | **no** |
| 15 | timeout | neither | early_swing_push_off | 6.436 | 0.130 | −0.272 | **no** |
| 16 | timeout | neither | early_swing_push_off | 6.359 | 0.095 | −0.273 | **no** |
| 17 | timeout | neither | mid_swing_rising | 5.094 | 0.160 | −0.278 | **no** |

## Observation vs. inference

- **Observed:** all gait-phase classifications and leg-speed metrics in the
  table above, computed directly from trajectory CSV joint-velocity and
  foot-velocity columns at each seed's trigger step.
- **Inference (not proven):** that high leg speed "causes" failure to reach
  crossing stages. The correlation is clear (neither-stage mean leg_speed 27%
  faster than stage-2 mean), but the causal mechanism is unknown. Alternatives:
  - Fast leg speed is a marker of a specific gait phase (e.g., late push-off)
    that happens to be incompatible with crossing-stage latching, and the
    causative factor is the phase itself, not the speed.
  - Fast leg speed and the pre-recovery entry conditions (e.g., base height,
    body rotation) are both consequences of some unobserved upstream cause
    (e.g., stochastic variations in the frozen walking policy's trajectory).

## Limitations

- The neither-stage cohort's high leg speed is driven largely by two extreme
  seeds (11, 13 with leg_speed > 7.7). Removing these outliers, the neither-
  stage mean drops to ~5.3, still higher than stage-2 but the gap narrows. The
  question of whether these are true outliers or representative remains open.
- Gait-phase classification uses heuristic thresholds (e.g., leg speed =
  |thigh_vel| + |calf_vel|, foot_z > −0.15 for "high"). No validation that
  these thresholds match the actual biomechanical or control phases of Go2's
  gait. A more rigorous phase detector (e.g., signal processing or learned
  classifiers) would be more reliable.
- The 20-seed sample remains small. The observed 5.8 vs. 4.6 leg-speed ratio
  may not generalize to a larger draw.
- No causal mechanism was tested — it remains unproven whether modifying the
  FSM's crossing-latch thresholds to account for entry leg speed would improve
  recovery success rate.

## Recommended next experiment (not executed)

**Phase-aware FSM modification**: Create two variants of the recovery FSM:

1. **Baseline** (existing, unchanged): crossing-latch thresholds fixed
   (clearance_margin=0.02 m, foot_crossing_margin=0.03 m).
2. **Phase-aware** (experimental): crossing-latch thresholds dynamically
   adjusted based on detected entry-phase leg speed. For example:
   - If entry leg_speed < 4.5: clearance_margin = 0.020 m, foot_crossing_margin = 0.030 m
   - If entry leg_speed ≥ 4.5 & < 6.5: clearance_margin = 0.025 m, foot_crossing_margin = 0.035 m
   - If entry leg_speed ≥ 6.5: clearance_margin = 0.030 m, foot_crossing_margin = 0.040 m

Run a cross-seed comparison (seeds 0–19, plus additional seeds 20–29) on both
variants at the default cap=48 configuration, and measure:
- Success rate (genuine completion) per variant.
- Whether the phase-aware variant improves success for the high-leg-speed
  seeds (11, 13, etc.) without degrading performance on low-speed seeds.

If phase-aware thresholds improve the overall recovery success rate, the
finding would support the hypothesis that leg-speed-dependent crossing-latch
dynamics are the root cause of the neither-stage-reaching failure mode.

This is **not implemented** per the stop condition. Do not modify the FSM or
train a policy; merely conduct the observational audit and recommend the
experiment.

## Stop-condition compliance

No controller changes, FSM modifications, caps, gains, frozen-policy updates,
or training were performed. No follow-up experiment was automatically launched.
The one recommended next experiment (phase-aware FSM variant comparison) is a
recommendation only.
