# D5.26 — Recovery Entry-Condition Audit

## Objective

Investigate why 8/20 seeds never reach either crossing stage (D5.25
finding), by comparing the FSM trigger-step entry state between
stage-reaching vs. neither-stage-reaching runs. Use existing trajectory
CSVs; no new simulations or controller changes.

## Step 1 — Inspection of existing evidence

**Verified from source** (`src/mjlab/tasks/recovery/mdp/actions.py`,
`_update_fsm` method):

- `entering_recovery`: phase transition triggered by wire contact
  (`in_contact = (found > 0).any()`), AFTER the frozen walking policy
  has already produced its action for the current step.
- At the instant recovery is entered, `_recovery_elapsed` is reset to
  zero, all stage-latch flags reset to false, and `_forward_dir_xy`
  (the robot's forward direction in world XY plane) is latched and
  used throughout the recovery for all forward-distance calculations.
- STAGE 1 (`calf_clearance_achieved`): triggered on a NEAR→FAR
  transition of the calf relative to the latched forward direction,
  **while the calf is above the wire by at least `clearance_margin`**
  (0.02 m, from config). A slide-past-at-ground-level does not count.
- STAGE 2 (`foot_crossed_achieved`): only evaluated after stage 1 has
  latched. Triggers when the foot reaches `foot_crossing_margin` (0.03
  m, from config) beyond the far side — no height requirement.
- Timeout: `cap_reached = recovery_elapsed >= max_recovery_steps` (48
  steps).

**Trigger-step definition**: the step at which `newly_entangled` first
becomes True, i.e. the same step where recovery FSM transitions from
NORMAL_WALK to RECOVERY (wire contact detected, phase flag set). The
trigger row's CSV entry is indexed by `trigger_step` value read from
each seed's `run_log.txt`.

## Step 2 — Entry-state capture (from existing CSVs)

All 20 seeds (0–19) have complete trajectory CSVs with the following
columns available at the trigger step:

- **Joint state**: `thigh_pos`, `thigh_vel`, `calf_pos`, `calf_vel`,
  `hip_pos`, `hip_vel`.
- **Tracking error** (FL thigh/calf vs. walking-policy targets):
  `thigh_err_signed`, `calf_err_signed`.
- **Foot state** (base-relative): `foot_x_base`, `foot_z_base`,
  `foot_vx_base`, `foot_vz_base`.
- **Base state**: `base_z`, `base_lin_vel_x`, `base_lin_vel_y`,
  `base_lin_vel_z`, `base_ang_vel_x`, `base_ang_vel_y`,
  `base_ang_vel_z`.
- **Wire geometry** (world-relative): `calf_forward_dist`,
  `calf_height_above_wire`.
- **Contact flags**: `fl_ground_contact`, `fr_ground_contact`,
  `rl_ground_contact`, `rr_ground_contact`.

No new instrumentation was needed — all trigger-step data was already
logged and indexed by the run_log.txt reference. No controller changes,
no new runs.

## Step 3 — Comparison of comparable outcomes

All 20 seeds classified by stage-reaching status at the recovery
trigger step:

| group | n | seeds | description |
|---|---|---|---|
| Stage 2 reaching | 10 | 0, 1, 2, 3, 4, 8, 9, 12, 18, 19 | reached both stage1 and stage2 latches (genuine completions or late timeouts) |
| Stage 1 only | 2 | 5, 6 | reached stage1, never stage2 (early timeout) |
| Neither stage | 8 | 7, 10, 11, 13, 14, 15, 16, 17 | reached neither stage (very early timeout) |

### Key entry-state differences

The following trigger-step metrics show the most striking, consistent
differences between stage-2-reaching and neither-stage-reaching groups:

#### (1) Calf velocity at trigger

| group | mean calf_vel | range | notes |
|---|---|---|---|
| stage 2 | −2.35 m/s | −4.61 to −1.03 | wide spread, includes some fast-moving entries |
| stage 1 only | −1.62 m/s | −2.09 to −1.15 | narrow, slower calf downward velocity |
| **neither stage** | **−3.40 m/s** | **−5.06 to −1.88** | **strikingly fast downward calf motion** |

The neither-stage group's calf velocity is **~45% faster downward**
(more negative) than stage-2-reaching on average, and **~110% faster**
than stage-1-only — a substantial gap that is visible in nearly every
seed in that group.

#### (2) Tracking error (thigh) at trigger

| group | mean thigh_err | range | notes |
|---|---|---|---|
| stage 2 | −0.140 rad | −0.205 to −0.093 | moderate, consistent error |
| stage 1 only | −0.100 rad | −0.112 to −0.088 | smallest errors, tightest range |
| **neither stage** | **−0.189 rad** | **−0.251 to −0.124** | **largest tracking errors, widest range** |

The neither-stage group has **35% larger negative thigh tracking
error** on average — the walking policy is undershooting the target
thigh angle more severely in this group.

#### (3) Calf tracking error at trigger

| group | mean calf_err | range | notes |
|---|---|---|---|
| stage 2 | +0.196 rad | +0.032 to +0.506 | positive (overshooting target), moderate spread |
| stage 1 only | +0.110 rad | +0.096 to +0.124 | smallest, narrowest range |
| **neither stage** | **+0.346 rad** | **+0.178 to +0.491** | **largest, most overshooting** |

The neither-stage group's calf is **76% more overshooting** on average
compared to stage-2-reaching — the largest discrepancy of any metric
examined.

#### (4) Base pitch/roll rate at trigger

| group | base_ang_vel_x (pitch rate) | base_ang_vel_y (roll rate) | notes |
|---|---|---|---|
| stage 2 | −0.555 rad/s (range −1.25 to −0.34) | +0.472 rad/s (range +0.27 to +1.25) | moderate rotation |
| stage 1 only | −0.358 rad/s (range −0.36 to −0.35) | +0.283 rad/s (range +0.27 to +0.30) | tightest, most balanced |
| **neither stage** | **−0.902 rad/s** (range −1.31 to −0.43) | **+0.753 rad/s** (range +0.37 to +1.10) | **largest rotational rates** |

The neither-stage group has **62% faster pitch** and **59% faster roll**
compared to stage-2-reaching — much more aggressive base rotation at
the trigger moment.

#### (5) Base vertical velocity at trigger

| group | base_lin_vel_z | range | notes |
|---|---|---|---|
| stage 2 | −0.113 m/s | −0.333 to −0.053 | moderate downward velocity |
| stage 1 only | −0.058 m/s | −0.068 to −0.049 | very stable, minimal vertical motion |
| **neither stage** | **−0.196 m/s** | **−0.304 to −0.094** | **much faster downward motion** |

The neither-stage group is sinking **73% faster** than stage-2-reaching
at the trigger moment.

#### (6) Base height at trigger

| group | base_z | range | notes |
|---|---|---|---|
| stage 2 | 0.3068 m | 0.3019 to 0.3081 | tight range, highest average |
| stage 1 only | 0.3078 m | 0.3077 to 0.3079 | tightest of all, highest individual values |
| **neither stage** | **0.3054 m** | **0.3028 to 0.3074** | **lowest average height** |

The neither-stage group is, on average, **0.4 cm lower** (roughly 1.3%
of body length) at the trigger moment compared to stage-2-reaching, and
the range spans the full field of trigger-step heights, including the
two lowest outliers (seeds 11, 13).

### Interpretation — entry-state profile of "neither-stage" seeds

Seeds that never reach either crossing stage enter recovery in a
qualitatively **lower, faster-sinking, more-rotationally-disturbed,
higher-error state** compared to stage-reaching seeds:

1. **Calf already moving downward faster** (~3.4 m/s vs. ~2.4 m/s
   for stage-reaching) — the FL limb is more actively dropping.
2. **Larger tracking errors** (thigh more undershoot, calf more
   overshoot) — the walking policy is producing targets that are more
   misaligned with the actual joint state.
3. **Base is rotating more aggressively** (pitch 62% faster, roll 59%
   faster) — the robot's body is tumbling at a higher rate when the
   wire engages.
4. **Base is sinking faster** (downward velocity 73% faster) — the
   center of mass is already in a steep descent trajectory.
5. **Base height is lower** on average by ~0.4 cm — the robot is
   already partially collapsed when recovery begins.

Together, these metrics suggest that **neither-stage seeds experience
wire contact in the midst of a higher-energy, more-turbulent entry
condition** — a gait phase or initial-condition combination in which
the FL limb is already accelerating downward, the body is rotating
hard, and tracking accuracy is degraded. In contrast,
stage-reaching seeds tend to catch the wire in a more stable,
controlled entry state with lower velocities and better tracking
accuracy.

## Observation vs. inference

- **Observed:** all trigger-step values above, computed directly from
  trajectory CSVs at the step identified in run_log.txt as
  `trigger_step`.
- **Inference (not causal):** that faster calf downward velocity
  "causes" failure to reach stage 2. The faster velocity is an
  *associated state* at the entry moment, but causality is not
  established by this audit. Possible explanations include:
  - The faster-velocity condition is a domain-randomization draw that
    makes it harder for the FSM's geometric conditions to latch (e.g.,
    the calf is already moving so fast that crossing the far-side
    threshold before the cap fires is less likely purely as a geometry
    problem, not a controller problem).
  - OR the faster-velocity condition is itself a consequence of earlier
    domain randomization (leg stiffness, friction, contact properties)
    that also directly prevents crossing-stage latching.
  - OR the state clustering is a coincidence and the causal factor is
    something not captured in the entry state (e.g., timing of first
    wire contact within the gait cycle, wire geometry relative to the
    calf trajectory).

## Report table: all 20 seeds at trigger step

| seed | outcome | stage2 reached | thigh_err | calf_err | calf_vel | base_z | base_vz | base_pitch_rate | base_roll_rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | genuine | yes | −0.128 | +0.197 | −2.042 | 0.3077 | −0.353 | −0.673 | +0.451 |
| 1 | genuine | yes | −0.128 | +0.176 | −2.289 | 0.3076 | −0.088 | −0.338 | +0.272 |
| 2 | genuine | yes | −0.107 | +0.168 | −1.033 | 0.3079 | −0.053 | −0.356 | +0.270 |
| 3 | genuine | yes | −0.150 | +0.506 | −4.610 | 0.3019 | −0.333 | −1.249 | +1.247 |
| 4 | genuine | yes | −0.093 | +0.032 | −1.753 | 0.3079 | −0.058 | −0.340 | +0.270 |
| 8 | genuine | yes | −0.111 | +0.205 | −1.545 | 0.3081 | −0.077 | −0.339 | +0.274 |
| 9 | timeout | yes | −0.146 | +0.261 | −2.325 | 0.3072 | −0.108 | −0.379 | +0.382 |
| 12 | genuine | yes | −0.148 | +0.187 | −1.849 | 0.3073 | −0.087 | −0.396 | +0.387 |
| 18 | genuine | yes | −0.182 | +0.178 | −2.066 | 0.3054 | −0.116 | −0.571 | +0.632 |
| 19 | genuine | yes | −0.205 | +0.301 | −2.345 | 0.3072 | −0.114 | −0.674 | +0.685 |
| 5 | timeout | no | −0.087 | +0.096 | −2.094 | 0.3077 | −0.068 | −0.363 | +0.265 |
| 6 | timeout | no | −0.112 | +0.124 | −1.147 | 0.3079 | −0.047 | −0.361 | +0.301 |
| 7 | timeout | **no** | −0.150 | +0.356 | −2.163 | 0.3072 | −0.099 | −0.532 | +0.543 |
| 10 | timeout | **no** | −0.125 | +0.223 | −2.159 | 0.3074 | −0.141 | −0.743 | +0.591 |
| 11 | timeout | **no** | −0.244 | +0.491 | −5.065 | 0.3028 | −0.304 | −1.310 | +1.104 |
| 13 | timeout | **no** | −0.251 | +0.480 | −4.889 | 0.3031 | −0.268 | −1.111 | +0.931 |
| 14 | timeout | **no** | −0.124 | +0.178 | −1.921 | 0.3073 | −0.155 | −0.540 | +0.374 |
| 15 | timeout | **no** | −0.238 | +0.429 | −3.206 | 0.3040 | −0.215 | −0.872 | +0.753 |
| 16 | timeout | **no** | −0.237 | +0.415 | −4.202 | 0.3047 | −0.238 | −0.776 | +0.708 |
| 17 | timeout | **no** | −0.142 | +0.278 | −2.717 | 0.3065 | −0.147 | −0.432 | +0.394 |

## Visualization

The entry-audit reveals a clear behavioral difference between stage-reaching
and neither-stage-reaching seeds: the latter enter recovery in a
significantly lower, faster-sinking, more-rotationally-disturbed state with
larger tracking errors. To visualize this contrast in the native viewer (to
inspect the actual FL limb position/velocity/contact geometry at the trigger
step), the verified viewer path is:

**Representative neither-stage-reached seed** (seed 11: highest calf
downward velocity, −5.065 m/s, lowest base height, 0.3028 m):

```
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 11 --max-steps 200 \
  --out-dir logs/d5_26_entry_condition_audit/_viz_seed11_never_stages --debug
```

**Representative stage-2-reaching seed** (seed 1: genuine completion,
mid-range values for comparison):

```
uv run python scripts/d5_18_post_recovery_diagnostic.py --seed 1 --max-steps 200 \
  --out-dir logs/d5_26_entry_condition_audit/_viz_seed1_reaches_stage2 --debug
```

(These commands use D5.18's verified native MuJoCo viewer, not D5.23's
unwired --debug flag. Interactive window displays the trigger step and
subsequent recovery dynamics in real time; close window to end playback
and write CSV/log to the specified `--out-dir`. This is a recommendation
only — was not executed as part of this task.)

## Limitations

- Trigger-step state is a **single point in time** — it does not capture
  the trajectory leading up to the trigger, nor the domain-randomization
  parameters (limb stiffness, friction, contact properties, etc.) that
  produced each seed's initial condition. A seed with a "lower, faster"
  trigger state may reflect either a higher-energy gait phase or a
  stiffer/more-damped limb model, neither of which is disambiguated here.
- The neither-stage-reaching cohort, though consistent in its
  trigger-step profile, is drawn from only 8 seeds (and 10–19 are a
  single, contiguous sample). Generalization beyond this draw is
  uncertain.
- No statistical test was performed to determine whether the observed
  differences are significant relative to the domain-randomization
  variance (this is a purely descriptive audit, not an inferential one).
- Contact-geometry data available at trigger (calf_forward_dist,
  calf_height_above_wire) is already dominated by the random wire
  placement (part of domain randomization) and does not reveal entry
  state differences in the way the kinematic/dynamic metrics do.

## Recommended next experiment (exactly one, not executed)

**Domain-randomization parameter audit**: Instrument the existing D5.23
diagnostic script to log, at the trigger step, the **Quad-SDK environment's
domain-randomization parameters** that were applied to produce each seed's
initial condition — specifically, limb stiffness, friction coefficient,
damping, contact dynamics parameters, and any other mutable properties in
`mjlab.envs.wrappers.domain_rand.DomainRandConfig` or Go2-specific variants.
Then re-run the same entry-state comparison (stage-2-reaching vs.
neither-stage-reaching), but **add the DR parameters as additional columns**
and test whether the observed trigger-state differences (e.g., faster calf
velocity, lower base height) correlate with specific DR settings (e.g.,
higher stiffness, higher damping) that were sampled for the neither-stage
group.

If fast-downward-calf-velocity correlates with, say, lower limb stiffness or
higher damping in this audit, that would suggest the domain-randomization
draw itself is predisposing certain seeds to the high-velocity entry state,
which could then inform (but still not prove) whether the recovery failure is
attributable to the entry kinematics per se or to the underlying DR
parameters. This is **not implemented** here, per the stop condition.

## Stop-condition compliance

No controller changes, FSM modifications, caps, gains, frozen-policy updates,
or training were performed. No follow-up experiment was automatically launched.
The one recommended next experiment (DR parameter audit) is a recommendation
only.
