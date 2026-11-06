# D5.12 — Isolating the position-actuator-disabling step from the auxiliary commands

## Objective

D5.11 held the position-actuator disabling fixed and varied only the
auxiliary thigh-velocity/calf-torque commands' onset profile, finding no
effect on D5.10's base-height sag. D5.12 isolates the remaining factor:
does disabling the FL thigh/calf position actuators at recovery entry —
independent of whether any auxiliary command is applied — reproduce the
sag?

## Setup

New, isolated script: [scripts/d5_12_position_actuator_ablation.py](scripts/d5_12_position_actuator_ablation.py),
verified via `--help` before running. Reuses `build_d510_env` unmodified
(same checkpoint/wire geometry/sim settings as D5.10/D5.11).

**Condition A (baseline):** unmodified `_apply_hybrid_actuators` —
reference point, identical to D5.10/D5.11's Condition A.

**Condition C (zero_aux):** the real, unmodified `_apply_hybrid_actuators`
still runs first (identical position-actuator gain/bias zeroing,
auxiliary-actuator gain-enabling) — then a wrapper overwrites
`hybrid_thigh_vel_ctrl`/`hybrid_calf_torque_ctrl` to exactly zero for
RECOVERY steps and re-issues the zero targets. The FL position actuators
are disabled exactly as in Condition A; no auxiliary assistance is ever
commanded. Note the auxiliary thigh actuator is a velocity actuator with
damping enabled by the unmodified call, so "command zero" still permits
passive damping forces, not literal zero torque — this is reported, not
hidden.

Commands used:
```
uv run python scripts/d5_12_position_actuator_ablation.py --seed 0 --max-steps 150 --condition baseline --out-dir logs/d5_12_position_actuator_ablation/seed0_condA_baseline
uv run python scripts/d5_12_position_actuator_ablation.py --seed 0 --max-steps 150 --condition zero_aux --out-dir logs/d5_12_position_actuator_ablation/seed0_condC_zeroaux
```

## Results (seed 0, window steps 45-75)

| | entry `base_z` | min `base_z` | step of min | steps to min | `fl_ground_contact=False` count |
|---|---|---|---|---|---|
| A baseline | 0.3077 | 0.2040 | 54 | 9 | 31/31 (100%) |
| C zero_aux | 0.3091 | 0.2113 | 56 | 11 | 13/31 (42%) |

Condition C still shows a substantial sag (drop ≈0.098 m, only slightly
smaller than Condition A's ≈0.104 m) and reaches a similar minimum
height — but the FL foot regains ground contact intermittently starting
at step 44 (one step after entry) and for roughly the first third of the
window, versus Condition A where FL ground contact is lost for the
*entire* 31-step window with no regain. Condition C's recovery also
fails to latch Stage 1/2 and exits via the 48-step cap_timeout, whereas
baseline succeeds.

Step-by-step auxiliary thigh force confirms the suppression worked as
intended: Condition C's commanded thigh velocity/calf torque are exactly
0.000 throughout (vs. A's constant +2.0/−5.0), and the actual auxiliary
thigh force in C is small, damping-only noise (roughly −2.7 to +1.9 N·m,
consistent with passive `kd*(0-qvel)` damping) rather than A's
commanded, larger, consistently-signed forces.

## Interpretation

**The base-height sag is NOT eliminated by suppressing the auxiliary
commands alone** — it persists at nearly the same magnitude (0.098 m vs
0.104 m) and nearly the same timing (11 vs 9 steps to minimum) even with
zero auxiliary torque/velocity commanded. This means the sag is
substantially attributable to the **position-actuator disabling itself**
(losing the FL leg's normal stance-stabilizing gain/bias at recovery
entry) — consistent with D5.11's elimination of the auxiliary-command
onset profile as a cause, and now directly confirmed rather than only
inferred by elimination.

However, the picture is not "100% position-actuator-disabling, 0%
auxiliary commands" — Condition C does show a meaningfully smaller drop
and, critically, intermittent FL ground-contact recovery during the
dip that Condition A never achieves. This suggests the auxiliary
commands (particularly the constant −2.0 rad/s thigh-lift command) are
actively holding/lifting the leg away from the ground throughout
Condition A's window, compounding the position-actuator-disabling
effect and fully preventing any ground-contact recovery that would
otherwise occur. The two factors appear to be: (1) the position-actuator
disabling as the dominant, sufficient cause of the sag's existence and
most of its magnitude, and (2) the auxiliary thigh-lift command as an
amplifying factor that prevents transient ground-contact recovery and
fully sustains the loss of support for the whole window (at the cost of
being necessary for Stage-1/2 latch and genuine recovery in this seed).

## Recommendation

The dominant contributor to the initial base-height sag is the FL
position-actuator disabling at recovery entry, not the auxiliary
commands' magnitude or onset speed (confirmed directly here, after
D5.11 ruled out onset speed). Any future mitigation work should
therefore focus on how the leg is supported immediately after the
position actuators are disabled (e.g., a brief, standard PD-hold at the
pre-entry joint position before engaging the auxiliary commands) rather
than on the auxiliary commands' ramp profile, which D5.11 already showed
has negligible effect. This observation is directly relevant to, but
does not resolve, the still-open D5.7–D5.9 calf-waiting-state design
question — D5.9 already rejected naively restoring the original
position actuator (Option C) due to force saturation; D5.12 reinforces
that *some* form of stance support during the disabled window is
important, without re-opening or re-litigating that prior rejection.

Per the task's instructions, this comparison stops here — no further
controller change is proposed or made automatically.
