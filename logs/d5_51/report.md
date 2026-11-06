# D5.51 — Consolidated Recovery Instability Root-Cause Synthesis

Read-only synthesis. No simulation was launched. No file in `src/`, no actuator
parameter, threshold, FSM logic, collision setting, checkpoint or walking policy was
touched. No viewer or video was created. All numbers below come from existing CSVs
and were re-read, not re-measured, with small ad-hoc column extractions only.

Decision table: [experiment_decision_table.csv](experiment_decision_table.csv)

## 0. Headline

1. **D5.49's bottom line ("wire-contact mechanics remain the stronger explanation than
   the hybrid handoff") is not supported by its own traces.** Its "first excursion"
   metric used a pre-entry reference computed from the short restored window
   (`pre_entry_abs_pitch_rate_max` = 0.018 rad/s vs 0.366 in baseline;
   `comparison_summary.csv`), so the counterfactual's "excursion at step 44" is a
   threshold artifact. The raw traces show something different (section B2).
2. **Handoff-aligned traces of baseline and delayed-handoff runs show the same runaway
   signature, locked to the hybrid takeover rather than to wire contact or absolute
   step.** Contact alone, with the walking position actuators held, did not produce it
   over 7 steps.
3. **What is not yet known** is whether the hybrid commands cause the runaway on their
   own, or only in combination with wire contact (or via the frozen policy's reaction).
   The one missing cell of the {contact} x {hybrid} design is the recommended experiment.

## A. Consolidated evidence (existing artifacts only)

### Verified facts

| # | Fact | Source |
|---|---|---|
| F1 | Sim `timestep=0.0025`, `decimation=8` → one logged step = 20 ms = 8 physics substeps. All sensors/logs are sampled once per policy step. | `env_cfg_residual.py:694,699` |
| F2 | Entry: `entering_recovery = is_normal & in_contact` from the `wire_contact` sensor (netforce, `found>0`), evaluated in `process_actions` *before* the actions of that step. Contact first seen after step 43 → hybrid active during the physics of step 44. | `actions.py:1434`, `actions.py:1769-1798` |
| F3 | At entry, only FL thigh and calf change: thigh/calf position actuator gain+bias zeroed; aux thigh velocity actuator (kd=2.0) enabled with ctrl=+2.0 rad/s; aux calf motor ctrl=−5.0 N·m (open loop, no damping). FL hip, and FR/RL/RR all joints, stay on the frozen walking policy. | `actions.py:1123-1292`; D5.47 report |
| F4 | The frozen policy still runs every step and its observation includes the **true** joint pos/vel of all 12 joints, incl. the hybrid-driven FL thigh/calf. Its FL thigh/calf outputs are discarded; its other 10 outputs are applied. | `actions.py:1725-1768, 1769-1830` |
| F5 | No actuator saturation: aux thigh force ≤2.4 N·m (limit 23.5), calf motor 5.0 N·m (limit 45). Commands identical across seeds. | D5.47 report |
| F6 | First contact: `wire_seg_2_geom <-> FL_calf2_collision`, step 43; raw force 130.5 N (D5.50) vs sensor 10.9 N same step. FL hip–wire contact first at step 51, base–wire at 52. | D5.50, D5.48 audit |
| F7 | Seed 0 first wire-contact: FL airborne (swing) before and through the sag; FR, RL on ground pre-entry (trot). | `d5_16/headless_seed0/trajectory.csv` |

### Not verified / inconsistent
- U-a. Sensor force 10.9 N vs raw pair force 130.5 N at step 43: sensor is a once-per-step sample, raw value is presumably an instantaneous substep peak. Not reconciled.
- U-b. Sign convention of `wire_contact_force_*` (on wire vs on robot) is not documented in any report.

## B. Mechanical trace of the initial instability (seed 0)

### B1. Entry-window timeline (rows = state after that policy step)

| step | event | pitch rate y (rad/s, + = nose-down) | base_z (m) | base vz (m/s) | FL thigh vel / calf vel (rad/s) | FL calf pos (rad) | feet on ground | wire F (N) |
|---|---|---|---|---|---|---|---|---|
| 43 | first calf–wire contact; still NORMAL | 0.018 | 0.309 | +0.005 | 0.26 / 0.44 | −1.731 | FR, RL | 10.9 |
| 44 | hybrid active | 0.188 | 0.309 | −0.030 | 1.94 / −1.57 | −1.747 | FR, RL | 23.4 |
| 45 | | 0.374 | 0.308 | −0.090 | 2.23 / −2.55 | −1.791 | FR, RL | 28.1 |
| 46 | | 0.703 | 0.305 | −0.183 | 2.21 / −3.52 | −1.854 | FR, RL, RR | 30.1 |
| 47 | **all support feet lose ground contact** (FR, RL) | 1.252 | 0.299 | −0.346 | 2.19 / −4.67 | −1.936 | RR only | 0 |
| 48–50 | flight, pitch-down, free fall | 1.61 → 1.76 | 0.289 → 0.257 | −0.55 → −0.96 | 2.4–2.7 / −6.5 → −9.0 | −2.05 → −2.36 | none | ~0–30 |
| 51 | 250.9 N wire spike | 1.621 | 0.234 | −1.138 | 1.9 / −10.8 | −2.557 | none | 250.9 |
| 52 | calf reaches soft limit (≈−2.58); pitch rate reverses | −1.098 | 0.210 | −0.750 | 1.1 / +2.1 | −2.584 | none | 32 |

Fact: from step 44 the pitch rate roughly doubles each step until the calf
reaches its soft joint limit at step 52, then reverses sign. The recorded vz reaches
−1.14 m/s over ~7 steps ≈ 0.14 s, i.e. close to ballistic fall (~8 m/s² average).
Facts about FL: the leg *retracts and lifts* immediately (foot z in base frame
−0.286 → −0.124 m by step 52; foot_x_base 0.191 → 0.177 → 0.199 — only a small x
retraction; D5.16's "0.107 m retraction" refers to the later window to step 81). It
was already airborne, so there is no FL support-loading change at entry.

Sources: `logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv`,
`logs/d5_47/seed0_trace.csv`.

### B2. The key comparison: baseline vs D5.49 delayed handoff, aligned on handoff

Rows k = steps since hybrid takeover (baseline k=0 ↔ step 44; hold run k=0 ↔ step 51).
Data: `logs/d5_49_recovery_override_counterfactual/{baseline,counterfactual}_trace.csv`.

| | pitch rate before handoff | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | flip (calf at limit) |
|---|---|---|---|---|---|---|---|---|
| baseline, handoff @44 | 0.018 | 0.188 | 0.374 | 0.703 | 1.252 | 1.610 | 1.804 | k=8 (step 52) |
| hold run, handoff @51 | 0.059 | 0.738 | 1.032 | 1.246 | 1.677 | 1.997 | 1.981 | k=7 (step 58) |

Hold-window rows (position actuators held, walking policy in control of FL, wire
contact continuing at 17–19 N from step 46): pitch rate 0.177, 0.068, 0.035, 0.031,
0.038, 0.052, 0.059 for steps 44–50; base_z 0.309 → 0.307; vz within ±0.02 m/s.

Facts:
- **The initial pitch-rate kick of ~0.18 rad/s at step 44 is present in both runs**
  (0.188 vs 0.177), including the run where the hybrid was *not* yet active. In the
  hold run it decays within 2 steps under position control. So the first-contact
  impulse (and the stiff-leg reaction) produces a small, self-limiting kick.
- **The runaway (pitch rate > 1 rad/s, vz < −0.3 m/s, calf slam to joint limit) begins
  at hybrid takeover in both runs, at a gait phase ~7 steps apart,** with a similar
  time-to-limit (8 vs 7 steps). The calf-limit event and pitch reversal are aligned
  to handoff, not to absolute step or wire force.
- Hold-run wire force during the hold (17–19 N) is the same order as baseline's
  (23–30 N) at the same stage, yet no runaway occurred during those 7 steps.

Interpretation (labelled): the hybrid takeover is **necessary in practice** for the
runaway under this contact condition; contact alone, as seen in the hold run, produced
a transient only. This weakens H1 (pure contact impulse). It does not show the hybrid
is *sufficient*: the contact condition at takeover differed from a no-contact case.

Caveats: one seed, one pair, one hold length; hold run's FL calf was held against the
wire by ~11 N·m of position torque, so the leg state at handoff was not the same as
baseline's (calf −1.71 vs −1.73, thigh 1.03 vs 0.90; FL thigh/calf vel differ);
D5.49 did not log FR/RL/RR ground contact, so support loss in the hold run is
unobserved; after handoff the hold run is also unstable (min base_z 0.1816, lower than
baseline's 0.204), consistent with the hybrid being destabilizing wherever applied.

### B3. Support-leg and body dynamics
Fact: FR and RL (the trot stance pair) leave the ground at step 47, 3 steps after
takeover, while the trunk is pitching nose-down at ~1.25 rad/s and descending. A body
falling onto the ground does not by itself lift its stance feet; for feet to lose contact
while z drops, the stance legs must flex faster than the body descends.
Interpretation: those joints' targets come from the frozen walking policy, whose
observation now contains FL calf velocity far outside its walking range (to −10.8
rad/s) and pitch rate up to 1.8 rad/s (F4). A policy-feedback path to the stance legs
is therefore plausible but **untested**; there is no log of the policy's 12-d raw
action, support-leg joint targets, or support-leg torques anywhere in D5.16–D5.50.

### B4. Is time alignment sufficient to separate the hypotheses?
No. All signals are 20 ms samples (F1). The first contact (step 43) and the hybrid
onset (step 44) are adjacent samples, contact persists 43–46 while hybrid is on, and the
contact force signal is both low-rate and sign-ambiguous (U-a, U-b). Attribution within
the entry window by timing alone is not possible. D5.49 separated the factors by
intervention instead of timing, which is why it is the most informative existing evidence
despite its flawed summary.

### B5. Missing measurements needed for causality
1. FR/RL/RR (and FL hip) joint position, velocity, applied torque, and the policy's
   raw 12-d action each step (support-leg response).
2. Per-foot ground contact force (not just a boolean) for all four feet.
3. Wire contact force at substep resolution (8/step) around steps 43–46 with a documented sign convention; reconciliation of 130 N raw vs 10.9 N sensor.
4. Joint-limit proximity/impact for FL calf (soft limit ≈ −2.58 from range −2.7227 × 0.9).
5. The (no-wire, hybrid) and (no-wire, no-hybrid) trajectories under the same state.

### B6. Differences from verified Quad-SDK that matter for the *initial* instability
Source: `robot_driver/robot_driver.yaml:23-30`, `underbrush_inverse_dynamics.cpp:345-462`.
- Command magnitudes (Spirit params): `retract_vel` 15 rad/s, `tau_push` 2.0 N·m, thigh kd
  = `swing_kd[1]` (1 for Spirit). mjlab/Go2: thigh 2.0 rad/s, kd 2.0, calf 5.0 N·m. The
  calf torque is 2.5× the SDK's (documented as a deliberate non-copy,
  `actions.py:316-326`); with no calf damping (same as SDK) that torque drives the
  calf to −10.8 rad/s and into its soft limit in 8 steps. Whether "−tau_push" is the
  same physical direction as mjlab's calf −5 N·m on the Spirit knee convention is not
  verified here.
- Trigger: SDK uses estimated body-torque thresholds with hysteresis, `min_switch` 0.1 s,
  and a `t_up` 0.04 s lockout after lift-off; mjlab uses any `found>0` calf contact with a
  one-step latency, so recovery starts at the very first, lightest contact (10.9 N).
- Both: ab/ad stays on the walking reference; thigh velocity + calf feedforward torque
  structure is the same (D5.16 verified).
No claim is made that copying the SDK trigger or magnitudes would remove the instability.

## C. Candidate evaluation and single recommendation

See [experiment_decision_table.csv](experiment_decision_table.csv). Criteria: (1) one
mechanism, (2) no unrelated contact/trigger change, (3) valid reproducible state,
(4) discriminates ≥2 explanations, (5) measurable.

Excluded: delayed handoff (D5.49 repeat), contact-filter / body-scoped ablations (D5.50),
threshold/gain sweeps, trigger copying.

### Recommended experiment — **E1: wire-removed forced-entry hybrid replay**

Design (not run here). Completes a 2×2 over {wire contact} × {hybrid takeover}:

| | no hybrid | hybrid at ~step 44 |
|---|---|---|
| wire contact | D5.49 hold (measured: transient only) | baseline (measured: runaway) |
| no wire | R0 null (new, free) | **R1 new** |

1. Seed 0, same frozen checkpoint and env config. In a *separate experiment script*
   (not `actions.py`), remove all robot–wire collisions for the run, using the
   existing validated mechanism in `scripts/d4_8_hip_wire_contact_ablation.py`
   (`add_exclude`), applied to all robot bodies × wire segments (broad scope is the
   intent here), or wire geom `contype/conaffinity=0`.
2. R1: force the existing FSM `entering_recovery` at step 44 (the baseline entry step),
   leaving hybrid commands/actuator handoff code untouched. R0: same run, no forced entry.
   R2: wire-on baseline reproduced through the same script path.
3. Preconditions to check first (abort if violated): R1/R0/R2 states at the end of step 42 bitwise
   equal (first contact is step 43, so earlier physics is wire-independent); R2
   reproduces `d5_49/baseline_trace.csv` pitch-rate/base_z columns to float tolerance; R1
   robot–wire contact count == 0 for all steps.
4. Window: handoff-aligned k = −3…+10; do not interpret beyond step 50 for wire-on
   comparisons (hip contact at 51).

Pre-registered outcomes (peak pitch rate over k=0..7 as fraction of baseline's 1.80 rad/s;
also min vz, base_z drop, step support feet leave ground, step FL calf reaches limit):

| R1 result | Interpretation |
|---|---|
| ≥70% of baseline peak, similar timing (calf limit near k=8), support feet lose contact | Runaway is **hybrid-intrinsic** (command + leg dynamics + possible policy feedback); contact is not needed. Next: E2 channel split and support-leg/policy-action analysis. |
| ≤25% of baseline, support feet stay, base_z within ~0.01 m of R0 | Runaway needs **contact × hybrid** interaction. Neither factor alone is sufficient (given D5.49 hold). Next: study wire reaction through the compliant leg. |
| 25–70% | Both contribute; contact amplifies a hybrid-driven instability; report the fraction. |

Discriminates: hybrid-intrinsic (H2) vs combination (H3), and with the D5.49 hold cell
bounds H1.

Required measurements (small logger addition in the experiment script, per policy step,
no change to `actions.py`):
- base pose, linear/angular velocity (world and body), base_z, pitch rate, projected gravity;
- all 12 joint pos, vel, ctrl, applied actuator torque (`qfrc_actuator`);
- `last_a_walk_raw` (12-d) and the FL entries of the policy observation;
- per-foot ground contact force for 4 feet;
- wire contact force with documented sign, and raw per-substep pair force for steps 43–46;
- phase flags, forced-entry flag, and robot–wire contact count.

Risks: wire removal changes everything contact-related after step 43, including the
first impulse — that is the factor being tested, so it is a feature, but it also means
R1 cannot isolate the *first impulse* separately from the sustained contact (infeasible per D5.50).
Single seed; seed 11 has a different entry state (D5.47/48), so use it only as a second
replicate after seed 0 gives a clear outcome.

## D. Conclusions

- **Leading explanation (moderate confidence, single seed):** the runaway (nose-down
  pitch to ~1.8 rad/s, loss of FR/RL support, ~0.1 m sag, FL calf slam into its soft limit)
  starts at hybrid takeover. Wire contact gives only a small ~0.18 rad/s kick that
  decays under position control. The runaway appears in two runs ~7 gait-steps apart
  with similar handoff-relative timing.
- **Strongest competitor:** a contact × hybrid interaction (or hybrid acting through the
  frozen policy's reaction to out-of-range FL observations), where the hybrid is not
  destabilizing without the wire. Not excluded by existing data.
- **Not claimed:** that the root cause is proven; that the stance-leg response is
  policy-driven; that the hybrid is the sole cause.
- **Single next experiment:** E1 above.
- **Exact measurements to interpret it:** the list under "Required measurements".

## E. Requirements for the future behavior-changing task prompt

The future prompt must require, for both baseline and experimental runs, on seed 0 and
(where applicable) seed 11, with durations covering the full behaviour (≥ 200 policy
steps for seed 0 recovery incl. exit at ~88 and settling to ~110; seed 11's cap
timeout at ~98 from entry at 51 → ≥ 200 steps):
- exact visualization/playback commands, **parser-verified** (run `--help` / dry-parse)
  and then actually run to completion to verify;
- the existing viewer flag is `--debug` in `scripts/d5_16_trajectory_audit.py` (D5.15 viewer;
  intermittent teardown crashes 134/139/124 recorded after complete output);
  `scripts/d5_49_recovery_override_counterfactual.py` has no viewer flag (flags: `--seed`,
  `--device`, `--max-steps` default 180, `--max-recovery-steps-override`, `--hold-steps`,
  `--out-dir`) — new scripts need their own;
- if the native viewer is unreliable, a verified headless playback alternative whose CSV
  matches the viewer run byte-for-byte (the D5.16 headless/viewer diff check is the template).

Note: `actions.py` and `env_cfg_residual.py` carry uncommitted modifications in the working
tree (default-off two-phase experiment flags). They were not touched here; baseline
equivalence rests on D5.49's baseline reproducing the earlier traces.
