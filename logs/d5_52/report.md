# D5.52 — E1 Wire-Removed Forced-Entry Hybrid Replay

Implements the E1 experiment specified in [D5.51](../d5_51/report.md). Seed 0, frozen
checkpoint `logs/rsl_rl/go2_velocity/2026-09-17_13-00-16/model_9900.pt`, same env build path
as D5.16/D5.49. No production file was edited (`actions.py`, `env_cfg_residual.py`
untouched; they still carry the pre-existing uncommitted default-off two-phase flags). No
training. All interventions live in
[scripts/d5_52_wire_removed_forced_entry.py](../../scripts/d5_52_wire_removed_forced_entry.py).

**Conclusion: 1 — hybrid-intrinsic runaway supported** for the first 10 policy steps after
handoff, with the qualifications in section 5. The body-level runaway reproduces with zero
robot–wire contact. The internal FL-leg trajectory is different, and the trajectory after
k≈7 diverges, because in the baseline the wire catches the falling body.

## 1. Conditions and interventions

| Condition | Wire collisions | Recovery entry | Source |
|---|---|---|---|
| `baseline` | on | natural contact trigger (step 44) | new run, reproduces D5.49 exactly |
| hold (D5.49) | on | step 44, handoff delayed to 51 | reused from `logs/d5_49_recovery_override_counterfactual/counterfactual_trace.csv`, not re-run |
| `wire_off_control` | off | none | new |
| `wire_off_forced` | off | forced at step 44 | new |

- **Wire removal:** `MjSpec.add_exclude` body-pair excludes between all 13 `robot/*`
  bodies and all 7 `wire_seg_*` bodies (91 pairs), composed onto the existing wire
  `spec_fn`. Same mechanism as `scripts/d4_8_hip_wire_contact_ablation.py`, with the scope
  deliberately widened. The D5.50 geom-contact callback is not used.
- **Forced entry:** only for the `_update_fsm` call of policy step 44, the FSM sees a proxy
  `wire_contact` sensor with `found=1`. Everything downstream runs the original code
  unchanged: `entering_recovery`, `_apply_hybrid_actuators` (gain/bias zeroing, aux
  thigh kd, +2.0 rad/s thigh, −5.0 N·m calf), thresholds, and exit logic.
- **Row convention (as D5.16/D5.49):** row `step=k` is the state after `env.step()` number
  k. k = step − 44 = steps since hybrid handoff (handoff is step 44 in both hybrid runs).
  For the hold run, k = step − 51.
- **Coordinates:** pitch rate = `root_link_ang_vel_b[1]` (+ = nose-down). vz is the world
  frame. Ground force is the per-leg netforce from the D5.16 `*_ground_contact_diag`
  sensors, whose `|Fz|` is reported as the support load. Wire force is the raw robot–wire
  contact force on the robot, world frame, decoded per physics substep from the
  mujoco_warp contact and efc arrays (pyramidal cone).

## 2. Preconditions — all passed ([validation.csv](validation.csv))

| Check | Result |
|---|---|
| Wire-off states bitwise equal to baseline through end of step 42 (full `qpos`, `qvel`) | PASS; first difference at step 43 for both wire-off runs, i.e. the first-contact step |
| Baseline reproduces D5.49 baseline (base_z, pitch rate, vz, FL thigh/calf pos, calf vel, wire force; 180 steps) | PASS, max abs diff 0.0 (tolerance 1e-5) |
| Zero robot–wire contacts, all substeps, wire-off runs | PASS, 0 / 0 over 2000 substeps each |
| Forced entry at 44 with intended controls | PASS: row 43 NORMAL_WALK; row 44 RECOVERY, pos gains 0, thigh cmd +2.0, calf cmd −5.0 |
| Control stays on walking path | PASS, 250/250 rows NORMAL_WALK with position gains active |

## 3. Results, handoff-aligned (k = 0..9)

[comparison_summary.csv](comparison_summary.csv), [aligned_trajectory.csv](aligned_trajectory.csv)
(steps 34–74, all measurements plus per-substep wire force lists).

| Metric (k0–9) | baseline | wire_off_forced | wire_off_control | D5.49 hold (k from 51) |
|---|---|---|---|---|
| Peak \|pitch rate\| (rad/s) | 1.80 @k5 | **1.88 @k8 (104%)** | 0.054 (3%) | 2.05 |
| Base-z drop from k−1 (m) | 0.105 | **0.088 (84%)** | 0.0002 | 0.102 |
| Min world vz (m/s) | −1.23 | **−1.10** | −0.02 | −1.13 (body frame) |
| Stance pair FR+RL both off ground | k3 | k3 | k3–4 (normal trot swap) | not logged |
| All of FR/RL/RR off ground | k4 | k9 | never | not logged |
| Peak \|FL calf vel\| (rad/s) | 10.8 | 18.7 | 0.65 | 10.1 |
| FL calf reaches soft limit (−2.628 rad) | not in window (stops at −2.584, k8) | k4 (overshoots to −2.836) | no | — |
| Max other-leg policy action change vs control | 3.71 (FR_calf) | 3.21 (FR_calf) | 0 | — |
| Min base z, whole run (m) | 0.204 | 0.168 | 0.304 | 0.182 |
| Crossing / release / exit | 84 / 88 / 88 crossing | 52 / 56 / 56 (geometric pass-through, wire non-colliding) | — | none (cap timeout) |
| Fall (`fell_over`) | no | no | no | no |

Pitch-rate trajectory, k = −1..8 (rad/s):

| k | −1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.018 | 0.188 | 0.374 | 0.703 | 1.252 | 1.610 | 1.804 | 1.763 | 1.621 | −1.098 |
| wire_off_forced | 0.008 | 0.061 | 0.212 | 0.526 | 0.924 | 1.215 | 1.188 | 1.468 | 1.734 | 1.881 |
| wire_off_control | 0.008 | 0.024 | −0.054 | 0.038 | 0.013 | 0.018 | −0.030 | −0.033 | −0.013 | 0.021 |

Total vertical ground support (sum of the four legs' \|Fz\|, N; robot weight = 15.2 kg × 9.81 = 149 N):

| k | −1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 146 | 146 | 133 | 116 | 28 | 0 | 0 | 0 | 0 | 0 | 202 |
| wire_off_forced | 146 | 143 | 130 | 115 | 28 | 34 | 50 | 61 | 53 | 56 | 0 |
| wire_off_control | 146 | 159 | 167 | 155 | 144 | 127 | 138 | 145 | 149 | 155 | 156 |
| of which FL (control) | 0 | 19 | 25 | 40 | 69 | 74 | 78 | 77 | 77 | 79 | 81 |

Baseline per-substep wire force on the robot ([substeps_baseline.csv](substeps_baseline.csv)):
- Step 43: 12 N in the last substep only.
- Steps 44–50: sustained 17–33 N on `FL_calf2_collision`, world Fz −10 to −30 N, i.e. the
  wire pushes the calf downward.
- Steps 51–53: FL hip and base hit the wire, with substep peaks of 2890 N (step 52) and
  1653 N (step 53), mostly +z on the robot.

## 4. Observations (measured)

O1. **Pitch-rate runaway reproduces without any wire contact:** 104% of baseline peak,
with the same doubling-per-step growth, lagging the baseline by about one step.
O2. **Base-height drop and downward velocity reproduce:** 84% and 89% of baseline within
k0–9. Without the wire the descent continues past k9 to 0.168 m (k12), lower than
baseline's 0.204 m.
O3. **Support loss reproduces.** In both hybrid runs FR and RL unload at k3, at the same
point in the trot cycle where the control also unloads them (k3–4). In the control, FL
touches down at k0 (step 44) and carries 19→81 N, so total support stays 127–167 N. In
both hybrid runs FL never touches down in the window, and total support falls to 28 N at
k3 and 0–61 N for k4–8. A deficit of roughly 90–150 N against 149 N of weight is
consistent with the measured descent rate (wire_off_forced: vz −0.25 → −0.92 m/s over
k3–8, ≈6.7 m/s²). Roll rate also grows to about −2 rad/s in both hybrid runs vs ≈0 in the
control: the body tips toward the front-left corner.
O4. **The FL calf path differs.** Without the wire, the calf folds at up to 18.7 rad/s and
slams through the soft limit (−2.628) into the hard range limit by k4–5. With the wire, it
reaches only 10.8 rad/s and stops at −2.584 at k8, during the hip/base/calf impact with
the wire (steps 51–53), short of the soft limit by 0.045 rad. **This corrects D5.51 B1,
which said the calf reached its soft limit at step 52.**
O5. **Frozen-policy outputs for the other legs change.** At k0 they are identical to the
control (same observation). They diverge from k1 (max 0.12 raw units, FR_calf), reach
0.8 at k3 and about 3 by k5–10. Most of the change is FR_calf in both hybrid runs.
O6. The first-step pitch kick at k0 differs: baseline 0.188, wire-off forced 0.061,
control 0.024. The ~0.13 rad/s extra kick in the baseline is contact-attributable, as
D5.51 inferred from the hold run.
O7. FL foot world height at step 43 ≈ 0.024 m (D5.16 `foot_z`), i.e. at ground-contact
height: the wire trigger fires at the end of FL swing, one step before scheduled
touchdown.

## 5. Interpretation (hypotheses, labelled)

- **H-support (supported, not proven):** the runaway comes from the hybrid retracting FL
  (thigh +2.0 rad/s, calf −5 N·m) at the moment the walking gait hands support from the
  FR/RL pair to the FL/RR pair. The policy unloads FR/RL on its normal schedule (O3, the
  control does the same), FL is not there to take the load, and the robot drops and tips
  toward the front-left. This explains O1–O3 without the wire, and explains why D5.49's
  delayed handoff (step 51) also ran away: in the hold run the FL foot sat at about ground
  height (`base_z` 0.307 + `foot_z_base` −0.286 ≈ 0.021 m, pitch ignored), so it was
  probably supporting before being retracted. That run did not log ground force, so this
  remains a hypothesis.
- **Same body trajectory, different internals:** O4 shows the FL leg does not move the
  same way, and the wire both adds a small kick (O6) and later catches the body (2.9 kN
  substep peak at step 52). So the wire changes the magnitude and ending of the event but
  is not required to start it.
- **Policy feedback is not shown to be causal:** the other-leg action changes (O5) are
  real and grow with the perturbation, but the stance unloading at k3 also happens in the
  unperturbed control. Whether the policy's reaction amplifies the fall cannot be
  separated from the support deficit with this design.
- Limits: one seed; one gait phase at entry (end of FL swing); 20 ms sampling for body and
  ground signals; ground-force sign is the sensor's netforce convention, used only as a
  magnitude.

Against the D5.51 pre-registered thresholds (≥70% of baseline peak pitch rate, similar
timing, support loss → hybrid-intrinsic): 104%, stance unloading at k3 in both runs, FL
calf limit reached → **conclusion 1**. The runaway is "intrinsic" to activating the
hybrid at this gait phase. These data do not show it would occur at every phase.

## 6. Visualization / playback — verified commands

The `--debug` flag opens the D5.15-pattern native viewer (callback never calls
`env.step()`). Each command was run to completion: 250 policy steps (5.0 s simulated),
covering contact (43), runaway (44–56), exit (88 / 56) and settling. Exit code 0 each
time, after the final formatting edits. Each viewer CSV is **byte-identical** to the
headless trace ([viewer_vs_headless.txt](viewer_vs_headless.txt)). A window cannot be
judged visually from here; the trace identity is the evidence that the viewer stepped the
same physics.

```sh
uv run python scripts/d5_52_wire_removed_forced_entry.py --condition baseline --debug --max-steps 250
uv run python scripts/d5_52_wire_removed_forced_entry.py --condition wire_off_control --debug --max-steps 250
uv run python scripts/d5_52_wire_removed_forced_entry.py --condition wire_off_forced --debug --max-steps 250
```

Headless (all conditions, validation, and every CSV in this directory; re-run on the
final script, output byte-identical):

```sh
uv run python scripts/d5_52_wire_removed_forced_entry.py
```

Parser check: `uv run python scripts/d5_52_wire_removed_forced_entry.py --help` lists
`--condition {all,baseline,wire_off_control,wire_off_forced}`, `--seed`, `--device`,
`--max-steps`, `--max-recovery-steps-override`, `--out-dir`, `--debug/--no-debug`.
Seed 11 is not part of E1. The script accepts `--seed 11`, but forced entry is
hard-coded to seed 0's step 44, so it must not be used for seed 11 without changing the
entry step.

## 7. Metrics not recorded / caveats

- Ground forces come from leg-geom-vs-terrain netforce sensors. They include
  thigh/calf/foot geoms and report a signed world-frame vector; the sign convention was
  not independently verified, so only magnitudes are interpreted.
- Wire force is per substep for robot–wire pairs only. D5.50's "130.5 N at step 43" was
  not reproduced: the raw decode gives 12 N in the last substep of step 43. The method
  difference is unexplained.
- The D5.49 hold run has no ground-force, support-leg or policy-action columns. It was
  reused, not re-run, per the task.
- `stance_pair_loss_k` is also triggered by normal trotting (control k4). Use
  `all_support_loss_k` and the support-force table to separate it.

## 8. Recommended next single controller experiment

**Touchdown-proximity guard (Quad-SDK `t_down` analogue), as an opt-in diagnostic variant.**
Quad-SDK refuses obstructed-swing mode when touchdown is imminent
(`t_TD - t_now < t_down`, `underbrush_inverse_dynamics.cpp:345-381`, `t_down` = 0.135 s in
`robot_driver.yaml:29`). mjlab has no planned touchdown time, so use a measured proxy: at
a would-be entry, if FL foot height above terrain is below a fixed small value and the
foot is descending (seed 0: foot at ≈0.024 m, landing next step in the control), keep FL
on the walking position path. Re-arm hybrid entry only after FL has been through stance
and lifted off again (Quad-SDK's `t_up` lockout analogue). Hybrid commands, gains and the
contact sensor stay unchanged. This is not a repeat of D5.49: the hold length is set by a
gait state, not a fixed 8 steps.

Discriminating outcomes:
- No runaway at first contact (peak pitch rate ≤25% of baseline over k0–9, total support
  stays ≥ ~120 N), and no runaway when the hybrid later activates mid-swing →
  **H-support confirmed**: phase-at-activation is the cause, not the commands.
- Runaway recurs whenever the hybrid activates, including mid-swing → the commands
  themselves (calf −5 N·m undamped, thigh +2 rad/s) destabilize the robot regardless of
  phase. Next step would be the D5.51 E2 channel split.
- Guard holds but FL lands on or behind the wire and crossing fails → stability and
  crossing conflict. Report both.

Required measurements: the D5.52 logger as-is, plus FL foot height above terrain at entry
and guard state per step.

The future prompt must require exact, parser-verified, actually-run viewer commands for
baseline and guard variant on **seed 0 and seed 11** (≥250 steps; seed 11 enters at 51
and times out ≈98), and a byte-identical headless trace comparison, as done here.

## Files

- [report.md](report.md), [comparison_summary.csv](comparison_summary.csv), [validation.csv](validation.csv)
- [aligned_trajectory.csv](aligned_trajectory.csv): steps 34–74, all conditions, k_from_handoff, all per-step measurements plus per-substep wire force list
- `trace_<condition>.csv` (250 rows × 153 cols), `substeps_<condition>.csv` (8 rows per step)
- `viewer_<condition>/trace.csv`, `viewer_<condition>.log`, [viewer_vs_headless.txt](viewer_vs_headless.txt), `headless_run.log`
