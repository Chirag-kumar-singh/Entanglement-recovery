# D5.48 — Quad-SDK Reference Audit and mjlab Baseline Root-Cause Audit

Scope:
- Part A: read-only audit of the Quad-SDK underbrush recovery reference in `/home/arc01/quad-sdk/quad-sdk`.
- Part B: read-only audit of the current `mjlab` baseline event sequence using existing artifacts wherever possible.
- Part C: decide whether a fresh FL hip–wire ablation is justified under the requested isolation constraints.

No controller code, gains, thresholds, checkpoints, wire placement, or collision settings were changed for this audit.

## Files used

Quad-SDK source inspected read-only:
- `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/controllers/underbrush_inverse_dynamics.cpp`
- `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/robot_driver.cpp`

mjlab source and artifacts inspected:
- [src/mjlab/tasks/recovery/mdp/actions.py](../../src/mjlab/tasks/recovery/mdp/actions.py)
- [src/mjlab/tasks/recovery/env_cfg_residual.py](../../src/mjlab/tasks/recovery/env_cfg_residual.py)
- [src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml](../../src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml)
- [scripts/d5_16_trajectory_audit.py](../../scripts/d5_16_trajectory_audit.py)
- [logs/d5_16_trajectory_audit/report.md](../d5_16_trajectory_audit/report.md)
- [logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv](../d5_16_trajectory_audit/headless_seed0/trajectory.csv)
- [logs/d4_6_support_descent_timing/seed0/contact_listing.txt](../d4_6_support_descent_timing/seed0/contact_listing.txt)
- [logs/d5_43/report.md](../d5_43/report.md)
- [logs/d5_47/report.md](../d5_47/report.md)

## Part A — Verified Quad-SDK reference behavior

### A1. What signal triggers the underbrush mode?

Verified source facts:
- The underbrush controller subscribes to a `quad_msgs::BodyForceEstimate` topic loaded from `topics/body_force/joint_torques` in `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/robot_driver.cpp` lines 26-27, 82-84, and forwarded via `RobotDriver::bodyForceEstimateCallback(...)` at lines 300-305.
- `UnderbrushInverseDynamicsController::updateBodyForceEstimate(...)` in `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/controllers/underbrush_inverse_dynamics.cpp` lines 13-15 simply stores the latest `BodyForceEstimate` message in `last_body_force_estimate_msg_`.
- `UnderbrushInverseDynamicsController::computeLegCommandArray(...)` refuses to run without both a local plan and a body-force-estimate message (`last_local_plan_msg_ == NULL || last_body_force_estimate_msg_ == NULL`) at lines 30-37.

Interpretation:
- The Quad-SDK reference does not use a direct wire-contact sensor in this controller. Its obstruction gate is based on estimated per-leg body wrench torques carried by the `BodyForceEstimate` message.

### A2. Exact enter / exit conditions for the obstructed-swing mode

Verified source facts from `computeLegCommandArray(...)`:
- Underbrush parameters are stored by `setUnderbrushParams(...)` at lines 18-28:
  - `retract_vel_`
  - `tau_push_`
  - `tau_contact_start_`
  - `tau_contact_end_`
  - `min_switch_`
  - `t_down_`
  - `t_up_`
- `RobotDriver::initLegController()` loads those values from ROS params `/underbrush_swing/retract_vel`, `/underbrush_swing/tau_push`, `/underbrush_swing/tau_contact_start`, `/underbrush_swing/tau_contact_end`, `/underbrush_swing/min_switch`, `/underbrush_swing/t_down`, `/underbrush_swing/t_up` at `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/robot_driver.cpp` lines 175-192.
- Swing timing bounds are derived from the current local plan:
  - lift-off times `t_LO_` are updated when the plan transitions contact -> swing at lines 109-117;
  - next touchdown times `t_TD_` are found from future planned contact transitions at lines 122-145.
- Binary mode switch logic appears at lines 345-381:
  - Exit obstructed mode when:
    - `force_mode_.at(i)` is true,
    - `t_now2 - t_switch_.at(i) > min_switch_`,
    - `last_body_force_estimate_msg_->body_wrenches.at(i).torque.z < tau_contact_end_`, and
    - `t_TD_.at(i) - t_now2 >= t_down_`.
  - Enter obstructed mode when:
    - `!force_mode_.at(i)`,
    - `t_now2 - t_switch_.at(i) > min_switch_`, and
    - either `torque.z > tau_contact_start_`
      or `(torque.y > tau_contact_start_ && t_now2 - t_LO_.at(i) > t_up_)`.
  - Force exit near touchdown when `force_mode_.at(i)` and `t_TD_.at(i) - t_now2 < t_down_`.
  - Early in swing, if `t_now2 - t_LO_.at(i) < t_up_` and `last_mode_.at(i)` is true, the previous obstructed mode is retained.

Interpretation:
- The reference behavior is a binary usual-swing vs obstructed-swing switch with hysteresis (`tau_contact_start_` vs `tau_contact_end_`) and timing guards (`min_switch_`, `t_up_`, `t_down_`).
- There is no explicit multi-state contact detector, no direct force threshold on a foot contact sensor, and no explicit “resume normal walking after release counter N” FSM.

### A3. Exact control law while obstructed

Verified source facts from `/home/arc01/quad-sdk/quad-sdk/robot_driver/src/controllers/underbrush_inverse_dynamics.cpp` lines 383-462:
- In usual swing mode (`!force_mode_`), each joint gets the local-plan swing reference:
  - `pos_setpoint = ref_state_msg_.joints.position[...]`
  - `vel_setpoint = ref_state_msg_.joints.velocity[...]`
  - `torque_ff = 0`
  - `kp = swing_kp_[j]`
  - `kd = swing_kd_[j]`
- In obstructed swing mode (`force_mode_`):
  - First, all three joints are initialized to:
    - `pos_setpoint = 0`
    - `torque_ff = 0`
    - `kp = 0`
    - `kd = swing_kd_[j]`
  - Then joint 0 (ab/ad) is restored to its usual swing reference:
    - `pos_setpoint = ref_state_msg_.joints.position.at(3 * i + 0)`
    - `vel_setpoint = ref_state_msg_.joints.velocity.at(3 * i + 0)`
    - `kp = swing_kp_.at(0)`
  - Joint 1 (hip-pitch / `thigh` in `mjlab` naming) gets:
    - `vel_setpoint = -retract_vel_`
    - `kp = 0`
    - `kd = swing_kd_[1]`
    - `torque_ff = 0`
  - Joint 2 (knee / `calf` in `mjlab` naming) gets:
    - `vel_setpoint = 0`
    - `kd = 0`
    - `torque_ff = -tau_push_`

Interpretation:
- The verified reference is a single constant-command obstructed-swing mode, not a scripted RETRACT -> ADVANCE-LIFT -> DESCEND sequence.
- The “sequence” seen in trajectory plots is emergent from one binary mode plus gait timing, not a literal internal stage machine.

### A4. Relation to current mjlab baseline

Verified `mjlab` facts:
- Recovery entry is contact-driven in [src/mjlab/tasks/recovery/mdp/actions.py](../../src/mjlab/tasks/recovery/mdp/actions.py): `entering_recovery = is_normal & in_contact`.
- The wire-contact sensor is defined in [src/mjlab/tasks/recovery/env_cfg_residual.py](../../src/mjlab/tasks/recovery/env_cfg_residual.py) as `wire_contact`, with the wire geoms as primary matches and `FL_calf2_collision` as the secondary geom.
- In the baseline hybrid recovery path, the thigh and calf auxiliary commands are applied in [src/mjlab/tasks/recovery/mdp/actions.py](../../src/mjlab/tasks/recovery/mdp/actions.py):
  - `thigh_vel_cmd = ...`
  - `calf_torque_cmd = -self.cfg.recovery_calf_torque` while in recovery.

Interpretation:
- `mjlab` matches the reference actuator structure (ab/ad stays on the walking path; thigh uses auxiliary velocity; calf uses auxiliary torque), but not the reference trigger signal. Quad-SDK triggers on estimated body torques during swing; `mjlab` triggers on a direct wire-contact sensor tied to `FL_calf2_collision`.

## Part B — Current mjlab baseline event sequence

### B1. First robot geom to contact the wire

Verified fact:
- The first robot–wire contact listed in [logs/d4_6_support_descent_timing/seed0/contact_listing.txt](../d4_6_support_descent_timing/seed0/contact_listing.txt) is at step 43:
  - `wire_seg_2_geom <-> robot/FL_calf2_collision`
- This is consistent with [logs/d5_43/report.md](../d5_43/report.md), which reports seed-0 first wire contact at step 43.

Answer:
- The first robot geom to contact the wire is `robot/FL_calf2_collision`, not the hip.

### B2. Does the FL hip contact the wire, and does it persist?

Verified facts from existing artifacts:
- Raw FL hip–wire contact is logged directly in [scripts/d5_16_trajectory_audit.py](../../scripts/d5_16_trajectory_audit.py) and recorded in [logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv](../d5_16_trajectory_audit/headless_seed0/trajectory.csv) as `fl_hip_wire_contact_raw`.
- The first raw hip contact occurs at step 51.
- Raw hip-contact intervals from the CSV are:
  - 51-52
  - 54-73
- Contact-listing evidence agrees: step 51 shows `wire_seg_3_geom <-> robot/FL_hip_collision` and `wire_seg_4_geom <-> robot/FL_hip_collision`.

Answer:
- Yes, the FL hip does contact the wire.
- It does not cause the initial entanglement. It first appears 8 steps after the first calf-wire contact and 7 steps after recovery entry, then persists intermittently and mostly continuously through the middle of recovery.

### B3. First measurable body instability

Operational definition used here:
- “First measurable body instability” means the first body-state signal that leaves its pre-contact step-0..42 range in the seed-0 baseline CSV.

Verified facts from [logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv](../d5_16_trajectory_audit/headless_seed0/trajectory.csv):
- Recovery-entry row is step 44.
- `base_ang_vel_y` first leaves its pre-contact range at step 45 (`0.3742`, just above the pre-contact max `0.3664`).
- `base_z` first leaves its pre-contact range at step 47 (`0.2988`, below the pre-contact min `0.3038`).
- The major body-height collapse then continues to the global minimum at step 54 (`base_z = 0.2040`), matching [logs/d5_16_trajectory_audit/report.md](../d5_16_trajectory_audit/report.md).

Answer:
- Using only existing telemetry, the earliest measurable body-instability signal is the pitch-rate excursion at step 45, followed by clear body-height loss at step 47.
- The large instability is therefore underway well before the later FL hip contact at step 51.

### B4. When does recovery activate relative to contact and instability?

Verified facts:
- First wire contact: step 43.
- First recovery row (`phase == 1`): step 44.
- Earliest measured instability by the criterion above: step 45 (`base_ang_vel_y`), then step 47 (`base_z`).

Answer:
- Recovery activates one step after the first calf-wire contact, and before the first strong body-instability excursion visible in the recorded base-state signals.

### B5. Does instability begin before recovery, or worsen during recovery?

Verified facts:
- There is no pre-entry FL hip contact.
- The first calf-wire contact is at step 43 while the major body sag has not yet begun.
- The large body-height drop occurs during recovery: step 47 is the first clear base-height excursion, and the nadir is step 54.
- Raw hip contact begins at step 51, after the sag is already well underway.

Answer:
- The existing data do not support “instability begins before recovery” as the main explanation.
- The data support a milder statement: the initial entanglement happens first, but the substantial body instability is introduced or strongly worsened during the recovery phase rather than by a pre-recovery hip contact.

### B6. Does stable forward motion resume after crossing release?

Verified facts:
- `crossing_achieved`, `foot_crossed_achieved`, and `calf_side_far` all first latch at step 84.
- `release_counter >= 5` first occurs at step 88, which is also the first `phase == 2` row.
- A conservative post-exit stability check on the existing CSV finds the first 10-step window with both:
  - `base_z` back inside the pre-contact range, and
  - `base_lin_vel_x` in a walking-like positive band `0.2..0.35 m/s`
  beginning at step 109.
- This matches the qualitative result in [logs/d5_16_trajectory_audit/report.md](../d5_16_trajectory_audit/report.md): forward walking recovers after the post-handoff dip.

Answer:
- Yes. There is a post-exit settling period, but stable forward motion resumes after `crossing_release`; this is not best explained as a persistent post-recovery walking failure.

## Part C — Is a fresh FL hip–wire ablation justified?

Verified facts:
- A technically valid pair-specific hip-wire isolation mechanism already exists in [scripts/d4_8_hip_wire_contact_ablation.py](../../scripts/d4_8_hip_wire_contact_ablation.py): `add_exclude(bodyname1="robot/FL_hip", bodyname2="wire_seg_i")`.
- The Go2 model in [src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml](../../src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml) defines a dedicated `FL_hip_collision` geom, so the isolation mechanism is specific rather than broad.
- In the current baseline, hip-wire contact begins at step 51, after:
  - calf-wire contact at step 43,
  - recovery entry at step 44,
  - pitch-rate instability at step 45, and
  - body-height collapse onset at step 47.

Decision:
- A fresh FL hip–wire ablation is **not justified** for the user’s root-cause question.
- It would be valid as an isolation mechanism, but it cannot explain the onset of the initial instability because the hip is not the first contacting geom and does not contact until after the instability is already in progress.
- Per the task constraints, no new ablation run was created.

## Summary answers to the user’s explicit questions

1. First robot geom to contact wire:
   - `robot/FL_calf2_collision` at step 43.
2. Does FL hip contact, and does it persist?
   - Yes; first at step 51, then 51-52 and 54-73.
3. First measurable body instability:
   - Step 45 by base pitch-rate excursion; step 47 by base-height excursion.
4. When does recovery activate relative to contact and instability?
   - Contact at 43, recovery at 44, measurable instability at 45/47.
5. Does instability begin before recovery or worsen during recovery?
   - The substantial instability worsens during recovery.
6. Does stable forward motion resume after `crossing_release`?
   - Yes; after a settling transient, stable forward motion resumes.

## Conclusion

Evidence supports instability introduced or worsened by the recovery maneuver.
