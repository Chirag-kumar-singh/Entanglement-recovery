D5.47 — Seed 11 Hybrid Actuation and Trajectory Feasibility

Method
- Read-only replay on the frozen hybrid baseline with the same checkpoint, wire geometry, and seed 0 / seed 11 setup.
- Logged raw MuJoCo `ctrl`, actuator force, joint state, wire contact, wire contact force, and wire-relative geometry per simulation step in `logs/d5_47/seed0_trace.csv` and `logs/d5_47/seed11_trace.csv`.
- Built `logs/d5_47/aligned_trace.csv` on recovery-relative time so the shared recovery maneuver could be compared directly.

What the code actually does
- During normal walking, FL hip/thigh/calf are all standard position actuators.
- During recovery, the FL thigh and calf position actuators are disabled by zeroing their gain/bias, and the recovery path switches to the auxiliary thigh velocity actuator plus the auxiliary calf motor.
- The FL hip position actuator stays on the frozen walking-policy path throughout recovery.
- The trace confirms that the recovery control values themselves are the intended ones: shared recovery-elapsed steps 1–44 have identical `fl_thigh_vel_ctrl = +2.0` and `fl_calf_motor_ctrl = -5.0` on both seeds, with no control mismatches in the shared recovery window.
- The same trace also confirms that the affected thigh/calf position actuators are not competing with the auxiliary path during recovery: `hybrid_thigh_pos_gain_active = False`, `fl_thigh_pos_gain = 0.0`, and `fl_calf_pos_gain = 0.0` in recovery.

First meaningful divergence
- The divergence is present at recovery entry, not in the control law.
- Seed 0 enters recovery at step 44 with wire contact still true, base forward velocity `+0.322 m/s`, base vertical velocity `-0.030 m/s`, calf forward distance `-0.0286 m`, and toe forward distance `+0.0072 m`.
- Seed 11 enters recovery at step 51 with wire contact already false on the first recorded recovery step, lower base forward velocity `+0.244 m/s`, much more negative base vertical velocity `-0.146 m/s`, calf forward distance `-0.0270 m`, and toe forward distance `+0.0019 m`.
- At shared recovery-elapsed step 8, the same actuator commands produce very different motion: seed 0 still has `base_lin_vel_b_x = +0.473 m/s` and `calf_forward_dist = +0.0109 m`, while seed 11 has `base_lin_vel_b_x = -0.102 m/s` and `calf_forward_dist = -0.0235 m`.
- Seed 0 eventually gets the calf across: first positive calf-forward step is 84, and the run reaches crossing/release.
- Seed 11 never gets the calf forward distance positive in the recorded window; its best calf-forward distance stays at `-0.0124 m`.
- The toe can be on the far side while the calf remains behind in both runs, so toe advance alone is not the missing ingredient.

Limits and saturation
- No actuator saturation is shown for the recovery actuators.
- The thigh velocity actuator force peaks at about `1.88 N·m` on seed 0 and `2.36 N·m` on seed 11, far below the stock thigh effort limit of `23.5 N·m`.
- The calf motor force stays at the commanded `5.0 N·m`, far below the calf effort limit of `45 N·m`.
- There is no evidence in this trace that the auxiliary actuators are being clipped before they can do work.

Interpretation
- The measurements do not support an implementation defect in the recovery actuator routing: the disabled position actuators are zeroed, the auxiliary thigh velocity and calf motor are active, and the commanded recovery values are identical across the shared recovery window.
- The measurements do not support a torque-authority failure: the actuators are not saturating.
- The main difference is the starting state at recovery entry. Seed 11 begins from a worse body state, with less forward speed and a much larger downward vertical velocity, and the same maneuver loses useful forward progress almost immediately.
- That is enough to explain why the same hybrid commands succeed on seed 0 but do not carry seed 11’s calf across the wire.

Recommendation
Use one controlled replay that starts seed 11 from seed 0’s recovery-entry body/joint state, while keeping the same hybrid commands, wire geometry, and checkpoint. If that replay crosses, the failure is entry-state sensitivity; if it still fails, the maneuver itself is mechanically insufficient from that state.

C. Test entry-state sensitivity.