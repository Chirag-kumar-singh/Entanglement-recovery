# D5.48 — Aligned Event Timeline

| Step | Category | Event | Evidence |
|---|---|---|---|
| 43 | Contact | First robot-wire contact | `wire_seg_2_geom <-> robot/FL_calf2_collision` in [logs/d4_6_support_descent_timing/seed0/contact_listing.txt](../d4_6_support_descent_timing/seed0/contact_listing.txt) |
| 44 | FSM | Recovery enters | `phase` first becomes `1` in [logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv](../d5_16_trajectory_audit/headless_seed0/trajectory.csv) |
| 45 | Instability | First body-rate excursion | `base_ang_vel_y=0.3742` exceeds the pre-contact max `0.3664` |
| 47 | Instability | First body-height excursion | `base_z=0.2988` falls below the pre-contact min `0.3038` |
| 48 | Contact | Short recontact block | `wire_in_contact=True` and `fl_calf_wire_contact_raw=True` |
| 51 | Contact | First FL hip-wire contact | `fl_hip_wire_contact_raw=True`; contact listing shows `wire_seg_3/4_geom <-> robot/FL_hip_collision` |
| 52 | Contact | Base also touches wire | contact listing shows `wire_seg_0_geom <-> robot/base1_collision` |
| 54 | Instability | Global body-height minimum | `base_z=0.2040` in [logs/d5_16_trajectory_audit/headless_seed0/trajectory.csv](../d5_16_trajectory_audit/headless_seed0/trajectory.csv) |
| 84 | Crossing | Calf clearance and foot crossing latch | `calf_side_far=True`, `foot_crossed_achieved=True`, `crossing_achieved=True` |
| 88 | Release | Release threshold satisfied and recovery exits | `release_counter` reaches `5`; `phase` first becomes `2` |
| 89 | Post | First FL ground contact after exit | reported in [logs/d5_16_trajectory_audit/report.md](../d5_16_trajectory_audit/report.md) |
| 92 | Post | Secondary post-handoff dip | reported in [logs/d5_16_trajectory_audit/report.md](../d5_16_trajectory_audit/report.md) |
| 109 | Post | Stable walking-like window begins | first 10-step post-exit window with `base_z` back in its pre-contact range and `base_lin_vel_x` in `0.2..0.35 m/s` |

## Key causal ordering

1. Calf-wire contact starts first.
2. Recovery activates one step later.
3. Measurable body instability appears after recovery entry.
4. FL hip-wire contact starts later, after the instability is already underway.
5. Crossing and sustained release succeed.
6. Walking stabilizes again after the post-handoff transient.
