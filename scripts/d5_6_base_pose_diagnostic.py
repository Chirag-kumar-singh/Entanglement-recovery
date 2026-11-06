"""D5.6 -- Base-pose diagnostic (why does seed 7's calf_forward_dist
plateau just short of zero while seed 0's crosses, despite near-identical
commanded joint motion?).

D5.6's seed7-vs-seed0 comparison (recovery-entry-aligned) found:
  - Commands (`thigh_vel_cmd`, `calf_torque_cmd`) are identical, constant
    open-loop values in both seeds.
  - `thigh_pos`/`calf_pos` (recovery-leg joint angles) track closely
    between the two seeds, with seed 7 if anything slightly MORE
    forward than seed 0 at matching relative steps.
  - Yet `calf_forward_dist` diverges starting around rel-step 20-30:
    seed 0 climbs past 0, seed 7 plateaus at ~-0.01 to -0.04 for the
    entire window.

Since `calf_forward_dist` is a signed distance of the CALF GEOM'S WORLD
XY position relative to a WIRE-PLANE projection, using a FORWARD
DIRECTION LATCHED AT RECOVERY ENTRY (see
`mjlab.tasks.recovery.mdp.actions._update_fsm`,
`wire_geometry.signed_forward_distance`), a divergence in this quantity
despite near-identical joint angles implicates the ROBOT BASE pose
(position and/or heading) relative to the wire, not the leg's own
motion.

THIS SCRIPT IS PURELY DIAGNOSTIC / ADDITIVE:
  - It is a byte-for-byte copy of `d5_3_extended_recovery_window.py`'s
    env-build and rollout logic (same `build_ablation_env`, same
    deterministic zero-residual-action rollout, same config knobs,
    same seeds/conditions).
  - The ONLY change is that 7 extra columns are added to the per-step
    CSV row, all of which are READS of quantities the action term
    (`act`) and entity data ALREADY COMPUTE internally every step
    (`act._forward_dir_xy`, `entity.data.root_link_pos_w`,
    `entity.data.heading_w`, `entity.data.geom_pos_w[..., calf_geom_idx]`).
    No new computation is introduced, no controller/FSM/gain/threshold
    is touched, and nothing is fed back into the action pipeline --
    these are read-only diagnostic observations appended to the log.
  - Output is written to a SEPARATE directory
    (`logs/d5_6_diagnostics/seed{N}/`) so none of the existing D5.3/D5.5
    results are touched or overwritten.

New columns:
  - base_x, base_y: robot base (root link) world XY position.
  - base_heading: robot's current world-frame heading (radians),
    recomputed every step from `entity.data.heading_w` (NOT the
    latched-at-entry value -- see `fwd_dir_x`/`fwd_dir_y` below for
    that).
  - calf_geom_x, calf_geom_y: calf geom world XY position (the same
    point used to compute `calf_forward_dist`).
  - fwd_dir_x, fwd_dir_y: the forward-direction unit vector LATCHED at
    the instant of recovery entry (`act._forward_dir_xy`), i.e. the
    actual reference direction used by `calf_forward_dist`'s
    projection.

D5.6 FOLLOW-UP (same run, same script): the first diagnostic pass found
that seed 0's base resumes forward translation after wire release while
seed 7's stalls/reverses, with greater net yaw drift. To test whether
this reflects a disturbance/orientation issue reaching the FROZEN
WALKING POLICY itself (rather than the recovery leg), the following
columns log EXACTLY the signals `act._build_walk_obs` feeds to that
policy (`act._imu_lin_vel.data`, `act._imu_ang_vel.data`,
`entity.data.projected_gravity_b`, `env.command_manager.get_command(...)`)
-- again, pure reads of already-computed internal state, nothing new is
computed and nothing is fed back into control:
  - imu_lin_vel_x/y/z: base linear velocity in the IMU/policy frame.
  - imu_ang_vel_x/y/z: base angular velocity in the IMU/policy frame.
  - projected_gravity_x/y/z: gravity vector in the base frame (tracks
    body tilt/orientation).
  - vel_cmd_x/y/yaw: the velocity command term the frozen policy is
    trying to track.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_3_extended_recovery_window import build_ablation_env  # noqa: E402


@dataclass
class D56Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 200
  max_recovery_steps_override: int = 48
  ablate_hip_wire: bool = False
  out_dir: str = ""


def run(cfg: D56Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_ablation_env(
    d3_cfg, device, cfg.ablate_hip_wire, cfg.max_recovery_steps_override
  )
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  mj_model = env.sim.mj_model

  def geom_name(gid: int) -> str:
    if gid < 0:
      return "<none>"
    n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom#{gid}"

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  state = {
    "enter_line": None,
    "exit_line": None,
    "stage1_first_step": None,
    "stage2_first_step": None,
  }

  def do_step(step: int, action: torch.Tensor) -> dict:
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    prev_phase = int(act.phase[0])
    env.step(action)
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      state["exit_line"] = (
        f"EXIT RECOVERY step={step} elapsed={int(act.recovery_elapsed[0])} "
        f"timeout={bool(act.recovery_failed_timeout[0])}"
      )

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())
    fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    fl_ground_found = bool((fl_ground.data.found[0].reshape(-1) > 0).any())
    fr_wire = env.scene.sensors["fr_wire_contact_diag"]
    fr_wire_found = bool((fr_wire.data.found[0].reshape(-1) > 0).any())

    contact = env.sim.data.contact
    nacon = int(env.sim.data.nacon[0])
    hip_wire_contact_now = False
    calf_wire_contact_now = False
    for ci in range(nacon):
      if int(contact.worldid[ci]) != 0:
        continue
      n0, n1 = geom_name(int(contact.geom[ci][0])), geom_name(int(contact.geom[ci][1]))
      pair = (n0, n1)
      if any("FL_hip_collision" in n for n in pair) and any("wire_seg" in n for n in pair):
        hip_wire_contact_now = True
      if any("FL_calf2_collision" in n for n in pair) and any("wire_seg" in n for n in pair):
        calf_wire_contact_now = True

    calf_forward_dist = float(act.calf_forward_dist[0])
    calf_height_above_wire = float(act.calf_height_above_wire[0])
    calf_side_far = bool(act.calf_side_far[0])
    foot_forward_dist = float(act.foot_forward_dist[0])
    calf_clearance_achieved = bool(act._calf_clearance_achieved[0])
    foot_crossed_achieved = bool(act._foot_crossed_achieved[0])
    crossing_achieved = bool(act._crossing_achieved[0])
    release_counter = int(act._release_counter[0])
    recovery_failed_timeout_flag = bool(act._recovery_failed_timeout[0])
    released_sustained = crossing_achieved and (
      release_counter >= act.cfg.release_required_steps
    )
    genuine_recovery_complete = crossing_achieved and released_sustained

    # --- D5.6 diagnostic-only additions: read-only, already-computed
    # internal state, NOT fed back into control. ---
    calf_geom_xy = entity.data.geom_pos_w[0, act._calf_geom_idx, :2].clone()
    fwd_dir_xy = act._forward_dir_xy[0].clone()
    current_heading = float(entity.data.heading_w[0])

    # D5.6 follow-up: exactly the signals the frozen walking policy
    # itself observes (same attributes read in `act._build_walk_obs`),
    # read-only, not fed back into control.
    imu_lin_vel = act._imu_lin_vel.data[0].clone()
    imu_ang_vel = act._imu_ang_vel.data[0].clone()
    projected_gravity_b = entity.data.projected_gravity_b[0].clone()
    vel_command = env.command_manager.get_command(act.cfg.command_name)[0].clone()

    if calf_clearance_achieved and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if foot_crossed_achieved and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step

    return {
        "step": step,
        "phase": new_phase,
        "recovery_elapsed": int(act.recovery_elapsed[0]),
        "foot_z": float(foot_pos[2]),
        "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
        "foot_height_above_wire": float(act.foot_height_above_wire[0]),
        "hip_pos": float(qpos[hip_i]),
        "thigh_pos": float(qpos[thigh_i]),
        "thigh_vel": float(qvel[thigh_i]),
        "calf_pos": float(qpos[calf_i]),
        "calf_vel": float(qvel[calf_i]),
        "base_z": float(base_pos[2]),
        "base_quat_w": float(base_quat[0]),
        "wire_in_contact": wire_in_contact,
        "fl_ground_contact": fl_ground_found,
        "fr_wire_contact": fr_wire_found,
        "fl_hip_wire_contact_raw": hip_wire_contact_now,
        "fl_calf_wire_contact_raw": calf_wire_contact_now,
        "pos_thigh_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._pos_thigh_ctrl_id]
        ),
        "aux_thigh_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._aux_thigh_ctrl_id]
        ),
        "pos_calf_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._pos_calf_ctrl_id]
        ),
        "aux_calf_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._aux_calf_ctrl_id]
        ),
        "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
        "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
        "calf_forward_dist": calf_forward_dist,
        "calf_height_above_wire": calf_height_above_wire,
        "calf_side_far": calf_side_far,
        "foot_forward_dist": foot_forward_dist,
        "calf_clearance_achieved": calf_clearance_achieved,
        "foot_crossed_achieved": foot_crossed_achieved,
        "crossing_achieved": crossing_achieved,
        "release_counter": release_counter,
        "recovery_failed_timeout_flag": recovery_failed_timeout_flag,
        "released_sustained": released_sustained,
        "genuine_recovery_complete": genuine_recovery_complete,
        # D5.6 additions:
        "base_x": float(base_pos[0]),
        "base_y": float(base_pos[1]),
        "base_heading": current_heading,
        "calf_geom_x": float(calf_geom_xy[0]),
        "calf_geom_y": float(calf_geom_xy[1]),
        "fwd_dir_x": float(fwd_dir_xy[0]),
        "fwd_dir_y": float(fwd_dir_xy[1]),
        "imu_lin_vel_x": float(imu_lin_vel[0]),
        "imu_lin_vel_y": float(imu_lin_vel[1]),
        "imu_lin_vel_z": float(imu_lin_vel[2]),
        "imu_ang_vel_x": float(imu_ang_vel[0]),
        "imu_ang_vel_y": float(imu_ang_vel[1]),
        "imu_ang_vel_z": float(imu_ang_vel[2]),
        "projected_gravity_x": float(projected_gravity_b[0]),
        "projected_gravity_y": float(projected_gravity_b[1]),
        "projected_gravity_z": float(projected_gravity_b[2]),
        "vel_cmd_x": float(vel_command[0]),
        "vel_cmd_y": float(vel_command[1]),
        "vel_cmd_yaw": float(vel_command[2]),
    }

  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    rows.append(do_step(step, action))

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"ablate_hip_wire={cfg.ablate_hip_wire}\n")
    f.write(f"max_recovery_steps_override={cfg.max_recovery_steps_override}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
  print(f"[D5.6] wrote {csv_path}")
  print(f"  {state['enter_line']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D56Config)
  run(cfg)


if __name__ == "__main__":
  main()
