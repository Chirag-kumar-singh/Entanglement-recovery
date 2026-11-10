"""D3 diagnostic: 2D FL thigh x calf kinematic foot-height map, overlaid
with the actual D3 baseline (thigh(t), calf(t)) trajectory.

PURELY DIAGNOSTIC. Does not modify the recovery controller, FSM, wire
geometry, or any default config/parameter.

Part A: re-run the exact D3 baseline trajectory (same as
  d3_thigh_geometry_diagnostic.py's Part 1) to get (thigh(t), calf(t))
  and the actual observed ranges.

Part B: build the SAME compiled Go2 recovery-task model used by the live
  experiment, hold base pose + FL ab/ad (hip) + all other joints at a
  representative recovery instant, and sweep FL thigh x FL calf jointly
  over a grid spanning the ACTUAL observed ranges from Part A. For each
  (thigh, calf) grid point, record foot x/y/z via mj_forward (kinematics
  only).

Part C: analyze the grid (global max, per-calf argmax-thigh, sign-change
  calf angle) and overlay the real trajectory on top of it to determine
  whether/where the trajectory enters then exits the high-clearance
  region.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY

SEED = 0
MAX_STEPS = 400
STOP_AFTER_POST_RECOVERY = 60
N_GRID = 40  # per axis


def run_baseline():
  torch.manual_seed(SEED)
  device = "cpu"
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
  )
  cfg.seed = SEED
  cfg.sim.device = device
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  act = env.action_manager._terms["joint_pos"]
  env.reset(seed=SEED)

  entity = act._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index("FL_hip_joint")
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")

  rows = []
  reached_post_recovery_at = None
  for i in range(MAX_STEPS):
    action = torch.zeros(1, act.action_dim, device=device)
    env.step(action)
    qpos = entity.data.joint_pos[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    phase = int(act.phase[0])
    rows.append(
      dict(
        step=i,
        phase=phase,
        recovery_elapsed=int(act.recovery_elapsed[0])
        if phase == PHASE_RECOVERY
        else -1,
        hip_pos=float(qpos[hip_i]),
        thigh_pos=float(qpos[thigh_i]),
        calf_pos=float(qpos[calf_i]),
        foot_x=float(foot_pos[0]),
        foot_z=float(foot_pos[2]),
        clearance=float(act.foot_height_above_wire[0]),
      )
    )
    if phase == 2 and reached_post_recovery_at is None:
      reached_post_recovery_at = i
    if (
      reached_post_recovery_at is not None
      and i - reached_post_recovery_at >= STOP_AFTER_POST_RECOVERY
    ):
      break

  env.close()
  return rows


def build_kinematic_model():
  device = "cpu"
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
  )
  cfg.seed = SEED
  cfg.sim.device = device
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  act = env.action_manager._terms["joint_pos"]
  env.reset(seed=SEED)
  entity = act._entity

  mj_model = env.sim.mj_model
  mj_data = mujoco.MjData(mj_model)
  mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  mj_data.qvel[:] = 0.0
  mujoco.mj_forward(mj_model, mj_data)

  entity_site_names = list(entity.site_names)
  foot_entity_local = entity_site_names[act._foot_site_idx]
  foot_site_name = None
  for i in range(mj_model.nsite):
    full_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
    if full_name and full_name.endswith("/" + foot_entity_local):
      foot_site_name = full_name
      break

  env.close()
  return mj_model, mj_data, foot_site_name


def joint_qpos_adr(mj_model, name):
  jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if jid < 0:
    raise ValueError(f"Joint {name} not found.")
  return mj_model.jnt_qposadr[jid]


def foot_world_pos(mj_model, mj_data, site_name):
  sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
  return mj_data.site_xpos[sid].copy()


def main():
  out_dir = Path("logs/d3_geometry_diagnostic")
  out_dir.mkdir(parents=True, exist_ok=True)

  print("=== Part A: re-running D3 baseline for (thigh(t), calf(t)) ===")
  rows = run_baseline()
  rec_rows = [r for r in rows if r["phase"] == PHASE_RECOVERY]
  thigh_vals = [r["thigh_pos"] for r in rec_rows]
  calf_vals = [r["calf_pos"] for r in rec_rows]
  hip_rep = rec_rows[len(rec_rows) // 2]["hip_pos"]
  heights = [r["foot_z"] for r in rec_rows]
  peak_idx = heights.index(max(heights))

  thigh_lo, thigh_hi = min(thigh_vals), max(thigh_vals)
  calf_lo, calf_hi = min(calf_vals), max(calf_vals)
  print(f"Actual thigh range: {thigh_lo:.4f} to {thigh_hi:.4f} rad")
  print(f"Actual calf  range: {calf_lo:.4f} to {calf_hi:.4f} rad")
  print(f"Hip held at (mid-recovery value): {hip_rep:.4f} rad")

  print("\n=== Part B: building kinematic model + sweeping thigh x calf grid ===")
  mj_model, mj_data, foot_site_name = build_kinematic_model()
  print(f"Using foot site: {foot_site_name}")

  hip_adr = joint_qpos_adr(mj_model, "robot/FL_hip_joint")
  thigh_adr = joint_qpos_adr(mj_model, "robot/FL_thigh_joint")
  calf_adr = joint_qpos_adr(mj_model, "robot/FL_calf_joint")
  mj_data.qpos[hip_adr] = hip_rep

  thighs = np.linspace(thigh_lo, thigh_hi, N_GRID)
  calfs = np.linspace(calf_lo, calf_hi, N_GRID)
  foot_z_grid = np.zeros((N_GRID, N_GRID))  # [calf_idx, thigh_idx]
  foot_x_grid = np.zeros((N_GRID, N_GRID))

  grid_rows = []
  for ci, cv in enumerate(calfs):
    mj_data.qpos[calf_adr] = cv
    for ti, tv in enumerate(thighs):
      mj_data.qpos[thigh_adr] = tv
      mujoco.mj_forward(mj_model, mj_data)
      pos = foot_world_pos(mj_model, mj_data, foot_site_name)
      foot_z_grid[ci, ti] = pos[2]
      foot_x_grid[ci, ti] = pos[0]
      grid_rows.append(
        dict(thigh=tv, calf=cv, foot_x=pos[0], foot_y=pos[1], foot_z=pos[2])
      )

  with open(out_dir / "thigh_calf_grid.csv", "w", newline="") as f:
    writer = csv.DictWriter(
      f, fieldnames=["thigh", "calf", "foot_x", "foot_y", "foot_z"]
    )
    writer.writeheader()
    writer.writerows(grid_rows)
  print(f"Grid CSV written: {out_dir / 'thigh_calf_grid.csv'} ({len(grid_rows)} rows)")

  print("\n=== Part C: analysis ===")
  flat_max_idx = np.unravel_index(np.argmax(foot_z_grid), foot_z_grid.shape)
  global_max_z = foot_z_grid[flat_max_idx]
  max_calf, max_thigh = calfs[flat_max_idx[0]], thighs[flat_max_idx[1]]
  print(f"1. Global max foot_z in observed region = {global_max_z:.4f} m")
  print(f"2. At thigh = {max_thigh:.4f} rad, calf = {max_calf:.4f} rad")

  print("\n3. For each calf row, argmax-thigh and whether it's at the high or low end:")
  argmax_thigh_per_calf = []
  for ci, cv in enumerate(calfs):
    row = foot_z_grid[ci, :]
    amax = np.argmax(row)
    argmax_thigh_per_calf.append(thighs[amax])
    if ci % 5 == 0 or ci == N_GRID - 1:
      end = (
        "LOW-end"
        if amax < N_GRID // 4
        else ("HIGH-end" if amax > 3 * N_GRID // 4 else "MID")
      )
      print(
        f"   calf={cv:+.4f}: argmax thigh={thighs[amax]:+.4f} ({end}), max_z={row[amax]:.4f}"
      )

  # Sign-change calf angle: find where argmax-thigh crosses from
  # high-end to low-end (or vice versa) as calf becomes more flexed.
  sign_change_calf = None
  for ci in range(1, N_GRID):
    prev_high_end = argmax_thigh_per_calf[ci - 1] > 0.5 * (thigh_lo + thigh_hi)
    cur_high_end = argmax_thigh_per_calf[ci] > 0.5 * (thigh_lo + thigh_hi)
    if prev_high_end != cur_high_end:
      sign_change_calf = 0.5 * (calfs[ci - 1] + calfs[ci])
      break
  print(
    f"\n4. Approximate calf angle where thigh-effect sign flips: "
    f"{sign_change_calf if sign_change_calf is not None else 'no flip detected in this range'}"
  )

  # Overlay: for each recovery row, interpolate the grid's foot_z at
  # that EXACT (thigh, calf) to see the kinematic-predicted height, vs
  # the actual dynamic foot_z (these differ because of velocity/momentum
  # effects, but should track the same qualitative trend).
  print("\n5/6/7. Overlay of actual trajectory on the grid (recovery steps):")
  print(
    f"{'elapsed':>7} {'thigh':>8} {'calf':>8} {'actual_z':>9} {'kin_pred_z':>11} {'in_top25pct':>12}"
  )
  z_thresh_top25 = np.percentile(foot_z_grid, 75)
  entered_high = False
  exit_step = None
  for r in rec_rows:
    # Nearest-grid-point lookup (grid is fine enough, N_GRID=40, for a
    # diagnostic -- no need for full bilinear interpolation).
    ti = int(np.clip(np.searchsorted(thighs, r["thigh_pos"]), 0, N_GRID - 1))
    ci = int(np.clip(np.searchsorted(calfs, r["calf_pos"]), 0, N_GRID - 1))
    kin_z = foot_z_grid[ci, ti]
    in_top25 = kin_z >= z_thresh_top25
    if in_top25 and not entered_high:
      entered_high = True
    if entered_high and not in_top25 and exit_step is None:
      exit_step = r["recovery_elapsed"]
    print(
      f"{r['recovery_elapsed']:>7} {r['thigh_pos']:>8.4f} {r['calf_pos']:>8.4f} "
      f"{r['foot_z']:>9.4f} {kin_z:>11.4f} {str(in_top25):>12}"
    )

  print(f"\nTop-25th-percentile foot_z threshold (this grid) = {z_thresh_top25:.4f} m")
  print(f"Trajectory entered top-25% region: {entered_high}")
  print(f"Trajectory exited top-25% region at recovery_elapsed = {exit_step}")

  print("\n==== SUMMARY ====")
  print(f"Thigh range used: {thigh_lo:.4f} to {thigh_hi:.4f}")
  print(f"Calf  range used: {calf_lo:.4f} to {calf_hi:.4f}")
  print(
    f"Global max foot_z = {global_max_z:.4f} at thigh={max_thigh:.4f}, calf={max_calf:.4f}"
  )
  print(f"Sign-change calf angle ~ {sign_change_calf}")
  print(
    f"Trajectory enters high-clearance region: {entered_high}, exits at elapsed={exit_step}"
  )

  # Save representative points for the visualization script.
  rep_points = {
    "A_early": (rec_rows[0]["thigh_pos"], rec_rows[0]["calf_pos"], hip_rep),
    "B_near_peak": (
      rec_rows[peak_idx]["thigh_pos"],
      rec_rows[peak_idx]["calf_pos"],
      hip_rep,
    ),
    "C_post_peak": (
      rec_rows[min(peak_idx + 4, len(rec_rows) - 1)]["thigh_pos"],
      rec_rows[min(peak_idx + 4, len(rec_rows) - 1)]["calf_pos"],
      hip_rep,
    ),
    "D_predicted_best": (max_thigh, max_calf, hip_rep),
  }
  with open(out_dir / "representative_points.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "thigh", "calf", "hip"])
    for label, (th, cf, hp) in rep_points.items():
      writer.writerow([label, th, cf, hp])
  print(f"\nRepresentative points written: {out_dir / 'representative_points.csv'}")
  for label, (th, cf, hp) in rep_points.items():
    print(f"  {label}: thigh={th:.4f}, calf={cf:.4f}, hip={hp:.4f}")


if __name__ == "__main__":
  main()
