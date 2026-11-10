"""D3 geometry diagnostic (Parts 1-3 of the thigh-geometry investigation).

PURELY DIAGNOSTIC. Does not modify the recovery controller, FSM, or any
default config/parameter. Three sub-experiments:

Part 1: re-run the exact D3 baseline (identical to
  scripts/d3_calf_torque_ablation.py's baseline path) and extract the
  hip/thigh/calf/foot trajectory around the foot-height peak, this time
  ALSO logging the FL ab/ad (hip) joint (not logged in the ablation
  script).

Part 2: controlled kinematic sweep. Loads the actual compiled Go2 model
  used by the recovery env, sets the base pose and all OTHER joints to
  their values at a representative recovery instant (captured from the
  live Part-1 run), then sweeps ONLY the FL thigh joint across the
  actual observed range from Part 1 (not an assumed range), calling
  `mj_forward` (kinematics only, no physics stepping) and reading the
  resulting foot site position at each angle. Repeated at three
  representative calf angles (early/peak/late recovery) to separate
  thigh-only geometry from thigh-calf coupling.

Part 3: same sweep methodology, but for the FL ab/ad (hip) joint, with
  thigh/calf held at representative recovery configurations.
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


def run_baseline_with_hip():
  """Part 1: identical baseline run to d3_calf_torque_ablation.py's
  baseline, with the FL ab/ad (hip) joint additionally logged."""
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
    qvel = entity.data.joint_vel[0]
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
        hip_vel=float(qvel[hip_i]),
        thigh_pos=float(qpos[thigh_i]),
        thigh_vel=float(qvel[thigh_i]),
        calf_pos=float(qpos[calf_i]),
        calf_vel=float(qvel[calf_i]),
        foot_x=float(foot_pos[0]),
        foot_y=float(foot_pos[1]),
        foot_z=float(foot_pos[2]),
        clearance=float(act.foot_height_above_wire[0]),
      )
    )
    if phase == 2 and reached_post_recovery_at is None:  # PHASE_POST_RECOVERY
      reached_post_recovery_at = i
    if (
      reached_post_recovery_at is not None
      and i - reached_post_recovery_at >= STOP_AFTER_POST_RECOVERY
    ):
      break

  # Snapshot full qpos/qvel at a representative mid-recovery instant,
  # for Part 2/3 kinematic sweeps (captured at the peak-foot-height step
  # itself, so "all other joints" reflects a genuine live recovery
  # configuration, not an arbitrary guess).
  rec_rows = [r for r in rows if r["phase"] == PHASE_RECOVERY]
  heights = [r["foot_z"] for r in rec_rows]
  peak_idx = heights.index(max(heights))
  peak_step = rec_rows[peak_idx]["step"]

  full_qpos_snapshot = entity.data.joint_pos[0].clone()  # last state (end of run)
  full_qvel_snapshot = entity.data.joint_vel[0].clone()

  env.close()
  return rows, rec_rows, peak_idx, joint_names, hip_i, thigh_i, calf_i


def build_kinematic_model():
  """Build the SAME compiled Go2 recovery-task model (hybrid mode) used
  by the live experiment, for kinematics-only sweeps (mj_forward, no
  stepping/physics/control)."""
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

  mj_model = env.sim.mj_model
  mj_data = mujoco.MjData(mj_model)

  # Seed mj_data from the live env's current (post-reset) qpos/qvel so
  # the base pose / all-other-joint values are genuinely representative,
  # not arbitrary defaults.
  mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  mj_data.qvel[:] = 0.0
  mujoco.mj_forward(mj_model, mj_data)

  # Resolve the actual MuJoCo-level site name (mjlab prefixes entity
  # element names with the entity's spec name, e.g. "robot/FL" -- NOT
  # the bare entity-local name "FL" used by `entity.site_names`).
  foot_site_names_entity = (
    list(entity.site_names) if hasattr(entity, "site_names") else None
  )
  foot_site_name = None
  if foot_site_names_entity is not None:
    entity_local_name = foot_site_names_entity[act._foot_site_idx]
    for i in range(mj_model.nsite):
      full_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
      if full_name and full_name.endswith("/" + entity_local_name):
        foot_site_name = full_name
        break

  env_info = dict(
    mj_model=mj_model,
    mj_data=mj_data,
    joint_names=joint_names,
    resolved_foot_site_name=foot_site_name,
  )
  env.close()
  return env_info


def joint_qpos_adr(mj_model, entity_joint_name_in_spec: str):
  """Resolve the MuJoCo-level qpos address for a named joint. Entity
  joint names in mjlab are typically the same as the MjSpec joint name
  for single-entity scenes, but we resolve robustly via mj_name2id."""
  jid = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_JOINT, entity_joint_name_in_spec
  )
  if jid < 0:
    raise ValueError(f"Joint {entity_joint_name_in_spec} not found in compiled model.")
  return mj_model.jnt_qposadr[jid]


def foot_site_world_pos(mj_model, mj_data, site_name: str) -> np.ndarray:
  sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
  if sid < 0:
    raise ValueError(f"Site {site_name} not found.")
  return mj_data.site_xpos[sid].copy()


def sweep_thigh(
  mj_model,
  mj_data,
  calf_val: float,
  hip_val: float,
  thigh_range,
  foot_site_name: str,
  n=40,
):
  hip_adr = joint_qpos_adr(mj_model, "robot/FL_hip_joint")
  thigh_adr = joint_qpos_adr(mj_model, "robot/FL_thigh_joint")
  calf_adr = joint_qpos_adr(mj_model, "robot/FL_calf_joint")

  mj_data.qpos[hip_adr] = hip_val
  mj_data.qpos[calf_adr] = calf_val

  thighs = np.linspace(thigh_range[0], thigh_range[1], n)
  results = []
  for th in thighs:
    mj_data.qpos[thigh_adr] = th
    mujoco.mj_forward(mj_model, mj_data)
    pos = foot_site_world_pos(mj_model, mj_data, foot_site_name)
    results.append((th, pos[0], pos[1], pos[2]))
  return results


def sweep_hip(
  mj_model,
  mj_data,
  thigh_val: float,
  calf_val: float,
  hip_range,
  foot_site_name: str,
  n=30,
):
  hip_adr = joint_qpos_adr(mj_model, "robot/FL_hip_joint")
  thigh_adr = joint_qpos_adr(mj_model, "robot/FL_thigh_joint")
  calf_adr = joint_qpos_adr(mj_model, "robot/FL_calf_joint")

  mj_data.qpos[thigh_adr] = thigh_val
  mj_data.qpos[calf_adr] = calf_val

  hips = np.linspace(hip_range[0], hip_range[1], n)
  results = []
  for hp in hips:
    mj_data.qpos[hip_adr] = hp
    mujoco.mj_forward(mj_model, mj_data)
    pos = foot_site_world_pos(mj_model, mj_data, foot_site_name)
    results.append((hp, pos[0], pos[1], pos[2]))
  return results


def main():
  out_dir = Path("logs/d3_geometry_diagnostic")
  out_dir.mkdir(parents=True, exist_ok=True)

  print("=== PART 1: re-running D3 baseline (with hip logged) ===")
  rows, rec_rows, peak_idx, joint_names, hip_i, thigh_i, calf_i = (
    run_baseline_with_hip()
  )

  thigh_vals = [r["thigh_pos"] for r in rec_rows]
  calf_vals = [r["calf_pos"] for r in rec_rows]
  hip_vals = [r["hip_pos"] for r in rec_rows]
  print(
    f"Actual thigh range during recovery: {min(thigh_vals):.4f} to {max(thigh_vals):.4f} rad"
  )
  print(
    f"Actual calf  range during recovery: {min(calf_vals):.4f} to {max(calf_vals):.4f} rad"
  )
  print(
    f"Actual hip   range during recovery: {min(hip_vals):.4f} to {max(hip_vals):.4f} rad"
  )

  lo = max(0, peak_idx - 3)
  hi = min(len(rec_rows), peak_idx + 11)
  print("\n--- Trajectory table around foot-height peak ---")
  print(
    f"{'elapsed':>7} {'hip':>8} {'thigh':>8} {'calf':>8} {'foot_x':>8} {'foot_z':>8} {'clearance':>10}"
  )
  for i in range(lo, hi):
    r = rec_rows[i]
    marker = "  <== PEAK" if i == peak_idx else ""
    print(
      f"{r['recovery_elapsed']:>7} {r['hip_pos']:>8.4f} {r['thigh_pos']:>8.4f} {r['calf_pos']:>8.4f} "
      f"{r['foot_x']:>8.4f} {r['foot_z']:>8.4f} {r['clearance']:>10.4f}{marker}"
    )

  with open(out_dir / "baseline_with_hip.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  # Representative calf angles: early / peak / late recovery.
  early_i = 0
  peak_i_local = peak_idx
  late_i = len(rec_rows) - 1
  calf_early = rec_rows[early_i]["calf_pos"]
  calf_peak = rec_rows[peak_i_local]["calf_pos"]
  calf_late = rec_rows[late_i]["calf_pos"]
  hip_rep = rec_rows[peak_i_local][
    "hip_pos"
  ]  # hold hip at its peak-instant value for Part 2

  thigh_lo, thigh_hi = min(thigh_vals), max(thigh_vals)

  print("\n=== PART 2: kinematic thigh sweep ===")
  env_info = build_kinematic_model()
  mj_model, mj_data = env_info["mj_model"], env_info["mj_data"]
  foot_site_name = env_info["resolved_foot_site_name"]
  print(f"Using foot site: {foot_site_name}")

  sweep_results = {}
  for label, calf_v in [
    ("early", calf_early),
    ("peak", calf_peak),
    ("late", calf_late),
  ]:
    res = sweep_thigh(
      mj_model, mj_data, calf_v, hip_rep, (thigh_lo, thigh_hi), foot_site_name
    )
    sweep_results[label] = res
    zs = [r[3] for r in res]
    max_i = zs.index(max(zs))
    print(f"\n-- calf={calf_v:.4f} rad ({label} recovery) --")
    print(f"   thigh range swept: {thigh_lo:.4f} to {thigh_hi:.4f} rad")
    print(f"   max foot_z = {zs[max_i]:.4f} at thigh = {res[max_i][0]:.4f} rad")
    monotonic_increasing = all(zs[i] <= zs[i + 1] + 1e-9 for i in range(len(zs) - 1))
    monotonic_decreasing = all(zs[i] >= zs[i + 1] - 1e-9 for i in range(len(zs) - 1))
    turning_point = not (monotonic_increasing or monotonic_decreasing)
    print(
      f"   monotonic increasing: {monotonic_increasing}  monotonic decreasing: {monotonic_decreasing}  "
      f"turning point present: {turning_point}"
    )

  with open(out_dir / "thigh_sweep.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["calf_label", "calf_val", "thigh", "foot_x", "foot_y", "foot_z"])
    for label, res in sweep_results.items():
      calf_v = {"early": calf_early, "peak": calf_peak, "late": calf_late}[label]
      for th, x, y, z in res:
        writer.writerow([label, calf_v, th, x, y, z])

  print("\n=== PART 3: kinematic ab/ad (hip) sweep ===")
  hip_lo, hip_hi = min(hip_vals), max(hip_vals)
  # Pad slightly if the observed hip range is degenerate (near-constant).
  if hip_hi - hip_lo < 0.02:
    mid = 0.5 * (hip_lo + hip_hi)
    hip_lo, hip_hi = mid - 0.1, mid + 0.1
    print(
      f"(observed hip range was nearly flat; padded to {hip_lo:.4f}..{hip_hi:.4f} for a visible sweep)"
    )

  hip_sweep_results = {}
  for label, (thigh_v, calf_v) in [
    ("early", (rec_rows[early_i]["thigh_pos"], calf_early)),
    ("peak", (rec_rows[peak_i_local]["thigh_pos"], calf_peak)),
    ("late", (rec_rows[late_i]["thigh_pos"], calf_late)),
  ]:
    res = sweep_hip(
      mj_model, mj_data, thigh_v, calf_v, (hip_lo, hip_hi), foot_site_name
    )
    hip_sweep_results[label] = res
    zs = [r[3] for r in res]
    dz = max(zs) - min(zs)
    print(
      f"-- thigh={thigh_v:.4f}, calf={calf_v:.4f} ({label}) -- "
      f"hip swept {hip_lo:.4f}..{hip_hi:.4f}: foot_z range = {dz:.4f} m (min={min(zs):.4f}, max={max(zs):.4f})"
    )

  with open(out_dir / "hip_sweep.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
      ["config_label", "thigh_val", "calf_val", "hip", "foot_x", "foot_y", "foot_z"]
    )
    for label, res in hip_sweep_results.items():
      thigh_v, calf_v = {
        "early": (rec_rows[early_i]["thigh_pos"], calf_early),
        "peak": (rec_rows[peak_i_local]["thigh_pos"], calf_peak),
        "late": (rec_rows[late_i]["thigh_pos"], calf_late),
      }[label]
      for hp, x, y, z in res:
        writer.writerow([label, thigh_v, calf_v, hp, x, y, z])

  print(f"\n[D3-geom] CSVs written to {out_dir}")


if __name__ == "__main__":
  main()
