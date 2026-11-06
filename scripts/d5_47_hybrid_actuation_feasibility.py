#!/usr/bin/env python
"""D5.47 seed-0 vs seed-11 hybrid actuation feasibility diagnostic.

This is a read-only replay: it uses the existing hybrid recovery
controller unchanged and logs the raw MuJoCo control/force path, joint
state response, wire-relative geometry, and a few saturation indicators
for seed 0 and seed 11.

Outputs are written under ``logs/d5_47/``:
  - ``seed0_trace.csv``
  - ``seed11_trace.csv``
  - ``aligned_trace.csv``
  - ``summary.csv``
  - ``report.md`` is written separately after the analysis step.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).parent))

import mjlab.tasks  # noqa: F401
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.tasks.recovery.mdp.actions import (
  LEG_NAMES,
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
  ResidualRecoveryAction,
)
from mjlab.utils.torch import configure_torch_backends

PHASE_NAMES = {
  PHASE_NORMAL_WALK: "NORMAL_WALK",
  PHASE_RECOVERY: "RECOVERY",
  PHASE_POST_RECOVERY: "POST_RECOVERY",
}


@dataclass(frozen=True)
class RunSpec:
  label: str
  seed: int


@dataclass(frozen=True)
class D547Config:
  device: str = "cpu"
  max_steps: int = 320
  out_dir: str = "logs/d5_47"
  runs: tuple[RunSpec, ...] = (
    RunSpec(label="seed0", seed=0),
    RunSpec(label="seed11", seed=11),
  )


def _quat_to_rpy_wxyz(quat: torch.Tensor) -> tuple[float, float, float]:
  w, x, y, z = [float(v) for v in quat]
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

  sinp = 2.0 * (w * y - z * x)
  sinp = max(-1.0, min(1.0, sinp))
  pitch = math.degrees(math.asin(sinp))

  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
  return roll, pitch, yaw


def _build_env(seed: int, device: str) -> ManagerBasedRlEnv:
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=False,
    max_recovery_steps=48,
  )
  env_cfg.seed = seed
  env_cfg.sim.device = device
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _find_first(rows: list[dict[str, Any]], key: str, value: Any) -> int:
  for row in rows:
    if row.get(key) == value:
      return int(row["step"])
  return -1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = sorted({key for row in rows for key in row.keys()})
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _collect_run(run: RunSpec, device: str, max_steps: int, out_dir: Path) -> dict[str, Any]:
  env = _build_env(run.seed, device)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  entity = act._entity
  wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]

  joint_names = list(entity.joint_names)
  site_names = list(entity.site_names)
  fl_leg = LEG_NAMES[0]

  fl_hip_joint = f"{fl_leg}_hip_joint"
  fl_thigh_joint = f"{fl_leg}_thigh_joint"
  fl_calf_joint = f"{fl_leg}_calf_joint"

  hip_joint_idx = joint_names.index(fl_hip_joint)
  thigh_joint_idx = joint_names.index(fl_thigh_joint)
  calf_joint_idx = joint_names.index(fl_calf_joint)
  fl_site_idx = site_names.index(fl_leg)

  hip_pos_ctrl_id = act._find_actuator_ctrl_id(fl_hip_joint, BuiltinPositionActuator)
  thigh_pos_ctrl_id = act._find_actuator_ctrl_id(fl_thigh_joint, BuiltinPositionActuator)
  calf_pos_ctrl_id = act._find_actuator_ctrl_id(fl_calf_joint, BuiltinPositionActuator)
  thigh_vel_ctrl_id = act._aux_thigh_ctrl_id
  calf_motor_ctrl_id = act._aux_calf_ctrl_id

  env.reset(seed=run.seed)

  rows: list[dict[str, Any]] = []
  entry_step = -1
  exit_step = -1
  outcome = "horizon_exhausted"

  def _wire_contact_row() -> tuple[bool, float, list[float]]:
    found = wire_sensor.data.found[0]
    contact = bool((found > 0).any())
    force_vec = torch.zeros(3, device=env.device)
    if wire_sensor.data.force is not None:
      force_vec = wire_sensor.data.force[0].reshape(-1, 3).sum(dim=0)
    return contact, float(torch.linalg.norm(force_vec)), [float(v) for v in force_vec]

  for step in range(max_steps):
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    phase = int(act.phase[0].item())
    if entry_step < 0 and phase == PHASE_RECOVERY:
      entry_step = step
    if bool(terminateds[0].item()):
      outcome = "terminated"
    elif bool(truncateds[0].item()):
      outcome = "truncated"

    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_pose_w[0, 3:7]
    base_lin_vel_w = entity.data.root_link_lin_vel_w[0]
    base_lin_vel_b = entity.data.root_link_lin_vel_b[0]
    base_ang_vel_w = entity.data.root_link_ang_vel_w[0]
    projected_gravity_b = entity.data.projected_gravity_b[0]
    roll_deg, pitch_deg, yaw_deg = _quat_to_rpy_wxyz(base_quat)

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    qfrc_actuator = entity.data.qfrc_actuator[0]
    actuator_force = entity.data.actuator_force[0]
    ctrl = env.sim.data.ctrl[0]

    joint_limits = entity.data.joint_pos_limits[0]
    soft_joint_limits = entity.data.soft_joint_pos_limits[0]

    wire_contact, wire_force_mag, wire_force_vec = _wire_contact_row()

    row = {
      "label": run.label,
      "seed": run.seed,
      "step": step,
      "sim_time_s": float(step * env.step_dt),
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": int(act.recovery_elapsed[0].item()),
      "post_recovery_elapsed": int(act.post_recovery_elapsed[0].item()),
      "newly_entangled": bool(act.newly_entangled[0].item()),
      "newly_recovered": bool(act.newly_recovered[0].item()),
      "newly_succeeded": bool(act.newly_succeeded[0].item()),
      "crossing_achieved": bool(act.crossing_achieved[0].item()),
      "calf_clearance_achieved": bool(act.calf_clearance_achieved[0].item()),
      "foot_crossed_achieved": bool(act.foot_crossed_achieved[0].item()),
      "release_achieved": bool(act.release_achieved[0].item()),
      "recovery_failed_timeout": bool(act.recovery_failed_timeout[0].item()),
      "wire_contact": wire_contact,
      "wire_contact_force_mag_n": wire_force_mag,
      "wire_contact_force_x_n": wire_force_vec[0],
      "wire_contact_force_y_n": wire_force_vec[1],
      "wire_contact_force_z_n": wire_force_vec[2],
      "base_x_m": float(base_pos[0].item()),
      "base_y_m": float(base_pos[1].item()),
      "base_z_m": float(base_pos[2].item()),
      "base_lin_vel_w_x_mps": float(base_lin_vel_w[0].item()),
      "base_lin_vel_w_y_mps": float(base_lin_vel_w[1].item()),
      "base_lin_vel_w_z_mps": float(base_lin_vel_w[2].item()),
      "base_lin_vel_b_x_mps": float(base_lin_vel_b[0].item()),
      "base_lin_vel_b_y_mps": float(base_lin_vel_b[1].item()),
      "base_lin_vel_b_z_mps": float(base_lin_vel_b[2].item()),
      "base_ang_vel_w_x_rps": float(base_ang_vel_w[0].item()),
      "base_ang_vel_w_y_rps": float(base_ang_vel_w[1].item()),
      "base_ang_vel_w_z_rps": float(base_ang_vel_w[2].item()),
      "base_roll_deg": roll_deg,
      "base_pitch_deg": pitch_deg,
      "base_yaw_deg": yaw_deg,
      "projected_gravity_b_x": float(projected_gravity_b[0].item()),
      "projected_gravity_b_y": float(projected_gravity_b[1].item()),
      "projected_gravity_b_z": float(projected_gravity_b[2].item()),
      "fl_foot_x_m": float(entity.data.site_pos_w[0, fl_site_idx, 0].item()),
      "fl_foot_y_m": float(entity.data.site_pos_w[0, fl_site_idx, 1].item()),
      "fl_foot_z_m": float(entity.data.site_pos_w[0, fl_site_idx, 2].item()),
      "fl_foot_vx_mps": float(entity.data.site_lin_vel_w[0, fl_site_idx, 0].item()),
      "fl_foot_vy_mps": float(entity.data.site_lin_vel_w[0, fl_site_idx, 1].item()),
      "fl_foot_vz_mps": float(entity.data.site_lin_vel_w[0, fl_site_idx, 2].item()),
      "fl_foot_speed_mps": float(torch.linalg.norm(entity.data.site_lin_vel_w[0, fl_site_idx]).item()),
      "fl_foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
      "fl_foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
      "fl_calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
      "fl_calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
      "fl_hip_pos_rad": float(qpos[hip_joint_idx].item()),
      "fl_thigh_pos_rad": float(qpos[thigh_joint_idx].item()),
      "fl_calf_pos_rad": float(qpos[calf_joint_idx].item()),
      "fl_hip_vel_radps": float(qvel[hip_joint_idx].item()),
      "fl_thigh_vel_radps": float(qvel[thigh_joint_idx].item()),
      "fl_calf_vel_radps": float(qvel[calf_joint_idx].item()),
      "fl_hip_qfrc_actuator_nm": float(qfrc_actuator[hip_joint_idx].item()),
      "fl_thigh_qfrc_actuator_nm": float(qfrc_actuator[thigh_joint_idx].item()),
      "fl_calf_qfrc_actuator_nm": float(qfrc_actuator[calf_joint_idx].item()),
      "fl_hip_ctrl": float(ctrl[hip_pos_ctrl_id].item()),
      "fl_thigh_pos_ctrl": float(ctrl[thigh_pos_ctrl_id].item()),
      "fl_calf_pos_ctrl": float(ctrl[calf_pos_ctrl_id].item()),
      "fl_thigh_vel_ctrl": float(ctrl[thigh_vel_ctrl_id].item()),
      "fl_calf_motor_ctrl": float(ctrl[calf_motor_ctrl_id].item()),
      "fl_hip_actuator_force": float(actuator_force[hip_pos_ctrl_id].item()),
      "fl_thigh_pos_actuator_force": float(actuator_force[thigh_pos_ctrl_id].item()),
      "fl_calf_pos_actuator_force": float(actuator_force[calf_pos_ctrl_id].item()),
      "fl_thigh_vel_actuator_force": float(actuator_force[thigh_vel_ctrl_id].item()),
      "fl_calf_motor_actuator_force": float(actuator_force[calf_motor_ctrl_id].item()),
      "fl_hip_pos_gain": float(env.sim.model.actuator_gainprm[0, hip_pos_ctrl_id, 0].item()),
      "fl_thigh_pos_gain": float(env.sim.model.actuator_gainprm[0, thigh_pos_ctrl_id, 0].item()),
      "fl_calf_pos_gain": float(env.sim.model.actuator_gainprm[0, calf_pos_ctrl_id, 0].item()),
      "fl_thigh_vel_gain": float(env.sim.model.actuator_gainprm[0, thigh_vel_ctrl_id, 0].item()),
      "fl_thigh_vel_bias_d": float(env.sim.model.actuator_biasprm[0, thigh_vel_ctrl_id, 2].item()),
      "fl_thigh_pos_bias_p": float(env.sim.model.actuator_biasprm[0, thigh_pos_ctrl_id, 1].item()),
      "fl_thigh_pos_bias_d": float(env.sim.model.actuator_biasprm[0, thigh_pos_ctrl_id, 2].item()),
      "fl_calf_pos_bias_p": float(env.sim.model.actuator_biasprm[0, calf_pos_ctrl_id, 1].item()),
      "fl_calf_pos_bias_d": float(env.sim.model.actuator_biasprm[0, calf_pos_ctrl_id, 2].item()),
      "fl_hip_pos_ctrlrange_lo": float(env.sim.model.actuator_ctrlrange[0, hip_pos_ctrl_id, 0].item()),
      "fl_hip_pos_ctrlrange_hi": float(env.sim.model.actuator_ctrlrange[0, hip_pos_ctrl_id, 1].item()),
      "fl_thigh_pos_ctrlrange_lo": float(env.sim.model.actuator_ctrlrange[0, thigh_pos_ctrl_id, 0].item()),
      "fl_thigh_pos_ctrlrange_hi": float(env.sim.model.actuator_ctrlrange[0, thigh_pos_ctrl_id, 1].item()),
      "fl_calf_pos_ctrlrange_lo": float(env.sim.model.actuator_ctrlrange[0, calf_pos_ctrl_id, 0].item()),
      "fl_calf_pos_ctrlrange_hi": float(env.sim.model.actuator_ctrlrange[0, calf_pos_ctrl_id, 1].item()),
      "fl_thigh_vel_ctrlrange_lo": float(env.sim.model.actuator_ctrlrange[0, thigh_vel_ctrl_id, 0].item()),
      "fl_thigh_vel_ctrlrange_hi": float(env.sim.model.actuator_ctrlrange[0, thigh_vel_ctrl_id, 1].item()),
      "fl_calf_motor_ctrlrange_lo": float(env.sim.model.actuator_ctrlrange[0, calf_motor_ctrl_id, 0].item()),
      "fl_calf_motor_ctrlrange_hi": float(env.sim.model.actuator_ctrlrange[0, calf_motor_ctrl_id, 1].item()),
      "fl_thigh_pos_forcerange_lo": float(env.sim.model.actuator_forcerange[0, thigh_pos_ctrl_id, 0].item()),
      "fl_thigh_pos_forcerange_hi": float(env.sim.model.actuator_forcerange[0, thigh_pos_ctrl_id, 1].item()),
      "fl_calf_pos_forcerange_lo": float(env.sim.model.actuator_forcerange[0, calf_pos_ctrl_id, 0].item()),
      "fl_calf_pos_forcerange_hi": float(env.sim.model.actuator_forcerange[0, calf_pos_ctrl_id, 1].item()),
      "fl_thigh_vel_forcerange_lo": float(env.sim.model.actuator_forcerange[0, thigh_vel_ctrl_id, 0].item()),
      "fl_thigh_vel_forcerange_hi": float(env.sim.model.actuator_forcerange[0, thigh_vel_ctrl_id, 1].item()),
      "fl_calf_motor_forcerange_lo": float(env.sim.model.actuator_forcerange[0, calf_motor_ctrl_id, 0].item()),
      "fl_calf_motor_forcerange_hi": float(env.sim.model.actuator_forcerange[0, calf_motor_ctrl_id, 1].item()),
      "fl_thigh_pos_soft_limit_lo": float(soft_joint_limits[thigh_joint_idx, 0].item()),
      "fl_thigh_pos_soft_limit_hi": float(soft_joint_limits[thigh_joint_idx, 1].item()),
      "fl_calf_pos_soft_limit_lo": float(soft_joint_limits[calf_joint_idx, 0].item()),
      "fl_calf_pos_soft_limit_hi": float(soft_joint_limits[calf_joint_idx, 1].item()),
      "fl_thigh_pos_hard_limit_lo": float(joint_limits[thigh_joint_idx, 0].item()),
      "fl_thigh_pos_hard_limit_hi": float(joint_limits[thigh_joint_idx, 1].item()),
      "fl_calf_pos_hard_limit_lo": float(joint_limits[calf_joint_idx, 0].item()),
      "fl_calf_pos_hard_limit_hi": float(joint_limits[calf_joint_idx, 1].item()),
      "hybrid_thigh_pos_gain_active": bool(act.hybrid_thigh_pos_gain_active[0].item()),
      "hybrid_two_phase_advance_active": bool(act.hybrid_two_phase_advance_active[0].item()),
      "two_phase_lift_counter": int(act.two_phase_lift_counter[0].item()),
      "two_phase_switch_step": int(act.two_phase_switch_step[0].item()),
      "entered_recovery": entry_step >= 0,
    }
    rows.append(row)

    if bool(act.newly_recovered[0].item()):
      exit_step = step
    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      break

  env.close()

  trace_path = out_dir / f"{run.label}_trace.csv"
  _write_csv(trace_path, rows)

  first_recovery_step = _find_first(rows, "phase", "RECOVERY")
  first_post_step = _find_first(rows, "phase", "POST_RECOVERY")
  first_cross_step = _find_first(rows, "crossing_achieved", True)
  first_stage1_step = _find_first(rows, "calf_clearance_achieved", True)
  first_stage2_step = _find_first(rows, "foot_crossed_achieved", True)

  summary = {
    "label": run.label,
    "seed": run.seed,
    "trace_csv": str(trace_path),
    "entry_step": first_recovery_step,
    "first_post_step": first_post_step,
    "first_stage1_step": first_stage1_step,
    "first_stage2_step": first_stage2_step,
    "first_cross_step": first_cross_step,
    "exit_step": exit_step,
    "outcome": outcome,
    "max_calf_forward_dist_m": max(float(row["fl_calf_forward_dist_m"]) for row in rows),
    "max_foot_forward_dist_m": max(float(row["fl_foot_forward_dist_m"]) for row in rows),
    "max_calf_height_above_wire_m": max(float(row["fl_calf_height_above_wire_m"]) for row in rows),
    "max_foot_height_above_wire_m": max(float(row["fl_foot_height_above_wire_m"]) for row in rows),
  }
  return summary


def _load_trace(path: str) -> dict[int, dict[str, Any]]:
  rows: list[dict[str, str]] = []
  with Path(path).open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows.append(row)

  first_entry = -1
  for row in rows:
    if row["phase"] == "RECOVERY" and first_entry < 0:
      first_entry = int(row["step"])

  normalized: dict[int, dict[str, Any]] = {}
  for row in rows:
    step = int(row["step"])
    rel = step - first_entry if first_entry >= 0 else step
    normalized[rel] = {
      "step": step,
      "rel_step": rel,
      "phase": row["phase"],
      "recovery_elapsed": int(row["recovery_elapsed"]),
      "wire_contact": row["wire_contact"] == "True",
      "wire_contact_force_mag_n": float(row["wire_contact_force_mag_n"]),
      "fl_thigh_vel_ctrl": float(row["fl_thigh_vel_ctrl"]),
      "fl_calf_motor_ctrl": float(row["fl_calf_motor_ctrl"]),
      "fl_thigh_vel_actuator_force": float(row["fl_thigh_vel_actuator_force"]),
      "fl_calf_motor_actuator_force": float(row["fl_calf_motor_actuator_force"]),
      "fl_thigh_vel_radps": float(row["fl_thigh_vel_radps"]),
      "fl_calf_vel_radps": float(row["fl_calf_vel_radps"]),
      "fl_calf_forward_dist_m": float(row["fl_calf_forward_dist_m"]),
      "fl_calf_height_above_wire_m": float(row["fl_calf_height_above_wire_m"]),
      "fl_foot_forward_dist_m": float(row["fl_foot_forward_dist_m"]),
      "fl_foot_height_above_wire_m": float(row["fl_foot_height_above_wire_m"]),
      "base_lin_vel_b_x_mps": float(row["base_lin_vel_b_x_mps"]),
      "base_lin_vel_b_z_mps": float(row["base_lin_vel_b_z_mps"]),
      "base_z_m": float(row["base_z_m"]),
      "base_roll_deg": float(row["base_roll_deg"]),
      "base_pitch_deg": float(row["base_pitch_deg"]),
      "hybrid_two_phase_advance_active": row["hybrid_two_phase_advance_active"] == "True",
      "hybrid_thigh_pos_gain_active": row["hybrid_thigh_pos_gain_active"] == "True",
    }
  return normalized


def _build_aligned_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
  summary_by_label = {row["label"]: row for row in summaries}
  seed0 = _load_trace(summary_by_label["seed0"]["trace_csv"])
  seed11 = _load_trace(summary_by_label["seed11"]["trace_csv"])
  rel_steps = sorted(set(seed0) | set(seed11))

  aligned: list[dict[str, Any]] = []
  for rel in rel_steps:
    row: dict[str, Any] = {"rel_step": rel}
    left = seed0.get(rel)
    right = seed11.get(rel)
    for label, src in (("seed0", left), ("seed11", right)):
      prefix = f"{label}_"
      if src is None:
        continue
      for key, value in src.items():
        row[prefix + key] = value
    if left is not None and right is not None:
      for key in (
        "fl_thigh_vel_ctrl",
        "fl_calf_motor_ctrl",
        "fl_thigh_vel_actuator_force",
        "fl_calf_motor_actuator_force",
        "fl_thigh_vel_radps",
        "fl_calf_vel_radps",
        "fl_calf_forward_dist_m",
        "fl_calf_height_above_wire_m",
        "fl_foot_forward_dist_m",
        "fl_foot_height_above_wire_m",
        "base_lin_vel_b_x_mps",
        "base_lin_vel_b_z_mps",
        "base_z_m",
        "base_roll_deg",
        "base_pitch_deg",
      ):
        row[f"delta_{key}"] = right[key] - left[key]
    aligned.append(row)
  return aligned


def main() -> None:
  configure_torch_backends()
  cfg = D547Config()
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summaries = [_collect_run(run, cfg.device, cfg.max_steps, out_dir) for run in cfg.runs]
  _write_csv(out_dir / "summary.csv", summaries)
  _write_csv(out_dir / "aligned_trace.csv", _build_aligned_rows(summaries))

  print(f"Wrote D5.47 outputs to {out_dir}")
  for summary in summaries:
    print(
      f"{summary['label']}: entry={summary['entry_step']} stage1={summary['first_stage1_step']} "
      f"stage2={summary['first_stage2_step']} cross={summary['first_cross_step']} "
      f"outcome={summary['outcome']} max_calf_forward={summary['max_calf_forward_dist_m']:.4f}"
    )


if __name__ == "__main__":
  main()