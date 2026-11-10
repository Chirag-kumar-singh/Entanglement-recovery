#!/usr/bin/env python
"""D5.48 controlled recovery entry-state swap diagnostic.

This is a read-only diagnostic. It replays the frozen hybrid baseline,
captures the last pre-recovery state for seed 0 and seed 11, then restarts
seed 11 from the seed 0 pre-recovery snapshot without changing the
controller, model, wire geometry, or recovery thresholds.

The experiment answers whether the recovery outcome is sensitive to the
entry state itself, as opposed to an implementation bug in the actuator
path.

Outputs go under ``logs/d5_48/``:
  - ``seed0_baseline_trace.csv``
  - ``seed11_baseline_trace.csv``
  - ``seed11_seed0_entry_swap_trace.csv``
  - ``entry_snapshots.csv``
  - ``aligned_recovery_window.csv``
  - ``summary.csv``
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
  swap_source: str | None = None


@dataclass(frozen=True)
class D548Config:
  device: str = "cpu"
  max_steps: int = 320
  out_dir: str = "logs/d5_48"
  runs: tuple[RunSpec, ...] = (
    RunSpec(label="seed0_baseline", seed=0),
    RunSpec(label="seed11_baseline", seed=11),
    RunSpec(label="seed11_seed0_entry_swap", seed=11, swap_source="seed0_baseline"),
  )


@dataclass
class EntrySnapshot:
  label: str
  seed: int
  pre_entry_step: int
  recovery_entry_step: int
  phase: str
  root_pose_w: torch.Tensor
  root_vel_w: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  qpos: torch.Tensor
  qvel: torch.Tensor
  wire_contact: bool
  wire_contact_force_mag_n: float
  base_lin_vel_w: torch.Tensor
  base_lin_vel_b: torch.Tensor
  base_ang_vel_w: torch.Tensor
  projected_gravity_b: torch.Tensor
  calf_forward_dist_m: float
  calf_height_above_wire_m: float
  foot_forward_dist_m: float
  foot_height_above_wire_m: float
  fl_hip_pos_rad: float
  fl_thigh_pos_rad: float
  fl_calf_pos_rad: float
  fl_hip_vel_radps: float
  fl_thigh_vel_radps: float
  fl_calf_vel_radps: float


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = sorted({key for row in rows for key in row.keys()})
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _flat(values: torch.Tensor) -> list[float]:
  return [float(value) for value in values.detach().cpu().reshape(-1)]


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
  return float(torch.max(torch.abs(left - right)).item())


def _capture_snapshot(
  env: ManagerBasedRlEnv,
  act: ResidualRecoveryAction,
  pre_step: bool,
  step: int,
  recovery_entry_step: int,
) -> dict[str, Any]:
  robot = env.scene["robot"]
  data = robot.data
  fl_leg = LEG_NAMES[0]
  joint_names = list(robot.joint_names)
  site_names = list(robot.site_names)
  fl_site_idx = site_names.index(fl_leg)
  hip_idx = joint_names.index(f"{fl_leg}_hip_joint")
  thigh_idx = joint_names.index(f"{fl_leg}_thigh_joint")
  calf_idx = joint_names.index(f"{fl_leg}_calf_joint")

  wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
  wire_force_mag = 0.0
  if wire_sensor.data.force is not None:
    force_vec = wire_sensor.data.force[0].reshape(-1, 3).sum(dim=0)
    wire_force_mag = float(torch.linalg.norm(force_vec).item())

  qpos = data.data.qpos[0].clone()
  qvel = data.data.qvel[0].clone()
  row = {
    "step": step,
    "phase": PHASE_NAMES[int(act.phase[0].item())],
    "recovery_elapsed": int(act.recovery_elapsed[0].item()),
    "post_recovery_elapsed": int(act.post_recovery_elapsed[0].item()),
    "entered_recovery": bool(act.phase[0].item() == PHASE_RECOVERY),
    "wire_contact": bool((wire_sensor.data.found[0] > 0).any()),
    "wire_contact_force_mag_n": wire_force_mag,
    "base_x_m": float(data.root_link_pos_w[0, 0].item()),
    "base_y_m": float(data.root_link_pos_w[0, 1].item()),
    "base_z_m": float(data.root_link_pos_w[0, 2].item()),
    "base_lin_vel_w_x_mps": float(data.root_link_lin_vel_w[0, 0].item()),
    "base_lin_vel_w_y_mps": float(data.root_link_lin_vel_w[0, 1].item()),
    "base_lin_vel_w_z_mps": float(data.root_link_lin_vel_w[0, 2].item()),
    "base_lin_vel_b_x_mps": float(data.root_link_lin_vel_b[0, 0].item()),
    "base_lin_vel_b_y_mps": float(data.root_link_lin_vel_b[0, 1].item()),
    "base_lin_vel_b_z_mps": float(data.root_link_lin_vel_b[0, 2].item()),
    "base_ang_vel_w_x_rps": float(data.root_link_ang_vel_w[0, 0].item()),
    "base_ang_vel_w_y_rps": float(data.root_link_ang_vel_w[0, 1].item()),
    "base_ang_vel_w_z_rps": float(data.root_link_ang_vel_w[0, 2].item()),
    "projected_gravity_b_x": float(data.projected_gravity_b[0, 0].item()),
    "projected_gravity_b_y": float(data.projected_gravity_b[0, 1].item()),
    "projected_gravity_b_z": float(data.projected_gravity_b[0, 2].item()),
    "fl_foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
    "fl_foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
    "fl_calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
    "fl_calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
    "fl_hip_pos_rad": float(data.joint_pos[0, hip_idx].item()),
    "fl_thigh_pos_rad": float(data.joint_pos[0, thigh_idx].item()),
    "fl_calf_pos_rad": float(data.joint_pos[0, calf_idx].item()),
    "fl_hip_vel_radps": float(data.joint_vel[0, hip_idx].item()),
    "fl_thigh_vel_radps": float(data.joint_vel[0, thigh_idx].item()),
    "fl_calf_vel_radps": float(data.joint_vel[0, calf_idx].item()),
    "root_pose_w": _flat(data.root_link_pose_w[0]),
    "root_vel_w": _flat(data.root_link_vel_w[0]),
    "joint_pos": _flat(data.joint_pos[0]),
    "joint_vel": _flat(data.joint_vel[0]),
    "qpos": _flat(qpos),
    "qvel": _flat(qvel),
  }
  if pre_step:
    row["pre_step"] = True
  return row


def _restore_snapshot(
  env: ManagerBasedRlEnv,
  snapshot: EntrySnapshot,
  wire_contact_sensor_name: str,
) -> dict[str, float]:
  robot = env.scene["robot"]
  all_joints = torch.arange(len(robot.joint_names), device=env.device)
  env_ids = torch.tensor([0], device=env.device)
  robot.write_root_link_pose_to_sim(snapshot.root_pose_w.unsqueeze(0), env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(
    snapshot.root_vel_w.unsqueeze(0), env_ids=env_ids
  )
  robot.write_joint_state_to_sim(
    snapshot.joint_pos.unsqueeze(0),
    snapshot.joint_vel.unsqueeze(0),
    joint_ids=all_joints,
    env_ids=env_ids,
  )
  env.sim.forward()

  actual_root_pose = robot.data.root_link_pose_w[0]
  actual_root_vel = robot.data.root_link_vel_w[0]
  actual_joint_pos = robot.data.joint_pos[0]
  actual_joint_vel = robot.data.joint_vel[0]
  actual_qpos = robot.data.data.qpos[0]
  actual_qvel = robot.data.data.qvel[0]
  actual_wire_contact = bool(
    (env.scene.sensors[wire_contact_sensor_name].data.found[0] > 0).any()
  )

  return {
    "root_pose_max_abs_diff": _max_abs_diff(actual_root_pose, snapshot.root_pose_w),
    "root_vel_max_abs_diff": _max_abs_diff(actual_root_vel, snapshot.root_vel_w),
    "joint_pos_max_abs_diff": _max_abs_diff(actual_joint_pos, snapshot.joint_pos),
    "joint_vel_max_abs_diff": _max_abs_diff(actual_joint_vel, snapshot.joint_vel),
    "qpos_max_abs_diff": _max_abs_diff(actual_qpos, snapshot.qpos),
    "qvel_max_abs_diff": _max_abs_diff(actual_qvel, snapshot.qvel),
    "wire_contact_after_restore": float(actual_wire_contact),
    "wire_contact_force_mag_n_after_restore": float(
      torch.linalg.norm(
        env.scene.sensors[wire_contact_sensor_name]
        .data.force[0]
        .reshape(-1, 3)
        .sum(dim=0)
      ).item()
      if env.scene.sensors[wire_contact_sensor_name].data.force is not None
      else 0.0
    ),
  }


def _run_episode(
  run: RunSpec,
  device: str,
  max_steps: int,
  out_dir: Path,
  swap_snapshot: EntrySnapshot | None = None,
) -> tuple[dict[str, Any], EntrySnapshot | None, list[dict[str, Any]]]:
  env = _build_env(run.seed, device)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  env.reset(seed=run.seed)

  if swap_snapshot is not None:
    applied = _restore_snapshot(env, swap_snapshot, act.cfg.wire_contact_sensor_name)
  else:
    applied = {
      "root_pose_max_abs_diff": float("nan"),
      "root_vel_max_abs_diff": float("nan"),
      "joint_pos_max_abs_diff": float("nan"),
      "joint_vel_max_abs_diff": float("nan"),
      "qpos_max_abs_diff": float("nan"),
      "qvel_max_abs_diff": float("nan"),
      "wire_contact_after_restore": float("nan"),
      "wire_contact_force_mag_n_after_restore": float("nan"),
    }

  rows: list[dict[str, Any]] = []
  entry_pre_snapshot: EntrySnapshot | None = None
  pre_snapshot_row: dict[str, Any] | None = None
  outcome = "horizon_exhausted"
  recovery_exit_reason = "not_entered"
  first_stage1_step = -1
  first_stage2_step = -1
  first_cross_step = -1
  first_positive_calf_step = -1
  max_calf_forward = float("-inf")
  max_foot_forward = float("-inf")
  max_base_forward = float("-inf")
  min_base_vertical = float("inf")
  recovery_entry_step = -1
  post_recovery_entered = False

  for step in range(max_steps):
    pre_snapshot_row = _capture_snapshot(
      env, act, pre_step=True, step=step, recovery_entry_step=recovery_entry_step
    )
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    row = _capture_snapshot(
      env, act, pre_step=False, step=step, recovery_entry_step=recovery_entry_step
    )
    row.update(
      {
        "label": run.label,
        "seed": run.seed,
        "variant": "swap" if run.swap_source else "baseline",
        "swap_source": run.swap_source or "",
      }
    )
    rows.append(row)

    phase = int(act.phase[0].item())
    max_calf_forward = max(max_calf_forward, float(act.calf_forward_dist[0].item()))
    max_foot_forward = max(max_foot_forward, float(act.foot_forward_dist[0].item()))
    max_base_forward = max(
      max_base_forward, float(env.scene["robot"].data.root_link_lin_vel_b[0, 0].item())
    )
    min_base_vertical = min(
      min_base_vertical, float(env.scene["robot"].data.root_link_lin_vel_b[0, 2].item())
    )
    if first_positive_calf_step < 0 and float(act.calf_forward_dist[0].item()) > 0.0:
      first_positive_calf_step = step
    if first_stage1_step < 0 and bool(act.calf_clearance_achieved[0].item()):
      first_stage1_step = step
    if first_stage2_step < 0 and bool(act.foot_crossed_achieved[0].item()):
      first_stage2_step = step
    if first_cross_step < 0 and bool(act.crossing_achieved[0].item()):
      first_cross_step = step

    if entry_pre_snapshot is None and phase == PHASE_RECOVERY:
      # The pre-step state is the actual recovery-entry state used by the controller.
      entry_pre_snapshot = EntrySnapshot(
        label=run.label,
        seed=run.seed,
        pre_entry_step=step,
        recovery_entry_step=step,
        phase=str(pre_snapshot_row["phase"]),
        root_pose_w=torch.tensor(pre_snapshot_row["root_pose_w"], device=env.device),
        root_vel_w=torch.tensor(pre_snapshot_row["root_vel_w"], device=env.device),
        joint_pos=torch.tensor(pre_snapshot_row["joint_pos"], device=env.device),
        joint_vel=torch.tensor(pre_snapshot_row["joint_vel"], device=env.device),
        qpos=torch.tensor(pre_snapshot_row["qpos"], device=env.device),
        qvel=torch.tensor(pre_snapshot_row["qvel"], device=env.device),
        wire_contact=bool(pre_snapshot_row["wire_contact"]),
        wire_contact_force_mag_n=float(pre_snapshot_row["wire_contact_force_mag_n"]),
        base_lin_vel_w=torch.tensor(
          [
            pre_snapshot_row["base_lin_vel_w_x_mps"],
            pre_snapshot_row["base_lin_vel_w_y_mps"],
            pre_snapshot_row["base_lin_vel_w_z_mps"],
          ],
          device=env.device,
        ),
        base_lin_vel_b=torch.tensor(
          [
            pre_snapshot_row["base_lin_vel_b_x_mps"],
            pre_snapshot_row["base_lin_vel_b_y_mps"],
            pre_snapshot_row["base_lin_vel_b_z_mps"],
          ],
          device=env.device,
        ),
        base_ang_vel_w=torch.tensor(
          [
            pre_snapshot_row["base_ang_vel_w_x_rps"],
            pre_snapshot_row["base_ang_vel_w_y_rps"],
            pre_snapshot_row["base_ang_vel_w_z_rps"],
          ],
          device=env.device,
        ),
        projected_gravity_b=torch.tensor(
          [
            pre_snapshot_row["projected_gravity_b_x"],
            pre_snapshot_row["projected_gravity_b_y"],
            pre_snapshot_row["projected_gravity_b_z"],
          ],
          device=env.device,
        ),
        calf_forward_dist_m=float(pre_snapshot_row["fl_calf_forward_dist_m"]),
        calf_height_above_wire_m=float(pre_snapshot_row["fl_calf_height_above_wire_m"]),
        foot_forward_dist_m=float(pre_snapshot_row["fl_foot_forward_dist_m"]),
        foot_height_above_wire_m=float(pre_snapshot_row["fl_foot_height_above_wire_m"]),
        fl_hip_pos_rad=float(pre_snapshot_row["fl_hip_pos_rad"]),
        fl_thigh_pos_rad=float(pre_snapshot_row["fl_thigh_pos_rad"]),
        fl_calf_pos_rad=float(pre_snapshot_row["fl_calf_pos_rad"]),
        fl_hip_vel_radps=float(pre_snapshot_row["fl_hip_vel_radps"]),
        fl_thigh_vel_radps=float(pre_snapshot_row["fl_thigh_vel_radps"]),
        fl_calf_vel_radps=float(pre_snapshot_row["fl_calf_vel_radps"]),
      )
      recovery_entry_step = step

    if bool(act.newly_recovered[0].item()):
      recovery_exit_reason = (
        "crossing_release"
        if bool(act.crossing_achieved[0].item())
        and bool(act.release_achieved[0].item())
        else "cap_timeout"
      )
      post_recovery_entered = True
      if bool(act.crossing_achieved[0].item()):
        first_cross_step = first_cross_step if first_cross_step >= 0 else step

    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      outcome = "terminated" if bool(terminateds[0].item()) else "truncated"
      break

  env.close()

  trace_path = out_dir / f"{run.label}_trace.csv"
  _write_csv(trace_path, rows)

  summary = {
    "label": run.label,
    "seed": run.seed,
    "variant": "swap" if run.swap_source else "baseline",
    "swap_source": run.swap_source or "",
    "trace_csv": str(trace_path),
    "entry_pre_step": int(entry_pre_snapshot.pre_entry_step)
    if entry_pre_snapshot
    else -1,
    "recovery_entry_step": int(entry_pre_snapshot.recovery_entry_step)
    if entry_pre_snapshot
    else -1,
    "entry_snapshot_applied": swap_snapshot is not None,
    "entry_restore_root_pose_max_abs_diff": applied["root_pose_max_abs_diff"],
    "entry_restore_root_vel_max_abs_diff": applied["root_vel_max_abs_diff"],
    "entry_restore_joint_pos_max_abs_diff": applied["joint_pos_max_abs_diff"],
    "entry_restore_joint_vel_max_abs_diff": applied["joint_vel_max_abs_diff"],
    "entry_restore_qpos_max_abs_diff": applied["qpos_max_abs_diff"],
    "entry_restore_qvel_max_abs_diff": applied["qvel_max_abs_diff"],
    "entry_restore_wire_contact": applied["wire_contact_after_restore"],
    "entry_restore_wire_contact_force_mag_n": applied[
      "wire_contact_force_mag_n_after_restore"
    ],
    "entry_snapshot_wire_contact": entry_pre_snapshot.wire_contact
    if entry_pre_snapshot
    else False,
    "entry_snapshot_wire_contact_force_mag_n": entry_pre_snapshot.wire_contact_force_mag_n
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_base_vx_mps": float(entry_pre_snapshot.base_lin_vel_b[0].item())
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_base_vz_mps": float(entry_pre_snapshot.base_lin_vel_b[2].item())
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_calf_forward_dist_m": entry_pre_snapshot.calf_forward_dist_m
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_foot_forward_dist_m": entry_pre_snapshot.foot_forward_dist_m
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_thigh_pos_rad": entry_pre_snapshot.fl_thigh_pos_rad
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_calf_pos_rad": entry_pre_snapshot.fl_calf_pos_rad
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_thigh_vel_radps": entry_pre_snapshot.fl_thigh_vel_radps
    if entry_pre_snapshot
    else float("nan"),
    "entry_snapshot_calf_vel_radps": entry_pre_snapshot.fl_calf_vel_radps
    if entry_pre_snapshot
    else float("nan"),
    "first_stage1_step": first_stage1_step,
    "first_stage2_step": first_stage2_step,
    "first_cross_step": first_cross_step,
    "first_positive_calf_step": first_positive_calf_step,
    "max_calf_forward_dist_m": max_calf_forward,
    "max_foot_forward_dist_m": max_foot_forward,
    "max_base_forward_vel_body_x_mps": max_base_forward,
    "min_base_vertical_vel_body_z_mps": min_base_vertical,
    "recovery_exit_reason": recovery_exit_reason,
    "outcome": outcome,
    "post_recovery_entered": post_recovery_entered,
    "full_crossing_achieved": first_cross_step >= 0,
  }
  return summary, entry_pre_snapshot, rows


def _aligned_window(
  traces: dict[str, list[dict[str, Any]]],
  summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
  def to_recovery_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    recovery_rows = [row for row in rows if row["phase"] == "RECOVERY"]
    if not recovery_rows:
      return {}
    entry_step = int(recovery_rows[0]["step"])
    return {int(row["step"]) - entry_step: row for row in recovery_rows}

  maps = {label: to_recovery_map(rows) for label, rows in traces.items()}
  rel_steps = sorted(set().union(*[set(mapping.keys()) for mapping in maps.values()]))
  aligned: list[dict[str, Any]] = []
  for rel_step in rel_steps:
    row: dict[str, Any] = {"rel_step": rel_step}
    for label, mapping in maps.items():
      source = mapping.get(rel_step)
      prefix = f"{label}_"
      if source is None:
        for key in (
          "step",
          "phase",
          "recovery_elapsed",
          "wire_contact",
          "base_lin_vel_b_x_mps",
          "base_lin_vel_b_z_mps",
          "fl_thigh_vel_radps",
          "fl_calf_vel_radps",
          "fl_calf_forward_dist_m",
          "fl_calf_height_above_wire_m",
          "fl_foot_forward_dist_m",
          "fl_foot_height_above_wire_m",
        ):
          row[prefix + key] = ""
      else:
        for key in (
          "step",
          "phase",
          "recovery_elapsed",
          "wire_contact",
          "base_lin_vel_b_x_mps",
          "base_lin_vel_b_z_mps",
          "fl_thigh_vel_radps",
          "fl_calf_vel_radps",
          "fl_calf_forward_dist_m",
          "fl_calf_height_above_wire_m",
          "fl_foot_forward_dist_m",
          "fl_foot_height_above_wire_m",
        ):
          row[prefix + key] = source[key]
    if all(maps[label].get(rel_step) is not None for label in maps):
      source0 = maps["seed0_baseline"][rel_step]
      source11 = maps["seed11_baseline"][rel_step]
      row["delta_base_lin_vel_b_x_mps"] = float(
        source11["base_lin_vel_b_x_mps"]
      ) - float(source0["base_lin_vel_b_x_mps"])
      row["delta_base_lin_vel_b_z_mps"] = float(
        source11["base_lin_vel_b_z_mps"]
      ) - float(source0["base_lin_vel_b_z_mps"])
      row["delta_fl_calf_forward_dist_m"] = float(
        source11["fl_calf_forward_dist_m"]
      ) - float(source0["fl_calf_forward_dist_m"])
      row["delta_fl_foot_forward_dist_m"] = float(
        source11["fl_foot_forward_dist_m"]
      ) - float(source0["fl_foot_forward_dist_m"])
    aligned.append(row)
  return aligned


def _snapshot_row(snapshot: EntrySnapshot) -> dict[str, Any]:
  return {
    "label": snapshot.label,
    "seed": snapshot.seed,
    "pre_entry_step": snapshot.pre_entry_step,
    "recovery_entry_step": snapshot.recovery_entry_step,
    "phase": snapshot.phase,
    "wire_contact": snapshot.wire_contact,
    "wire_contact_force_mag_n": snapshot.wire_contact_force_mag_n,
    "base_lin_vel_b_x_mps": float(snapshot.base_lin_vel_b[0].item()),
    "base_lin_vel_b_y_mps": float(snapshot.base_lin_vel_b[1].item()),
    "base_lin_vel_b_z_mps": float(snapshot.base_lin_vel_b[2].item()),
    "base_lin_vel_w_x_mps": float(snapshot.base_lin_vel_w[0].item()),
    "base_lin_vel_w_y_mps": float(snapshot.base_lin_vel_w[1].item()),
    "base_lin_vel_w_z_mps": float(snapshot.base_lin_vel_w[2].item()),
    "base_ang_vel_w_x_rps": float(snapshot.base_ang_vel_w[0].item()),
    "base_ang_vel_w_y_rps": float(snapshot.base_ang_vel_w[1].item()),
    "base_ang_vel_w_z_rps": float(snapshot.base_ang_vel_w[2].item()),
    "projected_gravity_b_x": float(snapshot.projected_gravity_b[0].item()),
    "projected_gravity_b_y": float(snapshot.projected_gravity_b[1].item()),
    "projected_gravity_b_z": float(snapshot.projected_gravity_b[2].item()),
    "calf_forward_dist_m": snapshot.calf_forward_dist_m,
    "calf_height_above_wire_m": snapshot.calf_height_above_wire_m,
    "foot_forward_dist_m": snapshot.foot_forward_dist_m,
    "foot_height_above_wire_m": snapshot.foot_height_above_wire_m,
    "fl_hip_pos_rad": snapshot.fl_hip_pos_rad,
    "fl_thigh_pos_rad": snapshot.fl_thigh_pos_rad,
    "fl_calf_pos_rad": snapshot.fl_calf_pos_rad,
    "fl_hip_vel_radps": snapshot.fl_hip_vel_radps,
    "fl_thigh_vel_radps": snapshot.fl_thigh_vel_radps,
    "fl_calf_vel_radps": snapshot.fl_calf_vel_radps,
  }


def main() -> None:
  configure_torch_backends()
  cfg = D548Config()
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summaries: dict[str, dict[str, Any]] = {}
  traces: dict[str, list[dict[str, Any]]] = {}
  snapshots: dict[str, EntrySnapshot] = {}

  # First pass: capture the natural seed 0 and seed 11 entry states.
  for run in cfg.runs[:2]:
    summary, snapshot, rows = _run_episode(run, cfg.device, cfg.max_steps, out_dir)
    summaries[run.label] = summary
    traces[run.label] = rows
    assert snapshot is not None, f"{run.label} never reached recovery entry"
    snapshots[run.label] = snapshot

  # Second pass: replay seed 11 from the seed 0 recovery-entry snapshot.
  swap_run = cfg.runs[2]
  swap_summary, swap_snapshot, swap_rows = _run_episode(
    swap_run,
    cfg.device,
    cfg.max_steps,
    out_dir,
    swap_snapshot=snapshots[swap_run.swap_source],
  )
  summaries[swap_run.label] = swap_summary
  traces[swap_run.label] = swap_rows
  snapshots[swap_run.label] = (
    swap_snapshot if swap_snapshot is not None else snapshots[swap_run.swap_source]
  )

  summary_csv = out_dir / "summary.csv"
  _write_csv(summary_csv, list(summaries.values()))

  entry_csv = out_dir / "entry_snapshots.csv"
  _write_csv(entry_csv, [_snapshot_row(snapshot) for snapshot in snapshots.values()])

  aligned_csv = out_dir / "aligned_recovery_window.csv"
  _write_csv(aligned_csv, _aligned_window(traces, summaries))

  print(f"Wrote D5.48 outputs to {out_dir}")
  for label in ("seed0_baseline", "seed11_baseline", "seed11_seed0_entry_swap"):
    summary = summaries[label]
    print(
      f"{label}: entry={summary['recovery_entry_step']} cross={summary['first_cross_step']} "
      f"stage1={summary['first_stage1_step']} stage2={summary['first_stage2_step']} "
      f"exit={summary['recovery_exit_reason']} outcome={summary['outcome']}"
    )


if __name__ == "__main__":
  main()
