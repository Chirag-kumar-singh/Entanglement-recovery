#!/usr/bin/env python
"""D5.49 -- Recovery override counterfactual.

Goal: isolate how much of the initial stumble is due to the hybrid
recovery override itself, versus wire contact alone.

Run A (baseline):
- current seed-0 baseline, unchanged.

Run B (override counterfactual):
- capture the exact last pre-recovery simulator state from Run A;
- restore that state into a fresh env instance;
- let the same first-contact -> RECOVERY trigger happen;
- keep the normal walking position actuators active for a short,
  predefined number of recovery steps before handing off to the existing
  hybrid override.

This is diagnostic-only. It does not modify production code, checkpoints,
wire geometry, trigger logic, or learned walking behavior.

Outputs go under ``logs/d5_49_recovery_override_counterfactual/``:
- ``baseline_trace.csv``
- ``counterfactual_trace.csv``
- ``entry_snapshot.csv``
- ``comparison_summary.csv``
- ``aligned_entry_window.csv``
- ``report.md``
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_16_trajectory_audit import build_env  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.recovery.mdp.actions import (  # noqa: E402
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
  ResidualRecoveryAction,
)
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

PHASE_NAMES = {
  PHASE_NORMAL_WALK: "NORMAL_WALK",
  PHASE_RECOVERY: "RECOVERY",
  PHASE_POST_RECOVERY: "POST_RECOVERY",
}


@dataclass(frozen=True)
class D549Config:
  seed: int = 0
  device: str = "cpu"
  max_steps: int = 180
  max_recovery_steps_override: int = 48
  hold_steps: int = 8
  out_dir: str = "logs/d5_49_recovery_override_counterfactual"


@dataclass
class EntrySnapshot:
  pre_entry_step: int
  recovery_entry_step: int
  root_pose_w: torch.Tensor
  root_vel_w: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  qpos: torch.Tensor
  qvel: torch.Tensor
  wire_contact: bool
  wire_contact_force_mag_n: float
  base_z_m: float
  base_ang_vel_y_rps: float
  base_lin_vel_x_mps: float
  base_lin_vel_z_mps: float
  fl_hip_pos_rad: float
  fl_thigh_pos_rad: float
  fl_calf_pos_rad: float
  fl_hip_vel_radps: float
  fl_thigh_vel_radps: float
  fl_calf_vel_radps: float


@dataclass
class RunResult:
  label: str
  trace_rows: list[dict[str, Any]]
  entry_snapshot: EntrySnapshot | None
  summary: dict[str, Any]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = sorted({key for row in rows for key in row.keys()})
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _build_env(seed: int, device: str, max_recovery_steps_override: int) -> ManagerBasedRlEnv:
  d3_cfg = D3GroundConfig(seed=seed, device=device, use_isolated_wire=True)
  return build_env(d3_cfg, device, max_recovery_steps_override)


def _wire_force_mag(env: ManagerBasedRlEnv, act: ResidualRecoveryAction) -> float:
  sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
  if sensor.data.force is None:
    return 0.0
  force_vec = sensor.data.force[0].reshape(-1, 3).sum(dim=0)
  return float(torch.linalg.norm(force_vec).item())


def _raw_wire_contacts(env: ManagerBasedRlEnv) -> tuple[bool, bool]:
  model = env.sim.mj_model
  contact = env.sim.data.contact
  nacon = int(env.sim.data.nacon[0])
  hip_wire = False
  calf_wire = False
  for ci in range(nacon):
    if int(contact.worldid[ci]) != 0:
      continue
    g0 = int(contact.geom[ci][0])
    g1 = int(contact.geom[ci][1])
    n0 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g0) or f"geom#{g0}"
    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom#{g1}"
    pair = (n0, n1)
    if any("FL_hip_collision" in name for name in pair) and any(
      "wire_seg" in name for name in pair
    ):
      hip_wire = True
    if any("FL_calf2_collision" in name for name in pair) and any(
      "wire_seg" in name for name in pair
    ):
      calf_wire = True
  return hip_wire, calf_wire


def _restore_snapshot(
  env: ManagerBasedRlEnv,
  snapshot: EntrySnapshot,
  wire_contact_sensor_name: str,
) -> dict[str, float]:
  robot = env.scene["robot"]
  env.sim.data.qpos[0] = snapshot.qpos.clone()
  env.sim.data.qvel[0] = snapshot.qvel.clone()
  env.sim.forward()
  env.scene.update(dt=env.physics_dt)
  env.sim.sense()

  actual_qpos = robot.data.data.qpos[0]
  actual_qvel = robot.data.data.qvel[0]
  actual_wire_contact = bool(
    (env.scene.sensors[wire_contact_sensor_name].data.found[0] > 0).any()
  )
  return {
    "qpos_max_abs_diff": float(torch.max(torch.abs(actual_qpos - snapshot.qpos)).item()),
    "qvel_max_abs_diff": float(torch.max(torch.abs(actual_qvel - snapshot.qvel)).item()),
    "wire_contact_after_restore": float(actual_wire_contact),
    "wire_contact_force_mag_n_after_restore": (
      float(
        torch.linalg.norm(
          env.scene.sensors[wire_contact_sensor_name]
          .data.force[0]
          .reshape(-1, 3)
          .sum(dim=0)
        ).item()
      )
      if env.scene.sensors[wire_contact_sensor_name].data.force is not None
      else 0.0
    ),
  }


def install_fixed_delay_handoff(
  act: ResidualRecoveryAction,
  hold_steps: int,
) -> dict[str, Any]:
  """Delay the hybrid handoff for a fixed number of recovery steps.

  During the hold, the RECOVERY phase is still active, but the thigh/calf
  position actuators remain enabled so the frozen walking controller keeps
  driving the leg normally. After `hold_steps`, the existing hybrid
  disable/aux-enable handoff is applied unchanged.
  """
  assert hold_steps >= 1, "hold_steps must be >= 1"

  device = act.device
  num_envs = act.num_envs
  sim = act._env.sim

  state: dict[str, Any] = {
    "holding": torch.zeros(num_envs, dtype=torch.bool, device=device),
    "handoff_step": torch.full((num_envs,), -1, dtype=torch.long, device=device),
    "current_step": -1,
  }

  def delayed_apply_hybrid_actuators() -> None:
    in_recovery = act._phase == PHASE_RECOVERY
    enter_ids = act.newly_entangled.nonzero(as_tuple=True)[0]
    exit_ids = act.newly_recovered.nonzero(as_tuple=True)[0]

    if enter_ids.numel() > 0:
      state["holding"][enter_ids] = True
      act._peak_armed[enter_ids] = False
      act._peak_running_max[enter_ids] = act.foot_height_above_wire[enter_ids]
      act._peak_entry_height[enter_ids] = act.foot_height_above_wire[enter_ids]
      act._peak_decline_counter[enter_ids] = 0
      act._peak_triggered[enter_ids] = False
      act.peak_trigger_step[enter_ids] = -1

    elapsed = act._recovery_elapsed
    transition_mask = state["holding"] & in_recovery & (elapsed >= hold_steps)
    transition_ids = transition_mask.nonzero(as_tuple=True)[0]
    if transition_ids.numel() > 0:
      gainprm_width = sim.model.actuator_gainprm.shape[-1]
      biasprm_width = sim.model.actuator_biasprm.shape[-1]
      zero_gain = torch.zeros(transition_ids.numel(), gainprm_width, device=device)
      zero_bias = torch.zeros(transition_ids.numel(), biasprm_width, device=device)
      sim.model.actuator_gainprm[transition_ids, act._pos_thigh_ctrl_id, :] = zero_gain
      sim.model.actuator_biasprm[transition_ids, act._pos_thigh_ctrl_id, :] = zero_bias
      sim.model.actuator_gainprm[transition_ids, act._pos_calf_ctrl_id, :] = zero_gain
      sim.model.actuator_biasprm[transition_ids, act._pos_calf_ctrl_id, :] = zero_bias
      sim.model.actuator_gainprm[transition_ids, act._aux_thigh_ctrl_id, 0] = (
        act.cfg.recovery_thigh_velocity_kd
      )
      sim.model.actuator_biasprm[transition_ids, act._aux_thigh_ctrl_id, 2] = (
        -act.cfg.recovery_thigh_velocity_kd
      )
      act.hybrid_thigh_pos_gain_active[transition_ids] = False
      state["holding"][transition_ids] = False
      state["handoff_step"][transition_ids] = state["current_step"] + 1

    if exit_ids.numel() > 0:
      sim.model.actuator_gainprm[exit_ids, act._pos_thigh_ctrl_id, :] = (
        act._orig_thigh_gainprm[exit_ids]
      )
      sim.model.actuator_biasprm[exit_ids, act._pos_thigh_ctrl_id, :] = (
        act._orig_thigh_biasprm[exit_ids]
      )
      sim.model.actuator_gainprm[exit_ids, act._pos_calf_ctrl_id, :] = (
        act._orig_calf_gainprm[exit_ids]
      )
      sim.model.actuator_biasprm[exit_ids, act._pos_calf_ctrl_id, :] = (
        act._orig_calf_biasprm[exit_ids]
      )
      sim.model.actuator_gainprm[exit_ids, act._aux_thigh_ctrl_id, 0] = 0.0
      sim.model.actuator_biasprm[exit_ids, act._aux_thigh_ctrl_id, 2] = 0.0
      act.hybrid_thigh_pos_gain_active[exit_ids] = True
      state["holding"][exit_ids] = False

    aux_active = in_recovery & (~state["holding"])
    thigh_vel_cmd = torch.where(
      aux_active,
      torch.full_like(act.hybrid_thigh_vel_ctrl, -act.cfg.recovery_thigh_velocity),
      torch.zeros_like(act.hybrid_thigh_vel_ctrl),
    )
    calf_torque_cmd = torch.where(
      aux_active,
      torch.full_like(act.hybrid_calf_torque_ctrl, -act.cfg.recovery_calf_torque),
      torch.zeros_like(act.hybrid_calf_torque_ctrl),
    )
    act.hybrid_thigh_vel_ctrl = thigh_vel_cmd
    act.hybrid_calf_torque_ctrl = calf_torque_cmd

    act._entity.set_joint_velocity_target(
      thigh_vel_cmd.unsqueeze(-1),
      joint_ids=act._hybrid_thigh_joint_ids,
    )
    act._entity.set_joint_effort_target(
      calf_torque_cmd.unsqueeze(-1),
      joint_ids=act._hybrid_calf_joint_ids,
    )

  act._apply_hybrid_actuators = delayed_apply_hybrid_actuators
  return state


def _capture_pre_entry_snapshot(
  env: ManagerBasedRlEnv,
  act: ResidualRecoveryAction,
  step: int,
) -> EntrySnapshot:
  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  hip_idx = joint_names.index("FL_hip_joint")
  thigh_idx = joint_names.index("FL_thigh_joint")
  calf_idx = joint_names.index("FL_calf_joint")
  data = robot.data
  return EntrySnapshot(
    pre_entry_step=step,
    recovery_entry_step=step,
    root_pose_w=data.root_link_pose_w[0].clone(),
    root_vel_w=data.root_link_vel_w[0].clone(),
    joint_pos=data.joint_pos[0].clone(),
    joint_vel=data.joint_vel[0].clone(),
    qpos=data.data.qpos[0].clone(),
    qvel=data.data.qvel[0].clone(),
    wire_contact=bool((env.scene.sensors[act.cfg.wire_contact_sensor_name].data.found[0] > 0).any()),
    wire_contact_force_mag_n=_wire_force_mag(env, act),
    base_z_m=float(data.root_link_pos_w[0, 2].item()),
    base_ang_vel_y_rps=float(data.root_link_ang_vel_b[0, 1].item()),
    base_lin_vel_x_mps=float(data.root_link_lin_vel_b[0, 0].item()),
    base_lin_vel_z_mps=float(data.root_link_lin_vel_b[0, 2].item()),
    fl_hip_pos_rad=float(data.joint_pos[0, hip_idx].item()),
    fl_thigh_pos_rad=float(data.joint_pos[0, thigh_idx].item()),
    fl_calf_pos_rad=float(data.joint_pos[0, calf_idx].item()),
    fl_hip_vel_radps=float(data.joint_vel[0, hip_idx].item()),
    fl_thigh_vel_radps=float(data.joint_vel[0, thigh_idx].item()),
    fl_calf_vel_radps=float(data.joint_vel[0, calf_idx].item()),
  )


def _collect_row(
  env: ManagerBasedRlEnv,
  act: ResidualRecoveryAction,
  step: int,
  label: str,
  rel_step_from_entry: int | None,
  holding: bool,
  handoff_step: int | None,
) -> dict[str, Any]:
  robot = env.scene["robot"]
  data = robot.data
  joint_names = list(robot.joint_names)
  site_names = list(robot.site_names)
  hip_idx = joint_names.index("FL_hip_joint")
  thigh_idx = joint_names.index("FL_thigh_joint")
  calf_idx = joint_names.index("FL_calf_joint")
  foot_site_idx = site_names.index("FL")

  foot_pos_w = data.site_pos_w[0, foot_site_idx, :].clone()
  base_pos_w = data.root_link_pos_w[0].clone()
  base_quat_w = data.root_link_quat_w[0].clone()
  foot_rel_w = (foot_pos_w - base_pos_w).unsqueeze(0)
  foot_pos_b = quat_apply_inverse(base_quat_w.unsqueeze(0), foot_rel_w)[0]

  wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
  wire_contact = bool((wire_sensor.data.found[0] > 0).any())
  wire_force_mag = _wire_force_mag(env, act)
  hip_wire_raw, calf_wire_raw = _raw_wire_contacts(env)

  return {
    "label": label,
    "step": step,
    "phase": PHASE_NAMES[int(act.phase[0].item())],
    "phase_id": int(act.phase[0].item()),
    "recovery_elapsed": int(act.recovery_elapsed[0].item()),
    "post_recovery_elapsed": int(act.post_recovery_elapsed[0].item()),
    "rel_step_from_entry": rel_step_from_entry if rel_step_from_entry is not None else "",
    "holding": holding,
    "handoff_step": handoff_step if handoff_step is not None else "",
    "pos_actuators_active": bool(act.hybrid_thigh_pos_gain_active[0].item()),
    "wire_contact": wire_contact,
    "wire_contact_force_mag_n": wire_force_mag,
    "fl_hip_wire_contact_raw": hip_wire_raw,
    "fl_calf_wire_contact_raw": calf_wire_raw,
    "base_x_m": float(base_pos_w[0].item()),
    "base_y_m": float(base_pos_w[1].item()),
    "base_z_m": float(base_pos_w[2].item()),
    "base_lin_vel_x_mps": float(data.root_link_lin_vel_b[0, 0].item()),
    "base_lin_vel_y_mps": float(data.root_link_lin_vel_b[0, 1].item()),
    "base_lin_vel_z_mps": float(data.root_link_lin_vel_b[0, 2].item()),
    "base_ang_vel_x_rps": float(data.root_link_ang_vel_b[0, 0].item()),
    "base_ang_vel_y_rps": float(data.root_link_ang_vel_b[0, 1].item()),
    "base_ang_vel_z_rps": float(data.root_link_ang_vel_b[0, 2].item()),
    "fl_hip_pos_rad": float(data.joint_pos[0, hip_idx].item()),
    "fl_thigh_pos_rad": float(data.joint_pos[0, thigh_idx].item()),
    "fl_calf_pos_rad": float(data.joint_pos[0, calf_idx].item()),
    "fl_hip_vel_radps": float(data.joint_vel[0, hip_idx].item()),
    "fl_thigh_vel_radps": float(data.joint_vel[0, thigh_idx].item()),
    "fl_calf_vel_radps": float(data.joint_vel[0, calf_idx].item()),
    "foot_x_base_m": float(foot_pos_b[0].item()),
    "foot_y_base_m": float(foot_pos_b[1].item()),
    "foot_z_base_m": float(foot_pos_b[2].item()),
    "fl_foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
    "fl_foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
    "fl_calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
    "fl_calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
    "crossing_achieved": bool(act.crossing_achieved[0].item()),
    "release_achieved": bool(act.release_achieved[0].item()),
    "release_counter": int(act._release_counter[0].item()),
    "recovery_failed_timeout": bool(act._recovery_failed_timeout[0].item()),
    "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0].item()),
    "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0].item()),
    "pos_thigh_actuator_force_global": float(
      env.sim.data.actuator_force[0, act._pos_thigh_ctrl_id].item()
    ),
    "aux_thigh_actuator_force_global": float(
      env.sim.data.actuator_force[0, act._aux_thigh_ctrl_id].item()
    ),
    "pos_calf_actuator_force_global": float(
      env.sim.data.actuator_force[0, act._pos_calf_ctrl_id].item()
    ),
    "aux_calf_actuator_force_global": float(
      env.sim.data.actuator_force[0, act._aux_calf_ctrl_id].item()
    ),
  }


def _run_baseline(cfg: D549Config, out_dir: Path) -> RunResult:
  env = _build_env(cfg.seed, cfg.device, cfg.max_recovery_steps_override)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  env.reset(seed=cfg.seed)

  rows: list[dict[str, Any]] = []
  snapshot: EntrySnapshot | None = None
  entry_step: int | None = None
  pre_entry_base_z_values: list[float] = []
  pre_entry_pitch_values: list[float] = []

  for step in range(cfg.max_steps):
    phase_before = int(act.phase[0].item())
    if phase_before == PHASE_NORMAL_WALK:
      pre_entry_base_z_values.append(float(env.scene["robot"].data.root_link_pos_w[0, 2].item()))
      pre_entry_pitch_values.append(float(env.scene["robot"].data.root_link_ang_vel_b[0, 1].item()))
    if snapshot is None and bool((env.scene.sensors[act.cfg.wire_contact_sensor_name].data.found[0] > 0).any()):
      snapshot = _capture_pre_entry_snapshot(env, act, step)
      entry_step = step

    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    env.step(actions)

    rel_step = None if entry_step is None else step - entry_step
    rows.append(
      _collect_row(
        env,
        act,
        step=step,
        label="baseline",
        rel_step_from_entry=rel_step,
        holding=False,
        handoff_step=entry_step,
      )
    )

  env.close()

  assert snapshot is not None and entry_step is not None, "Baseline never reached first contact"
  trace_path = out_dir / "baseline_trace.csv"
  _write_csv(trace_path, rows)

  summary = _summarize_run(
    label="baseline",
    rows=rows,
    pre_entry_base_z_values=pre_entry_base_z_values,
    pre_entry_pitch_values=pre_entry_pitch_values,
    entry_step=entry_step,
    handoff_step=entry_step,
    restore_info=None,
  )
  summary["trace_csv"] = str(trace_path)
  return RunResult("baseline", rows, snapshot, summary)


def _run_counterfactual(
  cfg: D549Config,
  out_dir: Path,
  snapshot: EntrySnapshot,
) -> RunResult:
  env = _build_env(cfg.seed, cfg.device, cfg.max_recovery_steps_override)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  hold_state = install_fixed_delay_handoff(act, cfg.hold_steps)
  env.reset(seed=cfg.seed)
  restore_info = _restore_snapshot(env, snapshot, act.cfg.wire_contact_sensor_name)

  rows: list[dict[str, Any]] = []
  entry_step: int | None = None
  handoff_step: int | None = None

  start_step = snapshot.pre_entry_step
  end_step = min(cfg.max_steps, start_step + cfg.max_steps)

  for step in range(start_step, end_step):
    hold_state["current_step"] = step
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    env.step(actions)

    if entry_step is None and int(act.phase[0].item()) == PHASE_RECOVERY:
      entry_step = step
    raw_handoff = int(hold_state["handoff_step"][0].item())
    if raw_handoff >= 0:
      handoff_step = raw_handoff

    rel_step = None if entry_step is None else step - entry_step
    rows.append(
      _collect_row(
        env,
        act,
        step=step,
        label="counterfactual",
        rel_step_from_entry=rel_step,
        holding=bool(hold_state["holding"][0].item()),
        handoff_step=handoff_step,
      )
    )

  env.close()

  assert entry_step is not None, "Counterfactual never entered recovery"
  assert handoff_step is not None, "Counterfactual never handed off to hybrid"

  trace_path = out_dir / "counterfactual_trace.csv"
  _write_csv(trace_path, rows)

  summary = _summarize_run(
    label="counterfactual",
    rows=rows,
    pre_entry_base_z_values=[snapshot.base_z_m],
    pre_entry_pitch_values=[snapshot.base_ang_vel_y_rps],
    entry_step=entry_step,
    handoff_step=handoff_step,
    restore_info=restore_info,
  )
  summary["trace_csv"] = str(trace_path)
  return RunResult("counterfactual", rows, snapshot, summary)


def _summarize_run(
  label: str,
  rows: list[dict[str, Any]],
  pre_entry_base_z_values: list[float],
  pre_entry_pitch_values: list[float],
  entry_step: int,
  handoff_step: int,
  restore_info: dict[str, float] | None,
) -> dict[str, Any]:
  entry_rows = [row for row in rows if row["step"] >= entry_step]
  baseline_base_z_min = min(pre_entry_base_z_values)
  baseline_pitch_abs_max = max(abs(v) for v in pre_entry_pitch_values)

  min_base_row = min(entry_rows, key=lambda row: row["base_z_m"])
  max_pitch_row = max(entry_rows, key=lambda row: abs(row["base_ang_vel_y_rps"]))
  max_wire_force_row = max(entry_rows, key=lambda row: row["wire_contact_force_mag_n"])

  first_base_drop_step = next(
    (
      row["step"]
      for row in entry_rows
      if row["base_z_m"] < baseline_base_z_min
    ),
    -1,
  )
  first_pitch_excursion_step = next(
    (
      row["step"]
      for row in entry_rows
      if abs(row["base_ang_vel_y_rps"]) > baseline_pitch_abs_max
    ),
    -1,
  )

  return {
    "label": label,
    "entry_step": entry_step,
    "handoff_step": handoff_step,
    "delay_steps": handoff_step - entry_step,
    "pre_entry_base_z_min_m": baseline_base_z_min,
    "pre_entry_abs_pitch_rate_max_rps": baseline_pitch_abs_max,
    "first_base_z_excursion_step": first_base_drop_step,
    "first_pitch_rate_excursion_step": first_pitch_excursion_step,
    "min_base_z_m": min_base_row["base_z_m"],
    "min_base_z_step": min_base_row["step"],
    "max_abs_pitch_rate_rps": abs(max_pitch_row["base_ang_vel_y_rps"]),
    "max_abs_pitch_rate_step": max_pitch_row["step"],
    "max_wire_force_mag_n": max_wire_force_row["wire_contact_force_mag_n"],
    "max_wire_force_step": max_wire_force_row["step"],
    "first_hip_wire_step": next(
      (row["step"] for row in entry_rows if row["fl_hip_wire_contact_raw"]),
      -1,
    ),
    "first_calf_wire_step": next(
      (row["step"] for row in entry_rows if row["fl_calf_wire_contact_raw"]),
      -1,
    ),
    "first_wire_contact_step": next(
      (row["step"] for row in entry_rows if row["wire_contact"]),
      -1,
    ),
    "wire_contact_steps": sum(1 for row in entry_rows if row["wire_contact"]),
    "crossing_achieved": any(row["crossing_achieved"] for row in entry_rows),
    "release_achieved": any(row["release_achieved"] for row in entry_rows),
    "restore_qpos_max_abs_diff": (
      restore_info["qpos_max_abs_diff"] if restore_info is not None else float("nan")
    ),
    "restore_qvel_max_abs_diff": (
      restore_info["qvel_max_abs_diff"] if restore_info is not None else float("nan")
    ),
    "restore_wire_contact": (
      restore_info["wire_contact_after_restore"] if restore_info is not None else float("nan")
    ),
    "restore_wire_force_mag_n": (
      restore_info["wire_contact_force_mag_n_after_restore"]
      if restore_info is not None
      else float("nan")
    ),
  }


def _aligned_window(
  baseline_rows: list[dict[str, Any]],
  counterfactual_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  def by_rel_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
      int(row["rel_step_from_entry"]): row
      for row in rows
      if row["rel_step_from_entry"] != ""
    }

  base_map = by_rel_step(baseline_rows)
  cf_map = by_rel_step(counterfactual_rows)
  rel_steps = sorted(set(base_map) | set(cf_map))
  aligned: list[dict[str, Any]] = []
  for rel_step in rel_steps:
    base = base_map.get(rel_step)
    cf = cf_map.get(rel_step)
    row: dict[str, Any] = {"rel_step": rel_step}
    for prefix, source in (("baseline", base), ("counterfactual", cf)):
      if source is None:
        for key in (
          "step",
          "phase",
          "wire_contact",
          "wire_contact_force_mag_n",
          "base_z_m",
          "base_ang_vel_y_rps",
          "base_lin_vel_z_mps",
          "fl_hip_pos_rad",
          "fl_thigh_pos_rad",
          "fl_calf_pos_rad",
          "fl_hip_vel_radps",
          "fl_thigh_vel_radps",
          "fl_calf_vel_radps",
          "holding",
          "pos_actuators_active",
        ):
          row[f"{prefix}_{key}"] = ""
      else:
        for key in (
          "step",
          "phase",
          "wire_contact",
          "wire_contact_force_mag_n",
          "base_z_m",
          "base_ang_vel_y_rps",
          "base_lin_vel_z_mps",
          "fl_hip_pos_rad",
          "fl_thigh_pos_rad",
          "fl_calf_pos_rad",
          "fl_hip_vel_radps",
          "fl_thigh_vel_radps",
          "fl_calf_vel_radps",
          "holding",
          "pos_actuators_active",
        ):
          row[f"{prefix}_{key}"] = source[key]
    if base is not None and cf is not None:
      row["delta_base_z_m"] = cf["base_z_m"] - base["base_z_m"]
      row["delta_base_ang_vel_y_rps"] = (
        cf["base_ang_vel_y_rps"] - base["base_ang_vel_y_rps"]
      )
      row["delta_wire_force_mag_n"] = (
        cf["wire_contact_force_mag_n"] - base["wire_contact_force_mag_n"]
      )
    aligned.append(row)
  return aligned


def _entry_snapshot_row(snapshot: EntrySnapshot) -> dict[str, Any]:
  return {
    "pre_entry_step": snapshot.pre_entry_step,
    "recovery_entry_step": snapshot.recovery_entry_step,
    "wire_contact": snapshot.wire_contact,
    "wire_contact_force_mag_n": snapshot.wire_contact_force_mag_n,
    "base_z_m": snapshot.base_z_m,
    "base_ang_vel_y_rps": snapshot.base_ang_vel_y_rps,
    "base_lin_vel_x_mps": snapshot.base_lin_vel_x_mps,
    "base_lin_vel_z_mps": snapshot.base_lin_vel_z_mps,
    "fl_hip_pos_rad": snapshot.fl_hip_pos_rad,
    "fl_thigh_pos_rad": snapshot.fl_thigh_pos_rad,
    "fl_calf_pos_rad": snapshot.fl_calf_pos_rad,
    "fl_hip_vel_radps": snapshot.fl_hip_vel_radps,
    "fl_thigh_vel_radps": snapshot.fl_thigh_vel_radps,
    "fl_calf_vel_radps": snapshot.fl_calf_vel_radps,
  }


def _write_report(
  out_dir: Path,
  cfg: D549Config,
  baseline: RunResult,
  counterfactual: RunResult,
) -> None:
  base = baseline.summary
  cf = counterfactual.summary

  def fmt_step(step: Any) -> str:
    return "n/a" if step in (-1, None, "") else str(step)

  lines = [
    "# D5.49 — Recovery Override Counterfactual",
    "",
    "Diagnostic only. This experiment compares the existing seed-0 baseline against a replay from the exact same verified pre-recovery simulator state, but with the normal walking position actuators kept active for a short fixed window after first contact before the existing hybrid override is allowed to take over.",
    "",
    "## Configuration",
    "",
    f"- Seed: {cfg.seed}",
    f"- Device: {cfg.device}",
    f"- Fixed post-contact hold: {cfg.hold_steps} recovery steps",
    f"- Recovery cap: {cfg.max_recovery_steps_override} steps",
    "",
    "## Artifacts",
    "",
    f"- Baseline trace: [baseline_trace.csv](baseline_trace.csv)",
    f"- Counterfactual trace: [counterfactual_trace.csv](counterfactual_trace.csv)",
    f"- Shared entry snapshot: [entry_snapshot.csv](entry_snapshot.csv)",
    f"- Summary metrics: [comparison_summary.csv](comparison_summary.csv)",
    f"- Aligned window: [aligned_entry_window.csv](aligned_entry_window.csv)",
    "",
    "## What changed in Run B",
    "",
    "- The same contact-triggered RECOVERY FSM still starts at first contact.",
    "- The wire, trigger, walking checkpoint, and hybrid commands are unchanged.",
    "- Only the timing of the actuator handoff changes: thigh/calf position actuators stay active for a short fixed window, then the original hybrid override takes over.",
    "",
    "## Restore validation",
    "",
    f"- Pre-entry snapshot step: {baseline.entry_snapshot.pre_entry_step if baseline.entry_snapshot else 'n/a'}",
    f"- Counterfactual restore qpos max abs diff: {cf['restore_qpos_max_abs_diff']:.3e}",
    f"- Counterfactual restore qvel max abs diff: {cf['restore_qvel_max_abs_diff']:.3e}",
    f"- Counterfactual restore wire contact / force: {bool(cf['restore_wire_contact'])} / {cf['restore_wire_force_mag_n']:.4f} N",
    "",
    "## Comparison",
    "",
    "| Metric | Baseline | Counterfactual |",
    "|---|---:|---:|",
    f"| Recovery entry step | {base['entry_step']} | {cf['entry_step']} |",
    f"| Hybrid handoff step | {base['handoff_step']} | {cf['handoff_step']} |",
    f"| Delay after entry | {base['delay_steps']} | {cf['delay_steps']} |",
    f"| First base-height excursion step | {fmt_step(base['first_base_z_excursion_step'])} | {fmt_step(cf['first_base_z_excursion_step'])} |",
    f"| First pitch-rate excursion step | {fmt_step(base['first_pitch_rate_excursion_step'])} | {fmt_step(cf['first_pitch_rate_excursion_step'])} |",
    f"| Minimum base height (m) | {base['min_base_z_m']:.4f} @ {base['min_base_z_step']} | {cf['min_base_z_m']:.4f} @ {cf['min_base_z_step']} |",
    f"| Peak |pitch rate| (rad/s) | {base['max_abs_pitch_rate_rps']:.4f} @ {base['max_abs_pitch_rate_step']} | {cf['max_abs_pitch_rate_rps']:.4f} @ {cf['max_abs_pitch_rate_step']} |",
    f"| Max wire force magnitude (N) | {base['max_wire_force_mag_n']:.4f} @ {base['max_wire_force_step']} | {cf['max_wire_force_mag_n']:.4f} @ {cf['max_wire_force_step']} |",
    f"| First calf-wire raw contact step | {fmt_step(base['first_calf_wire_step'])} | {fmt_step(cf['first_calf_wire_step'])} |",
    f"| First hip-wire raw contact step | {fmt_step(base['first_hip_wire_step'])} | {fmt_step(cf['first_hip_wire_step'])} |",
    f"| Crossing achieved | {base['crossing_achieved']} | {cf['crossing_achieved']} |",
    f"| Release achieved | {base['release_achieved']} | {cf['release_achieved']} |",
    "",
    "## Interpretation",
    "",
    (
      "- If the counterfactual substantially delays or reduces the first pitch-rate and base-height excursions while the wire-contact history stays comparable, that points to the hybrid handoff itself as a major amplifier of the stumble."
    ),
    (
      "- If both runs destabilize at nearly the same time and magnitude despite the delayed handoff, then wire contact / entanglement mechanics dominate and the override timing is secondary."
    ),
    "",
    "## Observed result",
    "",
    (
      f"- Baseline reaches its first pitch-rate excursion at step {fmt_step(base['first_pitch_rate_excursion_step'])} and first base-height excursion at step {fmt_step(base['first_base_z_excursion_step'])}."
    ),
    (
      f"- Counterfactual reaches its first pitch-rate excursion at step {fmt_step(cf['first_pitch_rate_excursion_step'])} and first base-height excursion at step {fmt_step(cf['first_base_z_excursion_step'])}."
    ),
    (
      f"- Baseline minimum base height is {base['min_base_z_m']:.4f} m versus {cf['min_base_z_m']:.4f} m in the counterfactual."
    ),
    (
      f"- Baseline peak |pitch rate| is {base['max_abs_pitch_rate_rps']:.4f} rad/s versus {cf['max_abs_pitch_rate_rps']:.4f} rad/s in the counterfactual."
    ),
    (
      f"- The counterfactual delays the hybrid handoff by {cf['delay_steps']} recovery steps."
    ),
  ]

  if cf["min_base_z_m"] > base["min_base_z_m"] and cf["max_abs_pitch_rate_rps"] < base["max_abs_pitch_rate_rps"]:
    lines.extend(
      [
        "",
        "## Bottom line",
        "",
        "- The delayed-handoff counterfactual reduces the severity of the initial stumble relative to the unchanged baseline, which supports the hypothesis that the hybrid override materially worsens the early instability after wire contact.",
      ]
    )
  else:
    lines.extend(
      [
        "",
        "## Bottom line",
        "",
        "- The delayed-handoff counterfactual does not materially reduce the early instability relative to baseline, so wire-contact mechanics remain the stronger explanation than the hybrid handoff itself.",
      ]
    )

  (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
  import tyro

  configure_torch_backends()
  cfg = tyro.cli(D549Config)
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  baseline = _run_baseline(cfg, out_dir)
  assert baseline.entry_snapshot is not None
  counterfactual = _run_counterfactual(cfg, out_dir, baseline.entry_snapshot)

  _write_csv(out_dir / "entry_snapshot.csv", [_entry_snapshot_row(baseline.entry_snapshot)])
  _write_csv(
    out_dir / "comparison_summary.csv",
    [baseline.summary, counterfactual.summary],
  )
  _write_csv(
    out_dir / "aligned_entry_window.csv",
    _aligned_window(baseline.trace_rows, counterfactual.trace_rows),
  )
  _write_report(out_dir, cfg, baseline, counterfactual)

  print(f"Wrote D5.49 outputs to {out_dir}")
  print(
    "baseline:",
    f"entry={baseline.summary['entry_step']}",
    f"min_base_z={baseline.summary['min_base_z_m']:.4f}",
    f"max_abs_pitch={baseline.summary['max_abs_pitch_rate_rps']:.4f}",
  )
  print(
    "counterfactual:",
    f"entry={counterfactual.summary['entry_step']}",
    f"handoff={counterfactual.summary['handoff_step']}",
    f"min_base_z={counterfactual.summary['min_base_z_m']:.4f}",
    f"max_abs_pitch={counterfactual.summary['max_abs_pitch_rate_rps']:.4f}",
  )


if __name__ == "__main__":
  main()
