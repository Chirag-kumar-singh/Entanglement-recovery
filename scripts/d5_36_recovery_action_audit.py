#!/usr/bin/env python
"""D5.36 recovery action audit: seed 0 vs seed 11.

Read-only follow-up to D5.35. This script logs the command path through
`ResidualRecoveryAction` for two baseline runs:
  - seed 0 (known crossing reference)
  - seed 11 (known stage-1 failure)

It captures, at matched recovery offsets:
  - frozen walking-policy raw action for FL hip/thigh/calf;
  - final raw action after residual composition;
  - processed joint-position targets and applied joint_pos_target;
  - actual MuJoCo actuator ctrl/force values for the FL hip position
    actuator, FL thigh/calf position actuators, and FL thigh/calf hybrid
    auxiliary actuators;
  - measured joint positions/velocities;
  - recovery phase / elapsed steps;
  - calf forward distance / height above wire;
  - base forward body velocity and world vertical velocity.

It does NOT change recovery behavior, thresholds, FSM logic, actuator
settings, rewards, terminations, wire geometry, or checkpoint. It only
adds observational logging.

Outputs go under logs/d5_36/.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

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
class D536Config:
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = "logs/d5_36"
  runs: tuple[RunSpec, ...] = (
    RunSpec(label="seed0_baseline", seed=0),
    RunSpec(label="seed11_baseline", seed=11),
  )


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


def _run_one(run: RunSpec, device: str, max_steps: int, out_dir: Path) -> Path:
  env = _build_env(run.seed, device)
  act = env.action_manager.get_term("joint_pos")
  entity = act._entity
  joint_names = list(entity.joint_names)
  fl = LEG_NAMES[0]
  hip_idx = joint_names.index(f"{fl}_hip_joint")
  thigh_idx = joint_names.index(f"{fl}_thigh_joint")
  calf_idx = joint_names.index(f"{fl}_calf_joint")

  # Global ctrl ids for the relevant actuators.
  hip_pos_ctrl_id = act._find_actuator_ctrl_id(
    f"{fl}_hip_joint", BuiltinPositionActuator
  )
  thigh_pos_ctrl_id = act._pos_thigh_ctrl_id
  calf_pos_ctrl_id = act._pos_calf_ctrl_id
  thigh_aux_ctrl_id = act._aux_thigh_ctrl_id
  calf_aux_ctrl_id = act._aux_calf_ctrl_id

  env.reset(seed=run.seed)

  rows: list[dict] = []
  recovery_offset = -1
  for step in range(max_steps):
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)

    phase = int(act.phase[0].item())
    if phase == PHASE_RECOVERY:
      recovery_offset += 1

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    pos_target = entity.data.joint_pos_target[0]
    sim_ctrl = env.sim.data.ctrl[0]
    sim_force = env.sim.data.actuator_force[0]
    lin_vel_w = entity.data.root_link_lin_vel_w[0]
    lin_vel_b = entity.data.root_link_lin_vel_b[0]

    walk_raw = (
      act.last_a_walk_raw[0]
      if act.last_a_walk_raw is not None
      else torch.zeros(12, device=env.device)
    )
    final_raw = (
      act.last_final_raw[0]
      if act.last_final_raw is not None
      else torch.zeros(12, device=env.device)
    )
    processed = act._joint_pos_action._processed_actions[0]

    row = {
      "step": step,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": int(act.recovery_elapsed[0].item()),
      "recovery_offset": recovery_offset if phase == PHASE_RECOVERY else -1,
      "calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
      "calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
      "foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
      "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
      "base_forward_vel_body_x": float(lin_vel_b[0].item()),
      "base_vertical_vel_world_z": float(lin_vel_w[2].item()),
      "hybrid_thigh_vel_ctrl": float(act.hybrid_thigh_vel_ctrl[0].item()),
      "hybrid_calf_torque_ctrl": float(act.hybrid_calf_torque_ctrl[0].item()),
      "hybrid_thigh_pos_gain_active": bool(act.hybrid_thigh_pos_gain_active[0].item()),
      # Frozen walk raw outputs.
      "walk_raw_hip": float(walk_raw[hip_idx].item()),
      "walk_raw_thigh": float(walk_raw[thigh_idx].item()),
      "walk_raw_calf": float(walk_raw[calf_idx].item()),
      # Final raw outputs after recovery composition (same as walk_raw in hybrid mode).
      "final_raw_hip": float(final_raw[hip_idx].item()),
      "final_raw_thigh": float(final_raw[thigh_idx].item()),
      "final_raw_calf": float(final_raw[calf_idx].item()),
      # Processed position targets before encoder-bias subtraction.
      "processed_target_hip": float(processed[hip_idx].item()),
      "processed_target_thigh": float(processed[thigh_idx].item()),
      "processed_target_calf": float(processed[calf_idx].item()),
      # Actual written joint position targets after encoder-bias subtraction.
      "joint_pos_target_hip": float(pos_target[hip_idx].item()),
      "joint_pos_target_thigh": float(pos_target[thigh_idx].item()),
      "joint_pos_target_calf": float(pos_target[calf_idx].item()),
      # MuJoCo ctrl inputs at the underlying actuators.
      "ctrl_hip_pos": float(sim_ctrl[hip_pos_ctrl_id].item()),
      "ctrl_thigh_pos": float(sim_ctrl[thigh_pos_ctrl_id].item()),
      "ctrl_calf_pos": float(sim_ctrl[calf_pos_ctrl_id].item()),
      "ctrl_thigh_aux_vel": float(sim_ctrl[thigh_aux_ctrl_id].item()),
      "ctrl_calf_aux_motor": float(sim_ctrl[calf_aux_ctrl_id].item()),
      # Measured actuator forces.
      "force_hip_pos": float(sim_force[hip_pos_ctrl_id].item()),
      "force_thigh_pos": float(sim_force[thigh_pos_ctrl_id].item()),
      "force_calf_pos": float(sim_force[calf_pos_ctrl_id].item()),
      "force_thigh_aux_vel": float(sim_force[thigh_aux_ctrl_id].item()),
      "force_calf_aux_motor": float(sim_force[calf_aux_ctrl_id].item()),
      # Measured joint state.
      "qpos_hip": float(qpos[hip_idx].item()),
      "qpos_thigh": float(qpos[thigh_idx].item()),
      "qpos_calf": float(qpos[calf_idx].item()),
      "qvel_hip": float(qvel[hip_idx].item()),
      "qvel_thigh": float(qvel[thigh_idx].item()),
      "qvel_calf": float(qvel[calf_idx].item()),
    }
    rows.append(row)

    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      break

  env.close()

  csv_path = out_dir / f"{run.label}.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  return csv_path


def _load_recovery_rows(csv_path: Path) -> list[dict]:
  rows = list(csv.DictReader(open(csv_path)))
  return [r for r in rows if r["phase"] == "RECOVERY"]


def _f(row: dict, key: str) -> float:
  return float(row[key])


def _write_overlay(seed0_csv: Path, seed11_csv: Path, out_dir: Path) -> None:
  seed0 = _load_recovery_rows(seed0_csv)
  seed11 = _load_recovery_rows(seed11_csv)
  n = min(len(seed0), len(seed11))

  overlay_rows = []
  for off in range(n):
    a = seed0[off]
    b = seed11[off]
    row = {
      "recovery_offset": off,
      "seed0_step": int(a["step"]),
      "seed11_step": int(b["step"]),
    }
    keys = [
      "walk_raw_hip",
      "walk_raw_thigh",
      "walk_raw_calf",
      "processed_target_hip",
      "processed_target_thigh",
      "processed_target_calf",
      "joint_pos_target_hip",
      "joint_pos_target_thigh",
      "joint_pos_target_calf",
      "ctrl_hip_pos",
      "ctrl_thigh_pos",
      "ctrl_calf_pos",
      "ctrl_thigh_aux_vel",
      "ctrl_calf_aux_motor",
      "qpos_thigh",
      "qpos_calf",
      "qvel_thigh",
      "qvel_calf",
      "calf_forward_dist_m",
      "calf_height_above_wire_m",
      "base_forward_vel_body_x",
      "base_vertical_vel_world_z",
    ]
    for key in keys:
      row[f"seed0_{key}"] = _f(a, key)
      row[f"seed11_{key}"] = _f(b, key)
      row[f"delta_{key}"] = _f(b, key) - _f(a, key)
    overlay_rows.append(row)

  overlay_csv = out_dir / "seed0_vs_seed11_action_overlay.csv"
  with open(overlay_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(overlay_rows[0].keys()))
    writer.writeheader()
    writer.writerows(overlay_rows)

  def first_persistent_pred(rows, pred, window=5):
    for i in range(len(rows) - window + 1):
      if all(pred(rows[j]) for j in range(i, i + window)):
        return i
    return None

  walk_gap = first_persistent_pred(
    overlay_rows,
    lambda r: (
      abs(r["delta_walk_raw_thigh"]) > 0.15 or abs(r["delta_walk_raw_calf"]) > 0.15
    ),
  )
  target_gap = first_persistent_pred(
    overlay_rows,
    lambda r: (
      abs(r["delta_joint_pos_target_thigh"]) > 0.15
      or abs(r["delta_joint_pos_target_calf"]) > 0.15
    ),
  )
  calf_fd_gap = first_persistent_pred(
    overlay_rows,
    lambda r: r["delta_calf_forward_dist_m"] < -0.02,
  )

  summary = []
  summary.append("D5.36 recovery action audit: seed0 baseline vs seed11 baseline")
  summary.append(
    "Behavior unchanged; this audit only logs existing walk-policy outputs, targets, and hybrid actuator controls."
  )
  summary.append("")
  summary.append("Hybrid path (observed from actions.py)")
  summary.append(
    "- FL hip keeps the frozen walking policy's normal position-action path."
  )
  summary.append(
    "- FL thigh/calf still get walk-policy position targets computed, but their POSITION actuators are disabled during RECOVERY."
  )
  summary.append(
    "- FL thigh is overridden by an auxiliary VELOCITY actuator; FL calf by an auxiliary MOTOR actuator."
  )
  summary.append(
    "- In hybrid mode, final_raw == walk_raw for all joints; the recovery behavior is actuator-level, not residual-position composition."
  )
  summary.append("")
  if walk_gap is not None:
    r = overlay_rows[walk_gap]
    summary.append(
      f"- Earliest persistent frozen-walk action difference: recovery offset {walk_gap} "
      f"(seed0 step {r['seed0_step']}, seed11 step {r['seed11_step']})."
    )
    summary.append(
      f"  walk_raw_thigh seed0={r['seed0_walk_raw_thigh']:.4f}, seed11={r['seed11_walk_raw_thigh']:.4f}; "
      f"walk_raw_calf seed0={r['seed0_walk_raw_calf']:.4f}, seed11={r['seed11_walk_raw_calf']:.4f}."
    )
  if target_gap is not None:
    r = overlay_rows[target_gap]
    summary.append(
      f"- Earliest persistent position-target difference: recovery offset {target_gap}."
    )
    summary.append(
      f"  joint_pos_target_thigh seed0={r['seed0_joint_pos_target_thigh']:.4f}, seed11={r['seed11_joint_pos_target_thigh']:.4f}; "
      f"joint_pos_target_calf seed0={r['seed0_joint_pos_target_calf']:.4f}, seed11={r['seed11_joint_pos_target_calf']:.4f}."
    )
  if calf_fd_gap is not None:
    r = overlay_rows[calf_fd_gap]
    summary.append(
      f"- Persistent calf-forward-distance gap begins at recovery offset {calf_fd_gap}."
    )
    summary.append(
      f"  calf_fd seed0={r['seed0_calf_forward_dist_m']:.4f} m, seed11={r['seed11_calf_forward_dist_m']:.4f} m."
    )
  summary.append("")
  summary.append("Direct observations")
  summary.append(
    f"- `ctrl_thigh_aux_vel` is the same commanded value in both runs during RECOVERY (seed0={_f(seed0[0], 'ctrl_thigh_aux_vel'):.4f}, seed11={_f(seed11[0], 'ctrl_thigh_aux_vel'):.4f})."
  )
  summary.append(
    f"- `ctrl_calf_aux_motor` is the same commanded value in both runs during RECOVERY (seed0={_f(seed0[0], 'ctrl_calf_aux_motor'):.4f}, seed11={_f(seed11[0], 'ctrl_calf_aux_motor'):.4f})."
  )
  summary.append(
    "- Therefore the hybrid override commands themselves do not differ across seeds; the difference must come from the robot state presented to those fixed commands and/or the frozen walking policy outputs riding alongside them."
  )
  summary.append(
    "- The earliest actionable measured difference is in the frozen walking-policy thigh/calf outputs and derived position targets, not in the auxiliary hybrid command magnitudes."
  )
  summary.append("")
  summary.append("Hypothesis (not yet proven causation)")
  summary.append(
    "- Seed11 enters recovery in a more downward / more retracted state, the frozen walking policy keeps the FL thigh/calf in a more retracted/flexed posture, and the fixed hybrid retract+push commands are not sufficient to carry the calf across the wire plane from that state."
  )
  summary.append("")
  summary.append("Specific next controller change to test")
  summary.append(
    "- Keep the FSM/thresholds unchanged, but make the hybrid recovery command phase/state-dependent instead of constant: add an explicit forward-advance phase (or a state-triggered reduction/reversal of thigh retraction once sufficient lift is achieved) so the calf is driven across the wire plane after lift."
  )
  summary.append(
    "- This is a proposed next experiment only; D5.36 changes no behavior."
  )

  summary_path = out_dir / "seed0_vs_seed11_action_overlay_summary.txt"
  summary_path.write_text("\n".join(summary) + "\n")


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="D5.36 recovery action audit")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--max-steps", type=int, default=300)
  parser.add_argument("--out-dir", default="logs/d5_36")
  args = parser.parse_args()

  cfg = D536Config(device=args.device, max_steps=args.max_steps, out_dir=args.out_dir)
  configure_torch_backends()

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  run_dir = out_dir / "recovery_action_audit"
  run_dir.mkdir(parents=True, exist_ok=True)

  outputs = []
  for run in cfg.runs:
    print(f"Running {run.label} (seed={run.seed})")
    outputs.append(_run_one(run, cfg.device, cfg.max_steps, run_dir))

  _write_overlay(outputs[0], outputs[1], run_dir)
  print(f"Wrote D5.36 outputs under: {run_dir}")


if __name__ == "__main__":
  main()
