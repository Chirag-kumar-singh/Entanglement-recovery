"""D3 ablation experiment: calf-torque timing/duration test.

TEMPORARY, non-permanent experiment. Does NOT modify
`mjlab.tasks.recovery.mdp.actions` or any default config. The ablation
is implemented entirely in this script via a thin runtime wrapper around
the action term's OWN `_apply_hybrid_actuators` method (monkey-patched
on the instance only, for the duration of this script's env, never
touching the class or the file on disk): the original method is called
unmodified every step (so entry/exit FSM logic, thigh velocity control,
gain toggling, and restoration are all exactly as validated in Test
A-D), and ONLY the calf feedforward torque is additionally zeroed once
`recovery_elapsed >= calf_torque_stop_after_steps`, by directly
overwriting `act.hybrid_calf_torque_ctrl` and re-issuing
`entity.set_joint_effort_target(0, ...)` for the affected envs.

Runs BOTH the baseline (calf torque active all 16 recovery steps) and
the ablation (calf torque active only for the first 5 recovery steps)
with identical seed/leg/checkpoint/wire/thigh-velocity, and reports a
step-by-step comparison plus the ablation's own interpretation
(STEP 4/5 of the spec).
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import (
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
)

PHASE_NAMES = {PHASE_NORMAL_WALK: "NORMAL_WALK", PHASE_RECOVERY: "RECOVERY", PHASE_POST_RECOVERY: "POST_RECOVERY"}
SEED = 0
MAX_STEPS = 400
STOP_AFTER_POST_RECOVERY = 60
CALF_TORQUE_STOP_AFTER_STEPS = 5  # per spec: baseline peak was ~5 steps in


def make_env(device: str):
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0, num_envs=1, recovery_mode="hybrid", policy_device=device, play=True
  )
  cfg.seed = SEED
  cfg.sim.device = device
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  return env


def install_calf_ablation(act, stop_after_steps: int):
  """Monkey-patch ONLY this instance's `_apply_hybrid_actuators` to zero
  the calf feedforward torque once `recovery_elapsed >= stop_after_steps`
  for envs currently in RECOVERY. The original bound method is called
  first, unmodified, so all gain-toggle/entry/exit logic is identical to
  the validated baseline; this wrapper only overrides the calf ctrl
  AFTER the original method has run, each step."""
  original = act._apply_hybrid_actuators
  entity = act._entity
  calf_joint_id = act._hybrid_calf_joint_id

  def wrapped():
    original()
    in_recovery = act._phase == PHASE_RECOVERY
    stop_mask = in_recovery & (act.recovery_elapsed >= stop_after_steps)
    if stop_mask.any():
      act.hybrid_calf_torque_ctrl = torch.where(
        stop_mask, torch.zeros_like(act.hybrid_calf_torque_ctrl), act.hybrid_calf_torque_ctrl
      )
      zero_col = torch.zeros(act.num_envs, 1, device=act.device)
      # Re-issue the effort target for ALL envs each call (cheap, 1 env
      # here); envs not in stop_mask keep getting the normal nonzero
      # torque from `original()` above, which already wrote the correct
      # per-env command into the entity's effort-target buffer via
      # set_joint_effort_target -- we only need to re-stomp the envs
      # that should now be zero.
      current = act.hybrid_calf_torque_ctrl.unsqueeze(-1)
      entity.set_joint_effort_target(current, joint_ids=[calf_joint_id])

  act._apply_hybrid_actuators = wrapped


def build_row(step, env, act, running_max_h):
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  qpos = entity.data.joint_pos[0]
  qvel = entity.data.joint_vel[0]
  foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
  phase = int(act.phase[0])
  sim = env.sim
  sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
  in_contact = bool((sensor.data.found[0] > 0).any())
  running_max_h = max(running_max_h, float(foot_pos[2]))
  base_pos = entity.data.root_link_pos_w[0].clone()
  base_quat = entity.data.root_link_quat_w[0].clone()
  row = {
    "step": step,
    "recovery_elapsed": int(act.recovery_elapsed[0]) if phase == PHASE_RECOVERY else -1,
    "phase": PHASE_NAMES[phase],
    "in_contact": in_contact,
    "thigh_pos": float(qpos[thigh_i]),
    "thigh_vel": float(qvel[thigh_i]),
    "calf_pos": float(qpos[calf_i]),
    "calf_vel": float(qvel[calf_i]),
    "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
    "aux_calf_actuator_force": float(sim.data.actuator_force[0, act._aux_calf_ctrl_id]),
    "foot_x": float(foot_pos[0]),
    "foot_z": float(foot_pos[2]),
    "foot_height_above_wire": float(act.foot_height_above_wire[0]),
    "foot_forward_dist_to_wire": float(act.foot_forward_dist[0]),
    "crossing_achieved": bool(act.crossing_achieved[0]),
    "base_z": float(base_pos[2]),
    "base_quat": base_quat.tolist(),
    "running_max_foot_height": running_max_h,
  }
  return row, running_max_h


def run_one(label: str, ablate: bool, device: str = "cpu"):
  torch.manual_seed(SEED)
  env = make_env(device)
  act = env.action_manager._terms["joint_pos"]
  if ablate:
    install_calf_ablation(act, CALF_TORQUE_STOP_AFTER_STEPS)

  env.reset(seed=SEED)
  rows = []
  running_max_h = -1e9
  reached_post_recovery_at = None
  for i in range(MAX_STEPS):
    action = torch.zeros(1, act.action_dim, device=device)
    env.step(action)
    row, running_max_h = build_row(i, env, act, running_max_h)
    rows.append(row)
    if row["phase"] == "POST_RECOVERY" and reached_post_recovery_at is None:
      reached_post_recovery_at = i
    if reached_post_recovery_at is not None and i - reached_post_recovery_at >= STOP_AFTER_POST_RECOVERY:
      break
  env.close()

  rec_rows = [r for r in rows if r["phase"] == "RECOVERY"]
  summary = dict(
    label=label,
    n_recovery_rows=len(rec_rows),
    max_foot_height=max((r["foot_z"] for r in rows), default=float("nan")),
    max_clearance=max((r["foot_height_above_wire"] for r in rec_rows), default=float("nan")),
    min_clearance=min((r["foot_height_above_wire"] for r in rec_rows), default=float("nan")),
    max_calf_excursion=(
      max((abs(r["calf_pos"]) for r in rec_rows), default=float("nan"))
    ),
    final_forward_dist=rec_rows[-1]["foot_forward_dist_to_wire"] if rec_rows else float("nan"),
    crossed=any(r["crossing_achieved"] for r in rows),
    reached_post_recovery=reached_post_recovery_at is not None,
    final_phase=rows[-1]["phase"] if rows else "NONE",
    calf_torque_zero_steps=sum(1 for r in rec_rows if r["calf_torque_cmd"] == 0.0),
    fell=any(r["base_z"] < 0.15 for r in rows),  # crude stability guard, informational only
  )
  return rows, summary


def main():
  out_dir = Path("logs/d3_calf_ablation")
  out_dir.mkdir(parents=True, exist_ok=True)

  print("=== Running BASELINE (calf torque active all 16 recovery steps) ===")
  base_rows, base_summary = run_one("baseline", ablate=False)
  print("=== Running ABLATION (calf torque stops after recovery step", CALF_TORQUE_STOP_AFTER_STEPS, ") ===")
  abl_rows, abl_summary = run_one("ablation_stop5", ablate=True)

  for rows, name in [(base_rows, "baseline"), (abl_rows, "ablation_stop5")]:
    csv_path = out_dir / f"{name}_trajectory.csv"
    with open(csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
    print(f"[D3-ablation] CSV written: {csv_path} ({len(rows)} rows)")

  print("\n==== PER-RECOVERY-STEP COMPARISON (step | foot_z base vs abl | clearance base vs abl | calf_torque base vs abl) ====")
  base_rec = [r for r in base_rows if r["phase"] == "RECOVERY"]
  abl_rec = [r for r in abl_rows if r["phase"] == "RECOVERY"]
  n = max(len(base_rec), len(abl_rec))
  for i in range(n):
    b = base_rec[i] if i < len(base_rec) else None
    a = abl_rec[i] if i < len(abl_rec) else None
    bz = f"{b['foot_z']:.4f}" if b else "  -   "
    az = f"{a['foot_z']:.4f}" if a else "  -   "
    bc = f"{b['foot_height_above_wire']:.4f}" if b else "  -   "
    ac = f"{a['foot_height_above_wire']:.4f}" if a else "  -   "
    bt = f"{b['calf_torque_cmd']:+.2f}" if b else "  -   "
    at = f"{a['calf_torque_cmd']:+.2f}" if a else "  -   "
    bcalf = f"{b['calf_pos']:.3f}" if b else "  -   "
    acalf = f"{a['calf_pos']:.3f}" if a else "  -   "
    print(f"  recov_step {i:2d}: foot_z {bz} / {az}   clearance {bc} / {ac}   calf_torque {bt} / {at}   calf_pos {bcalf} / {acalf}")

  print("\n==== SUMMARY: BASELINE ====")
  for k, v in base_summary.items():
    print(f"  {k}: {v}")
  print("\n==== SUMMARY: ABLATION (stop calf torque after step", CALF_TORQUE_STOP_AFTER_STEPS, ") ====")
  for k, v in abl_summary.items():
    print(f"  {k}: {v}")

  # Post-peak-collapse check: did foot height fall after its peak, and by how much, in each run?
  def post_peak_collapse(rec_rows):
    if not rec_rows:
      return None
    heights = [r["foot_z"] for r in rec_rows]
    peak_idx = heights.index(max(heights))
    if peak_idx == len(heights) - 1:
      return 0.0  # peak at the very end, no collapse observed
    return heights[peak_idx] - min(heights[peak_idx:])

  base_collapse = post_peak_collapse(base_rec)
  abl_collapse = post_peak_collapse(abl_rec)
  print(f"\nPost-peak foot-height collapse magnitude: baseline={base_collapse:.4f} m   ablation={abl_collapse:.4f} m")

  print("\n==== INTERPRETATION ====")
  if abl_collapse is not None and base_collapse is not None:
    if abl_collapse < base_collapse * 0.5:
      verdict = "Calf torque duration appears to be a major contributor"
    elif abs(abl_collapse - base_collapse) < 0.01:
      verdict = "Calf torque duration does not explain the collapse"
    else:
      verdict = "Result is inconclusive"
  else:
    verdict = "Result is inconclusive"
  print(f"VERDICT: {verdict}")


if __name__ == "__main__":
  main()
