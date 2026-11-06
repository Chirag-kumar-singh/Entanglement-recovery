#!/usr/bin/env python
"""D5.33 whole-robot post-recovery stability diagnostic.

Purpose (strictly OBSERVATIONAL, no controller/FSM/reward/termination/wire
changes): the D5.32A audit established that the isolated-FL-wire recovery
task only instruments the FRONT-LEFT (FL) leg's wire recovery. It has no
rear-left (RL) stuck detector and no whole-robot recovery-success metric.
Visual MuJoCo inspection suggested that on seed 0 the FL leg gets over the
wire but the RL leg subsequently "appears stuck". This script captures
per-step whole-robot + per-leg telemetry for a single seed-0 baseline
rollout so that claim can be examined against the actual simulation
geometry/contacts/kinematics rather than eyeballing the viewer.

It REUSES, unmodified:
  - the exact env builder `go2_wire_recovery_residual_env_cfg(...)` used by
    the D5.31 comparison and the D5.29 viewer (same checkpoint, same
    ISOLATED_FL_WIRE_CFG / spawn_y, same hybrid recovery mode,
    max_recovery_steps=48, baseline thresholds);
  - the existing `wire_contact` sensor for FL-wire contact;
  - the FSM's own latched milestone flags on the `joint_pos` action term.

It ADDS ONLY one purely-observational per-foot ground-contact sensor
(reduce="none", so per-foot identity is preserved) using the SAME additive
pattern already used by `scripts/d3_ground_contact_diagnostic.py` and the
velocity task's `feet_ground_contact` sensor. No existing sensor, reward,
termination, or actuator is touched.

Usage (single seed-0 baseline CPU rollout, writes CSV + timeline)::

    uv run python scripts/d5_33_whole_robot_post_recovery_diagnostic.py

Outputs under ``logs/d5_33/seed0_baseline/`` (fresh directory; historical
D5.3 / D5.31 logs are never touched).
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import mjlab.tasks  # noqa: F401  (populate task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.tasks.recovery.mdp.actions import (
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
LEG_NAMES = ("FL", "FR", "RL", "RR")
FEET_GROUND_SENSOR_NAME = "d5_33_feet_ground_contact"


@dataclass(frozen=True)
class D533Config:
  seed: int = 0
  variant: str = "baseline"  # "baseline" | "phase_aware"
  device: str = "cpu"
  # Enough steps to run well past recovery entry/exit AND keep observing
  # the robot through POST_RECOVERY. The environment may terminate earlier
  # (fell_over); we always stop at episode termination if it happens.
  max_steps: int = 300
  out_dir: str = ""


def _build_env(cfg: D533Config) -> ManagerBasedRlEnv:
  use_phase_aware = cfg.variant == "phase_aware"
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=cfg.device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=use_phase_aware,
    max_recovery_steps=48,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = cfg.device

  # Additive, purely-observational per-foot ground-contact sensor. Same
  # pattern as velocity task's `feet_ground_contact`, but reduce="none"
  # so each of the four feet keeps its own identity (via primary_names).
  foot_geoms = tuple(f"{leg}_foot_collision" for leg in LEG_NAMES)
  feet_ground_cfg = ContactSensorCfg(
    name=FEET_GROUND_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (feet_ground_cfg,)

  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


def run(cfg: D533Config) -> Path:
  configure_torch_backends()

  out_dir = Path(cfg.out_dir or f"logs/d5_33/seed{cfg.seed}_{cfg.variant}")
  out_dir.mkdir(parents=True, exist_ok=True)

  env = _build_env(cfg)
  act = env.action_manager.get_term("joint_pos")
  entity = act._entity
  joint_names = list(entity.joint_names)
  site_names = list(entity.site_names)

  # Per-leg foot site indices (feet sites are named by leg, e.g. "FL").
  foot_site_idx = {leg: site_names.index(leg) for leg in LEG_NAMES}

  # Per-leg joint column indices (hip/thigh/calf) into the 12-dim joint
  # arrays, for joint pos/vel reporting.
  leg_joint_idx = {}
  for leg in LEG_NAMES:
    leg_joint_idx[leg] = {
      "hip": joint_names.index(f"{leg}_hip_joint"),
      "thigh": joint_names.index(f"{leg}_thigh_joint"),
      "calf": joint_names.index(f"{leg}_calf_joint"),
    }

  feet_ground = env.scene.sensors[FEET_GROUND_SENSOR_NAME]
  # Map each contact-sensor primary row to its leg via the matched geom
  # name (e.g. ".../FL_foot_collision" -> "FL").
  primary_leg = []
  for pname in feet_ground.primary_names:
    short = pname.split("/")[-1]
    matched = next((leg for leg in LEG_NAMES if short.startswith(f"{leg}_")), "?")
    primary_leg.append(matched)

  wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]

  env.reset(seed=cfg.seed)

  rows: list[dict] = []
  final_exit_reason = "horizon_exhausted"
  env_terminated = False
  env_truncated = False

  def foot_contact_for_leg(leg: str) -> tuple[bool, float]:
    found_rows = feet_ground.data.found[0].reshape(-1) > 0
    force_mags = torch.linalg.norm(feet_ground.data.force[0].reshape(-1, 3), dim=-1)
    found_any = False
    fmax = 0.0
    for i, pleg in enumerate(primary_leg):
      if pleg != leg:
        continue
      if bool(found_rows[i]):
        found_any = True
        fmax = max(fmax, float(force_mags[i]))
    return found_any, fmax

  def build_row(step: int) -> dict:
    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    actuator_force = entity.data.actuator_force[0]

    phase = int(act.phase[0])
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]
    base_ang_vel = entity.data.root_link_ang_vel_w[0]
    base_lin_vel = entity.data.root_link_lin_vel_w[0]
    grav_b = entity.data.projected_gravity_b[0]

    wire_contact = bool((wire_sensor.data.found[0] > 0).any())

    command = env.command_manager.get_command(act.cfg.command_name)[0]

    recovery_elapsed = int(act.recovery_elapsed[0])
    exit_reason = ""
    if phase == PHASE_POST_RECOVERY:
      exit_reason = (
        "cap_timeout" if bool(act.recovery_failed_timeout[0]) else "crossing_release"
      )

    row = {
      "step": step,
      "sim_time": step * env.step_dt,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": recovery_elapsed,
      "stage1_calf_clearance": bool(act._calf_clearance_achieved[0]),
      "stage2_foot_crossed": bool(act._foot_crossed_achieved[0]),
      "crossing_achieved": bool(act._crossing_achieved[0]),
      "recovery_exit_reason": exit_reason,
      "recovery_failed_timeout": bool(act._recovery_failed_timeout[0]),
      "fl_foot_height_above_wire": float(act.foot_height_above_wire[0]),
      "wire_contact": wire_contact,
      # Base state.
      "base_x": float(base_pos[0]),
      "base_y": float(base_pos[1]),
      "base_z": float(base_pos[2]),
      "base_qw": float(base_quat[0]),
      "base_qx": float(base_quat[1]),
      "base_qy": float(base_quat[2]),
      "base_qz": float(base_quat[3]),
      "base_proj_grav_z": float(grav_b[2]),  # ~ -1 upright, -> 0 toppling
      "base_lin_vel_x": float(base_lin_vel[0]),
      "base_lin_vel_y": float(base_lin_vel[1]),
      "base_lin_vel_z": float(base_lin_vel[2]),
      "base_ang_vel_x": float(base_ang_vel[0]),
      "base_ang_vel_y": float(base_ang_vel[1]),
      "base_ang_vel_z": float(base_ang_vel[2]),
      # Walking command.
      "cmd_lin_vel_x": float(command[0]),
      "cmd_lin_vel_y": float(command[1]),
      "cmd_ang_vel_z": float(command[2]),
      # Internal recovery-success bookkeeping (strict FSM latch).
      "newly_succeeded": bool(act.newly_succeeded[0]),
      "already_succeeded_this_episode": bool(act._already_succeeded_this_episode[0]),
      "post_recovery_eligible": bool(act._post_recovery_eligible[0]),
      "stable_since_post_recovery": bool(act._stable_since_post_recovery[0]),
      "post_recovery_elapsed": int(act._post_recovery_elapsed[0]),
      "release_counter": int(act._release_counter[0]),
    }

    # Per-leg feet world pos/vel, contact, and joint state.
    for leg in LEG_NAMES:
      sidx = foot_site_idx[leg]
      fpos = entity.data.site_pos_w[0, sidx, :]
      fvel = entity.data.site_lin_vel_w[0, sidx, :]
      found, fmag = foot_contact_for_leg(leg)
      ji = leg_joint_idx[leg]
      row[f"{leg}_foot_x"] = float(fpos[0])
      row[f"{leg}_foot_y"] = float(fpos[1])
      row[f"{leg}_foot_z"] = float(fpos[2])
      row[f"{leg}_foot_vx"] = float(fvel[0])
      row[f"{leg}_foot_vy"] = float(fvel[1])
      row[f"{leg}_foot_vz"] = float(fvel[2])
      row[f"{leg}_foot_speed"] = float(torch.linalg.norm(fvel))
      row[f"{leg}_ground_contact"] = found
      row[f"{leg}_ground_force_mag"] = fmag
      for jn in ("hip", "thigh", "calf"):
        jidx = ji[jn]
        row[f"{leg}_{jn}_pos"] = float(qpos[jidx])
        row[f"{leg}_{jn}_vel"] = float(qvel[jidx])
        # NOTE: `entity.data.actuator_force` is indexed by actuator-
        # declaration order, which in hybrid mode (2 actuators per
        # recovery joint on the affected FL leg) does NOT line up with
        # joint order. Values for non-FL legs are reliable; FL
        # thigh/calf actuator-force here is approximate (reported for
        # completeness, not for precise FL-torque claims -- see
        # d3_ground_contact_diagnostic for the corrected global ctrl-id
        # method).
        torque = (
          float(actuator_force[jidx])
          if jidx < actuator_force.shape[0]
          else float("nan")
        )
        row[f"{leg}_{jn}_torque"] = torque

    return row

  # Record the initial state, then step.
  for step in range(cfg.max_steps):
    rows.append(build_row(step))
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)
    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      env_terminated = bool(terminateds[0].item())
      env_truncated = bool(truncateds[0].item())
      # Capture the post-termination state too.
      rows.append(build_row(step + 1))
      if env_truncated:
        final_exit_reason = "timeout"
      elif env_terminated:
        final_exit_reason = "fell_over"
      break
  else:
    final_exit_reason = "horizon_exhausted"

  env.close()

  # --- Write per-step CSV ---
  csv_path = out_dir / "telemetry.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  # --- Derive the event timeline + per-leg summary ---
  def first_step_where(pred) -> int:
    for r in rows:
      if pred(r):
        return r["step"]
    return -1

  entry_step = first_step_where(lambda r: r["phase"] == "RECOVERY")
  stage1_step = first_step_where(lambda r: r["stage1_calf_clearance"])
  stage2_step = first_step_where(lambda r: r["stage2_foot_crossed"])
  post_step = first_step_where(lambda r: r["phase"] == "POST_RECOVERY")
  post_exit_reason = next(
    (r["recovery_exit_reason"] for r in rows if r["phase"] == "POST_RECOVERY"),
    "none",
  )
  internal_success = any(r["already_succeeded_this_episode"] for r in rows)
  last = rows[-1]

  summary_lines = []
  summary_lines.append("=" * 70)
  summary_lines.append(
    f"D5.33 WHOLE-ROBOT DIAGNOSTIC  seed={cfg.seed} variant={cfg.variant}"
  )
  summary_lines.append("=" * 70)
  summary_lines.append("")
  summary_lines.append("--- EVENT TIMELINE (step, sim_time=step*step_dt) ---")
  summary_lines.append(f"  recovery entry (NORMAL->RECOVERY): step {entry_step}")
  summary_lines.append(f"  FL stage-1 (calf clearance) latch: step {stage1_step}")
  summary_lines.append(f"  FL stage-2 (foot crossing)  latch: step {stage2_step}")
  summary_lines.append(f"  RECOVERY exit (-> POST_RECOVERY):  step {post_step}")
  summary_lines.append(f"  RECOVERY exit reason:              {post_exit_reason}")
  summary_lines.append(f"  end of observation window:         step {last['step']}")
  summary_lines.append(f"  env outcome (this window):         {final_exit_reason}")
  summary_lines.append(
    f"  env_terminated={env_terminated}  env_truncated={env_truncated}"
  )
  summary_lines.append("")
  summary_lines.append("--- A. FL WIRE-CROSSING OUTCOME ---")
  summary_lines.append(f"  stage1 reached: {stage1_step != -1} (step {stage1_step})")
  summary_lines.append(f"  stage2 reached: {stage2_step != -1} (step {stage2_step})")
  min_fl_h = min(r["fl_foot_height_above_wire"] for r in rows)
  max_fl_h = max(r["fl_foot_height_above_wire"] for r in rows)
  summary_lines.append(
    f"  FL foot_height_above_wire range: [{min_fl_h:.4f}, {max_fl_h:.4f}] m"
  )
  summary_lines.append(
    f"  FSM exited RECOVERY: {post_step != -1}  reason: {post_exit_reason}"
  )
  summary_lines.append("")
  summary_lines.append("--- B. WHOLE-ROBOT POST-RECOVERY OUTCOME ---")
  summary_lines.append(f"  entered POST_RECOVERY: {post_step != -1}")
  summary_lines.append(
    f"  env internal recovery-success latch fired: {internal_success}"
  )
  summary_lines.append(
    f"  final base_z={last['base_z']:.4f} m  proj_grav_z={last['base_proj_grav_z']:.4f} "
    f"(<= -cos(70deg)=-0.342 => upright)"
  )
  upright_final = last["base_proj_grav_z"] <= -0.342
  summary_lines.append(f"  upright at end: {upright_final}")
  summary_lines.append("")
  summary_lines.append("--- PER-LEG BEHAVIOR AFTER FL STAGE-2 (foot crossing) ---")
  post_rows = [r for r in rows if stage2_step != -1 and r["step"] >= stage2_step]
  if post_rows:
    for leg in LEG_NAMES:
      speeds = [r[f"{leg}_foot_speed"] for r in post_rows]
      zs = [r[f"{leg}_foot_z"] for r in post_rows]
      contacts = [r[f"{leg}_ground_contact"] for r in post_rows]
      mean_speed = sum(speeds) / len(speeds)
      contact_frac = sum(1 for c in contacts if c) / len(contacts)
      summary_lines.append(
        f"  {leg}: mean_foot_speed={mean_speed:.4f} m/s  "
        f"foot_z[min,max]=[{min(zs):.4f},{max(zs):.4f}]  "
        f"ground_contact_fraction={contact_frac:.2f}"
      )
  else:
    summary_lines.append("  (stage-2 not reached; no post-crossing window)")
  summary_lines.append("")
  summary_lines.append(
    "NOTE: whole-robot recovery success remains UNDEFINED in the task. "
    "FL stage-2 alone does NOT establish whole-robot recovery. Any "
    "per-leg anomaly above is reported as evidence for investigation, "
    "not as a pass/fail verdict."
  )
  summary_lines.append("")
  summary_lines.append(f"CSV: {csv_path}")

  summary_text = "\n".join(summary_lines)
  print(summary_text)
  (out_dir / "summary.txt").write_text(summary_text + "\n")

  return out_dir


def main():
  import argparse

  parser = argparse.ArgumentParser(
    description="D5.33 whole-robot post-recovery stability diagnostic"
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--variant", type=str, choices=["baseline", "phase_aware"], default="baseline"
  )
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--max-steps", type=int, default=300)
  parser.add_argument("--out-dir", type=str, default="")
  args = parser.parse_args()

  cfg = D533Config(
    seed=args.seed,
    variant=args.variant,
    device=args.device,
    max_steps=args.max_steps,
    out_dir=args.out_dir,
  )
  run(cfg)


if __name__ == "__main__":
  main()
