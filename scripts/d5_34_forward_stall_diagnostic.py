#!/usr/bin/env python
"""D5.34 forward-stall / RL-contact diagnostic.

Follow-up to D5.33: seed 0 baseline showed the FL leg genuinely cross the
isolated wire (stage-1/stage-2 latches, RECOVERY exit via
`crossing_release`), yet base-frame forward progress stalled around step
140 despite a standing +0.30 m/s forward command, while the RL foot kept
moving (max recorded speed 3.17 m/s -- not obviously frozen). This script
adds READ-ONLY instrumentation to determine (a) whether the RL foot can
even geometrically contact the isolated FL wire, and (b) when forward
velocity tracking actually breaks down relative to recovery entry/exit,
WITHOUT inferring causation from timing alone.

Reused UNCHANGED:
  - `go2_wire_recovery_residual_env_cfg(...)` (same checkpoint, hybrid
    recovery mode, ISOLATED_FL_WIRE_CFG + spawn_y override, thresholds).
  - The existing `wire_contact` sensor (FL calf2 vs wire) -- NOT modified,
    NOT re-defined; read directly for the "FL foot-to-wire" comparison
    signal required by the task.

ADDED (purely observational, additive ContactSensorCfg entries, same
pattern as scripts/d3_ground_contact_diagnostic.py and D5.33's per-foot
ground sensor -- no existing sensor's trigger semantics are touched):
  - `d5_34_rl_wire_contact`: ALL RL leg collision geoms vs the wire's
    segment geoms (broader than the FL sensor's single calf2 geom, so a
    narrower match can't hide a real contact).
  - `d5_34_rl_terrain_contact`: ALL RL leg collision geoms vs terrain.
  - `d5_34_feet_ground_contact`: all four feet vs terrain (per-foot
    identity preserved, reduce="none"), for the cross-leg contact/
    velocity comparison table.

Usage::

    uv run python scripts/d5_34_forward_stall_diagnostic.py --seed 0 --variant baseline --device cpu
    uv run python scripts/d5_34_forward_stall_diagnostic.py --seed 1 --variant baseline --device cpu

Outputs under ``logs/d5_34/seed{N}_{variant}/`` (fresh directory; D5.3,
D5.31, and D5.33 outputs are never touched).
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

RL_WIRE_SENSOR_NAME = "d5_34_rl_wire_contact"
RL_TERRAIN_SENSOR_NAME = "d5_34_rl_terrain_contact"
FEET_GROUND_SENSOR_NAME = "d5_34_feet_ground_contact"

# Forward-velocity-error "well below command" threshold and persistence
# window used to flag the onset of a tracking breakdown (not a judgment
# of cause, just a consistent detection rule applied uniformly across
# pre-recovery / recovery / post-recovery).
STALL_ERROR_FRAC = 0.5  # actual < 50% of commanded forward speed
STALL_PERSIST_STEPS = 10


@dataclass(frozen=True)
class D534Config:
  seed: int = 0
  variant: str = "baseline"
  device: str = "cpu"
  max_steps: int = 300
  out_dir: str = ""


def _build_env(cfg: D534Config) -> ManagerBasedRlEnv:
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

  # Resolve the wire's actual segment geom names the SAME way the
  # existing (unmodified) `wire_contact` sensor does, so the new RL-wire
  # sensor is matched against the identical set of wire geoms.
  existing_wire_sensor = next(
    s for s in env_cfg.scene.sensors if s.name == "wire_contact"
  )
  wire_geom_pattern = existing_wire_sensor.primary

  rl_wire_cfg = ContactSensorCfg(
    name=RL_WIRE_SENSOR_NAME,
    primary=wire_geom_pattern,
    secondary=ContactMatch(mode="geom", pattern=r"RL_.*_collision\d*$", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  rl_terrain_cfg = ContactSensorCfg(
    name=RL_TERRAIN_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=r"RL_.*_collision\d*$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  foot_geoms = tuple(f"{leg}_foot_collision" for leg in LEG_NAMES)
  feet_ground_cfg = ContactSensorCfg(
    name=FEET_GROUND_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )

  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (
    rl_wire_cfg,
    rl_terrain_cfg,
    feet_ground_cfg,
  )

  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


def _contact_summary(sensor, leg_filter=None, primary_leg=None) -> tuple[bool, float]:
  found_rows = sensor.data.found[0].reshape(-1) > 0
  force_mags = torch.linalg.norm(sensor.data.force[0].reshape(-1, 3), dim=-1)
  found_any = False
  fmax = 0.0
  for i in range(found_rows.shape[0]):
    if (
      leg_filter is not None
      and primary_leg is not None
      and primary_leg[i] != leg_filter
    ):
      continue
    if bool(found_rows[i]):
      found_any = True
      fmax = max(fmax, float(force_mags[i]))
  return found_any, fmax


def run(cfg: D534Config) -> Path:
  configure_torch_backends()

  out_dir = Path(cfg.out_dir or f"logs/d5_34/seed{cfg.seed}_{cfg.variant}")
  out_dir.mkdir(parents=True, exist_ok=True)

  env = _build_env(cfg)
  act = env.action_manager.get_term("joint_pos")
  entity = act._entity
  joint_names = list(entity.joint_names)
  site_names = list(entity.site_names)
  foot_site_idx = {leg: site_names.index(leg) for leg in LEG_NAMES}
  leg_joint_idx = {
    leg: {
      "hip": joint_names.index(f"{leg}_hip_joint"),
      "thigh": joint_names.index(f"{leg}_thigh_joint"),
      "calf": joint_names.index(f"{leg}_calf_joint"),
    }
    for leg in LEG_NAMES
  }

  wire_sensor = env.scene.sensors[
    act.cfg.wire_contact_sensor_name
  ]  # FL vs wire (existing)
  rl_wire = env.scene.sensors[RL_WIRE_SENSOR_NAME]
  rl_terrain = env.scene.sensors[RL_TERRAIN_SENSOR_NAME]
  feet_ground = env.scene.sensors[FEET_GROUND_SENSOR_NAME]

  def _leg_map(sensor) -> list[str]:
    out = []
    for pname in sensor.primary_names:
      short = pname.split("/")[-1]
      matched = next((leg for leg in LEG_NAMES if short.startswith(f"{leg}_")), "?")
      out.append(matched)
    return out

  feet_primary_leg = _leg_map(feet_ground)

  # --- Report verified sensor geom coverage BEFORE running (per task
  # requirement #2: "Confirm that each new sensor measures the intended
  # geoms"). ---
  coverage_lines = []
  coverage_lines.append("=" * 70)
  coverage_lines.append("D5.34 SENSOR GEOM COVERAGE (verified at build time)")
  coverage_lines.append("=" * 70)
  coverage_lines.append(
    f"wire geom segments (shared by all wire-contact sensors): "
    f"{ISOLATED_FL_WIRE_CFG.geom_names()}"
  )
  coverage_lines.append(
    f"existing wire_contact (UNCHANGED) secondary pattern: "
    f"{wire_sensor.data.found[0].shape} rows "
    f"(FL calf2 vs wire; not re-defined by this script)"
  )
  coverage_lines.append(
    f"{RL_WIRE_SENSOR_NAME} primary (RL collision geoms matched): "
    f"{rl_wire.primary_names}"
  )
  coverage_lines.append(
    f"{RL_TERRAIN_SENSOR_NAME} primary (RL collision geoms matched): "
    f"{rl_terrain.primary_names}"
  )
  coverage_lines.append(
    f"{FEET_GROUND_SENSOR_NAME} primary (all 4 feet, matched->leg): "
    f"{list(zip(feet_ground.primary_names, feet_primary_leg, strict=True))}"
  )
  coverage_text = "\n".join(coverage_lines)
  print(coverage_text)

  env.reset(seed=cfg.seed)

  rows: list[dict] = []
  final_exit_reason = "horizon_exhausted"
  env_truncated = False

  prev_foot_contact = {leg: False for leg in LEG_NAMES}
  contact_transitions = {leg: 0 for leg in LEG_NAMES}
  prev_rl_wire_contact = False
  rl_wire_contact_transitions = 0
  rl_wire_ever_contacted = False

  def build_row(step: int) -> dict:
    nonlocal rl_wire_ever_contacted, prev_rl_wire_contact, rl_wire_contact_transitions
    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]

    phase = int(act.phase[0])
    base_pos = entity.data.root_link_pos_w[0]
    heading = float(entity.data.heading_w[0])
    lin_vel_b = entity.data.root_link_lin_vel_b[0]  # CORRECT frame: body
    base_ang_vel = entity.data.root_link_ang_vel_w[0]
    grav_b = entity.data.projected_gravity_b[0]

    command = env.command_manager.get_command(act.cfg.command_name)[0]
    cmd_fwd = float(command[0])
    actual_fwd = float(lin_vel_b[0])
    fwd_err = cmd_fwd - actual_fwd

    fl_wire_contact = bool((wire_sensor.data.found[0] > 0).any())
    rl_wire_found, rl_wire_fmag = _contact_summary(rl_wire)
    rl_terrain_found, rl_terrain_fmag = _contact_summary(rl_terrain)

    if rl_wire_found:
      rl_wire_ever_contacted = True
    if prev_rl_wire_contact and not rl_wire_found:
      rl_wire_contact_transitions += 1
    prev_rl_wire_contact = rl_wire_found

    exit_reason = ""
    if phase == PHASE_POST_RECOVERY:
      exit_reason = (
        "cap_timeout" if bool(act.recovery_failed_timeout[0]) else "crossing_release"
      )

    row = {
      "step": step,
      "sim_time": step * env.step_dt,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "recovery_exit_reason": exit_reason,
      "cmd_fwd_vel": cmd_fwd,
      "actual_fwd_vel_body": actual_fwd,
      "fwd_vel_error": fwd_err,
      "base_x": float(base_pos[0]),
      "base_y": float(base_pos[1]),
      "base_z": float(base_pos[2]),
      "heading_rad": heading,
      "base_proj_grav_z": float(grav_b[2]),
      "base_ang_vel_x": float(base_ang_vel[0]),
      "base_ang_vel_y": float(base_ang_vel[1]),
      "base_ang_vel_z": float(base_ang_vel[2]),
      "fl_wire_contact": fl_wire_contact,
      "rl_wire_contact": rl_wire_found,
      "rl_wire_force_mag": rl_wire_fmag,
      "rl_wire_ever_contacted_so_far": rl_wire_ever_contacted,
      "rl_wire_contact_loss_transitions_so_far": rl_wire_contact_transitions,
      "rl_terrain_contact": rl_terrain_found,
      "rl_terrain_force_mag": rl_terrain_fmag,
    }

    for leg in LEG_NAMES:
      sidx = foot_site_idx[leg]
      fpos = entity.data.site_pos_w[0, sidx, :]
      fvel = entity.data.site_lin_vel_w[0, sidx, :]
      found_any, fmax = False, 0.0
      found_rows = feet_ground.data.found[0].reshape(-1) > 0
      force_mags = torch.linalg.norm(feet_ground.data.force[0].reshape(-1, 3), dim=-1)
      for i, pleg in enumerate(feet_primary_leg):
        if pleg != leg:
          continue
        if bool(found_rows[i]):
          found_any = True
          fmax = max(fmax, float(force_mags[i]))
      if found_any != prev_foot_contact[leg]:
        contact_transitions[leg] += 1
      prev_foot_contact[leg] = found_any

      ji = leg_joint_idx[leg]
      row[f"{leg}_foot_x"] = float(fpos[0])
      row[f"{leg}_foot_y"] = float(fpos[1])
      row[f"{leg}_foot_z"] = float(fpos[2])
      row[f"{leg}_foot_vx"] = float(fvel[0])
      row[f"{leg}_foot_vy"] = float(fvel[1])
      row[f"{leg}_foot_vz"] = float(fvel[2])
      row[f"{leg}_foot_speed"] = float(torch.linalg.norm(fvel))
      row[f"{leg}_ground_contact"] = found_any
      row[f"{leg}_ground_force_mag"] = fmax
      row[f"{leg}_ground_contact_transitions_so_far"] = contact_transitions[leg]
      for jn in ("hip", "thigh", "calf"):
        jidx = ji[jn]
        row[f"{leg}_{jn}_pos"] = float(qpos[jidx])
        row[f"{leg}_{jn}_vel"] = float(qvel[jidx])

    return row

  for step in range(cfg.max_steps):
    rows.append(build_row(step))
    actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    _obs, _rew, terminateds, truncateds, _info = env.step(actions)
    if bool(terminateds[0].item()) or bool(truncateds[0].item()):
      env_truncated = bool(truncateds[0].item())
      rows.append(build_row(step + 1))
      final_exit_reason = "timeout" if env_truncated else "fell_over"
      break
  else:
    final_exit_reason = "horizon_exhausted"

  env.close()

  csv_path = out_dir / "telemetry.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  # --- Event timeline + stall-onset detection ---
  def first_step_where(pred) -> int:
    for r in rows:
      if pred(r):
        return r["step"]
    return -1

  entry_step = first_step_where(lambda r: r["phase"] == "RECOVERY")
  post_step = first_step_where(lambda r: r["phase"] == "POST_RECOVERY")
  post_exit_reason = next(
    (r["recovery_exit_reason"] for r in rows if r["phase"] == "POST_RECOVERY"), "none"
  )

  cmd = rows[0]["cmd_fwd_vel"] if rows else 0.0
  threshold = STALL_ERROR_FRAC * cmd if cmd > 0 else 0.0

  def persistent_stall_onset() -> int:
    """Onset of a stall that PERSISTS to the end of the observation
    window (not merely the first low-velocity window, which spuriously
    fires at step 0 while the robot is still accelerating from rest at
    reset). Computed via a rolling mean (window=STALL_PERSIST_STEPS) of
    body-frame forward velocity, then finding the LAST step at which that
    rolling mean was still >= threshold -- everything after that point
    never recovers to >= threshold for a sustained window. Returns -1 if
    no such persistent tail stall exists (i.e. the rolling mean recovers
    above threshold again before the window ends)."""
    if cmd <= 0 or len(rows) < STALL_PERSIST_STEPS:
      return -1
    vals = [r["actual_fwd_vel_body"] for r in rows]
    n = len(vals)
    last_ok = -1
    for i in range(n - STALL_PERSIST_STEPS + 1):
      window_mean = sum(vals[i : i + STALL_PERSIST_STEPS]) / STALL_PERSIST_STEPS
      if window_mean >= threshold:
        last_ok = i + STALL_PERSIST_STEPS - 1
    if last_ok == -1:
      # Never had a sustained window >= threshold at all.
      return rows[0]["step"]
    if last_ok >= n - 1:
      # The last sustained >=threshold window reaches the end -> no tail
      # stall.
      return -1
    return rows[last_ok + 1]["step"]

  stall_onset = persistent_stall_onset()

  def period_stats(lo: int, hi: int) -> tuple[float, float]:
    vals = [r["actual_fwd_vel_body"] for r in rows if lo <= r["step"] < hi]
    if not vals:
      return float("nan"), float("nan")
    return sum(vals) / len(vals), min(vals)

  pre_mean, pre_min = period_stats(0, entry_step if entry_step != -1 else 0)
  rec_mean, rec_min = period_stats(
    entry_step if entry_step != -1 else 0, post_step if post_step != -1 else 0
  )
  post_mean, post_min = period_stats(
    post_step if post_step != -1 else 0, rows[-1]["step"] + 1
  )

  rl_wire_ever = any(r["rl_wire_contact"] for r in rows)
  rl_wire_max_force = max((r["rl_wire_force_mag"] for r in rows), default=0.0)

  summary_lines = []
  summary_lines.append("=" * 70)
  summary_lines.append(
    f"D5.34 FORWARD-STALL / RL-CONTACT DIAGNOSTIC  seed={cfg.seed} variant={cfg.variant}"
  )
  summary_lines.append("=" * 70)
  summary_lines.append("")
  summary_lines.append(coverage_text)
  summary_lines.append("")
  summary_lines.append("--- EVENT TIMELINE ---")
  summary_lines.append(f"  recovery entry (NORMAL->RECOVERY): step {entry_step}")
  summary_lines.append(f"  RECOVERY exit (-> POST_RECOVERY):  step {post_step}")
  summary_lines.append(f"  RECOVERY exit reason:              {post_exit_reason}")
  summary_lines.append(
    f"  persistent (tail, never-recovers) fwd-vel stall onset "
    f"(rolling-{STALL_PERSIST_STEPS}-step mean actual < {STALL_ERROR_FRAC * 100:.0f}% of "
    f"cmd={cmd:.2f}, sustained through end of window): step {stall_onset} "
    f"({'none detected' if stall_onset == -1 else 'detected'})"
  )
  summary_lines.append(f"  end of observation window: step {rows[-1]['step']}")
  summary_lines.append(f"  env outcome: {final_exit_reason}")
  summary_lines.append("")
  summary_lines.append("--- FORWARD VELOCITY: ACTUAL (body-frame) vs COMMANDED ---")
  summary_lines.append(f"  commanded forward velocity (constant): {cmd:.3f} m/s")
  summary_lines.append(
    f"  pre-recovery  [0,{entry_step}): mean={pre_mean:.3f} min={pre_min:.3f}"
  )
  summary_lines.append(
    f"  recovery      [{entry_step},{post_step}): mean={rec_mean:.3f} min={rec_min:.3f}"
  )
  summary_lines.append(
    f"  post-recovery [{post_step},{rows[-1]['step'] + 1}): mean={post_mean:.3f} min={post_min:.3f}"
  )
  summary_lines.append("")
  summary_lines.append("--- RL-WIRE CONTACT ---")
  summary_lines.append(f"  RL-vs-wire contact ever observed: {rl_wire_ever}")
  summary_lines.append(
    f"  RL-vs-wire max force magnitude observed: {rl_wire_max_force:.4f} N"
  )
  if rl_wire_ever:
    summary_lines.append(
      "  CONCLUSION for this trajectory: RL-wire contact IS observed "
      "(not ruled out; not necessarily the stall's cause -- see below)."
    )
  else:
    summary_lines.append(
      "  CONCLUSION for this trajectory: RL-vs-wire contact sensor never "
      "fired (found=0 every step). RL-wire contact is RULED OUT as an "
      "explanation for this specific rollout's behavior."
    )
  summary_lines.append("")
  summary_lines.append(
    "--- PER-LEG SUMMARY (full post-recovery window, if reached) ---"
  )
  if post_step != -1:
    post_rows = [r for r in rows if r["step"] >= post_step]
    for leg in LEG_NAMES:
      speeds = [r[f"{leg}_foot_speed"] for r in post_rows]
      contacts = [r[f"{leg}_ground_contact"] for r in post_rows]
      trans = post_rows[-1][f"{leg}_ground_contact_transitions_so_far"]
      contact_frac = sum(1 for c in contacts if c) / len(contacts)
      summary_lines.append(
        f"  {leg}: mean_speed={sum(speeds) / len(speeds):.4f} m/s  "
        f"ground_contact_fraction={contact_frac:.2f}  "
        f"contact_transitions={trans}"
      )
  else:
    summary_lines.append("  (POST_RECOVERY not reached in this window)")
  summary_lines.append("")
  summary_lines.append(
    "NOTE: timing correlation between recovery exit and any forward-"
    "velocity breakdown does NOT by itself establish causation. The "
    "per-leg contact/velocity table above is reported as evidence, not "
    "as a verdict."
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
    description="D5.34 forward-stall / RL-contact diagnostic"
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--variant", type=str, choices=["baseline", "phase_aware"], default="baseline"
  )
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--max-steps", type=int, default=300)
  parser.add_argument("--out-dir", type=str, default="")
  args = parser.parse_args()

  cfg = D534Config(
    seed=args.seed,
    variant=args.variant,
    device=args.device,
    max_steps=args.max_steps,
    out_dir=args.out_dir,
  )
  run(cfg)


if __name__ == "__main__":
  main()
