"""D3 ground-contact / FR-confound dynamic diagnostic.

Resolves two open evidentiary questions from the D3 wire-recovery
investigation using ACTUAL MuJoCo contact data from a real dynamic run
(frozen walking policy -> real wire contact -> hybrid actuator-level
recovery -> real dynamics), NOT static kinematics and NOT inference from
foot-height alone:

1. Does FL foot/calf ACTUALLY contact the ground/terrain during the
   recovery window, and if so, does that contact's timing/force explain
   the foot-height collapse seen in the 2D kinematic map (static FK stays
   at 0.18-0.20 m while actual foot_z drops to 0.013-0.026 m)?
2. Does FR ACTUALLY contact the wire during this same dynamic run (which
   would mark the run CONFOUNDED), rather than relying on the earlier
   static/estimated ~3.1 cm minimum-distance check?

This script does NOT modify the controller, FSM, actuator architecture,
recovery parameters, wire geometry, or walking policy/checkpoint. It only
ADDS two new, purely observational `ContactSensorCfg` instances (same
pattern as the existing `wire_contact` sensor and the velocity task's
`feet_ground_contact` sensor) to a freshly-built env config, appended
AFTER `go2_wire_recovery_residual_env_cfg(...)` returns its (unmodified)
cfg -- i.e. exactly the same "additive sensor" pattern already used in
`mjlab.tasks.velocity.config.go2.env_cfgs`.

Dynamic (NOT static) visualization of the SAME run::

    uv run python scripts/d3_ground_contact_diagnostic.py --debug

Headless (CSV + causal-timeline summary only)::

    uv run python scripts/d3_ground_contact_diagnostic.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab.tasks  # noqa: F401
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

FL_GROUND_SENSOR_NAME = "fl_ground_contact_diag"
FL_GROUND_PERGEOM_SENSOR_NAME = "fl_ground_contact_pergeom_diag"
FR_WIRE_SENSOR_NAME = "fr_wire_contact_diag"


@dataclass(frozen=True)
class D3GroundConfig:
  affected_leg_idx: int = 0
  """0=FL -- same affected leg as the baseline D3 wire experiment."""
  seed: int = 0
  device: str | None = None
  max_steps: int = 400
  stop_steps_after_post_recovery: int = 60
  out_dir: str = ""
  debug: bool = False
  use_isolated_wire: bool = False
  """If True, use ISOLATED_FL_WIRE_CFG (narrowed to y in [0.30, 0.50],
  paired with spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y) instead of the
  default full-span wire, to verify the deconfounded single-leg-isolated
  geometry (FL still contacts, FR stays clear)."""
  experimental_peak_triggered_thigh_stop: bool = False
  """EXPERIMENTAL, opt-in, default False (preserves the validated clean
  baseline exactly). If True, zeros ONLY the recovery thigh velocity
  command once a genuine post-entry foot-height peak is detected (see
  ResidualRecoveryActionCfg docstring for the exact rule/thresholds).
  Calf torque, ab/ad, and the FSM are unaffected. Writes to a separate
  output directory by default so the baseline is never overwritten."""


def _affected_joint_names(leg: str) -> tuple[str, str, str]:
  return (f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint")


def build_env(cfg: D3GroundConfig, device: str):
  """Build the UNMODIFIED recovery env, then ADD two diagnostic-only
  contact sensors on top (purely observational, no controller/FSM
  change)."""
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG if cfg.use_isolated_wire else None,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y if cfg.use_isolated_wire else None,
    experimental_peak_triggered_thigh_stop=cfg.experimental_peak_triggered_thigh_stop,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  leg = LEG_NAMES[cfg.affected_leg_idx]
  other_leg = "FR" if leg == "FL" else "FL"

  # FL ground-contact sensor: ALL FL collision geoms vs terrain (not just
  # calf2), so we don't miss contact on the foot/thigh/hip geoms.
  fl_ground_cfg = ContactSensorCfg(
    name=FL_GROUND_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  # Per-geom variant (reduce="none"): preserves per-primary identity so
  # we can report exactly WHICH FL collision geom(s) touch terrain first
  # (foot vs calf1/2 vs thigh vs hip), per Phase 4's requirement to not
  # infer this from foot_z/aggregate force alone.
  fl_ground_pergeom_cfg = ContactSensorCfg(
    name=FL_GROUND_PERGEOM_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  # FR-wire confound sensor: same pattern as the existing `wire_contact`
  # sensor, but for the OTHER (unaffected) front leg's calf geom. Reuse
  # the existing sensor's already-resolved primary pattern so we don't
  # need to re-derive wire geometry/spec internals.
  existing_wire_sensor = next(s for s in env_cfg.scene.sensors if s.name == "wire_contact")
  fr_wire_cfg = ContactSensorCfg(
    name=FR_WIRE_SENSOR_NAME,
    primary=existing_wire_sensor.primary,
    secondary=ContactMatch(
      mode="geom", pattern=f"{other_leg}_calf2_collision", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (
    fl_ground_cfg,
    fl_ground_pergeom_cfg,
    fr_wire_cfg,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env


def run(cfg: D3GroundConfig) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  def _default_out_dir() -> str:
    if cfg.experimental_peak_triggered_thigh_stop:
      base = "logs/d3_peak_trigger_variant"
    elif cfg.use_isolated_wire:
      base = "logs/d3_ground_contact_diagnostic_isolated_fl_wire"
    else:
      base = "logs/d3_ground_contact_diagnostic"
    return base

  out_dir = Path(cfg.out_dir or _default_out_dir())
  out_dir.mkdir(parents=True, exist_ok=True)

  env = build_env(cfg, device)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)

  leg = LEG_NAMES[cfg.affected_leg_idx]
  other_leg = "FR" if leg == "FL" else "FL"
  hip_n, thigh_n, calf_n = _affected_joint_names(leg)
  hip_i = joint_names.index(hip_n)
  thigh_i = joint_names.index(thigh_n)
  calf_i = joint_names.index(calf_n)

  print(f"[D3-GROUND] affected_leg={leg}  other_leg={other_leg}")
  print(f"[D3-GROUND] FL-ground sensor: {FL_GROUND_SENSOR_NAME}")
  print(f"[D3-GROUND] FR-wire sensor:   {FR_WIRE_SENSOR_NAME}")
  if cfg.use_isolated_wire:
    print(
      f"[D3-GROUND] ISOLATED-FL wire: anchor_start={ISOLATED_FL_WIRE_CFG.anchor_start} "
      f"anchor_end={ISOLATED_FL_WIRE_CFG.anchor_end} radius={ISOLATED_FL_WIRE_CFG.radius} "
      f"spawn_y_override={ISOLATED_FL_WIRE_SPAWN_Y}"
    )
  else:
    wire = act.cfg.wire_geom_names
    print(f"[D3-GROUND] DEFAULT wire geom names: {wire}")

  other_leg_site_idx = list(entity.site_names).index(other_leg)

  rows: list[dict] = []
  entry_foot_pos: torch.Tensor | None = None
  running_max_h = -1e9
  contacted_wire = False
  recovery_activated = False
  fr_confounded = False

  env.reset(seed=cfg.seed)

  def build_row(step: int) -> dict:
    nonlocal running_max_h
    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    phase = int(act.phase[0])

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    in_contact = bool((wire_sensor.data.found[0] > 0).any())

    # `.force` has shape [N, 3] (N = num primary geoms matched * num_slots);
    # reduce per-row (per-geom) to a scalar magnitude, then take the max
    # across geoms -- NOT a flat reshape, which would silently mix rows.
    fl_ground = env.scene.sensors[FL_GROUND_SENSOR_NAME]
    fl_ground_found_rows = fl_ground.data.found[0].reshape(-1) > 0
    fl_ground_found = bool(fl_ground_found_rows.any())
    fl_ground_row_mags = torch.linalg.norm(fl_ground.data.force[0].reshape(-1, 3), dim=-1)
    fl_ground_fmag = float(fl_ground_row_mags.max()) if fl_ground_found else 0.0

    fr_wire = env.scene.sensors[FR_WIRE_SENSOR_NAME]
    fr_wire_found_rows = fr_wire.data.found[0].reshape(-1) > 0
    fr_wire_found = bool(fr_wire_found_rows.any())
    fr_wire_row_mags = torch.linalg.norm(fr_wire.data.force[0].reshape(-1, 3), dim=-1)
    fr_wire_fmag = float(fr_wire_row_mags.max()) if fr_wire_found else 0.0

    # Per-geom FL ground-contact identity (Phase 4): which specific FL
    # collision geom(s) are touching terrain right now, by name.
    fl_pergeom = env.scene.sensors[FL_GROUND_PERGEOM_SENSOR_NAME]
    pergeom_found = fl_pergeom.data.found[0].reshape(-1) > 0
    pergeom_force_mag = torch.linalg.norm(fl_pergeom.data.force[0].reshape(-1, 3), dim=-1)
    contacting_geoms = [
      name for name, found in zip(fl_pergeom.primary_names, pergeom_found.tolist()) if found
    ]
    contacting_geoms_forces = [
      float(pergeom_force_mag[i])
      for i, found in enumerate(pergeom_found.tolist())
      if found
    ]

    if entry_foot_pos is not None:
      running_max_h = max(running_max_h, float(foot_pos[2]))

    other_foot_pos = entity.data.site_pos_w[0, other_leg_site_idx, :].clone()

    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]
    base_lin_vel = entity.data.root_link_lin_vel_w[0]
    base_ang_vel = entity.data.root_link_ang_vel_w[0]

    # ORIGINAL (SUSPECT) INDEXING METHOD: `entity.data.actuator_force` is
    # sliced from the global actuator_force array using
    # `entity.indexing.ctrl_ids`, which orders entries by per-entity
    # ACTUATOR-DECLARATION order, not joint order. Indexing it with a
    # joint-name-derived index (`thigh_i`/`calf_i`, built from the 12-entry
    # `joint_names` list) is a category mismatch, especially in hybrid mode
    # where there are 2 actuators (pos + aux) per recovery joint. Kept here,
    # unmodified, purely for before/after comparison -- do not trust these
    # values for hybrid-mode recovery joints.
    actuator_force = entity.data.actuator_force[0]
    thigh_force = float(actuator_force[thigh_i]) if thigh_i < actuator_force.shape[0] else float("nan")
    calf_force = float(actuator_force[calf_i]) if calf_i < actuator_force.shape[0] else float("nan")

    # CORRECTED (GLOBAL CTRL-ID) INDEXING METHOD: `act._pos_thigh_ctrl_id`,
    # `act._pos_calf_ctrl_id`, `act._aux_thigh_ctrl_id`, `act._aux_calf_ctrl_id`
    # are confirmed (via `_find_actuator_ctrl_id` -> `Actuator.global_ctrl_ids`)
    # to be GLOBAL MuJoCo actuator ids. The raw `env.sim.data` (a `DataBridge`,
    # see `Simulation.data`) is indexed by true global actuator id directly,
    # so these ids can be used to index it without any entity-local
    # remapping. This is the correct way to read per-actuator measured force
    # and commanded ctrl input for the hybrid thigh/calf actuators.
    sim_actuator_force = env.sim.data.actuator_force[0]
    sim_ctrl = env.sim.data.ctrl[0]
    pos_thigh_force_global = float(sim_actuator_force[act._pos_thigh_ctrl_id])
    aux_thigh_force_global = float(sim_actuator_force[act._aux_thigh_ctrl_id])
    pos_calf_force_global = float(sim_actuator_force[act._pos_calf_ctrl_id])
    aux_calf_force_global = float(sim_actuator_force[act._aux_calf_ctrl_id])
    pos_thigh_ctrl_global = float(sim_ctrl[act._pos_thigh_ctrl_id])
    aux_thigh_ctrl_global = float(sim_ctrl[act._aux_thigh_ctrl_id])
    pos_calf_ctrl_global = float(sim_ctrl[act._pos_calf_ctrl_id])
    aux_calf_ctrl_global = float(sim_ctrl[act._aux_calf_ctrl_id])

    exit_reason = ""
    if phase == PHASE_POST_RECOVERY:
      exit_reason = "cap_timeout" if bool(act.recovery_failed_timeout[0]) else "crossing+release"

    recovery_elapsed = int(act.recovery_elapsed[0]) if hasattr(act, "recovery_elapsed") else -1

    return {
      "step": step,
      "sim_time": step * env.step_dt,
      "phase": PHASE_NAMES[phase],
      "recovery_elapsed": recovery_elapsed,
      "wire_in_contact": in_contact,
      "foot_x": float(foot_pos[0]),
      "foot_y": float(foot_pos[1]),
      "foot_z": float(foot_pos[2]),
      "foot_height_above_wire": float(act.foot_height_above_wire[0]),
      "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
      "other_leg_foot_x": float(other_foot_pos[0]),
      "other_leg_foot_y": float(other_foot_pos[1]),
      "other_leg_foot_z": float(other_foot_pos[2]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "hip_pos": float(qpos[hip_i]),
      "hip_vel": float(qvel[hip_i]),
      "fl_ground_contact_geoms": "|".join(g.split("/")[-1] for g in contacting_geoms),
      "fl_ground_contact_geom_forces": "|".join(f"{f:.2f}" for f in contacting_geoms_forces),
      "thigh_actuator_force_ORIGINAL_SUSPECT_INDEXING": thigh_force,
      "calf_actuator_force_ORIGINAL_SUSPECT_INDEXING": calf_force,
      "pos_thigh_actuator_force_global": pos_thigh_force_global,
      "aux_thigh_actuator_force_global": aux_thigh_force_global,
      "pos_calf_actuator_force_global": pos_calf_force_global,
      "aux_calf_actuator_force_global": aux_calf_force_global,
      "pos_thigh_ctrl_global": pos_thigh_ctrl_global,
      "aux_thigh_ctrl_global": aux_thigh_ctrl_global,
      "pos_calf_ctrl_global": pos_calf_ctrl_global,
      "aux_calf_ctrl_global": aux_calf_ctrl_global,
      "fl_ground_contact": fl_ground_found,
      "fl_ground_force_mag": fl_ground_fmag,
      "fr_wire_contact": fr_wire_found,
      "fr_wire_force_mag": fr_wire_fmag,
      "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
      "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
      "peak_armed": bool(act._peak_armed[0]),
      "peak_running_max": float(act._peak_running_max[0]),
      "peak_decline_counter": int(act._peak_decline_counter[0]),
      "peak_triggered": bool(act._peak_triggered[0]),
      "peak_trigger_step": int(act.peak_trigger_step[0]),
      "base_z": float(base_pos[2]),
      "base_quat_w": float(base_quat[0]),
      "base_lin_vel_z": float(base_lin_vel[2]),
      "base_ang_vel_x": float(base_ang_vel[0]),
      "base_ang_vel_y": float(base_ang_vel[1]),
      "running_max_foot_height": running_max_h,
      "recovery_exit_reason": exit_reason,
    }

  def do_step(step_idx: int, action: torch.Tensor) -> dict:
    nonlocal entry_foot_pos, running_max_h, contacted_wire, recovery_activated, fr_confounded
    env.step(action)
    if bool(act.newly_entangled[0]):
      entry_foot_pos = act._entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
      running_max_h = -1e9
      recovery_activated = True
    row = build_row(step_idx)
    if row["wire_in_contact"]:
      contacted_wire = True
    if row["fr_wire_contact"]:
      fr_confounded = True
    return row

  reached_post_recovery_at: int | None = None
  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    step_counter = {"i": 0}

    def viewer_policy(_obs):
      i = step_counter["i"]
      action = torch.zeros(1, act.action_dim, device=device)
      row = do_step(i, action)
      rows.append(row)
      if i % 20 == 0:
        print(
          f"[D3-GROUND] step={i} phase={row['phase']} wire={row['wire_in_contact']} "
          f"fl_ground={row['fl_ground_contact']}({row['fl_ground_force_mag']:.1f}N) "
          f"fr_wire={row['fr_wire_contact']} foot_z={row['foot_z']:.3f}"
        )
      step_counter["i"] += 1
      return action

    print("[INFO] Debug mode: native viewer (DYNAMIC, real physics). Close window to end run.")
    NativeMujocoViewer(env, viewer_policy).run()
  else:
    for i in range(cfg.max_steps):
      action = torch.zeros(1, act.action_dim, device=device)
      row = do_step(i, action)
      rows.append(row)
      if row["phase"] == "POST_RECOVERY" and reached_post_recovery_at is None:
        reached_post_recovery_at = i
      if (
        reached_post_recovery_at is not None
        and i - reached_post_recovery_at >= cfg.stop_steps_after_post_recovery
      ):
        break

  env.close()

  if not rows:
    raise RuntimeError("No steps logged.")

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[D3-GROUND] CSV written: {csv_path} ({len(rows)} rows)")

  _print_causal_timeline(rows, contacted_wire, fr_confounded)

  return csv_path


def _print_causal_timeline(rows: list[dict], contacted_wire: bool, fr_confounded: bool) -> None:
  print("\n==== D3 GROUND-CONTACT / FR-CONFOUND CAUSAL TIMELINE ====")

  def first(pred):
    return next((r for r in rows if pred(r)), None)

  wire_evt = first(lambda r: r["wire_in_contact"])
  rec_evt = first(lambda r: r["phase"] == "RECOVERY")
  recovery_rows = [r for r in rows if r["phase"] == "RECOVERY"]
  peak_row = max(recovery_rows, key=lambda r: r["foot_z"]) if recovery_rows else max(rows, key=lambda r: r["foot_z"])
  # NOTE: FL touches the ground constantly during normal stance/walking --
  # that is expected and uninformative. The scientifically relevant event
  # is ground contact occurring DURING RECOVERY, after the foot has
  # already lifted off (i.e. at/after the foot-height peak), since that is
  # the only contact that could explain a post-peak collapse.
  ground_evt = first(
    lambda r: r["fl_ground_contact"] and r["phase"] == "RECOVERY" and r["step"] >= peak_row["step"]
  )
  post_evt = first(lambda r: r["phase"] == "POST_RECOVERY")
  min_after_peak = None
  after = [r for r in rows if r["step"] >= peak_row["step"]]
  if after:
    min_after_peak = min(after, key=lambda r: r["foot_z"])

  def fmt(r, label):
    if r is None:
      print(f"  {label}: NOT OBSERVED in this run")
      return
    print(
      f"  {label}: step={r['step']} t={r['sim_time']:.3f}s phase={r['phase']} "
      f"foot_z={r['foot_z']:.4f}"
    )

  fmt(wire_evt, "1. FL-wire contact (recovery trigger)")
  fmt(rec_evt, "2. RECOVERY phase entry")
  fmt(peak_row, "3. Foot-height peak")
  fmt(ground_evt, "4. FIRST FL-ground contact DURING RECOVERY (post-peak)")
  if ground_evt is not None:
    ground_rows = [
      r for r in rows
      if r["fl_ground_contact"] and r["phase"] == "RECOVERY" and r["step"] >= peak_row["step"]
    ]
    peak_force = max(r["fl_ground_force_mag"] for r in ground_rows)
    duration = len(ground_rows) * (rows[1]["sim_time"] - rows[0]["sim_time"])
    print(f"     peak force = {peak_force:.2f} N, contact steps = {len(ground_rows)} (~{duration:.3f}s)")
  else:
    all_ground_rows = [r for r in rows if r["fl_ground_contact"]]
    print(
      f"     (FL does touch ground {len(all_ground_rows)}/{len(rows)} steps total, "
      "but NONE of that is during RECOVERY after the foot-height peak)"
    )
  fmt(min_after_peak, "5. Foot-height minimum (post-peak collapse)")
  fmt(post_evt, "6. POST_RECOVERY / walking-resume")

  print("\n  VERDICT -- ground contact explains collapse? ", end="")
  if ground_evt is not None:
    print(
      f"SUPPORTED: FL re-contacts ground at step {ground_evt['step']} "
      f"(foot-height peak was step {peak_row['step']}), i.e. {ground_evt['step'] - peak_row['step']} "
      "step(s) after the peak -- consistent with ground contact arresting/reversing the rise."
    )
  else:
    print(
      "NOT SUPPORTED: no FL-ground contact detected during RECOVERY after the "
      "foot-height peak in this run -- the collapse must have another cause "
      "(e.g. wire contact itself, gravity/inertia without ground support, or "
      "base-disturbance coupling). See CSV for fl_ground_contact column."
    )

  print(f"\n  FR-wire confound: {'CONFOUNDED (FR contacted wire)' if fr_confounded else 'clean (FR did not contact wire)'}")
  print(f"  FL contacted wire (recovery trigger fired): {contacted_wire}")
  print("\n  Dynamic (not static) visualization of this exact run:")
  print("    uv run python scripts/d3_ground_contact_diagnostic.py --debug")


def main() -> None:
  args = tyro.cli(D3GroundConfig, args=sys.argv[1:])
  run(args)


if __name__ == "__main__":
  main()
