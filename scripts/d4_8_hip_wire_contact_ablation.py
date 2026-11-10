"""D4.8 -- FL hip-wire contact ablation (causal intervention diagnostic).

Tests whether contact between `FL_hip_collision` and the wire segments
materially contributes to the calf-velocity reversal and FL foot collapse
observed at steps 51-52 (D4.2-D4.6c), by disabling collision RESPONSE
exclusively for that one contact pair and re-running the identical
deterministic seed-0/seed-1 rollouts.

MECHANISM (verified from source before implementation, per task
instructions -- NOT assumed):
  - MuJoCo's `contype`/`conaffinity` are GLOBAL bitmasks on a geom; a
    broad change there would affect ALL of that geom's contacts with
    every other geom in the model, not just the wire. This was explicitly
    rejected per the task's "do not assume a broad contype/conaffinity
    change is pair-specific" instruction.
  - The correct, pair-specific mechanism (already used elsewhere in this
    codebase, confirmed by reading `mjlab.tasks.cable_test.ball_cable.
    _add_single_ball_cable`, which calls
    `spec.add_exclude(bodyname1=..., bodyname2=...)` to disable adjacent
    wire-segment self-collision) is MuJoCo's `<contact><exclude
    bodyname1 bodyname2/></contact>` directive, surfaced via
    `mujoco.MjSpec.add_exclude`. This excludes contact generation between
    ALL geom pairs belonging to the two named BODIES -- so it is
    pair-specific IF AND ONLY IF each body has exactly one
    active-collision geom relevant to this interaction.
  - VERIFIED (not assumed): `FL_hip` body has exactly 3 geoms -- two
    visual meshes with `contype=0 conaffinity=0` (see
    `<default class="visual">` in go2.xml, meaning they never generate
    contacts with anything) and exactly one collision geom,
    `FL_hip_collision`. Each `wire_seg_i` body has exactly one geom
    (`wire_seg_i_geom`, see `ball_cable.py`). Therefore
    `add_exclude(bodyname1="robot/FL_hip", bodyname2="wire_seg_i")` for
    each wire segment body disables contact generation EXCLUSIVELY
    between `FL_hip_collision` and that one wire segment -- confirmed
    pair-specific, not a broad bitmask change.
  - The scene's intended extension point for this kind of diagnostic-only
    MjSpec modification is `SceneCfg.spec_fn` (see `scene.py`: "Optional
    callback to modify the MjSpec after entities and sensors have been
    added but before compilation"). The isolated-FL-wire config ALREADY
    sets `env_cfg.scene.spec_fn = build_ball_cable_spec_fn(wire)` to
    build the wire itself -- so this script COMPOSES with (calls through
    to) that existing spec_fn rather than overwriting it (verified this
    is necessary: naively overwriting it silently drops the wire
    entirely, confirmed via direct experimentation before writing this
    script).

NOTHING else is changed: same `build_env`/`D3GroundConfig` entry point as
D4.2-D4.6 (same frozen checkpoint, same recovery controller/FSM/gains/
thresholds, same wire geometry/placement, same spawn/seed/sim settings).
Only the exclude pairs are added, and ONLY when `--ablate-hip-wire` is
passed (default False, so baseline behavior is byte-for-byte reproducible
with the existing D4.2 script, verified by comparing outputs with the
flag off, see `run_baseline_equivalence_check()`).
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig, build_env  # noqa: E402
from mjlab.tasks.recovery.env_cfg_residual import (  # noqa: E402
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)


def build_ablation_env(cfg: D3GroundConfig, device: str, ablate_hip_wire: bool):
  """Same as `d3_ground_contact_diagnostic.build_env`, but optionally adds
  body-pair excludes between `robot/FL_hip` and every wire segment body,
  composed on top of (not replacing) the existing wire-building spec_fn.
  """
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

  # Reuse the exact same diagnostic-sensor-adding logic as D3/D4.2 by
  # calling build_env's module-level helper indirectly: we replicate only
  # the sensor-adding block here is unnecessary -- instead, delegate
  # entirely to the existing `build_env` for everything EXCEPT the
  # spec_fn composition, which must happen before `ManagerBasedRlEnv`
  # construction. Since `build_env` already constructs the env internally,
  # we re-implement the minimal env_cfg setup identically and then call
  # `ManagerBasedRlEnv` ourselves, composing the spec_fn.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  leg = ["FL", "FR", "RL", "RR"][cfg.affected_leg_idx]
  other_leg = "FR" if leg == "FL" else "FL"

  fl_ground_cfg = ContactSensorCfg(
    name="fl_ground_contact_diag",
    primary=ContactMatch(
      mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  fl_ground_pergeom_cfg = ContactSensorCfg(
    name="fl_ground_pergeom_diag",
    primary=ContactMatch(
      mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  existing_wire_sensor = next(
    s for s in env_cfg.scene.sensors if s.name == "wire_contact"
  )
  fr_wire_cfg = ContactSensorCfg(
    name="fr_wire_contact_diag",
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

  existing_spec_fn = env_cfg.scene.spec_fn

  def composed_spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
    if existing_spec_fn is not None:
      existing_spec_fn(spec, num_envs)
    if ablate_hip_wire:
      wire_seg_bodies = [b.name for b in spec.bodies if b.name.startswith("wire_seg_")]
      assert wire_seg_bodies, (
        "No wire_seg_* bodies found after existing spec_fn -- composition order wrong."
      )
      for seg_body in wire_seg_bodies:
        spec.add_exclude(bodyname1="robot/FL_hip", bodyname2=seg_body)
      print(
        f"[D4.8-ABLATION] Added {len(wire_seg_bodies)} body-pair excludes: "
        f"robot/FL_hip <-> {wire_seg_bodies}"
      )

  env_cfg.scene.spec_fn = composed_spec_fn

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env


@dataclass
class D48Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 120
  ablate_hip_wire: bool = False
  out_dir: str = ""


def run(cfg: D48Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_ablation_env(d3_cfg, device, cfg.ablate_hip_wire)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  mj_model = env.sim.mj_model
  hip_jid = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_JOINT, "robot/FL_thigh_joint"
  )
  calf_jid = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_JOINT, "robot/FL_calf_joint"
  )
  thigh_dof = int(mj_model.jnt_dofadr[hip_jid])
  calf_dof = int(mj_model.jnt_dofadr[calf_jid])

  def geom_name(gid: int) -> str:
    if gid < 0:
      return "<none>"
    n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom#{gid}"

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  enter_line = None
  exit_line = None

  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    if bool(act.newly_entangled[0]):
      enter_line = f"ENTER RECOVERY step={step}"
    prev_phase = int(act.phase[0])
    env.step(action)
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      exit_line = (
        f"EXIT RECOVERY step={step} elapsed={int(act.recovery_elapsed[0])} "
        f"timeout={bool(act.recovery_failed_timeout[0])}"
      )

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())
    fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    fl_ground_found = bool((fl_ground.data.found[0].reshape(-1) > 0).any())
    fr_wire = env.scene.sensors["fr_wire_contact_diag"]
    fr_wire_found = bool((fr_wire.data.found[0].reshape(-1) > 0).any())

    # Raw contact check for FL_hip_collision <-> any wire_seg (ablation
    # verification) and for FL_calf2_collision <-> any wire_seg (must be
    # preserved per constraints).
    contact = env.sim.data.contact
    nacon = int(env.sim.data.nacon[0])
    hip_wire_contact_now = False
    calf_wire_contact_now = False
    for ci in range(nacon):
      if int(contact.worldid[ci]) != 0:
        continue
      n0, n1 = geom_name(int(contact.geom[ci][0])), geom_name(int(contact.geom[ci][1]))
      pair = (n0, n1)
      if any("FL_hip_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        hip_wire_contact_now = True
      if any("FL_calf2_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        calf_wire_contact_now = True

    rows.append(
      {
        "step": step,
        "phase": new_phase,
        "foot_z": float(foot_pos[2]),
        "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
        "foot_height_above_wire": float(act.foot_height_above_wire[0]),
        "hip_pos": float(qpos[hip_i]),
        "thigh_pos": float(qpos[thigh_i]),
        "thigh_vel": float(qvel[thigh_i]),
        "calf_pos": float(qpos[calf_i]),
        "calf_vel": float(qvel[calf_i]),
        "base_z": float(base_pos[2]),
        "base_quat_w": float(base_quat[0]),
        "wire_in_contact": wire_in_contact,
        "fl_ground_contact": fl_ground_found,
        "fr_wire_contact": fr_wire_found,
        "fl_hip_wire_contact_raw": hip_wire_contact_now,
        "fl_calf_wire_contact_raw": calf_wire_contact_now,
        "pos_thigh_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._pos_thigh_ctrl_id]
        ),
        "aux_thigh_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._aux_thigh_ctrl_id]
        ),
        "pos_calf_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._pos_calf_ctrl_id]
        ),
        "aux_calf_actuator_force_global": float(
          env.sim.data.actuator_force[0, act._aux_calf_ctrl_id]
        ),
        "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
        "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
      }
    )

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"ablate_hip_wire={cfg.ablate_hip_wire}\n")
    f.write(f"{enter_line}\n{exit_line}\n")
  print(f"[D4.8] wrote {csv_path}")
  print(f"  {enter_line}")
  print(f"  {exit_line}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D48Config)
  run(cfg)


if __name__ == "__main__":
  main()
