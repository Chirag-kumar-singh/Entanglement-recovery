"""D4.2 contact/constraint diagnostic for the FL foot-height collapse.

Reuses the exact seed-0 isolated-FL-wire config from
`scripts/d3_ground_contact_diagnostic.py` (same `build_env`/`D3GroundConfig`,
same checkpoint, same wire geometry, same recovery controller -- NOTHING
about the controller/FSM/actuators/wire is changed here) and adds READ-ONLY
inspection of:

  1. Raw MuJoCo contact pairs (`env.sim.data.contact`): geom ids/names,
     penetration distance, contact position, for every step in a configurable
     window (default steps 49-54), not just the existing binary wire/ground
     ContactSensors.
  2. Per-DOF constraint force (`env.sim.data.qfrc_constraint`) for the FL
     thigh and FL calf joints, which aggregates both contact AND
     joint-limit constraint forces acting on that DOF (mujoco_warp does not
     expose a separately-attributed per-constraint-type force breakdown).
  3. FL thigh/calf joint position vs. the MJCF-defined `jnt_range` limits
     (read directly from `env.sim.mj_model`, not invented).

Does NOT modify the controller, FSM, actuator setup, gains, thresholds, or
wire geometry. Purely additive, read-only diagnostics appended to the same
per-step loop structure as the D3 script.

Usage::

    uv run python scripts/d4_contact_constraint_diagnostic.py \\
        --window-start 49 --window-end 54 \\
        --out-dir logs/d4_contact_constraint_check
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import (  # noqa: E402
  D3GroundConfig,
  build_env,
)


@dataclass
class D4ContactConfig:
  seed: int = 0
  device: str | None = None
  max_steps: int = 120
  out_dir: str = "logs/d4_contact_constraint_check"
  window_start: int = 44
  window_end: int = 62
  """Inclusive step range to print detailed contact listings for (the
  main CSV covers the full run regardless)."""


def run(cfg: D4ContactConfig) -> Path:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  d3_cfg = D3GroundConfig(
    seed=cfg.seed,
    device=device,
    use_isolated_wire=True,
    out_dir=cfg.out_dir,
  )
  env = build_env(d3_cfg, device)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  mj_model = env.sim.mj_model  # raw mujoco.MjModel, read-only reference

  leg = "FL"
  thigh_name = f"{leg}_thigh_joint"
  calf_name = f"{leg}_calf_joint"
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index(thigh_name)
  calf_i = joint_names.index(calf_name)

  # Raw mujoco.MjModel names are entity-qualified (e.g. "robot/FL_thigh_joint"),
  # unlike `entity.joint_names` which are bare. Confirmed empirically.
  thigh_jid = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{thigh_name}"
  )
  calf_jid = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{calf_name}"
  )
  assert thigh_jid >= 0 and calf_jid >= 0, "joint name lookup failed"
  thigh_dof = int(mj_model.jnt_dofadr[thigh_jid])
  calf_dof = int(mj_model.jnt_dofadr[calf_jid])
  thigh_range = tuple(float(x) for x in mj_model.jnt_range[thigh_jid])
  calf_range = tuple(float(x) for x in mj_model.jnt_range[calf_jid])

  print(f"[D4-CONTACT] FL_thigh_joint range (rad): {thigh_range}")
  print(f"[D4-CONTACT] FL_calf_joint  range (rad): {calf_range}")
  print(f"[D4-CONTACT] FL_thigh dof index: {thigh_dof}  FL_calf dof index: {calf_dof}")

  def geom_name(gid: int) -> str:
    if gid < 0:
      return "<none>"
    n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom#{gid}"

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  rows: list[dict] = []
  contact_log_lines: list[str] = []

  env.reset(seed=cfg.seed)

  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    env.step(action)

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    phase = int(act.phase[0])
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()

    qfrc = env.sim.data.qfrc_constraint
    thigh_qfrc = float(qfrc[0, thigh_dof])
    calf_qfrc = float(qfrc[0, calf_dof])

    # Raw contact inspection: filter contacts belonging to world 0.
    contact = env.sim.data.contact
    nacon = int(env.sim.data.nacon[0]) if hasattr(env.sim.data, "nacon") else 0
    world_ids = contact.worldid[:nacon] if nacon > 0 else None
    step_contacts = []
    if nacon > 0:
      for ci in range(nacon):
        if int(world_ids[ci]) != 0:
          continue
        g0, g1 = int(contact.geom[ci][0]), int(contact.geom[ci][1])
        n0, n1 = geom_name(g0), geom_name(g1)
        dist = float(contact.dist[ci])
        pos = [float(x) for x in contact.pos[ci]]
        step_contacts.append((n0, n1, dist, pos))

    involves_fl = [c for c in step_contacts if "FL_" in c[0] or "FL_" in c[1]]
    involves_wire = [
      c
      for c in step_contacts
      if "wire" in c[0].lower()
      or "wire" in c[1].lower()
      or "cable" in c[0].lower()
      or "cable" in c[1].lower()
    ]

    if cfg.window_start <= step <= cfg.window_end:
      contact_log_lines.append(
        f"step={step} phase={phase} n_contacts(world0)={len(step_contacts)}"
      )
      for n0, n1, dist, pos in step_contacts:
        contact_log_lines.append(f"    {n0}  <->  {n1}  dist={dist:.5f}  pos={pos}")
      if not step_contacts:
        contact_log_lines.append("    (no contacts detected in world 0)")

    rows.append(
      {
        "step": step,
        "phase": phase,
        "foot_z": float(foot_pos[2]),
        "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
        "foot_height_above_wire": float(act.foot_height_above_wire[0]),
        "thigh_pos": float(qpos[thigh_i]),
        "thigh_vel": float(qvel[thigh_i]),
        "calf_pos": float(qpos[calf_i]),
        "calf_vel": float(qvel[calf_i]),
        "thigh_range_lo": thigh_range[0],
        "thigh_range_hi": thigh_range[1],
        "calf_range_lo": calf_range[0],
        "calf_range_hi": calf_range[1],
        "thigh_dist_to_lo": float(qpos[thigh_i]) - thigh_range[0],
        "thigh_dist_to_hi": thigh_range[1] - float(qpos[thigh_i]),
        "calf_dist_to_lo": float(qpos[calf_i]) - calf_range[0],
        "calf_dist_to_hi": calf_range[1] - float(qpos[calf_i]),
        "thigh_qfrc_constraint": thigh_qfrc,
        "calf_qfrc_constraint": calf_qfrc,
        "n_contacts_world0": len(step_contacts),
        "n_contacts_fl": len(involves_fl),
        "n_contacts_wire": len(involves_wire),
        "fl_contact_pairs": "|".join(
          f"{n0}<->{n1}:{d:.4f}" for n0, n1, d, _ in involves_fl
        ),
        "wire_contact_pairs": "|".join(
          f"{n0}<->{n1}:{d:.4f}" for n0, n1, d, _ in involves_wire
        ),
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
      }
    )

  csv_path = out_dir / "contact_constraint_trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[D4-CONTACT] CSV written: {csv_path} ({len(rows)} rows)")

  log_path = out_dir / "contact_listing.txt"
  with open(log_path, "w") as f:
    f.write(f"FL_thigh_joint range (rad): {thigh_range}\n")
    f.write(f"FL_calf_joint  range (rad): {calf_range}\n\n")
    f.write("\n".join(contact_log_lines))
  print(f"[D4-CONTACT] Contact listing written: {log_path}")

  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D4ContactConfig)
  run(cfg)


if __name__ == "__main__":
  main()
