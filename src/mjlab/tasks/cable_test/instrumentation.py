"""Instrumentation helpers for the cable-entanglement test environment.

These functions read directly from the environment's entity data / sensors.
They are intentionally *not* wired into the observation manager, so they have
zero effect on the trained policy's observation layout — they exist purely
for logging/inspection (requirement: instrument, don't retrain).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_JOINT_SUFFIXES = ("hip_joint", "thigh_joint", "calf_joint")


def _leg_joint_names(leg: str) -> list[str]:
  return [f"{leg}_{suffix}" for suffix in LEG_JOINT_SUFFIXES]


def collect_instrumentation(env: "ManagerBasedRlEnv") -> dict[str, torch.Tensor]:
  """Collect a flat dict of instrumentation tensors, batched over envs.

  Returns tensors on ``env.device``; caller is responsible for moving to CPU
  / indexing a specific environment for printing or CSV logging.
  """
  robot = env.scene["robot"]
  data = robot.data

  out: dict[str, torch.Tensor] = {}

  # Root state.
  out["root_lin_vel_w"] = data.root_link_lin_vel_w  # [B, 3]
  out["root_ang_vel_w"] = data.root_link_ang_vel_w  # [B, 3]
  out["projected_gravity_b"] = data.projected_gravity_b  # [B, 3]
  out["root_pos_w"] = data.root_link_pos_w  # [B, 3]

  # Per-leg foot site positions, joint positions/velocities/torques.
  joint_names = list(robot.joint_names)
  site_names = list(robot.site_names)

  for leg in LEG_NAMES:
    if leg in site_names:
      idx = site_names.index(leg)
      out[f"foot_pos_{leg}"] = data.site_pos_w[:, idx, :]

    jids = [joint_names.index(n) for n in _leg_joint_names(leg) if n in joint_names]
    if jids:
      jids_t = torch.as_tensor(jids, device=env.device, dtype=torch.long)
      out[f"joint_pos_{leg}"] = data.joint_pos[:, jids_t]
      out[f"joint_vel_{leg}"] = data.joint_vel[:, jids_t]
      try:
        out[f"joint_torque_{leg}"] = data.actuator_force[:, jids_t]
      except (AttributeError, IndexError):
        pass

  # Cable state (raw mujoco access; the cable is not a scene Entity).
  cable_positions = []
  mj_model = env.sim.mj_model
  for i in range(64):  # generous upper bound on segment count.
    name = f"cable_seg{i}"
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid == -1:
      break
    cable_positions.append(env.sim.data.geom_xpos[:, gid, :])
  if cable_positions:
    out["cable_segment_pos"] = torch.stack(cable_positions, dim=1)  # [B, N, 3]
    out["cable_centroid"] = out["cable_segment_pos"].mean(dim=1)  # [B, 3]

  # Cable contact sensors (see env_cfg.py: one sensor per foot).
  total_contacts = None
  for leg in LEG_NAMES:
    sensor_name = f"cable_contact_{leg}"
    if sensor_name in env.scene.sensors:
      sdata = env.scene.sensors[sensor_name].data
      found = sdata.found  # [B, num_segments]
      in_contact = (found > 0).any(dim=-1)  # [B]
      out[f"cable_contact_{leg}"] = in_contact
      n_contacts = (found > 0).sum(dim=-1)  # [B]
      out[f"cable_contact_count_{leg}"] = n_contacts
      total_contacts = (
        n_contacts if total_contacts is None else total_contacts + n_contacts
      )
  if total_contacts is not None:
    out["cable_contact_count_total"] = total_contacts

  return out


def format_instrumentation_row(info: dict[str, torch.Tensor], env_idx: int = 0) -> dict:
  """Flatten the batched instrumentation dict to plain python scalars/lists
  for a single environment index, suitable for CSV logging or printing."""
  row: dict = {}
  for key, val in info.items():
    v = val[env_idx].detach().cpu()
    if v.numel() == 1:
      row[key] = v.item()
    else:
      row[key] = v.tolist()
  return row
