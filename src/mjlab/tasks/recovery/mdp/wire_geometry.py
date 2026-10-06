"""Geometry-aware wire clearance helpers for the residual recovery task.

Computes the wire's CURRENT (possibly deformed) centerline position near
a target world y-coordinate from the live ball-joint wire segment geom
positions (``env.sim.data.geom_xpos``), rather than assuming the wire's
static initial placement. This lets ``ResidualRecoveryAction`` determine
genuine geometric clearance/crossing of the affected foot -- "did the
foot actually go over the wire" -- as opposed to merely "did the contact
sensor stop firing", which was the root cause of the earlier
contact-release reward exploit (see chat history / audit).

The wire is a flexible chain (see ``mjlab.tasks.cable_test.ball_cable``),
so there is no fixed obstacle plane; instead, each segment's own live
midpoint position (``geom_xpos`` for a fromto-defined capsule is its
world-space midpoint) is used as a sample of the current centerline, and
the two segments whose y-coordinates bracket the foot's current y are
linearly interpolated to get a local (x, z) estimate right under/over the
foot -- not just "nearest segment" or "initial anchor position".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def resolve_wire_segment_geom_ids(
  env: "ManagerBasedRlEnv", geom_names: tuple[str, ...]
) -> torch.Tensor:
  """Resolve the wire's segment geom names to MuJoCo geom ids.

  Returns:
    Long tensor of shape [N] (N = number of wire segments), on env.device.
  """
  mj_model = env.sim.mj_model
  ids: list[int] = []
  for name in geom_names:
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid == -1:
      raise RuntimeError(f"Wire segment geom '{name}' not found in model.")
    ids.append(gid)
  return torch.tensor(ids, device=env.device, dtype=torch.long)


def wire_segment_world_pos(
  env: "ManagerBasedRlEnv", geom_ids: torch.Tensor
) -> torch.Tensor:
  """Current world positions of all wire segment geom midpoints.

  Returns:
    Tensor of shape [B, N, 3] (B = num_envs, N = num_segments). This
    reads the LIVE deformed wire state every call (nworld-batched, so
    each env's own independent wire copy is read correctly).
  """
  return env.sim.data.geom_xpos[:, geom_ids, :]


def local_wire_xz_at_y(
  seg_pos: torch.Tensor, target_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Interpolate the wire centerline's (x, z) at a given world y, from
  live (possibly-deformed) segment positions.

  Args:
    seg_pos: [B, N, 3] world positions of wire segment midpoints (see
      ``wire_segment_world_pos``).
    target_y: [B] world y-coordinate to interpolate at (typically the
      affected foot's current world y).

  Returns:
    (local_x, local_z): each [B], the interpolated wire centerline
    (x, z) nearest ``target_y``. Linearly interpolates between the two
    segments bracketing ``target_y``; clamps to the nearest segment's own
    (x, z) at/beyond the wire's endpoints (no bracketing pair exists).
  """
  seg_y = seg_pos[..., 1]  # [B, N]
  diff = seg_y - target_y.unsqueeze(-1)  # [B, N]

  nearest_idx = torch.argmin(diff.abs(), dim=-1)  # [B], endpoint fallback

  neg_inf = torch.full_like(diff, -1e9)
  pos_inf = torch.full_like(diff, 1e9)
  below_vals = torch.where(diff <= 0, diff, neg_inf)
  above_vals = torch.where(diff >= 0, diff, pos_inf)
  idx_below = torch.argmax(below_vals, dim=-1)  # closest with seg_y <= target_y
  idx_above = torch.argmin(above_vals, dim=-1)  # closest with seg_y >= target_y

  has_below = below_vals.gather(-1, idx_below.unsqueeze(-1)).squeeze(-1) > -1e9 + 1.0
  has_above = above_vals.gather(-1, idx_above.unsqueeze(-1)).squeeze(-1) < 1e9 - 1.0
  valid = has_below & has_above

  y_below = seg_y.gather(-1, idx_below.unsqueeze(-1)).squeeze(-1)
  y_above = seg_y.gather(-1, idx_above.unsqueeze(-1)).squeeze(-1)
  denom = y_above - y_below
  denom_safe = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)
  w = torch.clamp((target_y - y_below) / denom_safe, 0.0, 1.0)

  def _interp(coord_idx: int) -> torch.Tensor:
    c_below = seg_pos[..., coord_idx].gather(-1, idx_below.unsqueeze(-1)).squeeze(-1)
    c_above = seg_pos[..., coord_idx].gather(-1, idx_above.unsqueeze(-1)).squeeze(-1)
    c_nearest = seg_pos[..., coord_idx].gather(-1, nearest_idx.unsqueeze(-1)).squeeze(-1)
    interpolated = c_below + w * (c_above - c_below)
    return torch.where(valid, interpolated, c_nearest)

  local_x = _interp(0)
  local_z = _interp(2)
  return local_x, local_z
