"""D4.4 FL hip-wire geometry reconstruction (read-only, seeds 0 and 1 only).

Reuses the exact seed-0/seed-1 isolated-FL-wire deterministic rollout
(same `build_env`/`D3GroundConfig` from `scripts/d3_ground_contact_diagnostic.py`,
zero residual action throughout -- identical to the runs already analyzed
in D4.2/D4.3). No controller, FSM, gain, threshold, or wire-geometry change.

WHY THIS RE-RUNS THE SIMULATION INSTEAD OF PARSING EXISTING CSVs:
The existing logs (`logs/d3_peak_baseline_seed0/trajectory.csv`,
`logs/d4_contact_constraint_check/...`, `logs/d4_contact_constraint_seed1/...`)
do NOT contain the full base position (only `base_z`), the full base
quaternion (only `base_quat_w`), or any per-geom world pose -- all of
which are required to reconstruct the FL_hip_collision geom's world-frame
transform via manual forward kinematics. Rather than manually re-deriving
forward kinematics from an incomplete logged subset (error-prone and
redundant), this script reads the SAME quantities MuJoCo itself already
computes for the (identical, deterministic, already-validated) rollout:
`entity.data.geom_pose_w` (per-geom world pos+quat, computed by the
simulator's own forward kinematics) and the wire segment geom world poses
via `env.sim.data.geom_xpos`/`geom_xmat` (already the pattern used by
`mjlab.tasks.recovery.mdp.wire_geometry`). This is strictly more accurate
than a manual quaternion reconstruction and introduces no new physics,
seeds, or controller behavior -- it only reads MORE fields from the exact
same deterministic steps already analyzed.

Geometry model used for the minimum-distance calculation:
  - FL_hip_collision: MJCF cylinder, size=(radius=0.046, halflength=0.02),
    local offset pos=(0, 0.08, 0), local quat=(1,1,0,0) within the FL_hip
    body (see go2.xml). Modeled as a finite line SEGMENT of length
    2*halflength along the geom's local Z axis (world-transformed),
    inflated by `radius` -- i.e. treated as a capsule for the purposes of
    a conservative nearest-distance estimate. NOTE: a true cylinder has
    flat end-caps, so this capsule approximation can slightly
    UNDERESTIMATE distance very close to the cylinder's flat ends (by at
    most ~radius in the worst case); this is a known, reported
    approximation, not an exact cylinder-capsule solver.
  - Wire segments: MJCF capsules, radius=BallCableCfg.radius (0.006 for
    the isolated FL wire), half-length=segment_len/2, exact capsule
    geometry (matches the simulator's own contact-solver assumption).

Minimum distance between two finite 3D line segments uses the standard
closed-form segment-segment nearest-point algorithm, then subtracts the
sum of the two radii to get surface-to-surface clearance, directly
comparable to MuJoCo's own `contact.dist` (also a surface gap).
"""

from __future__ import annotations

import csv
import math
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
from mjlab.tasks.recovery.env_cfg_residual import ISOLATED_FL_WIRE_CFG  # noqa: E402
from mjlab.tasks.recovery.mdp.wire_geometry import (  # noqa: E402
  resolve_wire_segment_geom_ids,
)

FL_HIP_COLLISION_RADIUS = 0.046
FL_HIP_COLLISION_HALFLEN = 0.02
WIRE_RADIUS = ISOLATED_FL_WIRE_CFG.radius
WIRE_SEG_HALFLEN = ISOLATED_FL_WIRE_CFG.segment_len / 2.0


def quat_to_zaxis(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Local +Z axis of a geom, rotated into world frame, from a wxyz quat."""
  w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
  # Standard quat-to-rotation-matrix third column (R @ [0,0,1]).
  zx = 2 * (x * z + w * y)
  zy = 2 * (y * z - w * x)
  zz = 1 - 2 * (x * x + y * y)
  return torch.tensor([zx, zy, zz], dtype=quat_wxyz.dtype)


def segment_segment_min_dist(
  p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor
) -> float:
  """Closed-form minimum distance between finite 3D segments [p0,p1], [q0,q1]."""
  d1 = p1 - p0
  d2 = q1 - q0
  r = p0 - q0
  a = float(torch.dot(d1, d1))
  e = float(torch.dot(d2, d2))
  f = float(torch.dot(d2, r))

  if a <= 1e-12 and e <= 1e-12:
    return float(torch.norm(r))
  if a <= 1e-12:
    t = max(0.0, min(1.0, f / e))
    s = 0.0
  else:
    c = float(torch.dot(d1, r))
    if e <= 1e-12:
      t = 0.0
      s = max(0.0, min(1.0, -c / a))
    else:
      b = float(torch.dot(d1, d2))
      denom = a * e - b * b
      s = 0.0 if abs(denom) < 1e-12 else max(0.0, min(1.0, (b * f - c * e) / denom))
      t = (b * s + f) / e
      if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
      elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b - c) / a))
  closest_p = p0 + s * d1
  closest_q = q0 + t * d2
  return float(torch.norm(closest_p - closest_q))


@dataclass
class D4GeomConfig:
  seed: int = 0
  device: str | None = None
  max_steps: int = 60
  window_start: int = 49
  window_end: int = 53


def run(cfg: D4GeomConfig) -> list[dict]:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_env(d3_cfg, device)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity

  geom_names = list(entity.geom_names)
  hip_geom_idx = geom_names.index("FL_hip_collision")

  wire_geom_ids = resolve_wire_segment_geom_ids(env, ISOLATED_FL_WIRE_CFG.geom_names())
  wire_names = list(ISOLATED_FL_WIRE_CFG.geom_names())

  env.reset(seed=cfg.seed)

  rows = []
  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    env.step(action)

    if not (cfg.window_start <= step <= cfg.window_end):
      continue

    hip_pose = entity.data.geom_pose_w[0, hip_geom_idx]  # [7]: pos(3)+quat(4) wxyz
    hip_pos = hip_pose[0:3]
    hip_quat = hip_pose[3:7]
    hip_z = quat_to_zaxis(hip_quat)
    hip_p0 = hip_pos - hip_z * FL_HIP_COLLISION_HALFLEN
    hip_p1 = hip_pos + hip_z * FL_HIP_COLLISION_HALFLEN

    wire_pos = env.sim.data.geom_xpos[0, wire_geom_ids]  # [N, 3]
    wire_xmat = env.sim.data.geom_xmat[0, wire_geom_ids]  # [N, 3, 3] or [N,9]
    wire_xmat = wire_xmat.reshape(-1, 3, 3)

    best_dist = None
    best_seg = None
    for i in range(wire_pos.shape[0]):
      seg_pos = wire_pos[i]
      seg_z = wire_xmat[i][:, 2]
      q0 = seg_pos - seg_z * WIRE_SEG_HALFLEN
      q1 = seg_pos + seg_z * WIRE_SEG_HALFLEN
      centerline_dist = segment_segment_min_dist(hip_p0, hip_p1, q0, q1)
      clearance = centerline_dist - (FL_HIP_COLLISION_RADIUS + WIRE_RADIUS)
      if best_dist is None or clearance < best_dist:
        best_dist = clearance
        best_seg = wire_names[i]

    thigh_i = list(entity.joint_names).index("FL_thigh_joint")
    calf_i = list(entity.joint_names).index("FL_calf_joint")
    qpos = entity.data.joint_pos[0]
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]

    rows.append(
      {
        "seed": cfg.seed,
        "step": step,
        "hip_collision_world_pos": [round(float(x), 5) for x in hip_pos],
        "reconstructed_clearance": round(best_dist, 5),
        "nearest_wire_segment": best_seg,
        "thigh_pos": round(float(qpos[thigh_i]), 5),
        "calf_pos": round(float(qpos[calf_i]), 5),
        "base_pos": [round(float(x), 5) for x in base_pos],
        "base_quat_wxyz": [round(float(x), 5) for x in base_quat],
      }
    )

  env.close()
  return rows


def main() -> None:
  all_rows = []
  for seed in (0, 1):
    cfg = D4GeomConfig(seed=seed)
    rows = run(cfg)
    all_rows.extend(rows)
    print(f"\n==== seed {seed} ====")
    for r in rows:
      print(
        f"step={r['step']:3d}  hip_pos_w={r['hip_collision_world_pos']}  "
        f"clearance={r['reconstructed_clearance']:+.5f}  nearest={r['nearest_wire_segment']}  "
        f"thigh={r['thigh_pos']:.4f}  calf={r['calf_pos']:.4f}  "
        f"base_pos={r['base_pos']}  base_quat={r['base_quat_wxyz']}"
      )

  out_dir = Path("logs/d4_hip_wire_geometry_reconstruction")
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_path = out_dir / "reconstruction.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)
  print(f"\n[D4-GEOM] CSV written: {csv_path}")


if __name__ == "__main__":
  main()
