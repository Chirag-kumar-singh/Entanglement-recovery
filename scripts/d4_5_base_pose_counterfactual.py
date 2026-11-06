"""D4.5 counterfactual base-pose analysis for FL hip-wire clearance.

Read-only, reuses the EXACT same geometry model, clearance convention,
and quaternion/segment-distance math already validated in
`scripts/d4_hip_wire_geometry_reconstruction.py` (D4.4): FL_hip_collision
modeled as a capsule-approximated cylinder (radius=0.046, halflength=0.02),
wire segments modeled as true capsules (radius=0.006,
halflength=segment_len/2), clearance = segment-segment centerline
distance minus the sum of the two radii.

No new dynamic simulation beyond the SAME deterministic seed-0/seed-1
zero-residual rollout already run in D4.2/D4.3/D4.4. This script re-runs
that identical rollout (same `build_env`/`D3GroundConfig`) purely to read
MORE already-computed simulator fields (`entity.data.geom_pose_w` and
`entity.data.root_link_pose_w` at every step 44-53, not just the 49-53
window D4.4 reported) -- no controller, FSM, gain, threshold, or wire
change.

COUNTERFACTUAL METHOD (purely geometric algebra on recorded data, not a
new physical simulation):
  For each step s, the simulator gives us the REAL hip-collision-geom
  world pose `G(s)` and the REAL base-link world pose `B(s)`. We recover
  the "local chain" transform `L(s) = B(s)^-1 * G(s)` -- i.e. the
  transform from the base-link frame to the hip-collision-geom frame,
  which is entirely determined by the recorded hip/thigh joint angles at
  step s (the chain base->FL_hip->geom) and contains NO information about
  the base's own world pose. This is exact rigid-body algebra on the
  simulator's own already-computed poses, not a re-derivation or
  approximation of the kinematic chain.

  We then recompose this SAME recorded local chain with a counterfactual
  base pose to ask "where would this geom be if the base had a different
  world pose, holding the recorded joint configuration fixed":
    - fixed_base:        B_cf = B(44)              -> G_cf = B(44) * L(s)
    - translation_only:  B_cf = (B(44).pos, B(s).quat) -> G_cf = ...
    - rotation_only:     B_cf = (B(s).pos, B(44).quat) -> G_cf = ...
    - actual:            B_cf = B(s)                -> G_cf = G(s) (sanity check)

  These are GEOMETRIC "what-if" snapshots only. They do NOT represent a
  physically realizable or dynamically stable robot configuration (the
  real robot's joint angles would differ under a different base
  trajectory, due to contact forces, balance, actuator response, etc).
  They isolate the geometric contribution of recorded base translation
  vs. recorded base rotation to the hip-wire proximity, nothing more.

Wire segment poses are taken directly from the actual (unperturbed) run
at each step -- the wire's own physical state is not counterfactually
altered, only the hip geom's reconstructed position is.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig, build_env  # noqa: E402
from d4_hip_wire_geometry_reconstruction import (  # noqa: E402
  FL_HIP_COLLISION_HALFLEN,
  FL_HIP_COLLISION_RADIUS,
  WIRE_RADIUS,
  WIRE_SEG_HALFLEN,
  quat_to_zaxis,
  segment_segment_min_dist,
)
from mjlab.tasks.recovery.env_cfg_residual import ISOLATED_FL_WIRE_CFG  # noqa: E402
from mjlab.tasks.recovery.mdp.wire_geometry import (  # noqa: E402
  resolve_wire_segment_geom_ids,
)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  """Conjugate (inverse for unit quats) of a wxyz quaternion."""
  return torch.tensor([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product of two wxyz quaternions."""
  aw, ax, ay, az = a[0], a[1], a[2], a[3]
  bw, bx, by, bz = b[0], b[1], b[2], b[3]
  return torch.tensor(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dtype=a.dtype,
  )


def quat_rotate_vec(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate a 3-vector by a wxyz quaternion."""
  qv = torch.tensor([0.0, v[0], v[1], v[2]], dtype=q.dtype)
  r = quat_mul(quat_mul(q, qv), quat_conj(q))
  return r[1:4]


def compose(
  pos_a: torch.Tensor, quat_a: torch.Tensor, pos_b: torch.Tensor, quat_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compose transform A then B (B applied in A's local frame): A * B."""
  pos = pos_a + quat_rotate_vec(quat_a, pos_b)
  quat = quat_mul(quat_a, quat_b)
  return pos, quat


def invert(pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Invert a rigid transform (pos, quat)."""
  qinv = quat_conj(quat)
  pos_inv = -quat_rotate_vec(qinv, pos)
  return pos_inv, qinv


def clearance_for_hip_pose(
  hip_pos: torch.Tensor,
  hip_quat: torch.Tensor,
  wire_pos: torch.Tensor,
  wire_xmat: torch.Tensor,
  wire_names: list[str],
) -> tuple[float, str]:
  """Same clearance convention as D4.4: capsule-approx cylinder vs. wire capsules."""
  hip_z = quat_to_zaxis(hip_quat)
  hip_p0 = hip_pos - hip_z * FL_HIP_COLLISION_HALFLEN
  hip_p1 = hip_pos + hip_z * FL_HIP_COLLISION_HALFLEN

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
  return best_dist, best_seg


@dataclass
class D45Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 60
  ref_step: int = 44
  window_start: int = 44
  window_end: int = 53


def run(cfg: D45Config) -> list[dict]:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_env(d3_cfg, device)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity

  geom_names = list(entity.geom_names)
  hip_geom_idx = geom_names.index("FL_hip_collision")
  wire_geom_ids = resolve_wire_segment_geom_ids(env, ISOLATED_FL_WIRE_CFG.geom_names())
  wire_names = list(ISOLATED_FL_WIRE_CFG.geom_names())
  thigh_i = list(entity.joint_names).index("FL_thigh_joint")
  calf_i = list(entity.joint_names).index("FL_calf_joint")
  hip_i = list(entity.joint_names).index("FL_hip_joint")

  env.reset(seed=cfg.seed)

  # Per-step recorded state, captured for the FULL window including the
  # step-44 reference (needed for the fixed-base/translation/rotation
  # counterfactuals), not just the 49-53 reporting window.
  recorded: dict[int, dict] = {}

  for step in range(cfg.max_steps):
    action = torch.zeros(env.action_manager.action.shape, device=device)
    env.step(action)

    if not (cfg.window_start <= step <= cfg.window_end):
      continue

    hip_pose = entity.data.geom_pose_w[0, hip_geom_idx].clone()
    base_pos = entity.data.root_link_pos_w[0].clone()
    base_quat = entity.data.root_link_quat_w[0].clone()
    wire_pos = env.sim.data.geom_xpos[0, wire_geom_ids].clone()
    wire_xmat = env.sim.data.geom_xmat[0, wire_geom_ids].clone().reshape(-1, 3, 3)
    qpos = entity.data.joint_pos[0].clone()

    recorded[step] = {
      "hip_geom_pos": hip_pose[0:3],
      "hip_geom_quat": hip_pose[3:7],
      "base_pos": base_pos,
      "base_quat": base_quat,
      "wire_pos": wire_pos,
      "wire_xmat": wire_xmat,
      "hip_pos": float(qpos[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
    }

  env.close()

  assert cfg.ref_step in recorded, (
    f"Reference step {cfg.ref_step} not captured; widen --window-start."
  )
  ref = recorded[cfg.ref_step]
  ref_base_pos, ref_base_quat = ref["base_pos"], ref["base_quat"]

  rows = []
  for step in sorted(recorded.keys()):
    r = recorded[step]
    base_pos, base_quat = r["base_pos"], r["base_quat"]
    hip_geom_pos, hip_geom_quat = r["hip_geom_pos"], r["hip_geom_quat"]
    wire_pos, wire_xmat = r["wire_pos"], r["wire_xmat"]

    # Recover the base-relative "local chain" transform L(s) = B(s)^-1 * G(s).
    inv_pos, inv_quat = invert(base_pos, base_quat)
    local_pos, local_quat = compose(inv_pos, inv_quat, hip_geom_pos, hip_geom_quat)

    variants = {
      "actual": (base_pos, base_quat),
      "fixed_base": (ref_base_pos, ref_base_quat),
      "translation_only": (ref_base_pos, base_quat),
      "rotation_only": (base_pos, ref_base_quat),
    }

    result = {"seed": cfg.seed, "step": step}
    for name, (b_pos, b_quat) in variants.items():
      g_pos, g_quat = compose(b_pos, b_quat, local_pos, local_quat)
      clearance, seg = clearance_for_hip_pose(g_pos, g_quat, wire_pos, wire_xmat, wire_names)
      result[f"{name}_clearance"] = round(clearance, 5)
      result[f"{name}_nearest_seg"] = seg

    # Sanity check: "actual" variant must reproduce D4.4's reconstruction
    # (same geometry model) to within floating-point tolerance.
    actual_vs_direct = clearance_for_hip_pose(
      hip_geom_pos, hip_geom_quat, wire_pos, wire_xmat, wire_names
    )
    result["actual_matches_direct_geom_pose_w"] = (
      abs(result["actual_clearance"] - round(actual_vs_direct[0], 5)) < 1e-4
    )

    result["diff_fixed_base_vs_actual"] = round(
      result["fixed_base_clearance"] - result["actual_clearance"], 5
    )
    result["diff_translation_only_vs_actual"] = round(
      result["translation_only_clearance"] - result["actual_clearance"], 5
    )
    result["diff_rotation_only_vs_actual"] = round(
      result["rotation_only_clearance"] - result["actual_clearance"], 5
    )

    result["base_pos"] = [round(float(x), 5) for x in base_pos]
    result["base_quat_wxyz"] = [round(float(x), 5) for x in base_quat]
    result["hip_joint_pos"] = round(r["hip_pos"], 5)
    result["thigh_joint_pos"] = round(r["thigh_pos"], 5)
    result["calf_joint_pos"] = round(r["calf_pos"], 5)

    rows.append(result)

  return rows


def main() -> None:
  out_dir = Path("logs/d4_5_base_pose_counterfactual")
  out_dir.mkdir(parents=True, exist_ok=True)

  all_rows = []
  for seed in (0, 1):
    cfg = D45Config(seed=seed)
    rows = run(cfg)
    all_rows.extend(rows)
    print(f"\n==== seed {seed} (reference step = {cfg.ref_step}) ====")
    for r in rows:
      print(
        f"step={r['step']:3d}  actual={r['actual_clearance']:+.5f} "
        f"(sanity_ok={r['actual_matches_direct_geom_pose_w']})  "
        f"fixed_base={r['fixed_base_clearance']:+.5f} "
        f"(d={r['diff_fixed_base_vs_actual']:+.5f})  "
        f"transl_only={r['translation_only_clearance']:+.5f} "
        f"(d={r['diff_translation_only_vs_actual']:+.5f})  "
        f"rot_only={r['rotation_only_clearance']:+.5f} "
        f"(d={r['diff_rotation_only_vs_actual']:+.5f})"
      )

  csv_path = out_dir / "counterfactual_clearance.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)
  print(f"\n[D4.5] CSV written: {csv_path}")


if __name__ == "__main__":
  main()
