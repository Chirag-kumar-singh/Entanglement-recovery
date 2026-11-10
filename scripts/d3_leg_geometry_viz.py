"""D3 diagnostic: FL leg geometry visualization in the native MuJoCo
viewer. Steps the FL thigh/calf/hip through four representative
configurations identified by `scripts/d3_thigh_calf_2d_map.py`
(A_early -> B_near_peak -> C_post_peak -> D_predicted_best), holding
everything else FROZEN (base pose, all other joints) so the geometric
change is isolated and easy to see.

PURELY VISUAL/DIAGNOSTIC. Does not modify the recovery controller, FSM,
wire geometry, walking policy, or any default config. Uses
`mjlab.viewer.NativeMujocoViewer`, the SAME native-viewer entry point
already used by `scripts/d2_trajectory_check.py --debug` and
`scripts/d3_wire_recovery_experiment.py --debug`.

Reads representative points from
logs/d3_geometry_diagnostic/representative_points.csv (produced by
scripts/d3_thigh_calf_2d_map.py -- run that first if this file is
missing).

Usage::

    uv run python scripts/d3_leg_geometry_viz.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.viewer import NativeMujocoViewer

REP_POINTS_CSV = Path("logs/d3_geometry_diagnostic/representative_points.csv")
HOLD_SECONDS = 3.0  # how long to pause at each configuration


def load_points():
  if not REP_POINTS_CSV.exists():
    raise FileNotFoundError(
      f"{REP_POINTS_CSV} not found. Run scripts/d3_thigh_calf_2d_map.py first "
      "to generate it."
    )
  points = []
  with open(REP_POINTS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
      points.append(
        (row["label"], float(row["thigh"]), float(row["calf"]), float(row["hip"]))
      )
  return points


def main():
  device = "cpu"
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
  )
  cfg.sim.device = device
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  act = env.action_manager._terms["joint_pos"]
  env.reset()

  entity = act._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index("FL_hip_joint")
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")

  points = load_points()
  print("Loaded representative points:")
  for label, th, cf, hp in points:
    print(f"  {label}: thigh={th:.4f}, calf={cf:.4f}, hip={hp:.4f}")

  state = {"point_idx": 0, "t_entered": time.time(), "qpos_frozen": None}

  def viewer_policy(_obs):
    sim = env.sim
    if state["qpos_frozen"] is None:
      # Freeze the base pose + all other joints at the CURRENT
      # post-reset state, one time only, so only FL hip/thigh/calf move
      # across the sequence (everything else stays geometrically fixed
      # for a clean visual comparison).
      state["qpos_frozen"] = sim.data.qpos[0].clone()

    label, th, cf, hp = points[state["point_idx"]]
    qpos = state["qpos_frozen"].clone()
    # Overwrite ONLY the FL hip/thigh/calf entity-local joint values in
    # the full qpos vector. Entity joint i's qpos address within this
    # single-entity scene corresponds 1:1 with entity.data.joint_pos
    # ordering; resolve via the entity's own joint_pos indexing by
    # writing through set_joint_position (if available) instead of
    # poking qpos addresses directly, to stay consistent with the
    # entity's own bookkeeping.
    sim.data.qpos[0] = qpos
    entity.write_joint_state_to_sim(
      position=torch.tensor([[hp, th, cf]], device=device),
      velocity=torch.zeros(1, 3, device=device),
      joint_ids=[hip_i, thigh_i, calf_i],
    )
    sim.forward()

    elapsed = time.time() - state["t_entered"]
    if elapsed > HOLD_SECONDS:
      state["point_idx"] = (state["point_idx"] + 1) % len(points)
      state["t_entered"] = time.time()
      new_label = points[state["point_idx"]][0]
      print(f"[viz] -> now showing {new_label}")

    return torch.zeros(1, act.action_dim, device=device)

  print(
    "\n[INFO] Cycling through A_early -> B_near_peak -> C_post_peak -> "
    "D_predicted_best, holding each for "
    f"{HOLD_SECONDS:.0f}s. Base pose and all other joints are FROZEN at "
    "their post-reset values -- only FL hip/thigh/calf move. Close the "
    "viewer window to exit.\n"
  )
  NativeMujocoViewer(env, viewer_policy).run()
  env.close()


if __name__ == "__main__":
  main()
