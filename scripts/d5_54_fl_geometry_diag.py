#!/usr/bin/env python
"""D5.54 -- FL / wire geometry diagnostic for the D5.53 baseline and guarded runs.

Read-only diagnostic. Re-runs the four D5.53 conditions (baseline and
guard_enabled, seeds 0 and 11) through the D5.53 setup unchanged and logs, only
for the critical window, the geometry D5.53 did not record: true collision-geom
distances to the wire (``mj_geomDistance`` on the live qpos), FL foot position
relative to the interpolated wire centerline, FR foot position, and the robot
geoms touching the wire each step. It checks itself against the D5.53 traces.

Output: one compact CSV, ``<out_dir>/comparison_summary.csv``.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d5_53_touchdown_guard import D553Config, _setup  # noqa: E402

from mjlab.tasks.recovery.mdp import wire_geometry  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

VARIANTS = ("baseline", "guard_enabled")
SEEDS = (0, 11)
FL_GEOMS = (
  "FL_hip_collision",
  "FL_thigh_collision",
  "FL_calf1_collision",
  "FL_calf2_collision",
  "FL_foot_collision",
)
BODY_GEOMS = ("base1_collision", "base2_collision", "base3_collision")


@dataclass
class D554Config:
  start_step: int = 38
  max_steps: int = 112
  device: str = "cpu"
  out_dir: str = "logs/d5_54"
  ref_dir: str = "logs/d5_53"


def _short(name: str) -> str:
  return name.removeprefix("robot/").removesuffix("_collision")


class Geometry:
  def __init__(self, env):
    self.env = env
    self.m = env.sim.mj_model
    self.d = mujoco.MjData(self.m)
    gid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    self.fl = {g: gid(f"robot/{g}") for g in FL_GEOMS}
    self.body = [gid(f"robot/{g}") for g in BODY_GEOMS]
    self.fr_foot_site = list(env.scene["robot"].site_names).index("FR")
    self.wire = [
      gid(f"wire_seg_{i}_geom") for i in range(100) if gid(f"wire_seg_{i}_geom") >= 0
    ]
    assert all(v >= 0 for v in self.fl.values()) and all(b >= 0 for b in self.body)
    self.fromto = np.zeros(6)

  def refresh(self) -> None:
    self.d.qpos[:] = self.env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_kinematics(self.m, self.d)

  def min_dist(self, g: int) -> tuple[float, int]:
    best, seg = 1e9, -1
    for i, w in enumerate(self.wire):
      dist = mujoco.mj_geomDistance(self.m, self.d, g, w, 1.0, self.fromto)
      if dist < best:
        best, seg = dist, i
    return best, seg


def run(cfg: D554Config, variant: str, seed: int) -> list[dict[str, Any]]:
  d553 = D553Config(device=cfg.device)
  env, act, guard, probe, make_row = _setup(d553, variant, seed)
  env.reset(seed=seed)
  geo = Geometry(env)
  robot = env.scene["robot"]
  foot_site = list(robot.site_names).index("FL")
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  rows: list[dict[str, Any]] = []
  wire_r = float(act.cfg.wire_radius)
  for step in range(cfg.max_steps):
    probe.begin_step(step)
    env.step(zero)
    r = make_row(step)
    if step < cfg.start_step:
      continue
    geo.refresh()
    seg_pos = wire_geometry.wire_segment_world_pos(env, act._wire_geom_ids)
    foot = robot.data.site_pos_w[0, foot_site]
    fr = robot.data.site_pos_w[0, geo.fr_foot_site]
    base = robot.data.root_link_pos_w[0]
    wx, wz = wire_geometry.local_wire_xz_at_y(seg_pos, foot[1:2])
    calf2 = robot.data.geom_pos_w[0, act._calf_geom_idx]
    cx, cz = wire_geometry.local_wire_xz_at_y(seg_pos, calf2[1:2])
    dists = {g: geo.min_dist(i) for g, i in geo.fl.items()}
    near_g = min(dists, key=lambda k: dists[k][0])
    body_d = min(geo.min_dist(b)[0] for b in geo.body)
    quat = robot.data.root_link_quat_w[0:1]
    fr_base = quat_apply_inverse(quat, (fr - base).unsqueeze(0))[0]
    touching = sorted(
      {
        _short(g)
        for s in probe.rows
        if s["step"] == step
        for g in s["robot_geoms"].split("|")
        if g
      }
    )
    rows.append(
      {
        "seed": seed,
        "variant": variant,
        "step": step,
        "phase": r["phase"],
        "guard": r["guard_state"],
        "base_x": base[0].item(),
        "base_y": base[1].item(),
        "base_z": base[2].item(),
        "pitch_deg": r["base_pitch_deg"],
        "roll_deg": r["base_roll_deg"],
        "fl_thigh_pos": r["FL_thigh_pos_rad"],
        "fl_calf_pos": r["FL_calf_pos_rad"],
        "fl_thigh_vel": r["FL_thigh_vel_radps"],
        "fl_calf_vel": r["FL_calf_vel_radps"],
        "fl_foot_x": foot[0].item(),
        "fl_foot_y": foot[1].item(),
        "fl_foot_z": foot[2].item(),
        "fl_foot_dx_wire": r["fl_foot_forward_dist_m"],
        "fl_foot_dz_wire": foot[2].item() - wz.item(),
        "fl_calf2_dx_wire": r["fl_calf_forward_dist_m"],
        "fl_calf2_dz_wire": calf2[2].item() - cz.item(),
        "wire_x_at_foot": wx.item(),
        "wire_z_at_foot": wz.item(),
        "wire_radius": wire_r,
        "d_foot_wire": dists["FL_foot_collision"][0],
        "d_calf2_wire": dists["FL_calf2_collision"][0],
        "d_calf1_wire": dists["FL_calf1_collision"][0],
        "d_thigh_wire": dists["FL_thigh_collision"][0],
        "d_hip_wire": dists["FL_hip_collision"][0],
        "d_trunk_wire": body_d,
        "nearest_fl_geom": _short(near_g),
        "nearest_fl_seg": dists[near_g][1],
        "fr_foot_x": fr[0].item(),
        "fr_foot_y": fr[1].item(),
        "fr_foot_z": fr[2].item(),
        "fr_foot_x_base": fr_base[0].item(),
        "f_fl": abs(r["fl_ground_force_z_n"]),
        "f_fr": abs(r["fr_ground_force_z_n"]),
        "f_rl": abs(r["rl_ground_force_z_n"]),
        "f_rr": abs(r["rr_ground_force_z_n"]),
        "wire_touch": "|".join(touching),
        # for self-check vs D5.53 (dropped from the CSV below)
        "_ref_base_z": r["base_z_m"],
        "_ref_calf": r["FL_calf_pos_rad"],
        "_ref_hybrid": r["hybrid_mode_active"],
        "_ref_exit": r["newly_recovered"],
      }
    )
  env.close()
  return rows


def main() -> None:
  import tyro

  cfg = tyro.cli(D554Config)
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  all_rows: list[dict[str, Any]] = []
  checks = []
  for seed in SEEDS:
    for variant in VARIANTS:
      rows = run(cfg, variant, seed)
      ref = {
        int(x["step"]): x
        for x in csv.DictReader(open(f"{cfg.ref_dir}/trace_{variant}_seed{seed}.csv"))
      }
      worst = 0.0
      for r in rows:
        x = ref[r["step"]]
        worst = max(
          worst,
          abs(float(x["base_z_m"]) - r["_ref_base_z"]),
          abs(float(x["FL_calf_pos_rad"]) - r["_ref_calf"]),
          abs(abs(float(x["fl_ground_force_z_n"])) - r["f_fl"]),
          abs(abs(float(x["fr_ground_force_z_n"])) - r["f_fr"]),
        )
        assert x["phase"] == r["phase"] and x["guard_state"] == r["guard"]
      checks.append((seed, variant, len(rows), worst))
      print(
        f"[D5.54] seed{seed} {variant}: {len(rows)} rows, max diff vs D5.53 = {worst:.3e}"
      )
      all_rows += rows
  fields = [k for k in all_rows[0] if not k.startswith("_")]
  with (out / "comparison_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in all_rows:
      w.writerow(
        {
          k: (round(v, 5) if isinstance(v, float) else v)
          for k, v in r.items()
          if k in fields
        }
      )
  ok = all(c[3] == 0.0 for c in checks)
  print(f"[D5.54] {'PASS' if ok else 'FAIL'} reproduces D5.53 traces exactly")
  if not ok:
    sys.exit(2)


if __name__ == "__main__":
  main()
