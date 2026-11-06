#!/usr/bin/env python
"""D5.56 -- Diagnose the walking-policy stall after D5.55 bounded retraction.

Read-only diagnostic: no controller behavior is changed. Wrappers only *record*:
the 48-d observation (return value of ``act._build_walk_obs``), the raw 12-d
policy action (``act.last_a_walk_raw``), the processed joint targets, the applied
position-actuator controls, FL foot height, FL ground-contact state, and (stall
run only) the per-substep robot-wire force on the robot.

  reference  wire-off, no-hybrid control (D5.52 build): ordinary unperturbed
             trot, same checkpoint and seed. Windows start at FL touchdowns
             44, 66, 89 (steady gait).
  stall      D5.55 ``bounded_retraction`` (wire on); plateau = steps 84-108,
             after the post-release transient (steps ~72-83).

Everything stays in memory; the only output is ``logs/d5_56/summary.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import d5_52_wire_removed_forced_entry as d552  # noqa: E402
from d5_53_touchdown_guard import D553Config, _setup  # noqa: E402
from d5_55_bounded_retraction import BoundedRetraction  # noqa: E402

JOINTS = (
  "FL_hip",
  "FL_thigh",
  "FL_calf",
  "FR_hip",
  "FR_thigh",
  "FR_calf",
  "RL_hip",
  "RL_thigh",
  "RL_calf",
  "RR_hip",
  "RR_thigh",
  "RR_calf",
)
REF_STARTS = (44, 66, 89)
PLATEAU = slice(84, 109)


class Capture:
  def __init__(self, env, act):
    self.env, self.act = env, act
    self.obs: list[np.ndarray] = []
    orig = act._build_walk_obs

    def hook():
      o = orig()
      self.obs.append(o[0].detach().cpu().numpy().copy())
      return o

    act._build_walk_obs = hook
    self.raw: list[np.ndarray] = []
    self.target: list[np.ndarray] = []
    self.ctrl: list[np.ndarray] = []
    self.foot_z: list[float] = []
    self.base_x: list[float] = []
    self.scale = act._joint_pos_action._scale[0].detach().cpu().numpy().copy()
    mjm = env.sim.mj_model
    self.pos_ids = [
      next(i for i in range(mjm.nu) if mjm.actuator(i).name == f"robot/{j}_joint")
      for j in JOINTS
    ]
    self.fl_site = list(env.scene["robot"].site_names).index("FL")

  def record(self) -> None:
    act, env = self.act, self.env
    self.raw.append(act.last_a_walk_raw[0].detach().cpu().numpy().copy())
    self.target.append(
      act._joint_pos_action._processed_actions[0].detach().cpu().numpy().copy()
    )
    self.ctrl.append(env.sim.data.ctrl[0, self.pos_ids].detach().cpu().numpy().copy())
    robot = env.scene["robot"]
    self.foot_z.append(float(robot.data.site_pos_w[0, self.fl_site, 2]))
    self.base_x.append(float(robot.data.root_link_pos_w[0, 0]))


def run_reference(seed: int, n_steps: int):
  cfg = d552.D552Config(seed=seed, max_steps=n_steps)
  env, act, _forced, _probe, _rec = d552._setup(cfg, "wire_off_control")
  cap = Capture(env, act)
  env.reset(seed=seed)
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  for _ in range(n_steps):
    env.step(zero)
    cap.record()
  pol = act._policy
  env.close()
  return cap, pol


def run_stall(seed: int, n_steps: int):
  env, act, _guard, probe, _mk = _setup(D553Config(), "guard_enabled", seed)
  BoundedRetraction(env, act)
  cap = Capture(env, act)
  env.reset(seed=seed)
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  for step in range(n_steps):
    probe.begin_step(step)
    env.step(zero)
    cap.record()
  env.close()
  return cap, probe.rows


def p2p(a: np.ndarray) -> np.ndarray:
  return a.max(0) - a.min(0)


def main() -> None:
  rows: list[dict[str, Any]] = []

  def add(seed, metric, stall, ref, note=""):
    def fmt(v):
      return v if isinstance(v, str) else round(float(v), 3)

    rows.append(
      {
        "seed": seed,
        "metric": metric,
        "stall_plateau_84_108": fmt(stall),
        "reference_normal_walking": fmt(ref),
        "note": note,
      }
    )

  for seed in (0, 11):
    ref, pol = run_reference(seed, 140)
    stall, sub = run_stall(seed, 112)
    norm = pol.obs_normalizer
    mean = norm._mean.detach().cpu().numpy().reshape(-1)
    std = norm._std.detach().cpu().numpy().reshape(-1)

    def z(o, mean=mean, std=std):
      return (o - mean) / (std + 1e-2)

    so, ro = np.array(stall.obs), np.array(ref.obs)
    sraw, rraw = np.array(stall.raw), np.array(ref.raw)
    zs, zr = z(so), z(ro)
    sc = stall.scale

    ref_nn = [
      np.min(
        np.linalg.norm(zr[[u for u in range(140) if abs(u - t) > 6]] - zr[t], axis=1)
      )
      for t in range(44, 140)
    ]
    st_nn = [np.min(np.linalg.norm(zr - zs[t], axis=1)) for t in range(84, 109)]
    add(
      seed,
      "obs_nearest_neighbour_distance_normalized_median",
      np.median(st_nn),
      np.median(ref_nn),
      "stall->ref (140 ref steps incl. start from rest) vs ref->ref excluding +-6 steps",
    )
    add(
      seed,
      "obs_max_abs_z_over_48_dims",
      np.abs(zs[PLATEAU]).max(),
      np.abs(zr[44:140]).max(),
      "policy normalizer units",
    )
    add(
      seed,
      "imu_lin_vel_x_mps_mean",
      so[PLATEAU, 0].mean(),
      ro[44:140, 0].mean(),
      "command vx = 0.3 in both",
    )
    add(
      seed,
      "raw_action_p2p_mean_over_12_joints",
      p2p(sraw[PLATEAU]).mean(),
      np.mean([p2p(rraw[e : e + 25]).mean() for e in REF_STARTS]),
      "25-step windows",
    )
    ks = (14, 15, 16)
    for name, j in (("calf", 2), ("thigh", 1)):
      add(
        seed,
        f"FL_raw_{name}_mean",
        sraw[PLATEAU][:, j].mean(),
        np.mean([rraw[e + k][j] for e in REF_STARTS for k in ks]),
        "ref = swing-onset steps (k=14..16 after touchdown)",
      )
    add(
      seed,
      "FL_foot_z_max_m",
      max(stall.foot_z[PLATEAU]),
      max(ref.foot_z[40:140]),
      "resting z=0.022 (sphere radius); ref = swing apex",
    )

    def err(raw, obs, t, sc=sc):
      return sc * raw[t] - obs[t][9:21]

    add(
      seed,
      "FL_thigh_target_minus_actual_rad_mean",
      np.mean([err(sraw, so, t)[1] for t in range(84, 109)]),
      np.mean([err(rraw, ro, t)[1] for e in REF_STARTS for t in range(e, e + 13)]),
      "ref = FL stance",
    )
    add(
      seed,
      "FL_thigh_joint_pos_rel_default_rad",
      so[PLATEAU, 10].mean(),
      np.mean([ro[e : e + 13, 10].mean() for e in REF_STARTS]),
      "ref = FL stance",
    )
    add(
      seed,
      "applied_ctrl_minus_policy_target_max_all_joints",
      np.abs(np.array(stall.ctrl)[PLATEAU] - np.array(stall.target)[PLATEAU]).max(),
      np.abs(np.array(ref.ctrl)[44:140] - np.array(ref.target)[44:140]).max(),
      "12 position actuators",
    )
    in_win = [s for s in sub if 84 <= s["step"] < 109]
    steps = sorted({s["step"] for s in in_win})

    def per_step(key, in_win=in_win, steps=steps):
      return [np.mean([s[key] for s in in_win if s["step"] == t]) for t in steps]

    add(
      seed,
      "wire_force_on_robot_x_mean_N",
      np.mean(per_step("robot_wire_force_on_robot_x_n")),
      "no wire",
      "substep probe; command is +x",
    )
    add(
      seed,
      "wire_force_on_robot_z_mean_N",
      np.mean(per_step("robot_wire_force_on_robot_z_n")),
      "no wire",
      "substep probe",
    )
    add(
      seed,
      "wire_contact_fraction_of_steps",
      np.mean([n > 0 for n in per_step("robot_wire_contact_count")]),
      "no wire",
      "robot-wire contact in the plateau steps (FL_calf2 only)",
    )
    add(
      seed,
      "base_x_change_84_to_108_m",
      stall.base_x[108] - stall.base_x[84],
      ref.base_x[108] - ref.base_x[84],
      "negative = drifting backward; command is forward",
    )

  path = Path("logs/d5_56/summary.csv")
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print(f"[D5.56] wrote {path} ({len(rows)} rows)")
  for r in rows:
    print(
      r["seed"],
      r["metric"],
      r["stall_plateau_84_108"],
      "|",
      r["reference_normal_walking"],
    )


if __name__ == "__main__":
  main()
