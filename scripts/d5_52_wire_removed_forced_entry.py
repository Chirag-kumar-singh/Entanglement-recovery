#!/usr/bin/env python
"""D5.52 -- E1 wire-removed forced-entry hybrid replay (D5.51 spec).

Diagnostic only. No production code is modified: ``actions.py`` and
``env_cfg_residual.py`` are imported unchanged. Interventions live here.

Conditions (all seed 0, same checkpoint, same env build path as D5.16/D5.49):
  baseline           -- wire on, natural contact trigger, unchanged hybrid.
  wire_off_control   -- all robot-wire collisions excluded, no forced entry.
  wire_off_forced    -- all robot-wire collisions excluded, the EXISTING FSM
                        entry is forced at policy step 44 (the baseline entry
                        step); hybrid actuator path and commands unchanged.

Wire removal: ``MjSpec.add_exclude`` body-pair excludes between every
``robot/*`` body and every ``wire_seg_*`` body, composed on top of the
existing wire spec_fn (same mechanism as scripts/d4_8_hip_wire_contact_
ablation.py, scope widened on purpose to all robot bodies).

Forced entry: for the single ``_update_fsm`` call of policy step 44, the
``wire_contact`` sensor seen by the FSM is replaced by a proxy whose
``found`` is 1. Everything else (FSM transitions, thresholds, hybrid
handoff, commands) runs the original code path.

Row convention: row ``step=k`` is the state after ``env.step()`` number k
(same as D5.16/D5.49). The FSM decision for step k uses the sensor state
after step k-1, so baseline contact at row 43 -> RECOVERY row 44.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_16_trajectory_audit import build_env  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.recovery.mdp.actions import (  # noqa: E402
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
  WALK_JOINT_ORDER,
  ResidualRecoveryAction,
)
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

PHASE_NAMES = {
  PHASE_NORMAL_WALK: "NORMAL_WALK",
  PHASE_RECOVERY: "RECOVERY",
  PHASE_POST_RECOVERY: "POST_RECOVERY",
}
CONDITIONS = ("baseline", "wire_off_control", "wire_off_forced")
LEGS = ("FL", "FR", "RL", "RR")
FORCED_ENTRY_STEP = 44
CALF_LIMIT_EPS_RAD = 0.005


@dataclass
class D552Config:
  condition: Literal["all", "baseline", "wire_off_control", "wire_off_forced"] = "all"
  seed: int = 0
  device: str = "cpu"
  max_steps: int = 250
  max_recovery_steps_override: int = 48
  out_dir: str = "logs/d5_52"
  debug: bool = False
  """Open the D5.15-pattern native viewer for a single condition (the
  viewer callback never calls env.step() itself). CSV output is written to
  ``<out_dir>/viewer_<condition>/trace.csv`` for comparison with headless."""


def _quat_to_rpy_deg(q: torch.Tensor) -> tuple[float, float, float]:
  w, x, y, z = (float(v) for v in q)
  roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def build_condition_env(cfg: D552Config, wire_off: bool) -> ManagerBasedRlEnv:
  """D5.16 build path; for wire-off, compose robot<->wire body excludes."""
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=cfg.device, use_isolated_wire=True)
  if not wire_off:
    return build_env(d3_cfg, cfg.device, cfg.max_recovery_steps_override)

  import mjlab.envs as envs_mod

  original_cls = envs_mod.ManagerBasedRlEnv

  class _ExcludingEnv(original_cls):
    def __init__(self, cfg, device):  # noqa: ANN001
      # The frozen walking-policy loader also builds an env through this
      # class; only the recovery env (with the wire sensor) is modified.
      if not any(s.name == "wire_contact" for s in cfg.scene.sensors):
        super().__init__(cfg=cfg, device=device)
        return
      existing = cfg.scene.spec_fn

      def composed_spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
        if existing is not None:
          existing(spec, num_envs)
        wire = [b.name for b in spec.bodies if b.name.startswith("wire_seg_")]
        robot = [b.name for b in spec.bodies if b.name.startswith("robot/")]
        assert wire and robot, "wire/robot bodies missing after spec_fn"
        for rb in robot:
          for wb in wire:
            spec.add_exclude(bodyname1=rb, bodyname2=wb)
        print(f"[D5.52] added {len(robot) * len(wire)} robot<->wire excludes")

      cfg.scene.spec_fn = composed_spec_fn
      super().__init__(cfg=cfg, device=device)

  envs_mod.ManagerBasedRlEnv = _ExcludingEnv  # ty: ignore[invalid-assignment]
  try:
    return build_env(d3_cfg, cfg.device, cfg.max_recovery_steps_override)
  finally:
    envs_mod.ManagerBasedRlEnv = original_cls


def install_forced_entry(env: ManagerBasedRlEnv, act: ResidualRecoveryAction) -> dict:
  """Force the existing FSM's contact input to True on step 44 only."""
  state = {"fsm_calls": 0, "forced_step": -1}
  original_update_fsm = act._update_fsm
  name = act.cfg.wire_contact_sensor_name
  sensors = env.scene.sensors

  def forced_update_fsm() -> None:
    step = state["fsm_calls"]  # policy step index of this process_actions
    state["fsm_calls"] += 1
    if step != FORCED_ENTRY_STEP:
      original_update_fsm()
      return
    real = sensors[name]
    found = torch.ones_like(real.data.found)
    sensors[name] = cast(Any, SimpleNamespace(data=SimpleNamespace(found=found)))
    try:
      original_update_fsm()
    finally:
      sensors[name] = real
    state["forced_step"] = step

  act._update_fsm = forced_update_fsm  # ty: ignore[invalid-assignment]
  return state


class SubstepWireProbe:
  """Wraps sim.step to log raw robot<->wire contacts after each substep."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.mjm = env.sim.mj_model
    self.robot_bodies = {
      i for i in range(self.mjm.nbody) if self.mjm.body(i).name.startswith("robot/")
    }
    self.wire_bodies = {
      i for i in range(self.mjm.nbody) if self.mjm.body(i).name.startswith("wire_seg_")
    }
    self.policy_step = -1
    self.substep = 0
    self.rows: list[dict[str, Any]] = []
    original = env.sim.step

    def wrapped() -> None:
      original()
      self._record()

    env.sim.step = wrapped  # type: ignore[method-assign]

  def begin_step(self, step: int) -> None:
    self.policy_step = step
    self.substep = 0

  def _record(self) -> None:
    # Read warp contact/efc arrays directly (mjwarp.get_data_into returns
    # invalid contact geom ids in this mujoco_warp version). Pyramidal cone
    # decode as in MuJoCo's mju_decodePyramid: normal = sum of edge forces,
    # tangent_j = (f[2j] - f[2j+1]) * mu_j. Frame rows are (n, t1, t2); the
    # contact normal points geom1 -> geom2, so this is the force on geom2.
    simd = self.env.sim.data
    con = simd.contact
    n = int(simd.nacon[0])
    geom = con.geom[:n].tolist()
    efc_force = simd.efc.force[0]
    count = 0
    total = np.zeros(3)
    pairs = []
    for ci in range(n):
      if int(con.worldid[ci]) != 0:
        continue
      g1, g2 = geom[ci]
      b1 = int(self.mjm.geom_bodyid[g1])
      b2 = int(self.mjm.geom_bodyid[g2])
      robot_is_geom2 = b2 in self.robot_bodies and b1 in self.wire_bodies
      robot_is_geom1 = b1 in self.robot_bodies and b2 in self.wire_bodies
      if not (robot_is_geom1 or robot_is_geom2):
        continue
      count += 1
      dim = int(con.dim[ci])
      adr = [a for a in con.efc_address[ci].tolist() if a >= 0]
      f = efc_force[adr].double().numpy() if adr else np.zeros(1)
      mu = con.friction[ci].double().numpy()
      local = np.zeros(3)
      if dim == 1:
        local[0] = f[0]
      else:
        local[0] = f.sum()
        for j in range(min(2, dim - 1)):
          local[1 + j] = (f[2 * j] - f[2 * j + 1]) * mu[j]
      frame = con.frame[ci].double().numpy().reshape(3, 3)
      f_world = frame.T @ local
      total += f_world if robot_is_geom2 else -f_world
      pairs.append(self.mjm.geom(g2 if robot_is_geom2 else g1).name)
    self.rows.append(
      {
        "step": self.policy_step,
        "substep": self.substep,
        "robot_wire_contact_count": count,
        "robot_wire_force_on_robot_x_n": float(total[0]),
        "robot_wire_force_on_robot_y_n": float(total[1]),
        "robot_wire_force_on_robot_z_n": float(total[2]),
        "robot_wire_force_mag_n": float(np.linalg.norm(total)),
        "robot_geoms": "|".join(sorted(set(pairs))),
      }
    )
    self.substep += 1


class Recorder:
  """Per-policy-step row construction (state after env.step())."""

  def __init__(self, env: ManagerBasedRlEnv, act: ResidualRecoveryAction):
    self.env = env
    self.act = act
    self.robot = env.scene["robot"]
    self.joint_names = list(self.robot.joint_names)
    assert tuple(self.joint_names) == WALK_JOINT_ORDER
    mjm = env.sim.mj_model
    self.act_names = [mjm.actuator(i).name for i in range(mjm.nu)]
    self.dof_adr = [
      int(
        mjm.jnt_dofadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{j}")]
      )
      for j in self.joint_names
    ]
    self.calf_i = self.joint_names.index("FL_calf_joint")
    self.foot_site = list(self.robot.site_names).index("FL")

  def row(self, step: int, condition: str, forced_step: int) -> dict[str, Any]:
    env, act, data = self.env, self.act, self.robot.data
    simd = env.sim.data
    r: dict[str, Any] = {"condition": condition, "step": step}
    phase = int(act.phase[0])
    r["phase"] = PHASE_NAMES[phase]
    r["recovery_elapsed"] = int(act.recovery_elapsed[0])
    r["forced_entry_step"] = forced_step
    r["newly_entangled"] = bool(act.newly_entangled[0])
    r["newly_recovered"] = bool(act.newly_recovered[0])
    r["hybrid_mode_active"] = not bool(act.hybrid_thigh_pos_gain_active[0])

    pos = data.root_link_pos_w[0]
    quat = data.root_link_quat_w[0]
    for i, a in enumerate("xyz"):
      r[f"base_{a}_m"] = float(pos[i])
    for i, a in enumerate("wxyz"):
      r[f"base_quat_{a}"] = float(quat[i])
    roll, pitch, yaw = _quat_to_rpy_deg(quat)
    r["base_roll_deg"], r["base_pitch_deg"], r["base_yaw_deg"] = roll, pitch, yaw
    for i, a in enumerate("xyz"):
      r[f"base_lin_vel_b_{a}_mps"] = float(data.root_link_lin_vel_b[0, i])
      r[f"base_lin_vel_w_{a}_mps"] = float(data.root_link_lin_vel_w[0, i])
      r[f"base_ang_vel_b_{a}_rps"] = float(data.root_link_ang_vel_b[0, i])
      r[f"base_ang_vel_w_{a}_rps"] = float(data.root_link_ang_vel_w[0, i])
      r[f"projected_gravity_b_{a}"] = float(data.projected_gravity_b[0, i])

    for j, name in enumerate(self.joint_names):
      short = name.removesuffix("_joint")
      r[f"{short}_pos_rad"] = float(data.joint_pos[0, j])
      r[f"{short}_vel_radps"] = float(data.joint_vel[0, j])
      r[f"{short}_qfrc_actuator_nm"] = float(simd.qfrc_actuator[0, self.dof_adr[j]])
      r[f"{short}_a_walk_raw"] = float(cast(torch.Tensor, act.last_a_walk_raw)[0, j])
      r[f"{short}_final_raw"] = float(cast(torch.Tensor, act.last_final_raw)[0, j])
    for i, name in enumerate(self.act_names):
      short = name.removeprefix("robot/")
      r[f"act_{short}_ctrl"] = float(simd.ctrl[0, i])
      r[f"act_{short}_force"] = float(simd.actuator_force[0, i])

    m = env.sim.model
    r["fl_thigh_pos_gain"] = float(m.actuator_gainprm[0, act._pos_thigh_ctrl_id, 0])
    r["fl_calf_pos_gain"] = float(m.actuator_gainprm[0, act._pos_calf_ctrl_id, 0])
    r["fl_thigh_aux_vel_kd"] = float(m.actuator_gainprm[0, act._aux_thigh_ctrl_id, 0])
    r["thigh_vel_cmd"] = float(act.hybrid_thigh_vel_ctrl[0])
    r["calf_torque_cmd"] = float(act.hybrid_calf_torque_ctrl[0])
    soft_lo = float(data.soft_joint_pos_limits[0, self.calf_i, 0])
    r["fl_calf_soft_limit_lo_rad"] = soft_lo
    r["fl_calf_soft_limit_margin_rad"] = float(data.joint_pos[0, self.calf_i]) - soft_lo

    for leg in LEGS:
      s = env.scene.sensors[f"{leg.lower()}_ground_contact_diag"]
      f = s.data.force[0].reshape(-1, 3).sum(dim=0)
      r[f"{leg.lower()}_ground_contact"] = bool((s.data.found[0] > 0).any())
      r[f"{leg.lower()}_ground_force_z_n"] = float(f[2])
      r[f"{leg.lower()}_ground_force_mag_n"] = float(torch.linalg.norm(f))

    ws = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wf = ws.data.force[0].reshape(-1, 3).sum(dim=0)
    r["wire_contact_sensor"] = bool((ws.data.found[0] > 0).any())
    r["wire_contact_sensor_force_mag_n"] = float(torch.linalg.norm(wf))

    foot_rel = (data.site_pos_w[0, self.foot_site] - pos).unsqueeze(0)
    foot_b = quat_apply_inverse(quat.unsqueeze(0), foot_rel)[0]
    for i, a in enumerate("xyz"):
      r[f"fl_foot_{a}_base_m"] = float(foot_b[i])
    r["fl_calf_forward_dist_m"] = float(act.calf_forward_dist[0])
    r["fl_calf_height_above_wire_m"] = float(act.calf_height_above_wire[0])
    r["fl_foot_forward_dist_m"] = float(act.foot_forward_dist[0])
    r["fl_foot_height_above_wire_m"] = float(act.foot_height_above_wire[0])
    r["crossing_achieved"] = bool(act.crossing_achieved[0])
    r["release_achieved"] = bool(act.release_achieved[0])
    r["recovery_failed_timeout"] = bool(act._recovery_failed_timeout[0])
    r["fell_over"] = bool(env.reset_terminated[0])
    return r


def _setup(cfg: D552Config, condition: str):
  env = build_condition_env(cfg, wire_off=condition != "baseline")
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  forced = install_forced_entry(env, act) if condition == "wire_off_forced" else None
  probe = SubstepWireProbe(env)
  rec = Recorder(env, act)
  return env, act, forced, probe, rec


def run_condition(cfg: D552Config, condition: str) -> dict[str, Any]:
  env, act, forced, probe, rec = _setup(cfg, condition)
  env.reset(seed=cfg.seed)
  rows, states = [], []
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  for step in range(cfg.max_steps):
    probe.begin_step(step)
    env.step(zero)
    fs = forced["forced_step"] if forced else -1
    row = rec.row(step, condition, fs)
    rows.append(row)
    states.append((env.sim.data.qpos[0].clone(), env.sim.data.qvel[0].clone()))
    if row["fell_over"]:
      print(f"[D5.52] {condition}: fell_over termination at step {step}; stopping")
      break
  env.close()
  return {"rows": rows, "substeps": probe.rows, "states": states}


def run_viewer(cfg: D552Config, condition: str) -> Path:
  from mjlab.viewer import NativeMujocoViewer

  env, act, forced, probe, rec = _setup(cfg, condition)
  env.reset(seed=cfg.seed)
  rows: list[dict[str, Any]] = []
  pending: dict[str, Any] = {"step": None}

  def viewer_policy(_obs):  # noqa: ANN001
    # D5.15 pattern: never call env.step() here; finalize the previous row.
    if pending["step"] is not None:
      fs = forced["forced_step"] if forced else -1
      rows.append(rec.row(pending["step"], condition, fs))
      if pending["step"] % 25 == 0:
        print(
          f"[D5.52 viewer] {condition} step={pending['step']} phase={rows[-1]['phase']}"
        )
      if pending["step"] + 1 >= cfg.max_steps or rows[-1]["fell_over"]:
        raise SystemExit(0)
    step = 0 if pending["step"] is None else pending["step"] + 1
    probe.begin_step(step)
    pending["step"] = step
    return torch.zeros(env.action_manager.action.shape, device=env.device)

  try:
    NativeMujocoViewer(cast(Any, env), cast(Any, viewer_policy)).run()
  except SystemExit:
    pass
  env.close()
  out = Path(cfg.out_dir) / f"viewer_{condition}"
  out.mkdir(parents=True, exist_ok=True)
  path = out / "trace.csv"
  _write_csv(path, rows)
  print(f"[D5.52 viewer] wrote {path} ({len(rows)} rows)")
  return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fields = list(rows[0].keys())
  for r in rows:
    fields += [k for k in r if k not in fields]
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)


# ---------------------------------------------------------------- analysis


def _handoff_step(rows: list[dict[str, Any]]) -> int | None:
  for r in rows:
    if r["hybrid_mode_active"]:
      return int(r["step"])
  return None


def _first(rows, pred, start: int) -> int | None:
  for r in rows:
    if r["step"] >= start and pred(r):
      return int(r["step"])
  return None


def summarize(condition: str, rows: list[dict[str, Any]], window: int = 10) -> dict:
  handoff = _handoff_step(rows)
  align = handoff if handoff is not None else FORCED_ENTRY_STEP
  by_step = {r["step"]: r for r in rows}
  win = [by_step[s] for s in range(align, align + window) if s in by_step]
  pre = by_step.get(align - 1)
  stance = [
    leg for leg in ("FR", "RL", "RR") if pre and pre[f"{leg.lower()}_ground_contact"]
  ]
  peak = max(win, key=lambda r: abs(r["base_ang_vel_b_y_rps"]))
  zmin = min(win, key=lambda r: r["base_z_m"])
  vzmin = min(win, key=lambda r: r["base_lin_vel_w_z_mps"])

  def stance_lost(r):
    return bool(stance) and all(
      not r[f"{leg.lower()}_ground_contact"] for leg in stance
    )

  def all_support_lost(r):
    return all(not r[f"{leg}_ground_contact"] for leg in ("fr", "rl", "rr"))

  calf_lim = _first(
    rows, lambda r: r["fl_calf_soft_limit_margin_rad"] < CALF_LIMIT_EPS_RAD, align
  )
  entry = _first(rows, lambda r: r["phase"] == "RECOVERY", 0)
  exit_ = _first(rows, lambda r: r["newly_recovered"], 0)
  fell = _first(rows, lambda r: r["fell_over"], 0)
  return {
    "condition": condition,
    "rows": len(rows),
    "recovery_entry_step": entry,
    "hybrid_handoff_step": handoff,
    "alignment_step_k0": align,
    "pre_handoff_stance_legs": "|".join(stance),
    "pre_handoff_pitch_rate_rps": pre["base_ang_vel_b_y_rps"] if pre else None,
    "peak_abs_pitch_rate_k0_9_rps": abs(peak["base_ang_vel_b_y_rps"]),
    "peak_abs_pitch_rate_k": peak["step"] - align,
    "min_base_z_k0_9_m": zmin["base_z_m"],
    "min_base_z_k": zmin["step"] - align,
    "base_z_drop_k0_9_m": (pre["base_z_m"] - zmin["base_z_m"]) if pre else None,
    "min_base_vz_w_k0_9_mps": vzmin["base_lin_vel_w_z_mps"],
    "stance_pair_loss_k": (s - align)
    if (s := _first(rows, stance_lost, align)) is not None
    else None,
    "all_support_loss_k": (s - align)
    if (s := _first(rows, all_support_lost, align)) is not None
    else None,
    "peak_abs_fl_calf_vel_k0_9_radps": max(abs(r["FL_calf_vel_radps"]) for r in win),
    "fl_calf_soft_limit_k": (calf_lim - align) if calf_lim is not None else None,
    "min_base_z_full_run_m": min(r["base_z_m"] for r in rows),
    "first_crossing_step": _first(rows, lambda r: r["crossing_achieved"], 0),
    "first_release_step": _first(rows, lambda r: r["release_achieved"], 0),
    "recovery_exit_step": exit_,
    "recovery_exit_timeout": bool(by_step[exit_]["recovery_failed_timeout"])
    if exit_ is not None
    else None,
    "fell_over_step": fell,
  }


def policy_diff(a: list[dict], b: list[dict], align: int, window: int = 10) -> dict:
  """Max |a_walk_raw| difference on the 9 non-FL joints + FL hip."""
  joints = ["FL_hip"] + [
    f"{leg}_{j}" for leg in ("FR", "RL", "RR") for j in ("hip", "thigh", "calf")
  ]
  ba = {r["step"]: r for r in a}
  bb = {r["step"]: r for r in b}
  out = {}
  for k in range(window):
    s = align + k
    if s in ba and s in bb:
      out[k] = max(
        abs(ba[s][f"{j}_a_walk_raw"] - bb[s][f"{j}_a_walk_raw"]) for j in joints
      )
  return out


def main() -> None:
  import tyro

  cfg = tyro.cli(D552Config)
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  if cfg.debug:
    assert cfg.condition != "all", "--debug requires a single --condition"
    run_viewer(cfg, cfg.condition)
    return

  conditions = CONDITIONS if cfg.condition == "all" else (cfg.condition,)
  results = {c: run_condition(cfg, c) for c in conditions}
  for c, res in results.items():
    _write_csv(out / f"trace_{c}.csv", res["rows"])
    _write_csv(out / f"substeps_{c}.csv", res["substeps"])
    print(f"[D5.52] wrote trace_{c}.csv ({len(res['rows'])} rows)")
  if cfg.condition != "all":
    return

  # Precondition: bitwise-equal states through end of step 42.
  checks: list[dict[str, Any]] = []
  base_states = results["baseline"]["states"]
  for c in ("wire_off_control", "wire_off_forced"):
    st = results[c]["states"]
    first_diff = None
    for k in range(min(len(st), len(base_states))):
      if not (
        torch.equal(st[k][0], base_states[k][0])
        and torch.equal(st[k][1], base_states[k][1])
      ):
        first_diff = k
        break
    checks.append(
      {
        "check": f"bitwise_equal_through_step42_{c}",
        "value": f"first_differing_step={first_diff}",
        "pass": first_diff is None or first_diff >= 43,
      }
    )

  # Precondition: baseline reproduces D5.49 baseline.
  ref_path = Path("logs/d5_49_recovery_override_counterfactual/baseline_trace.csv")
  ref = {int(r["step"]): r for r in csv.DictReader(ref_path.open())}
  pairs = {
    "base_z_m": "base_z_m",
    "base_ang_vel_y_rps": "base_ang_vel_b_y_rps",
    "base_lin_vel_z_mps": "base_lin_vel_b_z_mps",
    "fl_thigh_pos_rad": "FL_thigh_pos_rad",
    "fl_calf_pos_rad": "FL_calf_pos_rad",
    "fl_calf_vel_radps": "FL_calf_vel_radps",
    "wire_contact_force_mag_n": "wire_contact_sensor_force_mag_n",
  }
  max_diff = 0.0
  n = 0
  for r in results["baseline"]["rows"]:
    if r["step"] in ref:
      n += 1
      for rk, mk in pairs.items():
        max_diff = max(max_diff, abs(float(ref[r["step"]][rk]) - float(r[mk])))
  tol = 1e-5
  checks.append(
    {
      "check": "baseline_reproduces_d5_49",
      "value": f"max_abs_diff={max_diff:.3e} over {n} steps x {len(pairs)} cols; tol={tol}",
      "pass": max_diff <= tol,
    }
  )

  # Precondition: zero robot-wire contacts in wire-off runs (all substeps).
  for c in ("wire_off_control", "wire_off_forced"):
    total = sum(s["robot_wire_contact_count"] for s in results[c]["substeps"])
    checks.append(
      {
        "check": f"zero_robot_wire_contacts_{c}",
        "value": f"total={total}",
        "pass": total == 0,
      }
    )

  # Precondition: forced entry at 44 with intended controls.
  fr = {r["step"]: r for r in results["wire_off_forced"]["rows"]}
  r44 = fr.get(FORCED_ENTRY_STEP)
  ok = (
    r44 is not None
    and r44["phase"] == "RECOVERY"
    and r44["hybrid_mode_active"]
    and r44["forced_entry_step"] == FORCED_ENTRY_STEP
    and abs(r44["thigh_vel_cmd"] - 2.0) < 1e-6
    and abs(r44["calf_torque_cmd"] + 5.0) < 1e-6
    and r44["fl_thigh_pos_gain"] == 0.0
    and r44["fl_calf_pos_gain"] == 0.0
    and fr[FORCED_ENTRY_STEP - 1]["phase"] == "NORMAL_WALK"
  )
  checks.append(
    {
      "check": "forced_entry_at_44_with_hybrid_controls",
      "value": (
        f"phase={r44 and r44['phase']} hybrid={r44 and r44['hybrid_mode_active']} "
        f"thigh_cmd={r44 and r44['thigh_vel_cmd']} calf_cmd={r44 and r44['calf_torque_cmd']}"
      ),
      "pass": bool(ok),
    }
  )
  ctrl_rows = results["wire_off_control"]["rows"]
  ok = all(
    r["phase"] == "NORMAL_WALK" and not r["hybrid_mode_active"] for r in ctrl_rows
  )
  checks.append(
    {
      "check": "control_stays_on_walking_actuator_path",
      "value": f"all {len(ctrl_rows)} rows NORMAL_WALK with position gains active",
      "pass": ok,
    }
  )
  _write_csv(out / "validation.csv", checks)
  for ch in checks:
    print(f"[D5.52] {'PASS' if ch['pass'] else 'FAIL'} {ch['check']}: {ch['value']}")

  summaries = [summarize(c, results[c]["rows"]) for c in conditions]
  base_peak = summaries[0]["peak_abs_pitch_rate_k0_9_rps"]
  for s in summaries:
    s["peak_pitch_rate_fraction_of_baseline"] = (
      s["peak_abs_pitch_rate_k0_9_rps"] / base_peak
    )
  pd_fc = policy_diff(
    results["wire_off_forced"]["rows"], results["wire_off_control"]["rows"], 44
  )
  pd_bc = policy_diff(
    results["baseline"]["rows"], results["wire_off_control"]["rows"], 44
  )
  for s in summaries:
    s["max_other_leg_policy_action_diff_vs_control_k0_9"] = (
      max(pd_fc.values())
      if s["condition"] == "wire_off_forced"
      else max(pd_bc.values())
      if s["condition"] == "baseline"
      else 0.0
    )
  _write_csv(out / "comparison_summary.csv", summaries)
  for s in summaries:
    print(s)

  # Aligned trajectory: k = step - 44 (handoff is 44 in both hybrid runs).
  aligned = []
  for c in conditions:
    sub = {}
    for s in results[c]["substeps"]:
      d = sub.setdefault(s["step"], {"n": 0, "fmax": 0.0, "f": []})
      d["n"] = max(d["n"], s["robot_wire_contact_count"])
      d["fmax"] = max(d["fmax"], s["robot_wire_force_mag_n"])
      d["f"].append(f"{s['robot_wire_force_mag_n']:.2f}")
    for r in results[c]["rows"]:
      if 34 <= r["step"] <= 74:
        a = {"k_from_handoff": r["step"] - FORCED_ENTRY_STEP, **r}
        d = sub.get(r["step"], {"n": 0, "fmax": 0.0, "f": []})
        a["substep_robot_wire_contact_count_max"] = d["n"]
        a["substep_robot_wire_force_mag_max_n"] = d["fmax"]
        a["substep_robot_wire_force_mag_list_n"] = " ".join(d["f"])
        aligned.append(a)
  _write_csv(out / "aligned_trajectory.csv", aligned)
  print(f"[D5.52] wrote aligned_trajectory.csv ({len(aligned)} rows)")
  if not all(ch["pass"] for ch in checks):
    print("[D5.52] PRECONDITION FAILURE -- results must not be interpreted")
    sys.exit(2)


if __name__ == "__main__":
  main()
