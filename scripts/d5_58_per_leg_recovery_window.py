"""D5.58 -- Per-leg extended recovery-window experiment (FL / FR / RL / RR).

Generalization of D5.3 (`scripts/d5_3_extended_recovery_window.py`) from
the front-left leg to ALL FOUR legs. D5.3 itself is NOT modified, and
neither is any production default, checkpoint, gain, threshold, or wire
geometry constant: this script builds its own isolated env_cfg instance
via the existing, already-plumbed factory parameters
(`affected_leg_idx`, `wire_cfg`, `spawn_y_override`,
`max_recovery_steps`) plus an additive `spec_fn` composition, exactly as
D5.3 does.

What is the same as D5.3
------------------------
  - Same factory (`go2_wire_recovery_residual_env_cfg`), same
    `recovery_mode="hybrid"`, same `play=True`, same frozen walking
    policy checkpoint, same deterministic zero-residual-action rollout.
  - Same `max_recovery_steps_override=48` (3x the production 16-step
    cap), so the post-cap trend of `calf_forward_dist` /
    `calf_height_above_wire` is observable.
  - Same logged columns and same per-step pre/post split
    (`do_step_pre`/`do_step_post`) including the D5.15 viewer
    double-step fix.
  - For `--leg FL`, the wire geometry and spawn are D5.3's
    `ISOLATED_FL_WIRE_CFG` / `ISOLATED_FL_WIRE_SPAWN_Y` verbatim.

What is per-leg
---------------
  - `affected_leg_idx`, and therefore the FSM's tracked calf geom, foot
    site, hybrid auxiliary actuators, observations and rewards, all
    follow the target leg (the production factory and action term are
    already leg-generic -- see `LEG_NAMES[affected_leg_idx]` usages in
    `env_cfg_residual.py` / `mdp/actions.py`).
  - Wire placement comes from `scripts/per_leg_wire.py`: D5.3's FL wire
    for the left legs, its mirror image about the spawn centreline for
    the right legs. Robot spawn is the SAME for every leg
    (`PER_LEG_SPAWN_Y = 0.25`), so no leg's absolute corridor moves.
  - The three NON-target legs are contact-excluded from the wire, so the
    wire interacts with the target leg only. This is required for the
    rear legs (RL/RR share their side's lateral corridor with the front
    leg on that side, which would otherwise always trip first); it is a
    no-op in practice for FL, whose isolated wire the other legs already
    cannot reach. See `scripts/per_leg_wire.py` for the full rationale.
  - Default rollout length: rear legs reach the wire plane roughly 65-70
    steps later than front legs (measured by
    `scripts/d5_58_leg_corridor_survey.py`: calf crosses x=0 at step
    ~57/51 for FL/FR vs. ~122/126 for RL/RR), so `--max-steps` defaults
    to 200 for front legs and 320 for rear legs.

Isolation is VERIFIED, not assumed: the CSV logs a wire-contact sensor
for every non-target leg's calf, and the run log asserts/reports whether
any of them ever fired.

Usage
-----
    uv run python scripts/d5_58_per_leg_recovery_window.py --leg RL --seed 0
    uv run python scripts/d5_58_per_leg_recovery_window.py --leg FR --debug
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from per_leg_wire import (  # noqa: E402
  LEG_IDX,
  LEG_NAMES,
  PER_LEG_SPAWN_Y,
  PER_LEG_WIRE_CFG,
  leg_body_names,
  make_leg_isolating_spec_fn,
)

from mjlab.tasks.recovery.env_cfg_residual import (  # noqa: E402
  go2_wire_recovery_residual_env_cfg,
)

FRONT_LEGS = ("FL", "FR")


def default_max_steps(leg: str) -> int:
  """Front legs reach the wire ~55 steps in, rear legs ~125 (measured,
  see module docstring). 200 steps was D5.3's window for FL; rear legs
  get the same post-contact observation budget, shifted."""
  return 200 if leg in FRONT_LEGS else 320


@dataclass
class D558Config:
  leg: str = "FL"
  """Target leg: FL, FR, RL or RR."""
  seed: int = 0
  device: str | None = None
  max_steps: int = 0
  """0 = auto (200 front / 320 rear, see `default_max_steps`)."""
  max_recovery_steps_override: int = 48
  """Same 3x-the-production-cap observation window D5.3 uses."""
  ablate_hip_wire: bool = False
  """D4.8/D5.3-style ablation: additionally exclude the TARGET leg's hip
  from the wire, leaving only thigh/calf interaction."""
  recovery_thigh_velocity: float = -2.0
  """Production default. The sign encodes the sweep direction: cfg=-2.0
  writes ctrl=+2.0, i.e. RETRACT (thigh angle increases, foot swings
  backward/up). cfg=+2.0 inverts it to PROTRACT (foot swings forward/up),
  which is the direction a REAR leg needs -- see `run` docstring."""
  recovery_calf_torque: float = 5.0
  """Production default open-loop calf flexion torque magnitude."""
  out_dir: str = ""
  debug: bool = False
  """Open the native MuJoCo viewer (real physics, same deterministic
  rollout and identical CSV output) instead of running headless."""


def build_leg_env(cfg: D558Config, device: str):
  """D5.3's `build_ablation_env`, generalized over the target leg.

  Differences from D5.3's version, all additive:
    - wire geometry / affected leg follow `cfg.leg`;
    - the three non-target legs are contact-excluded from the wire;
    - a wire-contact diagnostic sensor is added for EVERY non-target leg
      (D5.3 added one, for FR only) so isolation can be verified.
  """
  leg = cfg.leg
  wire = PER_LEG_WIRE_CFG[leg]
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=LEG_IDX[leg],
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=wire,
    spawn_y_override=PER_LEG_SPAWN_Y,
    max_recovery_steps=cfg.max_recovery_steps_override,
    recovery_thigh_velocity=cfg.recovery_thigh_velocity,
    recovery_calf_torque=cfg.recovery_calf_torque,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  other_legs = tuple(name for name in LEG_NAMES if name != leg)

  ground_cfg = ContactSensorCfg(
    name="target_ground_contact_diag",
    primary=ContactMatch(
      mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  ground_pergeom_cfg = ContactSensorCfg(
    name="target_ground_pergeom_diag",
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
  other_wire_sensors = tuple(
    ContactSensorCfg(
      name=f"{other}_wire_contact_diag",
      primary=existing_wire_sensor.primary,
      secondary=ContactMatch(
        mode="geom", pattern=f"{other}_calf2_collision", entity="robot"
      ),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    )
    for other in other_legs
  )
  env_cfg.scene.sensors = (
    tuple(env_cfg.scene.sensors) + (ground_cfg, ground_pergeom_cfg) + other_wire_sensors
  )

  isolating_spec_fn = make_leg_isolating_spec_fn(env_cfg.scene.spec_fn, leg)

  def composed_spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
    isolating_spec_fn(spec, num_envs)
    if cfg.ablate_hip_wire:
      seg_bodies = [b.name for b in spec.bodies if b.name.startswith("wire_seg_")]
      hip_body = leg_body_names(leg)[0]
      for seg_body in seg_bodies:
        spec.add_exclude(bodyname1=hip_body, bodyname2=seg_body)
      print(
        f"[D5.58-ABLATION] Added {len(seg_bodies)} body-pair excludes: "
        f"{hip_body} <-> wire_seg_*"
      )

  env_cfg.scene.spec_fn = composed_spec_fn

  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def run(cfg: D558Config) -> Path:
  if cfg.leg not in LEG_IDX:
    raise ValueError(f"--leg must be one of {LEG_NAMES}, got {cfg.leg!r}")
  leg = cfg.leg
  device = cfg.device or "cpu"
  max_steps = cfg.max_steps or default_max_steps(leg)
  other_legs = tuple(name for name in LEG_NAMES if name != leg)

  env = build_leg_env(cfg, device)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index(f"{leg}_thigh_joint")
  calf_i = joint_names.index(f"{leg}_calf_joint")
  hip_i = joint_names.index(f"{leg}_hip_joint")

  mj_model = env.sim.mj_model

  def geom_name(gid: int) -> str:
    if gid < 0:
      return "<none>"
    n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom#{gid}"

  out_dir = Path(cfg.out_dir or f"logs/d5_58_per_leg_recovery/{leg}_seed{cfg.seed}")
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  state = {
    "enter_line": None,
    "exit_line": None,
    "stage1_first_step": None,
    "stage2_first_step": None,
  }

  def do_step_pre(step: int) -> int:
    """Pre-step bookkeeping (identical in role to D5.3's). Must be called
    with env state reflecting the END of the previous step."""
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    return int(act.phase[0])

  def do_step_post(step: int, prev_phase: int) -> dict:
    """Post-step row construction (same columns as D5.3, with the target
    leg substituted for FL and one wire-contact column per other leg)."""
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      state["exit_line"] = (
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
    ground = env.scene.sensors["target_ground_contact_diag"]
    ground_found = bool((ground.data.found[0].reshape(-1) > 0).any())
    other_wire_found = {}
    for other in other_legs:
      sensor = env.scene.sensors[f"{other}_wire_contact_diag"]
      other_wire_found[other] = bool((sensor.data.found[0].reshape(-1) > 0).any())

    contact = env.sim.data.contact
    nacon = int(env.sim.data.nacon[0])
    hip_wire_contact_now = False
    calf_wire_contact_now = False
    for ci in range(nacon):
      if int(contact.worldid[ci]) != 0:
        continue
      n0, n1 = geom_name(int(contact.geom[ci][0])), geom_name(int(contact.geom[ci][1]))
      pair = (n0, n1)
      if any(f"{leg}_hip_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        hip_wire_contact_now = True
      if any(f"{leg}_calf2_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        calf_wire_contact_now = True

    calf_forward_dist = float(act.calf_forward_dist[0])
    calf_height_above_wire = float(act.calf_height_above_wire[0])
    calf_side_far = bool(act.calf_side_far[0])
    foot_forward_dist = float(act.foot_forward_dist[0])
    calf_clearance_achieved = bool(act._calf_clearance_achieved[0])
    foot_crossed_achieved = bool(act._foot_crossed_achieved[0])
    crossing_achieved = bool(act._crossing_achieved[0])
    release_counter = int(act._release_counter[0])
    recovery_failed_timeout_flag = bool(act._recovery_failed_timeout[0])
    released_sustained = crossing_achieved and (
      release_counter >= act.cfg.release_required_steps
    )
    genuine_recovery_complete = crossing_achieved and released_sustained

    if calf_clearance_achieved and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if foot_crossed_achieved and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step

    row = {
      "step": step,
      "leg": leg,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "foot_z": float(foot_pos[2]),
      "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
      "foot_height_above_wire": float(act.foot_height_above_wire[0]),
      "hip_pos": float(qpos[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "base_x": float(base_pos[0]),
      "base_z": float(base_pos[2]),
      "base_quat_w": float(base_quat[0]),
      "wire_in_contact": wire_in_contact,
      "target_ground_contact": ground_found,
      "target_hip_wire_contact_raw": hip_wire_contact_now,
      "target_calf_wire_contact_raw": calf_wire_contact_now,
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
      "calf_forward_dist": calf_forward_dist,
      "calf_height_above_wire": calf_height_above_wire,
      "calf_side_far": calf_side_far,
      "foot_forward_dist": foot_forward_dist,
      "calf_clearance_achieved": calf_clearance_achieved,
      "foot_crossed_achieved": foot_crossed_achieved,
      "crossing_achieved": crossing_achieved,
      "release_counter": release_counter,
      "recovery_failed_timeout_flag": recovery_failed_timeout_flag,
      "released_sustained": released_sustained,
      "genuine_recovery_complete": genuine_recovery_complete,
    }
    for other in other_legs:
      row[f"{other}_wire_contact"] = other_wire_found[other]
    return row

  def do_step(step: int, action: torch.Tensor) -> dict:
    prev_phase = do_step_pre(step)
    env.step(action)
    return do_step_post(step, prev_phase)

  def log_row(step: int, row: dict) -> None:
    if step % 10 == 0 or row["calf_clearance_achieved"] or row["crossing_achieved"]:
      print(
        f"[D5.58/{leg}] step={step} phase={row['phase']} "
        f"calf_forward_dist={row['calf_forward_dist']:.4f} "
        f"calf_height_above_wire={row['calf_height_above_wire']:.4f} "
        f"calf_side_far={row['calf_side_far']} "
        f"stage1={row['calf_clearance_achieved']} "
        f"stage2={row['foot_crossed_achieved']}"
      )

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    pending: dict = {"step": None, "prev_phase": None}

    def viewer_policy(_obs):
      """Same D5.15 double-step fix as D5.3: the viewer calls
      `env.step()` itself exactly once per tick, so this callback must
      only do bookkeeping and return an action. It finishes the PREVIOUS
      step's row first, then pre-registers the step about to run."""
      if pending["step"] is not None:
        row = do_step_post(pending["step"], pending["prev_phase"])
        rows.append(row)
        log_row(pending["step"], row)
        if pending["step"] + 1 >= max_steps:
          raise SystemExit(0)

      step = 0 if pending["step"] is None else pending["step"] + 1
      pending["prev_phase"] = do_step_pre(step)
      pending["step"] = step
      return torch.zeros(env.action_manager.action.shape, device=device)

    print(
      "[INFO] Debug mode: native viewer (DYNAMIC, real physics). "
      "Close window to end run."
    )
    try:
      NativeMujocoViewer(env, viewer_policy).run()
    except SystemExit:
      pass
  else:
    for step in range(max_steps):
      action = torch.zeros(env.action_manager.action.shape, device=device)
      row = do_step(step, action)
      rows.append(row)
      log_row(step, row)

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  leakage = {
    other: sum(1 for r in rows if r[f"{other}_wire_contact"]) for other in other_legs
  }
  target_contact_steps = sum(1 for r in rows if r["wire_in_contact"])

  lines = [
    f"leg={leg}",
    f"seed={cfg.seed}",
    f"max_steps={max_steps}",
    f"ablate_hip_wire={cfg.ablate_hip_wire}",
    f"max_recovery_steps_override={cfg.max_recovery_steps_override}",
    f"recovery_thigh_velocity={cfg.recovery_thigh_velocity}",
    f"recovery_calf_torque={cfg.recovery_calf_torque}",
    f"wire_cfg={PER_LEG_WIRE_CFG[leg]}",
    f"spawn_y={PER_LEG_SPAWN_Y}",
    f"{state['enter_line']}",
    f"{state['exit_line']}",
    f"stage1_first_latch_step={state['stage1_first_step']}",
    f"stage2_first_latch_step={state['stage2_first_step']}",
    f"target_leg_wire_contact_steps={target_contact_steps}",
    f"other_leg_wire_contact_steps={leakage}",
    "ISOLATION_OK=" + str(all(v == 0 for v in leakage.values())),
  ]
  summary = "\n".join(lines)
  (out_dir / "run_log.txt").write_text(summary + "\n")
  print(f"[D5.58] wrote {csv_path}")
  print(summary)
  return csv_path


def main() -> None:
  import tyro

  run(tyro.cli(D558Config))


if __name__ == "__main__":
  main()
