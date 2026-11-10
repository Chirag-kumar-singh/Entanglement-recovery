"""D5.3 -- Extended recovery-window diagnostic (does calf_forward_dist
resume advancing, plateau, or keep regressing beyond the 16-step cap?).

Directly answers D5.2's identified open question: the logged 16-step
RECOVERY window ends with `calf_forward_dist` actively regressing in all
4 runs (seed0/seed1, baseline/ablated); this script observes what the
SAME constant open-loop actuator commands produce if the FSM's hard cap
is raised, so the regression can be followed for longer, WITHOUT
altering any production default.

HOW THIS STAYS DIAGNOSTIC-ONLY (no controller/production change):
  - `go2_wire_recovery_residual_env_cfg(..., max_recovery_steps=N)` is an
    EXISTING, already-plumbed-through parameter of the factory function
    (see `env_cfg_residual.py` line ~174/262) -- raising it here is
    passing a different value to an existing config knob, in an
    isolated, hand-built env_cfg instance used only by this script. It
    does NOT edit `ResidualRecoveryActionCfg`'s class-level default
    (still 16), any checkpoint, any gain, any threshold besides this one
    cap, or wire geometry. No other D5.1/D4.8 script or its outputs are
    touched.
  - The recovery-entry trigger, exit/crossing logic, actuator gains, and
    all other thresholds (`clearance_margin`, `foot_crossing_margin`,
    `release_required_steps`, `wire_radius`) are UNCHANGED -- raising
    `max_recovery_steps` only delays the timeout-cap exit; it does not
    change what counts as genuine completion.
  - This is explicitly an extend-the-observation-window diagnostic, not
    a tuning change: results here must NOT be read as "the real recovery
    works for N>16 steps" -- only as evidence about the trend of
    `calf_forward_dist`/`calf_height_above_wire` under the existing
    constant commands, consistent with D5.2's stated next step.

Same build pattern as D4.8/D5.1 (build_ablation_env), same deterministic
zero-residual-action rollout, same seeds (0, 1), same baseline/ablated
conditions, same logged columns as D5.1 (full gate-variable
instrumentation), extended to `max_steps=200` and
`max_recovery_steps_override=48` (3x the original cap) so the
post-16-step trend is visible well past the original cutoff.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from mjlab.tasks.recovery.env_cfg_residual import (  # noqa: E402
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)


def build_ablation_env(
  cfg: D3GroundConfig,
  device: str,
  ablate_hip_wire: bool,
  max_recovery_steps_override: int,
):
  """Same as D5.1's `build_ablation_env`, plus passing
  `max_recovery_steps_override` through to the factory's existing
  `max_recovery_steps` parameter. Nothing else differs."""
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG if cfg.use_isolated_wire else None,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y if cfg.use_isolated_wire else None,
    experimental_peak_triggered_thigh_stop=cfg.experimental_peak_triggered_thigh_stop,
    max_recovery_steps=max_recovery_steps_override,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  leg = ["FL", "FR", "RL", "RR"][cfg.affected_leg_idx]
  other_leg = "FR" if leg == "FL" else "FL"

  fl_ground_cfg = ContactSensorCfg(
    name="fl_ground_contact_diag",
    primary=ContactMatch(
      mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  fl_ground_pergeom_cfg = ContactSensorCfg(
    name="fl_ground_pergeom_diag",
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
  fr_wire_cfg = ContactSensorCfg(
    name="fr_wire_contact_diag",
    primary=existing_wire_sensor.primary,
    secondary=ContactMatch(
      mode="geom", pattern=f"{other_leg}_calf2_collision", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors) + (
    fl_ground_cfg,
    fl_ground_pergeom_cfg,
    fr_wire_cfg,
  )

  existing_spec_fn = env_cfg.scene.spec_fn

  def composed_spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
    if existing_spec_fn is not None:
      existing_spec_fn(spec, num_envs)
    if ablate_hip_wire:
      wire_seg_bodies = [b.name for b in spec.bodies if b.name.startswith("wire_seg_")]
      assert wire_seg_bodies, (
        "No wire_seg_* bodies found after existing spec_fn -- composition order wrong."
      )
      for seg_body in wire_seg_bodies:
        spec.add_exclude(bodyname1="robot/FL_hip", bodyname2=seg_body)
      print(
        f"[D5.3-ABLATION] Added {len(wire_seg_bodies)} body-pair excludes: "
        f"robot/FL_hip <-> {wire_seg_bodies}"
      )

  env_cfg.scene.spec_fn = composed_spec_fn

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env


@dataclass
class D53Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 200
  max_recovery_steps_override: int = 48
  ablate_hip_wire: bool = False
  out_dir: str = ""
  debug: bool = False
  """If True, open a dynamic native MuJoCo viewer (real physics, same
  pattern as d3_ground_contact_diagnostic.py's --debug) instead of
  running headless. CSV/log output is identical either way -- this only
  adds a live visualization of the exact same deterministic rollout."""


def run(cfg: D53Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_ablation_env(
    d3_cfg, device, cfg.ablate_hip_wire, cfg.max_recovery_steps_override
  )
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  mj_model = env.sim.mj_model

  def geom_name(gid: int) -> str:
    if gid < 0:
      return "<none>"
    n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom#{gid}"

  out_dir = Path(cfg.out_dir)
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
    """D5.15: pre-step bookkeeping, split out of the old combined
    `do_step` so the viewer path (below) can invoke it WITHOUT calling
    `env.step()` itself. Must be called with env state reflecting the
    END of the PREVIOUS step (or the initial reset state, for step=0)
    -- i.e., strictly BEFORE this step's own `env.step()` call. Checks
    the wire-contact-triggered RECOVERY-entry flag (`newly_entangled`,
    which is set during the PRIOR step's `_update_fsm()` call) and
    returns the phase this step starts in, for later use by
    `do_step_post`."""
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    return int(act.phase[0])

  def do_step_post(step: int, prev_phase: int) -> dict:
    """D5.15: post-step row construction, split out of the old combined
    `do_step`. Must be called with env state reflecting the state
    immediately AFTER this step's `env.step()` has executed exactly
    once. Unchanged in content/semantics from the original inline body
    -- only the call-site/timing relative to `env.step()` differs
    between the headless (`do_step`) and viewer (`viewer_policy`)
    paths."""
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
    fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    fl_ground_found = bool((fl_ground.data.found[0].reshape(-1) > 0).any())
    fr_wire = env.scene.sensors["fr_wire_contact_diag"]
    fr_wire_found = bool((fr_wire.data.found[0].reshape(-1) > 0).any())

    contact = env.sim.data.contact
    nacon = int(env.sim.data.nacon[0])
    hip_wire_contact_now = False
    calf_wire_contact_now = False
    for ci in range(nacon):
      if int(contact.worldid[ci]) != 0:
        continue
      n0, n1 = geom_name(int(contact.geom[ci][0])), geom_name(int(contact.geom[ci][1]))
      pair = (n0, n1)
      if any("FL_hip_collision" in n for n in pair) and any(
        "wire_seg" in n for n in pair
      ):
        hip_wire_contact_now = True
      if any("FL_calf2_collision" in n for n in pair) and any(
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

    return {
      "step": step,
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
      "base_z": float(base_pos[2]),
      "base_quat_w": float(base_quat[0]),
      "wire_in_contact": wire_in_contact,
      "fl_ground_contact": fl_ground_found,
      "fr_wire_contact": fr_wire_found,
      "fl_hip_wire_contact_raw": hip_wire_contact_now,
      "fl_calf_wire_contact_raw": calf_wire_contact_now,
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

  def do_step(step: int, action: torch.Tensor) -> dict:
    """Headless-path convenience wrapper: performs the pre-step check,
    calls `env.step()` itself exactly once, then performs the post-step
    row construction. Used ONLY by the headless loop below. The viewer
    path (`viewer_policy`) does NOT use this wrapper -- see its
    docstring for why (D5.15 fix)."""
    prev_phase = do_step_pre(step)
    env.step(action)
    return do_step_post(step, prev_phase)

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    pending: dict = {"step": None, "prev_phase": None}

    def viewer_policy(_obs):
      """D5.15 FIX: this callback must NOT call `env.step()` itself.
      `NativeMujocoViewer._execute_step` already calls
      `env.step(actions)` exactly once per rendered tick, immediately
      after calling this policy callback with the pre-step observation
      (see `mjlab/viewer/base.py`'s `_execute_step`: `actions =
      self.policy(obs); self.env.step(actions)`). The PRE-FIX version
      of this callback called `do_step` (which itself calls
      `env.step()`) and then returned an action that the viewer stepped
      AGAIN -- silently double-stepping the environment every tick.

      Fix: this callback is now stateful across calls. Each invocation
      FIRST finalizes the row for the PREVIOUS actual environment step
      using `do_step_post` -- which is safe to call here because the
      CURRENT env state (as seen by this callback, via `get_observations`
      called by the viewer just before invoking us) is EXACTLY the state
      produced by the viewer's own `env.step()` call for that previous
      step (or the initial post-reset state, on the very first call,
      before any step has happened). It THEN performs `do_step_pre` for
      the step about to be executed, and finally returns the (zero)
      action for the viewer to apply via its own, single `env.step()`
      call. This produces byte-identical `rows` content/ordering to the
      headless path, merely shifted by one callback invocation (see the
      D5.15 report for the before/after headless-equivalence check)."""
      if pending["step"] is not None:
        row = do_step_post(pending["step"], pending["prev_phase"])
        rows.append(row)
        if (
          pending["step"] % 10 == 0
          or row["calf_clearance_achieved"]
          or row["crossing_achieved"]
        ):
          print(
            f"[D5.3] step={pending['step']} phase={row['phase']} "
            f"calf_forward_dist={row['calf_forward_dist']:.4f} "
            f"calf_height_above_wire={row['calf_height_above_wire']:.4f} "
            f"calf_side_far={row['calf_side_far']} "
            f"stage1={row['calf_clearance_achieved']} stage2={row['foot_crossed_achieved']}"
          )
        if pending["step"] + 1 >= cfg.max_steps:
          raise SystemExit(0)

      step = 0 if pending["step"] is None else pending["step"] + 1
      prev_phase = do_step_pre(step)
      pending["step"] = step
      pending["prev_phase"] = prev_phase
      return torch.zeros(env.action_manager.action.shape, device=device)

    print(
      "[INFO] Debug mode: native viewer (DYNAMIC, real physics). Close window to end run."
    )
    try:
      NativeMujocoViewer(env, viewer_policy).run()
    except SystemExit:
      pass
  else:
    for step in range(cfg.max_steps):
      action = torch.zeros(env.action_manager.action.shape, device=device)
      rows.append(do_step(step, action))

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"ablate_hip_wire={cfg.ablate_hip_wire}\n")
    f.write(f"max_recovery_steps_override={cfg.max_recovery_steps_override}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
  print(f"[D5.3] wrote {csv_path}")
  print(f"  {state['enter_line']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D53Config)
  run(cfg)


if __name__ == "__main__":
  main()
