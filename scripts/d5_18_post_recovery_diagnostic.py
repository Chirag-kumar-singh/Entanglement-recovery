"""D5.18 -- Post-recovery handoff and landing diagnostic.

Diagnostic-only. Investigates the SECONDARY base-height dip observed
after D5.3's RECOVERY phase exits (first characterized in D5.16's
trajectory audit: a ~0.064 m dip bottoming at step 92, 4 steps after the
position-actuator restoration at step 88; also visible, with a larger
magnitude, in D5.17's stance_deferred condition -- but per this task's
explicit instruction, THIS script uses ONLY the immediate-entry D5.3
baseline, reusing D5.17's existing `seed0_immediate` logs/condition
where possible rather than re-deriving a different baseline).

Reuses D5.3's exact env-build path (`build_ablation_env`, same
`go2_wire_recovery_residual_env_cfg` factory call, same isolated FL
wire, same 48-step recovery cap, same frozen checkpoint) UNMODIFIED --
no monkeypatch, no FSM/actuator override of any kind (unlike D5.13/
D5.17, this script installs nothing on `act`). Adds purely observational
logging: all four legs' ground-contact sensors (same `ContactSensorCfg`
pattern as D3/D5.16), and the actual per-step position-actuator CTRL
target (`env.sim.data.ctrl[..., pos_thigh_ctrl_id/pos_calf_ctrl_id]`,
i.e. what the frozen policy commanded that step, regardless of whether
the actuator's gain is currently zeroed) alongside the already-existing
force/torque columns, so a true "target discontinuity" at handoff can be
measured directly rather than inferred.

Uses the D5.15/D5.16/D5.17-corrected viewer stepping pattern (callback
never calls `env.step()` itself) if `--debug` is requested.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

PHASE_RECOVERY = 1
PHASE_POST_RECOVERY = 2


@dataclass
class D518Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 200
  max_recovery_steps_override: int = 48
  out_dir: str = ""
  debug: bool = False
  """If True, use the D5.15/D5.16/D5.17-corrected native MuJoCo viewer
  (callback never calls env.step() itself) instead of running headless.
  CSV/log output is identical either way."""


def main() -> None:
  import tyro

  cfg = tyro.cli(D518Config)
  run(cfg)


def run(cfg: D518Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)

  # Build directly (mirrors D5.3's build_ablation_env body) so the extra
  # FR/RL/RR ground-contact sensors can be appended BEFORE env
  # construction -- same approach D5.16 used, reusing the identical
  # go2_wire_recovery_residual_env_cfg(...) factory call and isolated
  # wire config, with NO FSM/actuator modification of any kind (pure
  # observation-only additions).
  import mujoco
  from mjlab.sensor import ContactMatch, ContactSensorCfg
  from mjlab.tasks.recovery.env_cfg_residual import (
    ISOLATED_FL_WIRE_CFG,
    ISOLATED_FL_WIRE_SPAWN_Y,
    go2_wire_recovery_residual_env_cfg,
  )

  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=d3_cfg.affected_leg_idx,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG if d3_cfg.use_isolated_wire else None,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y if d3_cfg.use_isolated_wire else None,
    experimental_peak_triggered_thigh_stop=d3_cfg.experimental_peak_triggered_thigh_stop,
    max_recovery_steps=cfg.max_recovery_steps_override,
  )
  env_cfg.seed = d3_cfg.seed
  env_cfg.sim.device = device

  from mjlab.envs import ManagerBasedRlEnv

  leg = "FL"
  other_leg = "FR"

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
  all_ground_cfgs = [fl_ground_cfg]
  for leg_name in ["FR", "RL", "RR"]:
    all_ground_cfgs.append(
      ContactSensorCfg(
        name=f"{leg_name.lower()}_ground_contact_diag",
        primary=ContactMatch(
          mode="geom", pattern=rf"{leg_name}_.*_collision\d*$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
      )
    )

  env_cfg.scene.sensors = (
    tuple(env_cfg.scene.sensors) + tuple(all_ground_cfgs) + (fr_wire_cfg,)
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index("FL_hip_joint")
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  state = {
    "trigger_step": None,
    "exit_line": None,
    "stage1_first_step": None,
    "stage2_first_step": None,
    "first_actuator_restore_step": None,
    "first_sustained_fl_ground_step": None,
    "_fl_ground_run": 0,
    "_seen_actuators_disabled": False,
  }

  def do_step_pre(step: int) -> int:
    if bool(act.newly_entangled[0]):
      state["trigger_step"] = step
    return int(act.phase[0])

  def do_step_post(step: int, prev_phase: int) -> dict:
    new_phase = int(act.phase[0])
    if prev_phase == PHASE_RECOVERY and new_phase == PHASE_POST_RECOVERY:
      state["exit_line"] = (
        f"EXIT RECOVERY step={step} elapsed={int(act.recovery_elapsed[0])} "
        f"timeout={bool(act.recovery_failed_timeout[0])}"
      )

    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos_w = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    foot_vel_w = entity.data.site_lin_vel_w[0, act._foot_site_idx, :].clone()
    base_pos_w = entity.data.root_link_pos_w[0].clone()
    base_quat_w = entity.data.root_link_quat_w[0].clone()
    base_lin_vel_b = entity.data.root_link_lin_vel_b[0].clone()
    base_ang_vel_b = entity.data.root_link_ang_vel_b[0].clone()

    foot_rel_w = (foot_pos_w - base_pos_w).unsqueeze(0)
    foot_pos_b = quat_apply_inverse(base_quat_w.unsqueeze(0), foot_rel_w)[0]
    foot_vel_b = quat_apply_inverse(base_quat_w.unsqueeze(0), foot_vel_w.unsqueeze(0))[
      0
    ]

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())

    ground_contacts = {}
    for leg_name in ["FL", "FR", "RL", "RR"]:
      sensor = env.scene.sensors[f"{leg_name.lower()}_ground_contact_diag"]
      ground_contacts[leg_name] = bool((sensor.data.found[0].reshape(-1) > 0).any())
    fr_wire_found = bool(
      (env.scene.sensors["fr_wire_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )

    calf_forward_dist = float(act.calf_forward_dist[0])
    calf_height_above_wire = float(act.calf_height_above_wire[0])
    calf_side_far = bool(act.calf_side_far[0])
    foot_forward_dist = float(act.foot_forward_dist[0])
    foot_height_above_wire = float(act.foot_height_above_wire[0])
    calf_clearance_achieved = bool(act._calf_clearance_achieved[0])
    foot_crossed_achieved = bool(act._foot_crossed_achieved[0])
    crossing_achieved = bool(act._crossing_achieved[0])
    release_counter = int(act._release_counter[0])
    recovery_failed_timeout_flag = bool(act._recovery_failed_timeout[0])
    released_sustained = crossing_achieved and (
      release_counter >= act.cfg.release_required_steps
    )
    genuine_recovery_complete = crossing_achieved and released_sustained
    pos_actuators_active = bool(act.hybrid_thigh_pos_gain_active[0])

    if calf_clearance_achieved and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if foot_crossed_achieved and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step
    # Only track the restoration event AFTER we have actually observed
    # the position actuators go inactive at least once (i.e. strictly
    # after a genuine RECOVERY entry) -- otherwise the pre-entry
    # steps (where pos_actuators_active is trivially True by default)
    # would spuriously register as "restoration" at step 0.
    if not pos_actuators_active:
      state["_seen_actuators_disabled"] = True
    if (
      pos_actuators_active
      and state["_seen_actuators_disabled"]
      and state["first_actuator_restore_step"] is None
    ):
      state["first_actuator_restore_step"] = step
    # Only track sustained FL ground contact AFTER recovery entry has
    # been observed (same rationale -- FL is trivially on the ground
    # during ordinary pre-entry walking stance phases).
    if state["trigger_step"] is not None:
      if ground_contacts["FL"]:
        state["_fl_ground_run"] += 1
      else:
        state["_fl_ground_run"] = 0
      if (
        state["_fl_ground_run"] >= 3 and state["first_sustained_fl_ground_step"] is None
      ):
        state["first_sustained_fl_ground_step"] = step - 2

    sim_ctrl = env.sim.data.ctrl[0]
    pos_thigh_ctrl_target = float(sim_ctrl[act._pos_thigh_ctrl_id])
    pos_calf_ctrl_target = float(sim_ctrl[act._pos_calf_ctrl_id])

    return {
      "step": step,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "pos_actuators_active": pos_actuators_active,
      "foot_z": float(foot_pos_w[2]),
      "foot_vz": float(foot_vel_w[2]),
      "foot_height_above_wire": foot_height_above_wire,
      "foot_x_base": float(foot_pos_b[0]),
      "foot_y_base": float(foot_pos_b[1]),
      "foot_z_base": float(foot_pos_b[2]),
      "foot_vx_base": float(foot_vel_b[0]),
      "foot_vy_base": float(foot_vel_b[1]),
      "foot_vz_base": float(foot_vel_b[2]),
      "hip_pos": float(qpos[hip_i]),
      "hip_vel": float(qvel[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "pos_thigh_ctrl_target": pos_thigh_ctrl_target,
      "pos_calf_ctrl_target": pos_calf_ctrl_target,
      "base_x": float(base_pos_w[0]),
      "base_y": float(base_pos_w[1]),
      "base_z": float(base_pos_w[2]),
      "base_quat_w": float(base_quat_w[0]),
      "base_quat_x": float(base_quat_w[1]),
      "base_quat_y": float(base_quat_w[2]),
      "base_quat_z": float(base_quat_w[3]),
      "base_lin_vel_x": float(base_lin_vel_b[0]),
      "base_lin_vel_y": float(base_lin_vel_b[1]),
      "base_lin_vel_z": float(base_lin_vel_b[2]),
      "base_ang_vel_x": float(base_ang_vel_b[0]),
      "base_ang_vel_y": float(base_ang_vel_b[1]),
      "base_ang_vel_z": float(base_ang_vel_b[2]),
      "wire_in_contact": wire_in_contact,
      "fl_ground_contact": ground_contacts["FL"],
      "fr_ground_contact": ground_contacts["FR"],
      "rl_ground_contact": ground_contacts["RL"],
      "rr_ground_contact": ground_contacts["RR"],
      "fr_wire_contact": fr_wire_found,
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
    prev_phase = do_step_pre(step)
    env.step(action)
    return do_step_post(step, prev_phase)

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    pending: dict = {"step": None, "prev_phase": None}

    def viewer_policy(_obs):
      """D5.15/D5.16/D5.17-pattern fix: never calls env.step() itself."""
      if pending["step"] is not None:
        row = do_step_post(pending["step"], pending["prev_phase"])
        rows.append(row)
        if (
          pending["step"] % 10 == 0
          or row["calf_clearance_achieved"]
          or row["crossing_achieved"]
        ):
          print(
            f"[D5.18] step={pending['step']} phase={row['phase']} "
            f"pos_act={row['pos_actuators_active']} fl_ground={row['fl_ground_contact']} "
            f"base_z={row['base_z']:.4f}"
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
    f.write(f"max_recovery_steps_override={cfg.max_recovery_steps_override}\n")
    f.write(f"trigger_step={state['trigger_step']}\n")
    f.write(f"{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
    f.write(f"first_actuator_restore_step={state['first_actuator_restore_step']}\n")
    f.write(
      f"first_sustained_fl_ground_step={state['first_sustained_fl_ground_step']}\n"
    )
  print(f"[D5.18] wrote {csv_path}")
  print(f"  trigger_step={state['trigger_step']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  print(f"  first_actuator_restore_step={state['first_actuator_restore_step']}")
  print(f"  first_sustained_fl_ground_step={state['first_sustained_fl_ground_step']}")
  return csv_path


if __name__ == "__main__":
  main()
