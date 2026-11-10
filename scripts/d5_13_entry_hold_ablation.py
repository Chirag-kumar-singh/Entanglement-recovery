"""D5.13 -- Test D5.12's recommended mitigation: a brief, FIXED-TARGET
PD hold at the FL leg's recovery-entry joint pose, before handing off to
the existing position-actuator-disabling + auxiliary-command override,
to see whether it reduces the base-height sag isolated in D5.10-D5.12.

BACKGROUND: D5.11 showed ramping the auxiliary commands' onset has no
effect on the sag. D5.12 showed suppressing the auxiliary commands
entirely still leaves most of the sag -- the position-actuator
DISABLING itself (losing the FL leg's normal stance-stabilizing
gain/bias at recovery entry) is the dominant cause. D5.12 recommended
testing a brief hold at the pre-entry pose before disabling, as the next
step.

THIS IS NOT A REVIVAL OF D5.9's REJECTED "OPTION C": Option C restored
the position actuator to track pi_walk's LIVE, continuously-diverging
calf target for the WHOLE recovery window, which D5.9 showed saturates
at the +/-45 Nm effort limit within 4-5 steps. Condition D here instead
holds the position actuator at a FIXED target equal to the leg's ACTUAL
measured position at the instant of recovery entry (zero initial
tracking error by construction) for a SHORT, FIXED, EXPERIMENTAL
duration (`hold_steps`, default 5), using the leg's EXISTING, UNCHANGED
stiffness/damping -- then hands off to the existing, unmodified
disabling+auxiliary-command override exactly as in baseline, merely
delayed by `hold_steps`. This is explicitly a new, separate, isolated
diagnostic experiment, not a production controller change, and does not
reopen or reverse D5.9's Option-C rejection.

CONDITION A -- baseline (identical to D5.10/D5.11/D5.12's Condition A):
included again purely as the reference point.

CONDITION D -- entry_hold: on recovery entry, the FL thigh/calf
POSITION actuators' gain/bias are left COMPLETELY UNTOUCHED (still the
entity's original stiffness/damping) for `hold_steps` RECOVERY steps.
During the hold, this leg's position-actuator ctrl target is forcibly
overridden (AFTER `self._joint_pos_action.apply_actions()` writes the
live walking-policy target into `sim.data.ctrl`, so the live, diverging
policy target never physically takes effect) to the FIXED position
captured at the instant of entry. No auxiliary thigh-velocity/calf-
torque command is applied during the hold (aux actuator gain stays
disabled, matching its normal out-of-RECOVERY state). At
`recovery_elapsed == hold_steps`, the position actuators are disabled
and the auxiliary commands engaged at FULL magnitude immediately (no
ramp -- D5.11 already showed ramping doesn't matter), i.e. exactly
baseline's enter-transition logic, just time-shifted by `hold_steps`.
Exit/restore logic is completely unmodified and unaffected (occurs well
after any hold_steps in practice).

HOW THIS IS IMPLEMENTED (isolated, non-production):
  - `act._apply_hybrid_actuators` is replaced by a reimplementation that
    mirrors the real method's enter/exit/command logic (verified against
    source), with ONE change: the position-actuator-disabling and
    auxiliary-gain-enabling that normally happens on `newly_entangled`
    is deferred until `recovery_elapsed == hold_steps`; before that, the
    affected envs are marked `holding=True` and receive no gain changes
    and no auxiliary command.
  - `act.apply_actions` (the wrapping action term's per-step ctrl-write
    entrypoint) is wrapped to call the original
    `self._joint_pos_action.apply_actions()` first (unchanged), then,
    for any env still `holding`, overwrite
    `sim.data.ctrl[env_id, pos_thigh_ctrl_id/pos_calf_ctrl_id]` with the
    FIXED entry-position targets captured at entry.
  - No FSM, threshold, gain, wire-geometry, checkpoint, or
    `max_recovery_steps` change. No production file edited.

Reuses `build_d510_env` from d5_10_stumble_ablation.py UNCHANGED, so all
conditions here share identical initial state with
D5.10/D5.11/D5.12.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d3_ground_contact_diagnostic import D3GroundConfig  # noqa: E402
from d5_10_stumble_ablation import build_d510_env  # noqa: E402

PHASE_RECOVERY = 1


def install_entry_hold(act, hold_steps: int, thigh_i: int, calf_i: int) -> dict:
  """Install the entry-hold variant of `_apply_hybrid_actuators` plus a
  ctrl-override wrapper on `apply_actions`, per the module docstring.
  Returns a small state dict (exposed for logging) tracking per-env
  `holding` status and the frozen hold targets."""
  assert hold_steps >= 1, "hold_steps must be >= 1"
  entity = act._entity
  device = act.device
  num_envs = act.num_envs

  state = {
    "holding": torch.zeros(num_envs, dtype=torch.bool, device=device),
    "hold_target_thigh": torch.zeros(num_envs, device=device),
    "hold_target_calf": torch.zeros(num_envs, device=device),
  }

  def entry_hold_apply_hybrid_actuators() -> None:
    in_recovery = act._phase == PHASE_RECOVERY
    sim = act._env.sim

    enter_ids = act.newly_entangled.nonzero(as_tuple=True)[0]
    exit_ids = act.newly_recovered.nonzero(as_tuple=True)[0]

    if enter_ids.numel() > 0:
      qpos = entity.data.joint_pos
      state["hold_target_thigh"][enter_ids] = qpos[enter_ids, thigh_i]
      state["hold_target_calf"][enter_ids] = qpos[enter_ids, calf_i]
      state["holding"][enter_ids] = True
      # Deliberately do NOT zero position-actuator gain/bias and do NOT
      # enable the auxiliary actuator's gain here -- deferred to the
      # transition block below, once hold_steps have elapsed.
      act._peak_armed[enter_ids] = False
      act._peak_running_max[enter_ids] = act.foot_height_above_wire[enter_ids]
      act._peak_entry_height[enter_ids] = act.foot_height_above_wire[enter_ids]
      act._peak_decline_counter[enter_ids] = 0
      act._peak_triggered[enter_ids] = False
      act.peak_trigger_step[enter_ids] = -1

    elapsed = act._recovery_elapsed
    transition_mask = state["holding"] & in_recovery & (elapsed >= hold_steps)
    transition_ids = transition_mask.nonzero(as_tuple=True)[0]
    if transition_ids.numel() > 0:
      gainprm_width = sim.model.actuator_gainprm.shape[-1]
      biasprm_width = sim.model.actuator_biasprm.shape[-1]
      zero_gain = torch.zeros(transition_ids.numel(), gainprm_width, device=device)
      zero_bias = torch.zeros(transition_ids.numel(), biasprm_width, device=device)
      sim.model.actuator_gainprm[transition_ids, act._pos_thigh_ctrl_id, :] = zero_gain
      sim.model.actuator_biasprm[transition_ids, act._pos_thigh_ctrl_id, :] = zero_bias
      sim.model.actuator_gainprm[transition_ids, act._pos_calf_ctrl_id, :] = zero_gain
      sim.model.actuator_biasprm[transition_ids, act._pos_calf_ctrl_id, :] = zero_bias
      sim.model.actuator_gainprm[transition_ids, act._aux_thigh_ctrl_id, 0] = (
        act.cfg.recovery_thigh_velocity_kd
      )
      sim.model.actuator_biasprm[
        transition_ids, act._aux_thigh_ctrl_id, 2
      ] = -act.cfg.recovery_thigh_velocity_kd
      act.hybrid_thigh_pos_gain_active[transition_ids] = False
      state["holding"][transition_ids] = False

    if exit_ids.numel() > 0:
      # Unmodified restore -- identical to the real method.
      sim.model.actuator_gainprm[exit_ids, act._pos_thigh_ctrl_id, :] = (
        act._orig_thigh_gainprm[exit_ids]
      )
      sim.model.actuator_biasprm[exit_ids, act._pos_thigh_ctrl_id, :] = (
        act._orig_thigh_biasprm[exit_ids]
      )
      sim.model.actuator_gainprm[exit_ids, act._pos_calf_ctrl_id, :] = (
        act._orig_calf_gainprm[exit_ids]
      )
      sim.model.actuator_biasprm[exit_ids, act._pos_calf_ctrl_id, :] = (
        act._orig_calf_biasprm[exit_ids]
      )
      sim.model.actuator_gainprm[exit_ids, act._aux_thigh_ctrl_id, 0] = 0.0
      sim.model.actuator_biasprm[exit_ids, act._aux_thigh_ctrl_id, 2] = 0.0
      act.hybrid_thigh_pos_gain_active[exit_ids] = True
      state["holding"][exit_ids] = False

    # Auxiliary commands: zero while holding (gain disabled anyway, but
    # zeroed explicitly for clean logging), full magnitude once the hold
    # has ended -- no ramp (D5.11 showed ramp has no effect).
    aux_active = in_recovery & (~state["holding"])
    thigh_vel_cmd = torch.where(
      aux_active,
      torch.full_like(act.hybrid_thigh_vel_ctrl, -act.cfg.recovery_thigh_velocity),
      torch.zeros_like(act.hybrid_thigh_vel_ctrl),
    )
    calf_torque_cmd = torch.where(
      aux_active,
      torch.full_like(act.hybrid_calf_torque_ctrl, -act.cfg.recovery_calf_torque),
      torch.zeros_like(act.hybrid_calf_torque_ctrl),
    )
    act.hybrid_thigh_vel_ctrl = thigh_vel_cmd
    act.hybrid_calf_torque_ctrl = calf_torque_cmd

    act._entity.set_joint_velocity_target(
      thigh_vel_cmd.unsqueeze(-1), joint_ids=[act._hybrid_thigh_joint_id]
    )
    act._entity.set_joint_effort_target(
      calf_torque_cmd.unsqueeze(-1), joint_ids=[act._hybrid_calf_joint_id]
    )

  act._apply_hybrid_actuators = entry_hold_apply_hybrid_actuators

  original_apply_actions = act.apply_actions

  def ctrl_override_apply_actions() -> None:
    original_apply_actions()
    holding_ids = state["holding"].nonzero(as_tuple=True)[0]
    if holding_ids.numel() > 0:
      sim = act._env.sim
      sim.data.ctrl[holding_ids, act._pos_thigh_ctrl_id] = state["hold_target_thigh"][
        holding_ids
      ]
      sim.data.ctrl[holding_ids, act._pos_calf_ctrl_id] = state["hold_target_calf"][
        holding_ids
      ]

  act.apply_actions = ctrl_override_apply_actions

  return state


@dataclass
class D513Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 150
  """Hard diagnostic-window cap, matching D5.10/D5.11/D5.12 for direct comparability."""
  condition: Literal["baseline", "entry_hold"] = "baseline"
  """'baseline' = Condition A (existing hybrid controller, unmodified).
  'entry_hold' = Condition D (brief fixed-target PD hold at the
  recovery-entry pose for `hold_steps` steps, then normal baseline
  disabling+auxiliary-command engagement)."""
  hold_steps: int = 5
  """Number of RECOVERY steps the position actuators are held at the
  frozen entry pose before handing off to the normal disabling+
  auxiliary-command override. Experimental parameter, not validated --
  chosen to match D5.11's ramp_steps default (~0.1 s at the 0.02 s
  control step) for rough comparability. Ignored when
  condition=baseline."""
  out_dir: str = ""
  debug: bool = False
  """If True, open a dynamic native MuJoCo viewer. NOTE: D5.11 found
  this viewer mode diverges from the headless rollout and crashes on
  teardown in this environment -- use with that caveat."""


def run(cfg: D513Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_d510_env(d3_cfg, device, max_recovery_steps_override=48)

  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  hold_state = None
  if cfg.condition == "entry_hold":
    hold_state = install_entry_hold(act, cfg.hold_steps, thigh_i, calf_i)

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
    "terminated_step": None,
    "terminated_reason": None,
  }

  def read_row(step: int, new_phase: int) -> dict:
    qpos = entity.data.joint_pos[0]
    qvel = entity.data.joint_vel[0]
    foot_pos = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    base_pos = entity.data.root_link_pos_w[0]
    base_quat = entity.data.root_link_quat_w[0]
    base_lin_vel_w = entity.data.root_link_vel_w[0, :3].clone()
    base_ang_vel_w = entity.data.root_link_vel_w[0, 3:].clone()

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())
    fl_ground_found = bool(
      (env.scene.sensors["fl_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    fr_ground_found = bool(
      (env.scene.sensors["fr_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    rl_ground_found = bool(
      (env.scene.sensors["rl_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    rr_ground_found = bool(
      (env.scene.sensors["rr_ground_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )
    fr_wire_found = bool(
      (env.scene.sensors["fr_wire_contact_diag"].data.found[0].reshape(-1) > 0).any()
    )

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

    holding_now = bool(hold_state["holding"][0]) if hold_state is not None else False

    return {
      "step": step,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
      "holding": holding_now,
      "hip_pos": float(qpos[hip_i]),
      "thigh_pos": float(qpos[thigh_i]),
      "thigh_vel": float(qvel[thigh_i]),
      "calf_pos": float(qpos[calf_i]),
      "calf_vel": float(qvel[calf_i]),
      "foot_z": float(foot_pos[2]),
      "foot_height_above_wire": float(act.foot_height_above_wire[0]),
      "base_x": float(base_pos[0]),
      "base_y": float(base_pos[1]),
      "base_z": float(base_pos[2]),
      "base_vz_world": float(base_lin_vel_w[2]),
      "base_quat_w": float(base_quat[0]),
      "base_ang_vel_x_world": float(base_ang_vel_w[0]),
      "base_ang_vel_y_world": float(base_ang_vel_w[1]),
      "base_ang_vel_z_world": float(base_ang_vel_w[2]),
      "wire_in_contact": wire_in_contact,
      "fl_ground_contact": fl_ground_found,
      "fr_ground_contact": fr_ground_found,
      "rl_ground_contact": rl_ground_found,
      "rr_ground_contact": rr_ground_found,
      "fr_wire_contact": fr_wire_found,
      "fl_hip_wire_contact_raw": hip_wire_contact_now,
      "fl_calf_wire_contact_raw": calf_wire_contact_now,
      "thigh_vel_cmd": float(act.hybrid_thigh_vel_ctrl[0]),
      "calf_torque_cmd": float(act.hybrid_calf_torque_ctrl[0]),
      "aux_thigh_actuator_force_actual": float(
        env.sim.data.actuator_force[0, act._aux_thigh_ctrl_id]
      ),
      "aux_calf_actuator_force_actual": float(
        env.sim.data.actuator_force[0, act._aux_calf_ctrl_id]
      ),
      "pos_thigh_actuator_force_actual": float(
        env.sim.data.actuator_force[0, act._pos_thigh_ctrl_id]
      ),
      "pos_calf_actuator_force_actual": float(
        env.sim.data.actuator_force[0, act._pos_calf_ctrl_id]
      ),
      "calf_forward_dist": float(act.calf_forward_dist[0]),
      "calf_height_above_wire": float(act.calf_height_above_wire[0]),
      "calf_clearance_achieved": bool(act._calf_clearance_achieved[0]),
      "foot_crossed_achieved": bool(act._foot_crossed_achieved[0]),
      "crossing_achieved": bool(act._crossing_achieved[0]),
      "recovery_failed_timeout_flag": bool(act._recovery_failed_timeout[0]),
    }

  def do_step_pre(step: int) -> int:
    """D5.15: pre-step bookkeeping, split out of the old combined
    `do_step` so the viewer path (below) can invoke it WITHOUT calling
    `env.step()` itself. Must be called with env state reflecting the
    END of the PREVIOUS step (or the initial reset state, for step=0)
    -- i.e., strictly BEFORE this step's own `env.step()` call."""
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    return int(act.phase[0])

  def do_step_post(step: int, prev_phase: int) -> dict:
    """D5.15: post-step row construction, split out of the old combined
    `do_step`. Must be called with env state reflecting the state
    immediately AFTER this step's `env.step()` has executed exactly
    once. Unchanged in content/semantics from the original inline body."""
    new_phase = int(act.phase[0])
    if prev_phase == 1 and new_phase == 2:
      state["exit_line"] = (
        f"EXIT RECOVERY step={step} elapsed={int(act.recovery_elapsed[0])} "
        f"timeout={bool(act.recovery_failed_timeout[0])}"
      )
    row = read_row(step, new_phase)
    if act._calf_clearance_achieved[0] and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if act._foot_crossed_achieved[0] and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step
    return row

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
      """D5.15 FIX: this callback must NOT call `env.step()` itself --
      see `d5_3_extended_recovery_window.py`'s `viewer_policy` docstring
      for the full rationale (the pre-fix version here had the same
      double-stepping bug: it called `do_step`, which itself calls
      `env.step()`, and then returned an action the viewer stepped
      AGAIN). This callback is now stateful across calls, finalizing the
      PREVIOUS actual step's row (via `do_step_post`, using the env
      state the viewer's own prior `env.step()` call -- or the initial
      reset -- already produced) before performing `do_step_pre` for the
      step about to be executed and returning the action for the viewer
      to apply via its own, single `env.step()` call."""
      if pending["step"] is not None:
        row = do_step_post(pending["step"], pending["prev_phase"])
        rows.append(row)
        print(
          f"[D5.13] step={pending['step']} phase={row['phase']} holding={row['holding']} "
          f"base_z={row['base_z']:.4f} thigh_vel_cmd={row['thigh_vel_cmd']:.3f} "
          f"calf_torque_cmd={row['calf_torque_cmd']:.3f} fl_ground={row['fl_ground_contact']}"
        )
        if pending["step"] + 1 >= cfg.max_steps or bool(env.reset_buf[0]):
          if bool(env.reset_buf[0]):
            reason = "fell_over" if bool(env.reset_terminated[0]) else "time_out"
            state["terminated_step"] = pending["step"]
            state["terminated_reason"] = reason
            print(f"[D5.13] env0 TERMINATED at step={pending['step']} reason={reason}")
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
      if bool(env.reset_buf[0]):
        reason = "fell_over" if bool(env.reset_terminated[0]) else "time_out"
        state["terminated_step"] = step
        state["terminated_reason"] = reason
        print(f"[D5.13] env0 TERMINATED at step={step} reason={reason}")
        break

  env.close()

  if not rows:
    print(
      "[D5.13] WARNING: no rows recorded (viewer likely exited before any step completed)."
    )
    return out_dir

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"condition={cfg.condition}\n")
    f.write(f"hold_steps={cfg.hold_steps}\n")
    f.write(f"seed={cfg.seed}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
    f.write(f"terminated_step={state['terminated_step']}\n")
    f.write(f"terminated_reason={state['terminated_reason']}\n")
  print(f"[D5.13] wrote {csv_path}")
  print(f"  {state['enter_line']}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  print(
    f"  terminated_step={state['terminated_step']} reason={state['terminated_reason']}"
  )
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D513Config)
  run(cfg)


if __name__ == "__main__":
  main()
