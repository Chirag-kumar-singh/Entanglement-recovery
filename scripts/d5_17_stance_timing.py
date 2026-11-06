"""D5.17 -- Recovery entry timing: stance-deferred actuator handoff vs.
immediate handoff.

Diagnostic-only, non-invasive experiment. Reuses D5.3's exact env-build
path (`build_ablation_env`, same `go2_wire_recovery_residual_env_cfg`
factory call, same isolated FL wire, same 48-step recovery cap, same
frozen checkpoint, same `fl_ground_contact_diag` sensor already added by
that function) and the D5.15/D5.16-corrected viewer stepping pattern
(viewer callback never calls `env.step()` itself).

============================================================================
BEFORE-IMPLEMENTATION ANALYSIS (required by the D5.17 task before running
anything -- documented here, not just in the report, so the rationale
travels with the code)
============================================================================

1. INSPECTION: are the FSM trigger and the actuator handoff the SAME
   event in the current production code?

   YES, confirmed by direct source reading of
   `ResidualRecoveryAction.process_actions` (src/mjlab/tasks/recovery/
   mdp/actions.py): it calls `self._update_fsm()` (which sets
   `self._phase = PHASE_RECOVERY` and `self.newly_entangled` the instant
   wire contact is detected) and THEN, in the same call, for
   `recovery_mode == "hybrid"`, calls `self._apply_hybrid_actuators()`,
   which disables the thigh/calf POSITION actuators and enables the
   auxiliary thigh-velocity/calf-torque actuators keyed OFF THE SAME
   `self.newly_entangled` flag just set by `_update_fsm`. Both the phase
   transition (trigger) and the actuator-level handoff (disable position
   actuators + enable auxiliary actuators) currently happen in the exact
   same environment step, driven by the exact same boolean mask. This
   experiment's Condition B deliberately separates these two events,
   which is why both a "trigger step" and a "handoff step" are logged
   and reported separately below.

2. RELIABLE FL STANCE SIGNAL (not velocity-based):
   Reuses the EXACT SAME `fl_ground_contact_diag` `ContactSensorCfg`
   already added by D5.3's `build_ablation_env` (geom-vs-terrain
   contact, `reduce="netforce"`, `fields=("found","force")`) -- this is
   a genuine MuJoCo contact-detection signal (`sensor.data.found`),
   identical in construction to the sensor D5.16's audit already used to
   establish that FL is airborne (`fl_ground_contact=False`) throughout
   the entire baseline RECOVERY window (steps 44-88, confirmed in
   logs/d5_16_trajectory_audit/report.md). NOT inferred from foot
   velocity, joint velocity, or any kinematic proxy.

3. BOUNDED WAIT AND FALLBACK (values proposed and documented here,
   BEFORE the first run, per the task's explicit requirement):

   `stance_max_wait_steps = 20` RECOVERY steps.

   Rationale: `ResidualRecoveryActionCfg.max_recovery_steps`'s own
   docstring (unchanged, read verbatim from source) states its value
   of 16 was "chosen as roughly 2x the ~9-step natural FL swing
   duration measured for plain walking." This experiment reuses that
   SAME established reference point (~9-step natural swing) and the
   SAME "~2x" conservative-margin convention already used elsewhere in
   this codebase, rather than inventing a new, unrelated number: if the
   affected leg is simply left under the frozen walking policy's own
   gait-phase-driven position command (exactly what happens during the
   deferred-handoff waiting interval -- see point 4 below), it should
   complete its own natural swing-to-stance transition within roughly
   9-10 steps under normal circumstances; doubling that to 20 steps is
   a generous, non-arbitrary bound that still leaves most of the
   existing 48-step recovery budget (`max_recovery_steps_override=48`,
   UNCHANGED from D5.3/D5.16) for the actual retract+cross maneuver
   after handoff.

   FALLBACK: if FL ground contact is not observed within
   `stance_max_wait_steps` RECOVERY-elapsed steps, the SAME handoff
   (disable position actuators, enable auxiliary actuators) that would
   have fired immediately in Condition A is instead forced at that
   bound step, exactly reproducing baseline behavior merely time-
   shifted. This guarantees Condition B can never get permanently stuck
   waiting and always eventually applies the existing, unmodified
   actuator-level recovery behavior within a bounded number of steps.

4. WHAT HAPPENS DURING THE WAIT (deliberately: NOTHING NEW):
   While `holding=True` for an env, this script's monkeypatched
   `_apply_hybrid_actuators` override does NOT disable the position
   actuators and does NOT enable the auxiliary actuators -- it simply
   skips that branch for the held envs. No ctrl override, no fixed
   target, no blending, no new control law of any kind is introduced
   (this experiment's README/constraints explicitly forbid "actuator
   blending or another hold strategy" -- note this is intentionally
   DIFFERENT from D5.13's entry_hold, which actively overrode ctrl to a
   FIXED captured target; here, nothing is overridden at all). Because
   `process_actions`'s hybrid branch already composes `final_raw` as
   `a_walk_raw` (frozen policy's own 12-dim output) with a zero residual
   on the affected leg, the simple absence of the disable call means the
   affected leg's thigh/calf position actuators continue tracking
   whatever the frozen walking policy would have commanded for that leg
   if it were not in RECOVERY at all -- i.e. the leg keeps walking
   normally through the policy's own gait cycle until EITHER it reaches
   stance (detected via the sensor) OR the bounded wait expires. This is
   the smallest possible change: it modifies only the TIMING of the one
   existing `if newly_entangled: disable actuators` branch, with no new
   control law for the interim.

5. DOES THE ORIGINAL CROSSING/RELEASE FSM STILL FUNCTION CORRECTLY WHEN
   TRIGGER AND HANDOFF DIFFER?
   YES, confirmed by direct source reading of `_update_fsm`: the
   two-stage crossing condition (`calf_clearance_achieved`,
   `foot_crossed_achieved`, `crossing_achieved`) and the release counter
   are computed EVERY step purely from CURRENT GEOMETRY (calf/foot
   position relative to the wire's live centerline) and the wire-contact
   sensor, with NO reference anywhere in that method to which actuators
   are currently active. `recovery_elapsed`, `max_recovery_steps` (the
   cap), and the phase-exit logic are likewise actuator-state-agnostic.
   Deferring the actuator handoff therefore cannot, by construction,
   break or bypass the crossing/release/exit logic -- it only changes
   what is physically driving the leg's motion during the early part of
   the (unmodified) RECOVERY window. The one real interaction to measure
   (not just assert) is whether `recovery_elapsed` being "spent" on the
   wait leaves enough of the 48-step budget for the subsequent
   retract+cross maneuver -- this is why both conditions are run to
   completion and crossing success is explicitly reported for both.

============================================================================
IMPLEMENTATION (non-invasive, instance-level monkeypatch -- same pattern
already used and documented in scripts/d5_13_entry_hold_ablation.py's
`install_entry_hold`; NO production file is edited)
============================================================================
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
from d5_3_extended_recovery_window import build_ablation_env  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

PHASE_RECOVERY = 1
PHASE_POST_RECOVERY = 2

STANCE_MAX_WAIT_STEPS = 20
"""See module docstring point 3 for the full rationale (~2x the ~9-step
natural FL swing duration, same convention as `max_recovery_steps`'s own
docstring)."""


def install_stance_deferred_handoff(act, stance_max_wait_steps: int) -> dict:
  """Install the stance-deferred variant of `_apply_hybrid_actuators`.
  Returns a state dict (exposed for logging) tracking, per env:
  `holding` (bool), `trigger_step` (the step `newly_entangled` fired),
  `handoff_step` (the step the position-actuator disable/aux-enable
  actually executed -- may equal `trigger_step` under
  `condition="immediate"`, or be later under `condition="stance_
  deferred"`), and `handoff_reason` ("stance_detected" or
  "bound_reached")."""
  device = act.device
  num_envs = act.num_envs

  state = {
    "holding": torch.zeros(num_envs, dtype=torch.bool, device=device),
    "trigger_step": torch.full((num_envs,), -1, dtype=torch.long, device=device),
    "handoff_step": torch.full((num_envs,), -1, dtype=torch.long, device=device),
    "handoff_reason": [""] * num_envs,
    "current_step": -1,
  }

  def stance_deferred_apply_hybrid_actuators() -> None:
    in_recovery = act._phase == PHASE_RECOVERY
    sim = act._env.sim
    step_now = state["current_step"]

    enter_ids = act.newly_entangled.nonzero(as_tuple=True)[0]
    exit_ids = act.newly_recovered.nonzero(as_tuple=True)[0]

    if enter_ids.numel() > 0:
      state["holding"][enter_ids] = True
      for i in enter_ids.tolist():
        state["trigger_step"][i] = step_now
      # Deliberately do NOT disable position actuators / enable aux here
      # -- deferred to the transition block below (point 4 in the module
      # docstring: no new control law during the wait).
      act._peak_armed[enter_ids] = False
      act._peak_running_max[enter_ids] = act.foot_height_above_wire[enter_ids]
      act._peak_entry_height[enter_ids] = act.foot_height_above_wire[enter_ids]
      act._peak_decline_counter[enter_ids] = 0
      act._peak_triggered[enter_ids] = False
      act.peak_trigger_step[enter_ids] = -1

    # Stance detection: reuse the EXACT existing fl_ground_contact_diag
    # sensor (added by build_ablation_env, same construction D5.16
    # audited).
    fl_ground = act._env.scene.sensors["fl_ground_contact_diag"]
    fl_ground_found = (fl_ground.data.found.reshape(num_envs, -1) > 0).any(dim=-1)

    elapsed = act._recovery_elapsed
    stance_detected = state["holding"] & in_recovery & fl_ground_found
    bound_reached = state["holding"] & in_recovery & (elapsed >= stance_max_wait_steps)
    transition_mask = stance_detected | bound_reached
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
      sim.model.actuator_biasprm[transition_ids, act._aux_thigh_ctrl_id, 2] = (
        -act.cfg.recovery_thigh_velocity_kd
      )
      act.hybrid_thigh_pos_gain_active[transition_ids] = False
      state["holding"][transition_ids] = False
      for i in transition_ids.tolist():
        # NOTE: +1 here to match the SAME delayed labeling convention
        # already used (via `do_step_pre`'s one-step-delayed read of
        # `newly_entangled`) for `trigger_step` throughout D5.3/D5.13/
        # D5.16 (e.g. "ENTER RECOVERY step=N" prints) -- this is a
        # physical env.step() call index + 1, NOT a different
        # convention. Keeping both labels on the SAME convention means
        # "trigger_step == handoff_step" genuinely means "same physical
        # step," and any difference directly reports the number of
        # physical steps the handoff was deferred by.
        state["handoff_step"][i] = step_now + 1
        state["handoff_reason"][i] = (
          "stance_detected" if bool(stance_detected[i]) else "bound_reached"
        )

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

    # Auxiliary commands: zero while holding (gain disabled anyway), full
    # magnitude once handed off -- identical constants to baseline, no
    # ramp, no blend.
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

  act._apply_hybrid_actuators = stance_deferred_apply_hybrid_actuators
  return state


@dataclass
class D517Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 200
  max_recovery_steps_override: int = 48
  condition: Literal["immediate", "stance_deferred"] = "immediate"
  """'immediate' = Condition A, the UNMODIFIED D5.3 baseline (actuator
  handoff fires the same step as the FSM trigger, exactly as in
  production). 'stance_deferred' = Condition B (handoff deferred until
  FL ground contact is confirmed, bounded by `stance_max_wait_steps`)."""
  stance_max_wait_steps: int = STANCE_MAX_WAIT_STEPS
  out_dir: str = ""
  debug: bool = False


def run(cfg: D517Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_ablation_env(
    d3_cfg, device, ablate_hip_wire=False,
    max_recovery_steps_override=cfg.max_recovery_steps_override,
  )
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  hip_i = joint_names.index("FL_hip_joint")
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")

  handoff_state = None
  if cfg.condition == "stance_deferred":
    handoff_state = install_stance_deferred_handoff(act, cfg.stance_max_wait_steps)

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  state = {
    "trigger_step": None,
    "exit_line": None,
    "stage1_first_step": None,
    "stage2_first_step": None,
  }

  def do_step_pre(step: int) -> int:
    if handoff_state is not None:
      handoff_state["current_step"] = step
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
    base_pos_w = entity.data.root_link_pos_w[0].clone()
    base_quat_w = entity.data.root_link_quat_w[0].clone()
    base_lin_vel_b = entity.data.root_link_lin_vel_b[0].clone()
    base_ang_vel_b = entity.data.root_link_ang_vel_b[0].clone()

    foot_rel_w = (foot_pos_w - base_pos_w).unsqueeze(0)
    foot_pos_b = quat_apply_inverse(base_quat_w.unsqueeze(0), foot_rel_w)[0]

    wire_sensor = env.scene.sensors[act.cfg.wire_contact_sensor_name]
    wire_in_contact = bool((wire_sensor.data.found[0] > 0).any())
    fl_ground = env.scene.sensors["fl_ground_contact_diag"]
    fl_ground_found = bool((fl_ground.data.found[0].reshape(-1) > 0).any())
    fr_wire = env.scene.sensors["fr_wire_contact_diag"]
    fr_wire_found = bool((fr_wire.data.found[0].reshape(-1) > 0).any())

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
    holding_now = (
      bool(handoff_state["holding"][0]) if handoff_state is not None else False
    )

    if calf_clearance_achieved and state["stage1_first_step"] is None:
      state["stage1_first_step"] = step
    if foot_crossed_achieved and state["stage2_first_step"] is None:
      state["stage2_first_step"] = step

    return {
        "step": step,
        "phase": new_phase,
        "recovery_elapsed": int(act.recovery_elapsed[0]),
        "pos_actuators_active": pos_actuators_active,
        "holding": holding_now,
        "foot_z": float(foot_pos_w[2]),
        "foot_vz": float(entity.data.site_lin_vel_w[0, act._foot_site_idx, 2]),
        "foot_height_above_wire": foot_height_above_wire,
        "foot_x_base": float(foot_pos_b[0]),
        "foot_y_base": float(foot_pos_b[1]),
        "foot_z_base": float(foot_pos_b[2]),
        "hip_pos": float(qpos[hip_i]),
        "hip_vel": float(qvel[hip_i]),
        "thigh_pos": float(qpos[thigh_i]),
        "thigh_vel": float(qvel[thigh_i]),
        "calf_pos": float(qpos[calf_i]),
        "calf_vel": float(qvel[calf_i]),
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
        "fl_ground_contact": fl_ground_found,
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
      """D5.15/D5.16-pattern fix: never calls env.step() itself."""
      if pending["step"] is not None:
        row = do_step_post(pending["step"], pending["prev_phase"])
        rows.append(row)
        if pending["step"] % 10 == 0 or row["calf_clearance_achieved"] or row["crossing_achieved"]:
          print(
            f"[D5.17] step={pending['step']} phase={row['phase']} "
            f"holding={row['holding']} pos_act={row['pos_actuators_active']} "
            f"calf_forward_dist={row['calf_forward_dist']:.4f} "
            f"stage1={row['calf_clearance_achieved']} stage2={row['foot_crossed_achieved']}"
          )
        if pending["step"] + 1 >= cfg.max_steps:
          raise SystemExit(0)

      step = 0 if pending["step"] is None else pending["step"] + 1
      prev_phase = do_step_pre(step)
      pending["step"] = step
      pending["prev_phase"] = prev_phase
      return torch.zeros(env.action_manager.action.shape, device=device)

    print("[INFO] Debug mode: native viewer (DYNAMIC, real physics). Close window to end run.")
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

  trigger_step = state["trigger_step"]
  handoff_step = (
    int(handoff_state["handoff_step"][0]) if handoff_state is not None else trigger_step
  )
  handoff_reason = (
    handoff_state["handoff_reason"][0] if handoff_state is not None else "immediate"
  )

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"condition={cfg.condition}\n")
    f.write(f"stance_max_wait_steps={cfg.stance_max_wait_steps}\n")
    f.write(f"max_recovery_steps_override={cfg.max_recovery_steps_override}\n")
    f.write(f"trigger_step={trigger_step}\n")
    f.write(f"handoff_step={handoff_step}\n")
    f.write(f"handoff_reason={handoff_reason}\n")
    f.write(f"{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
  print(f"[D5.17] wrote {csv_path}")
  print(f"  condition={cfg.condition} trigger_step={trigger_step} handoff_step={handoff_step} reason={handoff_reason}")
  print(f"  {state['exit_line']}")
  print(f"  stage1_first_latch_step={state['stage1_first_step']}")
  print(f"  stage2_first_latch_step={state['stage2_first_step']}")
  return csv_path


def main() -> None:
  import tyro

  cfg = tyro.cli(D517Config)
  run(cfg)


if __name__ == "__main__":
  main()
