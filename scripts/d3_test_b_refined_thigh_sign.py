"""D3 Test B refinement: isolated, instantaneous-signal thigh/hip
velocity-sign verification for the hybrid actuator-level recovery
controller.

Does NOT modify the recovery controller, FSM, velocity magnitude, calf
torque, walking policy, or wire environment. Only the *measurement*
harness changes relative to the original `d3_free_space_sign_check.py`:
instead of inferring direction from net joint-position change over 16
steps (contaminated by the frozen walking gait), this script records
instantaneous qvel / actuator_force / ctrl / gain state immediately
before and in the first few steps after the auxiliary thigh velocity
actuator is enabled, for two otherwise-identical runs that differ only
in the sign of `recovery_thigh_velocity`.

Both runs use the same seed, same number of warm-up walking steps, same
affected leg, and the gain toggle is triggered via a direct call to
`act._apply_hybrid_actuators()` (NOT via `env.step()`, which would let
`_update_fsm()` clobber the manually-set `newly_entangled` flag before
the toggle logic runs -- see the Test D / original Test B post-mortem).
"""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import PHASE_NORMAL_WALK, PHASE_RECOVERY

N_WARMUP = 5
N_SAMPLE_STEPS = 3  # short, isolated post-activation window
SEED = 0


def run_one(thigh_vel_sign: float):
  torch.manual_seed(SEED)
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device="cpu",
    recovery_thigh_velocity=2.0 * thigh_vel_sign,
    # Calf torque sign already conclusively verified; hold it fixed and
    # irrelevant to this measurement by leaving it at its validated
    # default (do not touch per instructions).
    play=True,
  )
  cfg.sim.device = "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  act = env.action_manager._terms["joint_pos"]
  env.reset()

  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")

  zero = torch.zeros(1, 3)
  for _ in range(N_WARMUP):
    env.step(zero)

  sim = env.sim

  # --- State immediately BEFORE activation. ---
  qvel_before = float(entity.data.joint_vel[0, thigh_i])
  qpos_before = float(entity.data.joint_pos[0, thigh_i])
  foot_before = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
  pos_gain_before = sim.model.actuator_gainprm[0, act._pos_thigh_ctrl_id, :3].clone()
  pos_bias_before = sim.model.actuator_biasprm[0, act._pos_thigh_ctrl_id, :3].clone()
  aux_gain_before = sim.model.actuator_gainprm[0, act._aux_thigh_ctrl_id, :3].clone()
  # The frozen walking policy's own thigh ctrl target for context (what
  # the position actuator *would* have been tracking had it stayed
  # enabled -- read from the action manager's processed/raw action for
  # this joint, for informational purposes only).
  walk_policy_thigh_cmd = float(act._raw_actions[0, thigh_i])

  # --- Trigger entry directly (bypass _update_fsm(), see module docstring). ---
  act._phase[:] = PHASE_RECOVERY
  act._recovery_elapsed[:] = 0
  act.newly_entangled[:] = True
  act._apply_hybrid_actuators()
  act.newly_entangled[:] = False

  # State immediately AFTER the gain toggle but BEFORE any physics step.
  pos_gain_after_toggle = sim.model.actuator_gainprm[
    0, act._pos_thigh_ctrl_id, :3
  ].clone()
  pos_bias_after_toggle = sim.model.actuator_biasprm[
    0, act._pos_thigh_ctrl_id, :3
  ].clone()
  aux_gain_after_toggle = sim.model.actuator_gainprm[
    0, act._aux_thigh_ctrl_id, :3
  ].clone()
  aux_bias_after_toggle = sim.model.actuator_biasprm[
    0, act._aux_thigh_ctrl_id, :3
  ].clone()
  commanded_thigh_vel_ctrl = float(act.hybrid_thigh_vel_ctrl[0])

  samples = []
  for step_i in range(N_SAMPLE_STEPS):
    # Re-apply the per-step command (in_recovery is already latched via
    # self._phase, so this recomputes the same ctrl each step -- mirrors
    # what process_actions()'s hybrid branch would do, without invoking
    # _update_fsm()).
    act._apply_hybrid_actuators()
    env.step(zero)
    qvel = float(entity.data.joint_vel[0, thigh_i])
    qpos = float(entity.data.joint_pos[0, thigh_i])
    actuator_force_aux = float(sim.data.actuator_force[0, act._aux_thigh_ctrl_id])
    actuator_force_pos = float(sim.data.actuator_force[0, act._pos_thigh_ctrl_id])
    ctrl_aux = float(sim.data.ctrl[0, act._aux_thigh_ctrl_id])
    foot_now = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
    samples.append(
      dict(
        step=step_i,
        qvel=qvel,
        qpos=qpos,
        actuator_force_aux=actuator_force_aux,
        actuator_force_pos=actuator_force_pos,
        ctrl_aux=ctrl_aux,
        foot_delta=(foot_now - foot_before).tolist(),
      )
    )

  # Release recovery (restore gains), to leave the env in a clean state.
  act.newly_recovered[:] = True
  act._phase[:] = PHASE_NORMAL_WALK
  act._apply_hybrid_actuators()
  act.newly_recovered[:] = False
  pos_gain_restored = sim.model.actuator_gainprm[0, act._pos_thigh_ctrl_id, :3].clone()
  aux_gain_restored = sim.model.actuator_gainprm[0, act._aux_thigh_ctrl_id, :3].clone()

  env.close()

  return dict(
    thigh_vel_sign=thigh_vel_sign,
    qvel_before=qvel_before,
    qpos_before=qpos_before,
    walk_policy_thigh_cmd=walk_policy_thigh_cmd,
    pos_gain_before=pos_gain_before.tolist(),
    pos_bias_before=pos_bias_before.tolist(),
    aux_gain_before=aux_gain_before.tolist(),
    pos_gain_after_toggle=pos_gain_after_toggle.tolist(),
    pos_bias_after_toggle=pos_bias_after_toggle.tolist(),
    aux_gain_after_toggle=aux_gain_after_toggle.tolist(),
    aux_bias_after_toggle=aux_bias_after_toggle.tolist(),
    commanded_thigh_vel_ctrl=commanded_thigh_vel_ctrl,
    samples=samples,
    pos_gain_restored=pos_gain_restored.tolist(),
    aux_gain_restored=aux_gain_restored.tolist(),
  )


def report(label, r):
  print(
    f"\n===== {label} (recovery_thigh_velocity sign = {r['thigh_vel_sign']:+.0f}) ====="
  )
  print("  walk_policy_thigh_cmd (context only)     =", r["walk_policy_thigh_cmd"])
  print("  qvel_before activation                   =", r["qvel_before"])
  print(
    "  pos actuator gain BEFORE (should be active, e.g. [20,0,0]) =",
    r["pos_gain_before"],
  )
  print(
    "  pos actuator gain AFTER TOGGLE (should be [0,0,0])         =",
    r["pos_gain_after_toggle"],
  )
  print(
    "  pos actuator bias AFTER TOGGLE (should be [0,0,0])         =",
    r["pos_bias_after_toggle"],
  )
  print(
    "  aux  actuator gain BEFORE (should be [0,0,0], inert)       =",
    r["aux_gain_before"],
  )
  print(
    "  aux  actuator gain AFTER TOGGLE (should be nonzero kd)     =",
    r["aux_gain_after_toggle"],
  )
  print(
    "  aux  actuator bias AFTER TOGGLE (should be nonzero -kd)    =",
    r["aux_bias_after_toggle"],
  )
  print(
    "  commanded_thigh_vel_ctrl (actual ctrl written this step)   =",
    r["commanded_thigh_vel_ctrl"],
  )
  print("  -- per-step samples (instantaneous) --")
  for s in r["samples"]:
    print(
      f"    step {s['step']}: qvel={s['qvel']:+.5f}  "
      f"actuator_force_aux={s['actuator_force_aux']:+.5f}  "
      f"actuator_force_pos={s['actuator_force_pos']:+.5f}  "
      f"ctrl_aux={s['ctrl_aux']:+.5f}  "
      f"foot_delta(x,y,z)={[f'{v:+.5f}' for v in s['foot_delta']]}"
    )
  print(
    "  pos actuator gain RESTORED after exit (should match BEFORE) =",
    r["pos_gain_restored"],
  )
  print(
    "  aux  actuator gain RESTORED after exit (should be [0,0,0])   =",
    r["aux_gain_restored"],
  )


r_pos = run_one(thigh_vel_sign=1.0)
report("TEST B1: thigh_vel_sign=+1", r_pos)

r_neg = run_one(thigh_vel_sign=-1.0)
report("TEST B2: thigh_vel_sign=-1", r_neg)

print("\n===== CROSS-CHECK =====")
qvel_b1_step0 = r_pos["samples"][0]["qvel"]
qvel_b2_step0 = r_neg["samples"][0]["qvel"]
print(f"B1 step0 qvel = {qvel_b1_step0:+.5f}   B2 step0 qvel = {qvel_b2_step0:+.5f}")
print("Opposite sign qvel response:", (qvel_b1_step0 > 0) != (qvel_b2_step0 > 0))
force_b1 = r_pos["samples"][0]["actuator_force_aux"]
force_b2 = r_neg["samples"][0]["actuator_force_aux"]
print(
  f"B1 step0 actuator_force_aux = {force_b1:+.5f}   B2 step0 actuator_force_aux = {force_b2:+.5f}"
)
print("Opposite sign actuator force:", (force_b1 > 0) != (force_b2 > 0))
foot_x_b1 = r_pos["samples"][-1]["foot_delta"][0]
foot_x_b2 = r_neg["samples"][-1]["foot_delta"][0]
foot_z_b1 = r_pos["samples"][-1]["foot_delta"][2]
foot_z_b2 = r_neg["samples"][-1]["foot_delta"][2]
print(f"B1 final foot_delta x,z = {foot_x_b1:+.5f}, {foot_z_b1:+.5f}")
print(f"B2 final foot_delta x,z = {foot_x_b2:+.5f}, {foot_z_b2:+.5f}")
