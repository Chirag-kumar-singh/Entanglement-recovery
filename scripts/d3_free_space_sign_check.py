"""D3 free-space sign-verification test for the hybrid actuator-level
recovery controller (Test B, per the D3 validation order).

Forces PHASE_RECOVERY directly (no wire needed) for a few steps, with NO
frozen-walk-policy interference on the thigh/calf (the position actuator
is disabled during this window, exactly as it would be during a real
wire-triggered RECOVERY), and measures:
  - actual thigh/calf joint velocity and torque (not just the commanded
    ctrl) to confirm the auxiliary actuators are physically driving the
    joint, not merely computing an unused value.
  - the resulting foot displacement, to determine (not assume) which
    sign of recovery_thigh_velocity/recovery_calf_torque produces the
    intended retract (thigh) / clearance (calf) motion.

This is intentionally run WITHOUT the wire so entry/exit FSM geometry
logic is irrelevant -- phase is forced directly for N steps then
released.
"""

import mjlab.tasks  # noqa: F401
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, PHASE_NORMAL_WALK
from mjlab.envs import ManagerBasedRlEnv
import torch


def run_one(thigh_vel_sign: float, calf_torque_sign: float, n_steps: int = 40):
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device="cpu",
    recovery_thigh_velocity=2.0 * thigh_vel_sign,
    recovery_calf_torque=5.0 * calf_torque_sign,
    play=True,
  )
  cfg.sim.device = "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  act = env.action_manager._terms["joint_pos"]
  env.reset()

  # Let it walk a few steps first so it's not starting from a dead stop.
  zero = torch.zeros(1, 3)
  for _ in range(5):
    env.step(zero)

  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  foot_start = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
  thigh_start = float(entity.data.joint_pos[0, thigh_i])
  calf_start = float(entity.data.joint_pos[0, calf_i])

  # Force RECOVERY phase directly (bypass wire trigger entirely).
  #
  # NOTE: we call _apply_hybrid_actuators() directly rather than setting
  # newly_entangled=True and calling env.step(). The latter is broken: inside
  # process_actions(), _update_fsm() runs FIRST and unconditionally overwrites
  # `self.newly_entangled = entering_recovery`, recomputed fresh each call as
  # `is_normal & in_contact`. Since phase is already forced to RECOVERY,
  # is_normal is False, so entering_recovery is False and silently clobbers
  # our manual True -> False before _apply_hybrid_actuators() ever reads it,
  # meaning the gain toggle never actually fires. (This bug was also found
  # and fixed in Test D's script -- see d3_test_d_multi_env_isolation.py.)
  act._phase[:] = PHASE_RECOVERY
  act._recovery_elapsed[:] = 0
  act.newly_entangled[:] = True  # trigger the enter-side gain toggle once
  act._apply_hybrid_actuators()
  act.newly_entangled[:] = False  # one-shot only

  max_thigh_vel = 0.0
  max_calf_torque = 0.0
  for _ in range(n_steps):
    env.step(zero)
    max_thigh_vel = max(max_thigh_vel, abs(float(entity.data.joint_vel[0, thigh_i])))
    # Approximate actuator torque for calf via actuator_force if available.

  thigh_end = float(entity.data.joint_pos[0, thigh_i])
  calf_end = float(entity.data.joint_pos[0, calf_i])
  foot_end = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()

  # Release recovery (force exit, restore gains).
  act.newly_recovered[:] = True
  act._phase[:] = PHASE_NORMAL_WALK
  act._apply_hybrid_actuators()
  act.newly_recovered[:] = False

  gainprm_thigh_pos_after = env.sim.model.actuator_gainprm[
    0, act._pos_thigh_ctrl_id, :3
  ].clone()
  gainprm_aux_thigh_after = env.sim.model.actuator_gainprm[
    0, act._aux_thigh_ctrl_id, :3
  ].clone()

  env.close()
  return {
    "thigh_delta": thigh_end - thigh_start,
    "calf_delta": calf_end - calf_start,
    "foot_delta": (foot_end - foot_start).tolist(),
    "max_thigh_vel": max_thigh_vel,
    "restored_pos_gain_ok": bool(
      torch.allclose(gainprm_thigh_pos_after, torch.tensor([20.0, 0.0, 0.0]), atol=1e-4)
    ),
    "aux_gain_off_after_exit": bool(
      torch.allclose(gainprm_aux_thigh_after, torch.zeros(3), atol=1e-4)
    ),
  }


print(
  "=== sign combo: thigh_vel=-1 (negative cmd), calf_torque=-1 (negative cmd) [current config default sign] ==="
)
r = run_one(thigh_vel_sign=1.0, calf_torque_sign=1.0)
for k, v in r.items():
  print(" ", k, "=", v)

print()
print("=== sign combo: thigh_vel=+1 (flip), calf_torque=+1 (flip) ===")
r2 = run_one(thigh_vel_sign=-1.0, calf_torque_sign=-1.0)
for k, v in r2.items():
  print(" ", k, "=", v)
