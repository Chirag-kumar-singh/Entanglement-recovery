"""Minimal Test B sanity check using the CORRECTED default config
(recovery_thigh_velocity = -2.0, see actions.py docstring for the sign
correction derivation). Does not sweep signs -- that was already
conclusively established in scripts/d3_test_b_refined_thigh_sign.py.

Confirms, using the actual default config (no override):
  - position actuator disabled during recovery
  - auxiliary velocity actuator active during recovery
  - commanded thigh ctrl == +2.0
  - measured qvel > 0
  - foot motion consistent with retract direction
"""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import PHASE_NORMAL_WALK, PHASE_RECOVERY

torch.manual_seed(0)
cfg = go2_wire_recovery_residual_env_cfg(
  affected_leg_idx=0, num_envs=1, recovery_mode="hybrid", policy_device="cpu", play=True
)
cfg.sim.device = "cpu"
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
act = env.action_manager._terms["joint_pos"]
env.reset()

entity = act._entity
thigh_i = list(entity.joint_names).index("FL_thigh_joint")
sim = env.sim

zero = torch.zeros(1, 3)
for _ in range(5):
  env.step(zero)

foot_before = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()

act._phase[:] = PHASE_RECOVERY
act._recovery_elapsed[:] = 0
act.newly_entangled[:] = True
act._apply_hybrid_actuators()
act.newly_entangled[:] = False

pos_gain = sim.model.actuator_gainprm[0, act._pos_thigh_ctrl_id, :3].tolist()
aux_gain = sim.model.actuator_gainprm[0, act._aux_thigh_ctrl_id, :3].tolist()
ctrl_written = float(act.hybrid_thigh_vel_ctrl[0])

for _ in range(3):
  act._apply_hybrid_actuators()
  env.step(zero)

qvel = float(entity.data.joint_vel[0, thigh_i])
foot_after = entity.data.site_pos_w[0, act._foot_site_idx, :].clone()
foot_dx = float((foot_after - foot_before)[0])

act.newly_recovered[:] = True
act._phase[:] = PHASE_NORMAL_WALK
act._apply_hybrid_actuators()
act.newly_recovered[:] = False

print("RESULT pos_gain_disabled (expect [0,0,0]):", pos_gain)
print("RESULT aux_gain_active (expect nonzero):", aux_gain)
print("RESULT commanded_ctrl (expect +2.0):", ctrl_written)
print("RESULT measured_qvel (expect >0):", qvel)
print("RESULT foot_dx_over_3steps:", foot_dx)
print(
  "RESULT PASS:",
  pos_gain[0] == 0.0
  and aux_gain[0] != 0.0
  and abs(ctrl_written - 2.0) < 1e-6
  and qvel > 0.0,
)
env.close()
