"""D3 Test D: multi-environment isolation check.

Runs 2 envs, one with affected_leg_idx forced into RECOVERY and the other
kept in NORMAL, and verifies that env B's actuator gains/joint behavior
are completely unaffected by env A's recovery state.
"""

import mjlab.tasks  # noqa: F401
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import PHASE_RECOVERY, PHASE_NORMAL_WALK
from mjlab.envs import ManagerBasedRlEnv
import torch

cfg = go2_wire_recovery_residual_env_cfg(
  affected_leg_idx=0, num_envs=2, recovery_mode="hybrid", policy_device="cpu", play=True
)
cfg.sim.device = "cpu"
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
act = env.action_manager._terms["joint_pos"]
env.reset()

zero = torch.zeros(2, 3)
for _ in range(5):
  env.step(zero)

# Force ONLY env 0 into RECOVERY.
#
# NOTE: we deliberately do NOT trigger this via env.step(), because
# process_actions() calls self._update_fsm() first, and _update_fsm()
# unconditionally overwrites `self.newly_entangled = entering_recovery`
# (recomputed fresh from `is_normal & in_contact`). Since we've already
# forced phase to RECOVERY, `is_normal` is False and that overwrite would
# silently clobber our manual True -> False before _apply_hybrid_actuators()
# ever reads it. So we call the toggle method directly here instead.
act._phase[0] = PHASE_RECOVERY
act._phase[1] = PHASE_NORMAL_WALK
act._recovery_elapsed[:] = 0
act.newly_entangled[:] = torch.tensor([True, False])
act._apply_hybrid_actuators()
act.newly_entangled[:] = False

for _ in range(10):
  env.step(zero)

gainprm_thigh_env0 = env.sim.model.actuator_gainprm[
  0, act._pos_thigh_ctrl_id, :3
].tolist()
gainprm_thigh_env1 = env.sim.model.actuator_gainprm[
  1, act._pos_thigh_ctrl_id, :3
].tolist()
aux_gain_env0 = env.sim.model.actuator_gainprm[0, act._aux_thigh_ctrl_id, :3].tolist()
aux_gain_env1 = env.sim.model.actuator_gainprm[1, act._aux_thigh_ctrl_id, :3].tolist()

print(
  "RESULT env0_thigh_pos_gain (should be 0,0,0 -- in RECOVERY):", gainprm_thigh_env0
)
print(
  "RESULT env1_thigh_pos_gain (should be [20,0,0] -- unaffected):", gainprm_thigh_env1
)
print("RESULT env0_aux_thigh_gain (should be nonzero -- active):", aux_gain_env0)
print("RESULT env1_aux_thigh_gain (should be 0,0,0 -- inactive):", aux_gain_env1)
print(
  "RESULT isolation_ok:",
  gainprm_thigh_env1[0] == 20.0
  and aux_gain_env1[0] == 0.0
  and gainprm_thigh_env0[0] == 0.0,
)

env.close()
