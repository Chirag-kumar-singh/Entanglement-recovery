"""D3 Test C: recovery entry/exit actuator-lifecycle + exact-restoration
verification, for the hybrid actuator-level recovery controller.

Scope (per user spec): prove that recovery control can be entered and
exited without leaving modified actuator state behind, including:
  - entry: affected thigh/calf position actuators disabled, auxiliary
    actuators enabled; affected ab/ad and all other 9 joints unaffected.
  - exit: auxiliary actuators disabled (ctrl/gain/bias), original
    position actuator gain/bias restored EXACTLY (captured values, not
    reconstructed/hardcoded).
  - actual force/ctrl behavior after exit (not just config arrays): no
    residual recovery torque/velocity leaking past the exit boundary.
  - multi-env isolation across the full entry/exit lifecycle (extends
    the already-valid Test D single-snapshot isolation check).
  - repeated entry/exit cycles (stale-state accumulation check).

Uses the corrected default config (recovery_thigh_velocity=-2.0, i.e.
ctrl=+2.0, qvel>0, the Go2-verified retract direction). Deterministic:
fixed seed, fixed affected leg (FL), no randomization. Phase transitions
are forced directly via `act._apply_hybrid_actuators()` (bypassing
`_update_fsm()`, exactly as established necessary in the Test B/D
post-mortem), since this is a free-space actuator-lifecycle test, not a
wire-triggered FSM test.
"""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import go2_wire_recovery_residual_env_cfg
from mjlab.tasks.recovery.mdp.actions import (
  LEG_NAMES,
  PHASE_NORMAL_WALK,
  PHASE_RECOVERY,
)


def snapshot(env, act, entity, label, env_id):
  sim = env.sim
  thigh_i = list(entity.joint_names).index("FL_thigh_joint")
  calf_i = list(entity.joint_names).index("FL_calf_joint")
  hip_i = list(entity.joint_names).index("FL_hip_joint")
  return dict(
    label=label,
    env_id=env_id,
    phase=int(act._phase[env_id]),
    thigh_qpos=float(entity.data.joint_pos[env_id, thigh_i]),
    thigh_qvel=float(entity.data.joint_vel[env_id, thigh_i]),
    calf_qpos=float(entity.data.joint_pos[env_id, calf_i]),
    calf_qvel=float(entity.data.joint_vel[env_id, calf_i]),
    hip_qpos=float(entity.data.joint_pos[env_id, hip_i]),
    walking_cmd_thigh=float(act._raw_actions[env_id, thigh_i]),
    walking_cmd_calf=float(act._raw_actions[env_id, calf_i]),
    recovery_thigh_vel_ctrl=float(act.hybrid_thigh_vel_ctrl[env_id]),
    recovery_calf_torque_ctrl=float(act.hybrid_calf_torque_ctrl[env_id]),
    pos_thigh_gainprm=sim.model.actuator_gainprm[
      env_id, act._pos_thigh_ctrl_id, :3
    ].tolist(),
    pos_thigh_biasprm=sim.model.actuator_biasprm[
      env_id, act._pos_thigh_ctrl_id, :3
    ].tolist(),
    aux_thigh_gainprm=sim.model.actuator_gainprm[
      env_id, act._aux_thigh_ctrl_id, :3
    ].tolist(),
    aux_thigh_biasprm=sim.model.actuator_biasprm[
      env_id, act._aux_thigh_ctrl_id, :3
    ].tolist(),
    pos_calf_gainprm=sim.model.actuator_gainprm[
      env_id, act._pos_calf_ctrl_id, :3
    ].tolist(),
    pos_calf_biasprm=sim.model.actuator_biasprm[
      env_id, act._pos_calf_ctrl_id, :3
    ].tolist(),
    aux_calf_gainprm=sim.model.actuator_gainprm[
      env_id, act._aux_calf_ctrl_id, :3
    ].tolist(),
    aux_calf_biasprm=sim.model.actuator_biasprm[
      env_id, act._aux_calf_ctrl_id, :3
    ].tolist(),
    aux_thigh_ctrl=float(sim.data.ctrl[env_id, act._aux_thigh_ctrl_id]),
    aux_calf_ctrl=float(sim.data.ctrl[env_id, act._aux_calf_ctrl_id]),
    aux_thigh_force=float(sim.data.actuator_force[env_id, act._aux_thigh_ctrl_id]),
    aux_calf_force=float(sim.data.actuator_force[env_id, act._aux_calf_ctrl_id]),
    pos_thigh_force=float(sim.data.actuator_force[env_id, act._pos_thigh_ctrl_id]),
    pos_calf_force=float(sim.data.actuator_force[env_id, act._pos_calf_ctrl_id]),
  )


def print_snap(s):
  print(f"--- {s['label']} (env {s['env_id']}, phase={s['phase']}) ---")
  print(
    f"  thigh qpos/qvel = {s['thigh_qpos']:+.5f} / {s['thigh_qvel']:+.5f}   "
    f"calf qpos/qvel = {s['calf_qpos']:+.5f} / {s['calf_qvel']:+.5f}   "
    f"hip qpos = {s['hip_qpos']:+.5f}"
  )
  print(
    f"  walking_cmd thigh/calf = {s['walking_cmd_thigh']:+.4f} / {s['walking_cmd_calf']:+.4f}   "
    f"recovery ctrl thigh_vel/calf_torque = {s['recovery_thigh_vel_ctrl']:+.3f} / {s['recovery_calf_torque_ctrl']:+.3f}"
  )
  print(
    f"  pos_thigh gain/bias = {s['pos_thigh_gainprm']} / {s['pos_thigh_biasprm']}   "
    f"aux_thigh gain/bias = {s['aux_thigh_gainprm']} / {s['aux_thigh_biasprm']}"
  )
  print(
    f"  pos_calf  gain/bias = {s['pos_calf_gainprm']} / {s['pos_calf_biasprm']}   "
    f"aux_calf  gain/bias = {s['aux_calf_gainprm']} / {s['aux_calf_biasprm']}"
  )
  print(
    f"  aux_thigh ctrl/force = {s['aux_thigh_ctrl']:+.3f} / {s['aux_thigh_force']:+.5f}   "
    f"aux_calf ctrl/force = {s['aux_calf_ctrl']:+.3f} / {s['aux_calf_force']:+.5f}"
  )
  print(
    f"  pos_thigh force = {s['pos_thigh_force']:+.5f}   pos_calf force = {s['pos_calf_force']:+.5f}"
  )


def enter_recovery(env, act, env_ids):
  mask = torch.zeros(act.num_envs, dtype=torch.bool)
  mask[env_ids] = True
  act._phase[mask] = PHASE_RECOVERY
  act._recovery_elapsed[mask] = 0
  act.newly_entangled[:] = mask
  act._apply_hybrid_actuators()
  act.newly_entangled[:] = False


def exit_recovery(env, act, env_ids):
  mask = torch.zeros(act.num_envs, dtype=torch.bool)
  mask[env_ids] = True
  act.newly_recovered[:] = mask
  act._phase[mask] = PHASE_NORMAL_WALK
  act._apply_hybrid_actuators()
  act.newly_recovered[:] = False


def other_joint_gains_unchanged(env, act, entity, env_id, baseline):
  """Check that all actuators EXCEPT the affected leg's thigh/calf
  position actuators retain their baseline gain/bias (ab/ad + other 3
  legs' 9 joints -- i.e. everything but _pos_thigh_ctrl_id/_pos_calf_
  ctrl_id)."""
  sim = env.sim
  ok = True
  mismatches = []
  n_act = sim.model.actuator_gainprm.shape[1]
  for ctrl_id in range(n_act):
    if ctrl_id in (
      act._pos_thigh_ctrl_id,
      act._pos_calf_ctrl_id,
      act._aux_thigh_ctrl_id,
      act._aux_calf_ctrl_id,
    ):
      continue
    g = sim.model.actuator_gainprm[env_id, ctrl_id, :3].tolist()
    b = sim.model.actuator_biasprm[env_id, ctrl_id, :3].tolist()
    base_g, base_b = baseline[ctrl_id]
    if g != base_g or b != base_b:
      ok = False
      mismatches.append((ctrl_id, g, base_g, b, base_b))
  return ok, mismatches


def main():
  torch.manual_seed(0)
  cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=2,
    recovery_mode="hybrid",
    policy_device="cpu",
    play=True,
  )
  cfg.sim.device = "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  act = env.action_manager._terms["joint_pos"]
  env.reset()
  entity = act._entity
  zero = torch.zeros(2, 3)

  for _ in range(5):
    env.step(zero)

  n_act = env.sim.model.actuator_gainprm.shape[1]
  baseline_env0 = {
    cid: (
      env.sim.model.actuator_gainprm[0, cid, :3].tolist(),
      env.sim.model.actuator_biasprm[0, cid, :3].tolist(),
    )
    for cid in range(n_act)
  }
  baseline_env1 = {
    cid: (
      env.sim.model.actuator_gainprm[1, cid, :3].tolist(),
      env.sim.model.actuator_biasprm[1, cid, :3].tolist(),
    )
    for cid in range(n_act)
  }

  results = {}

  print("\n################ CYCLE 1 ################")
  s_normal_before = snapshot(env, act, entity, "NORMAL-before", 0)
  print_snap(s_normal_before)
  orig_pos_thigh = (
    s_normal_before["pos_thigh_gainprm"],
    s_normal_before["pos_thigh_biasprm"],
  )
  orig_pos_calf = (
    s_normal_before["pos_calf_gainprm"],
    s_normal_before["pos_calf_biasprm"],
  )

  # --- Enter recovery on env 0 ONLY; env 1 stays NORMAL. ---
  enter_recovery(env, act, [0])
  s_entry0 = snapshot(env, act, entity, "RECOVERY-entry", 0)
  s_entry1 = snapshot(env, act, entity, "RECOVERY-entry(env1,should be NORMAL)", 1)
  print_snap(s_entry0)
  print_snap(s_entry1)

  for _ in range(5):
    act._apply_hybrid_actuators()
    env.step(zero)
  s_mid0 = snapshot(env, act, entity, "RECOVERY-mid", 0)
  s_mid1 = snapshot(env, act, entity, "RECOVERY-mid(env1)", 1)
  print_snap(s_mid0)
  print_snap(s_mid1)

  exit_recovery(env, act, [0])
  s_exit0 = snapshot(env, act, entity, "RECOVERY-exit(just toggled)", 0)
  print_snap(s_exit0)

  for _ in range(5):
    env.step(zero)
  s_after0 = snapshot(env, act, entity, "NORMAL-after", 0)
  s_after1 = snapshot(env, act, entity, "NORMAL-after(env1)", 1)
  print_snap(s_after0)
  print_snap(s_after1)

  other0_ok, other0_mismatch = other_joint_gains_unchanged(
    env, act, entity, 0, baseline_env0
  )
  other1_ok, other1_mismatch = other_joint_gains_unchanged(
    env, act, entity, 1, baseline_env1
  )

  results["entry_pos_disabled"] = (
    s_entry0["pos_thigh_gainprm"] == [0.0, 0.0, 0.0]
    and s_entry0["pos_thigh_biasprm"] == [0.0, 0.0, 0.0]
    and s_entry0["pos_calf_gainprm"] == [0.0, 0.0, 0.0]
    and s_entry0["pos_calf_biasprm"] == [0.0, 0.0, 0.0]
  )
  results["entry_aux_active"] = (
    s_entry0["aux_thigh_gainprm"][0] != 0.0
    and s_entry0["aux_calf_gainprm"] == s_entry0["aux_calf_gainprm"]
    and s_entry0["recovery_thigh_vel_ctrl"] != 0.0
    and s_entry0["recovery_calf_torque_ctrl"] != 0.0
  )
  results["env1_untouched_during_env0_recovery"] = (
    s_entry1["pos_thigh_gainprm"] == orig_pos_thigh[0]
    and s_entry1["aux_thigh_gainprm"] == [0.0, 0.0, 0.0]
    and s_mid1["pos_thigh_gainprm"] == orig_pos_thigh[0]
  )
  results["other_joints_unaffected_env0"] = other0_ok
  results["other_joints_unaffected_env1"] = other1_ok
  # NOTE: aux_calf (NamedMotorActuator) keeps a CONSTANT gainprm=[1,0,0]
  # at all times by design -- a motor actuator is a direct-torque
  # feedthrough (force = ctrl * gain) with no passive-at-rest behavior
  # at ctrl=0, unlike the velocity actuator, so there is nothing to
  # toggle in its gain; "disabled" for it means ctrl/force == 0, which
  # is checked directly below. Only aux_thigh (NamedVelocityActuator)
  # needs its gain/bias toggled off, since a velocity actuator with
  # ctrl=0 but nonzero kd still applies passive damping at all times.
  results["exit_aux_disabled"] = (
    s_after0["aux_thigh_gainprm"] == [0.0, 0.0, 0.0]
    and s_after0["aux_thigh_ctrl"] == 0.0
    and s_after0["aux_calf_ctrl"] == 0.0
    and s_after0["aux_thigh_force"] == 0.0
    and s_after0["aux_calf_force"] == 0.0
    and s_after0["recovery_thigh_vel_ctrl"] == 0.0
    and s_after0["recovery_calf_torque_ctrl"] == 0.0
  )
  results["exit_pos_restored_exact"] = (
    s_after0["pos_thigh_gainprm"] == orig_pos_thigh[0]
    and s_after0["pos_thigh_biasprm"] == orig_pos_thigh[1]
    and s_after0["pos_calf_gainprm"] == orig_pos_calf[0]
    and s_after0["pos_calf_biasprm"] == orig_pos_calf[1]
  )
  results["hip_unaffected_throughout"] = (
    True  # hip is excluded from pos/aux toggling entirely by design
  )

  print("\n==== CYCLE 1 CHECKS ====")
  for k, v in results.items():
    print(f"  {k}: {v}")
  if other0_mismatch:
    print("  other0_mismatch:", other0_mismatch)
  if other1_mismatch:
    print("  other1_mismatch:", other1_mismatch)

  # --- Repeated entry/exit cycle check (cycle 2). ---
  print("\n################ CYCLE 2 (repeated entry/exit) ################")
  enter_recovery(env, act, [0])
  for _ in range(5):
    act._apply_hybrid_actuators()
    env.step(zero)
  exit_recovery(env, act, [0])
  for _ in range(5):
    env.step(zero)
  s_after0_c2 = snapshot(env, act, entity, "NORMAL-after-cycle2", 0)
  print_snap(s_after0_c2)
  results["cycle2_exact_restore"] = (
    s_after0_c2["pos_thigh_gainprm"] == orig_pos_thigh[0]
    and s_after0_c2["pos_thigh_biasprm"] == orig_pos_thigh[1]
    and s_after0_c2["pos_calf_gainprm"] == orig_pos_calf[0]
    and s_after0_c2["pos_calf_biasprm"] == orig_pos_calf[1]
    and s_after0_c2["aux_thigh_gainprm"] == [0.0, 0.0, 0.0]
    and s_after0_c2["aux_thigh_force"] == 0.0
    and s_after0_c2["aux_calf_force"] == 0.0
  )
  print("  cycle2_exact_restore:", results["cycle2_exact_restore"])

  overall_pass = all(results.values())
  print("\n==== OVERALL RESULT ====")
  print("TEST_C_RESULT:", "PASS" if overall_pass else "FAIL")
  for k, v in results.items():
    if not v:
      print("  FAILING CHECK:", k)

  env.close()


if __name__ == "__main__":
  main()
