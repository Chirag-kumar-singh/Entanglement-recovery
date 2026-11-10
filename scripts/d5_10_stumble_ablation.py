"""D5.10 -- Causal ablation: does the hybrid recovery actuator override
cause/worsen the initial post-wire-contact stumble, or is it already
present under walking-only control?

CONDITION A (existing baseline): unmodified hybrid recovery, identical
to every prior D5.x script (`build_ablation_env`, recovery_mode="hybrid",
`_apply_hybrid_actuators` runs normally).

CONDITION B (recovery override disabled): the EXACT SAME env build
(`build_ablation_env`, same seed, same wire geometry, same sim settings,
same checkpoint, same max_recovery_steps_override=48) -- the FSM/phase-
tracking, wire-contact entry trigger, and all geometric instrumentation
are therefore computed IDENTICALLY in both conditions. Only the
actuator-level override itself is suppressed, so the FL leg stays under
the frozen walking policy's own control throughout.

HOW CONDITION B IS IMPLEMENTED (verified against actions.py before
writing this script):
  - `_apply_hybrid_actuators` is the ONLY method that (a) zeros the
    affected leg's thigh/calf POSITION actuator gain/bias, and (b)
    issues the auxiliary thigh-velocity / calf-torque commands. It is
    called from exactly one place -- `process_actions`, and ONLY when
    `cfg.recovery_mode == "hybrid"`.
  - In hybrid mode, `delta_a` is always zero, so `final_raw` (what is
    passed into the existing, unmodified `JointPositionAction.
    process_actions`) is identical to the raw walking-policy action in
    BOTH conditions -- the only thing that differs is whether the FL
    thigh/calf position actuators' gain/bias get zeroed out.
  - Condition B monkeypatches the bound method
    `act._apply_hybrid_actuators` to a no-op on the already-constructed
    action-term instance, for this script's env only. No production
    file is edited, and `_update_fsm()` (phase transitions, wire-contact
    entry trigger, all geometric gate state) is left completely
    unmodified and still runs identically in both conditions -- this is
    what lets both conditions align on the same "recovery entry" (first
    wire contact) event for comparison, even though Condition B's
    "RECOVERY phase" bookkeeping has no physical actuator effect.
  - Caveat (reported, not hidden): because `_update_fsm` still runs in
    Condition B, `recovery_elapsed`/stage-latch bookkeeping still
    advances and the FSM will still eventually flag a timeout at the
    48-step cap -- this has no actuator effect in Condition B and is
    logged for alignment purposes only.

AUTO-RESET / STOP CONDITION: `env.cfg.auto_reset` is set to False (an
existing, documented `ManagerBasedRlEnv` flag). Each step checks
`env.reset_buf`; the rollout loop stops IMMEDIATELY the first time
termination/time-out fires (before calling `env.step` again, per
auto_reset=False's documented contract), rather than silently
auto-resetting and continuing on synthetic post-reset state. This
implements the requirement to never continue a run indefinitely once
the robot becomes unstable.

Reuses `build_ablation_env` from d5_3_extended_recovery_window.py
UNCHANGED (same env-construction path used by every prior D5.x script),
plus a small post-construction patch adding RL/RR/FR ground-contact
sensors (FL ground + FR wire already exist via that function; this task
additionally needs FR/RL/RR-vs-terrain for the "all four legs" ground-
contact requirement).
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


def build_d510_env(cfg: D3GroundConfig, device: str, max_recovery_steps_override: int):
  """Same construction path as `d5_3_extended_recovery_window.
  build_ablation_env` (reused, unmodified pattern, used by every prior
  D5.x script), plus FR/RL/RR-vs-terrain ground-contact sensors added
  BEFORE construction (contact sensors must be present in the MjSpec at
  build time -- they cannot be added post-construction). FL-vs-terrain
  and FR-vs-wire are already added by this same pattern; this adds the
  remaining 3 legs' ground contact so all 4 legs have a readout, per
  this task's measurement requirements. `ablate_hip_wire` is always
  False here (irrelevant/out of scope for this experiment)."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

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
  env_cfg.auto_reset = (
    False  # existing, documented flag -- never silently reset past a fall
  )

  extra_sensors = []
  for leg in ("FL", "FR", "RL", "RR"):
    extra_sensors.append(
      ContactSensorCfg(
        name=f"{leg.lower()}_ground_contact_diag",
        primary=ContactMatch(
          mode="geom", pattern=rf"{leg}_.*_collision\d*$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
      )
    )
  existing_wire_sensor = next(
    s for s in env_cfg.scene.sensors if s.name == "wire_contact"
  )
  fr_wire_cfg = ContactSensorCfg(
    name="fr_wire_contact_diag",
    primary=existing_wire_sensor.primary,
    secondary=ContactMatch(mode="geom", pattern="FR_calf2_collision", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  env_cfg.scene.sensors = (
    tuple(env_cfg.scene.sensors) + tuple(extra_sensors) + (fr_wire_cfg,)
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env


@dataclass
class D510Config:
  seed: int = 0
  device: str | None = None
  max_steps: int = 150
  """Hard diagnostic-window cap, separate from the 48-step RECOVERY FSM
  cap. Generous enough to see stumble onset + either recovery or a fall
  well past first wire contact for the previously-characterized seeds."""
  disable_hybrid_override: bool = False
  """False = Condition A (existing baseline, unmodified). True =
  Condition B (FL thigh/calf position actuators never disabled; frozen
  walking policy controls FL throughout; FSM phase-tracking unaffected)."""
  out_dir: str = ""


def run(cfg: D510Config) -> Path:
  device = cfg.device or "cpu"
  d3_cfg = D3GroundConfig(seed=cfg.seed, device=device, use_isolated_wire=True)
  env = build_d510_env(d3_cfg, device, max_recovery_steps_override=48)

  act = env.action_manager._terms["joint_pos"]
  entity = act._entity
  joint_names = list(entity.joint_names)
  thigh_i = joint_names.index("FL_thigh_joint")
  calf_i = joint_names.index("FL_calf_joint")
  hip_i = joint_names.index("FL_hip_joint")

  if cfg.disable_hybrid_override:
    # Condition B: suppress ONLY the actuator-level override (see module
    # docstring for the verification that this is the sole site doing
    # gain-zeroing / auxiliary-command issuance).
    act._apply_hybrid_actuators = lambda: None

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

    return {
      "step": step,
      "phase": new_phase,
      "recovery_elapsed": int(act.recovery_elapsed[0]),
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
      "base_quat_w": float(base_quat[0]),
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
      "calf_forward_dist": float(act.calf_forward_dist[0]),
      "calf_height_above_wire": float(act.calf_height_above_wire[0]),
      "calf_clearance_achieved": bool(act._calf_clearance_achieved[0]),
      "foot_crossed_achieved": bool(act._foot_crossed_achieved[0]),
      "crossing_achieved": bool(act._crossing_achieved[0]),
      "recovery_failed_timeout_flag": bool(act._recovery_failed_timeout[0]),
    }

  for step in range(cfg.max_steps):
    if bool(act.newly_entangled[0]):
      state["enter_line"] = f"ENTER RECOVERY step={step}"
    prev_phase = int(act.phase[0])

    action = torch.zeros(env.action_manager.action.shape, device=device)
    env.step(action)

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
    rows.append(row)

    # Stop IMMEDIATELY on genuine termination/time-out, before calling
    # env.step again -- auto_reset=False's documented contract.
    if bool(env.reset_buf[0]):
      reason = "fell_over" if bool(env.reset_terminated[0]) else "time_out"
      state["terminated_step"] = step
      state["terminated_reason"] = reason
      print(f"[D5.10] env0 TERMINATED at step={step} reason={reason}")
      break

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  log_path = out_dir / "run_log.txt"
  with open(log_path, "w") as f:
    f.write(f"disable_hybrid_override={cfg.disable_hybrid_override}\n")
    f.write(f"seed={cfg.seed}\n")
    f.write(f"{state['enter_line']}\n{state['exit_line']}\n")
    f.write(f"stage1_first_latch_step={state['stage1_first_step']}\n")
    f.write(f"stage2_first_latch_step={state['stage2_first_step']}\n")
    f.write(f"terminated_step={state['terminated_step']}\n")
    f.write(f"terminated_reason={state['terminated_reason']}\n")
  print(f"[D5.10] wrote {csv_path}")
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

  cfg = tyro.cli(D510Config)
  run(cfg)


if __name__ == "__main__":
  main()
