"""TEMPORARY diagnostic script (CONTROL experiment): run the existing
trained Go2 walking policy through a single RIGID CAPSULE ROD in place of
the articulated test cable.

This is a control experiment only. It does **not** modify the cable
implementation (``mjlab/tasks/cable_test/cable.py``), the cable test
environment (``mjlab/tasks/cable_test/env_cfg.py``), the trained Go2
walking policy, the TCN, or the recovery environment. It builds a
standalone diagnostic environment (``env_cfg_rod_diagnostic.py`` /
``rod_diagnostic.py``, both new files) that keeps everything else about the
cable test identical -- same policy, same walking command, same robot
reset, same timestep/decimation, same terrain, same camera -- and replaces
only the cable object with a single static rigid capsule of the same center
position, orientation, total length, radius (0.01 m), center height
(0.10 m), and collision masks.

It does not register a permanent task in the global task registry; the
environment config is built and used directly, so running this script has
no side effects on the existing cable test setup.

Example (visual inspection, native viewer)::

    uv run python scripts/rod_diagnostic_test.py \\
        --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<iter>.pt \\
        --debug

Example (headless, N steps, CSV log)::

    uv run python scripts/rod_diagnostic_test.py \\
        --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<iter>.pt \\
        --num-steps 500 --log-file logs/rod_diagnostic/run1.csv
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.cable_test.env_cfg_rod_diagnostic import (
  DEFAULT_ROD_CFG,
  go2_rod_diagnostic_env_cfg,
)
from mjlab.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class RodDiagnosticConfig:
  checkpoint_file: str
  """Path to the existing trained Go2 velocity-walking checkpoint (.pt)."""
  device: str | None = None
  debug: bool = False
  """Debug mode: forces num_envs=1 and launches the native viewer for
  visual inspection of the robot-vs-rod interaction."""
  num_envs: int = 1
  num_steps: int = 400
  """Number of environment steps to run (ignored in --debug)."""
  log_file: str | None = None
  """Optional CSV path to write per-step instrumentation to."""
  print_every: int = 25
  """Print an instrumentation summary line every N steps (0 disables)."""


def _isfinite_all(*tensors: torch.Tensor) -> bool:
  return all(bool(torch.isfinite(t).all()) for t in tensors if t is not None)


def collect_row(env: ManagerBasedRlEnv, env_idx: int = 0) -> dict:
  robot = env.scene["robot"]
  data = robot.data

  root_lin_vel = data.root_link_lin_vel_w[env_idx].detach().cpu()
  root_ang_vel = data.root_link_ang_vel_w[env_idx].detach().cpu()
  root_pos = data.root_link_pos_w[env_idx].detach().cpu()
  proj_grav = data.projected_gravity_b[env_idx].detach().cpu()
  joint_pos = data.joint_pos[env_idx].detach().cpu()
  joint_vel = data.joint_vel[env_idx].detach().cpu()

  sensor = env.scene.sensors["rod_contact"]
  found = sensor.data.found  # [B, P]
  force = sensor.data.force  # [B, P, 3]
  primary_names = sensor.primary_names

  found_i = found[env_idx].detach().cpu()
  force_i = force[env_idx].detach().cpu() if force is not None else None

  contact_mask = found_i > 0
  contacting_geoms = [primary_names[i] for i in torch.nonzero(contact_mask).flatten().tolist()]
  contact_forces = {}
  if force_i is not None:
    for i in torch.nonzero(contact_mask).flatten().tolist():
      contact_forces[primary_names[i]] = force_i[i].tolist()

  finite = _isfinite_all(
    root_lin_vel,
    root_ang_vel,
    root_pos,
    proj_grav,
    joint_pos,
    joint_vel,
    found_i,
    force_i if force_i is not None else torch.zeros(1),
  )

  return {
    "root_lin_vel_w": root_lin_vel.tolist(),
    "root_ang_vel_w": root_ang_vel.tolist(),
    "root_pos_w": root_pos.tolist(),
    "in_contact": bool(contact_mask.any()),
    "contacting_geoms": contacting_geoms,
    "contact_forces": contact_forces,
    "all_finite": finite,
  }


def _classify_behavior(rows: list[dict]) -> str:
  """Heuristic post-hoc classification of the robot's response to the rod."""
  contact_idxs = [i for i, r in enumerate(rows) if r["in_contact"]]
  if not contact_idxs:
    return "no contact recorded during run"

  first, last = contact_idxs[0], contact_idxs[-1]
  before = rows[max(0, first - 10) : first]
  during = rows[first : last + 1]
  after = rows[last + 1 : last + 11]

  def mean_vx(chunk: list[dict]) -> float | None:
    if not chunk:
      return None
    return sum(r["root_lin_vel_w"][0] for r in chunk) / len(chunk)

  vx_before = mean_vx(before)
  vx_during = mean_vx(during)
  vx_after = mean_vx(after)

  z_before = rows[max(0, first - 1)]["root_pos_w"][2]
  z_min_during = min(r["root_pos_w"][2] for r in during)
  fell = (z_before - z_min_during) > 0.08  # trunk dropped notably -> fall.

  if fell:
    return (
      f"FALL: root height dropped {z_before - z_min_during:.3f} m during "
      f"contact (vx before={vx_before}, during={vx_during}, after={vx_after})"
    )
  if vx_during is not None and vx_before is not None and vx_during < 0.15 * (vx_before or 1e-6):
    if vx_after is not None and vx_after > 0.5 * (vx_before or 1e-6):
      return (
        f"STEPS OVER / BRIEF STOP: vx dropped during contact then recovered "
        f"(before={vx_before}, during={vx_during}, after={vx_after})"
      )
    return (
      f"STOPS: forward velocity collapsed and did not recover "
      f"(before={vx_before}, during={vx_during}, after={vx_after})"
    )
  return (
    f"STEPS OVER / PUSHES THROUGH: forward velocity roughly maintained "
    f"(before={vx_before}, during={vx_during}, after={vx_after})"
  )


def run(cfg: RodDiagnosticConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  num_envs = 1 if cfg.debug else cfg.num_envs

  env_cfg = go2_rod_diagnostic_env_cfg(num_envs=num_envs, rod_cfg=DEFAULT_ROD_CFG)
  agent_cfg = unitree_go2_ppo_runner_cfg()

  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = VelocityOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  raw_env: ManagerBasedRlEnv = wrapped.unwrapped

  rows: list[dict] = []
  first_contact_step: int | None = None
  nan_step: int | None = None

  csv_writer = None
  csv_file = None
  if cfg.log_file:
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_path, "w", newline="")

  def instrumented_policy(obs):
    nonlocal first_contact_step, nan_step, csv_writer
    action = policy(obs)
    row = collect_row(raw_env)
    row["step"] = instrumented_policy.step  # type: ignore[attr-defined]
    rows.append(row)

    if not row["all_finite"] and nan_step is None:
      nan_step = row["step"]
      print(f"[WARNING] NaN/Inf detected at step {nan_step}!")

    if row["in_contact"] and first_contact_step is None:
      first_contact_step = row["step"]
      print(
        f"[CONTACT] First robot/rod contact at step {first_contact_step}: "
        f"geoms={row['contacting_geoms']} forces={row['contact_forces']} "
        f"root_lin_vel_w={row['root_lin_vel_w']}"
      )

    if csv_file is not None:
      flat = {
        "step": row["step"],
        "root_lin_vel_x": row["root_lin_vel_w"][0],
        "root_lin_vel_y": row["root_lin_vel_w"][1],
        "root_lin_vel_z": row["root_lin_vel_w"][2],
        "root_pos_x": row["root_pos_w"][0],
        "root_pos_y": row["root_pos_w"][1],
        "root_pos_z": row["root_pos_w"][2],
        "in_contact": row["in_contact"],
        "contacting_geoms": ";".join(row["contacting_geoms"]),
        "all_finite": row["all_finite"],
      }
      if csv_writer is None:
        csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(flat.keys()))
        csv_writer.writeheader()
      csv_writer.writerow(flat)

    if cfg.print_every and instrumented_policy.step % cfg.print_every == 0:  # type: ignore[attr-defined]
      print(
        f"[step {instrumented_policy.step:5d}] "  # type: ignore[attr-defined]
        f"root_lin_vel_w={row['root_lin_vel_w']} in_contact={row['in_contact']} "
        f"finite={row['all_finite']}"
      )

    instrumented_policy.step += 1  # type: ignore[attr-defined]
    return action

  instrumented_policy.step = 0  # type: ignore[attr-defined]

  if cfg.debug:
    from mjlab.viewer import NativeMujocoViewer

    print(
      "[INFO] Debug mode: launching native viewer with 1 environment. "
      "Close the viewer window to end the run."
    )
    NativeMujocoViewer(wrapped, instrumented_policy).run()
  else:
    obs, _ = wrapped.reset()
    for _ in range(cfg.num_steps):
      with torch.no_grad():
        actions = instrumented_policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      if bool(dones.any()):
        pass  # Auto-reset is handled internally by the environment.

  if csv_file is not None:
    csv_file.close()
    print(f"[INFO] Instrumentation log written to {cfg.log_file}")

  print("\n===== ROD DIAGNOSTIC SUMMARY =====")
  print(f"Total steps recorded: {len(rows)}")
  print(f"First contact step: {first_contact_step}")
  print(f"NaN/Inf detected: {'YES at step ' + str(nan_step) if nan_step is not None else 'no'}")
  if rows:
    print(f"Behavior: {_classify_behavior(rows)}")
  print("===================================\n")

  wrapped.close()


def main() -> None:
  args = tyro.cli(RodDiagnosticConfig, args=sys.argv[1:])
  run(args)


if __name__ == "__main__":
  main()
