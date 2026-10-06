"""Run the existing trained Go2 walking policy through the cable-entanglement
test environment (Phase 3A).

This script does NOT train anything. It loads the existing trained Go2
velocity-walking checkpoint and drives the *new* cable-entanglement test
environment (``Mjlab-Go2-Cable-Entanglement-Test``) with it, logging
instrumentation every step: foot positions, joint positions/velocities/
torques, root velocity/orientation, cable position, and cable contact
counts (especially RL foot vs. cable).

Example (single-env, visual debug mode, native viewer)::

    uv run python scripts/cable_entanglement_test.py \\
        --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<iter>.pt \\
        --debug

Example (headless, N steps, CSV log only)::

    uv run python scripts/cable_entanglement_test.py \\
        --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<iter>.pt \\
        --num-steps 500 --log-file logs/cable_test/run1.csv
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.cable_test.instrumentation import (
  collect_instrumentation,
  format_instrumentation_row,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Go2-Cable-Entanglement-Test"


@dataclass(frozen=True)
class CableTestConfig:
  checkpoint_file: str
  """Path to a trained Go2 velocity-walking checkpoint (.pt)."""
  device: str | None = None
  debug: bool = False
  """Debug mode: forces num_envs=1 and launches the native viewer for visual
  inspection of the cable interaction."""
  num_envs: int = 1
  num_steps: int = 400
  """Number of environment steps to run (ignored in --debug, which runs
  until the viewer window is closed)."""
  log_file: str | None = None
  """Optional CSV path to write per-step instrumentation to."""
  print_every: int = 25
  """Print a instrumentation summary line every N steps (0 disables)."""


def _print_summary(step: int, row: dict) -> None:
  rl_contact = row.get("cable_contact_RL", False)
  total = row.get("cable_contact_count_total", 0)
  rl_vel = row.get("joint_vel_RL")
  rl_torque = row.get("joint_torque_RL")
  print(
    f"[step {step:5d}] RL_foot={row.get('foot_pos_RL')} "
    f"RL_cable_contact={bool(rl_contact)} total_cable_contacts={total} "
    f"RL_joint_vel={rl_vel} RL_joint_torque={rl_torque} "
    f"root_lin_vel={row.get('root_lin_vel_w')}"
  )


def run(cfg: CableTestConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  num_envs = 1 if cfg.debug else cfg.num_envs

  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = num_envs

  resume_path = Path(cfg.checkpoint_file)
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  from dataclasses import asdict

  runner_cls = load_runner_cls(TASK_ID)
  assert runner_cls is not None
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  raw_env: ManagerBasedRlEnv = wrapped.unwrapped

  csv_writer = None
  csv_file = None
  if cfg.log_file:
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_path, "w", newline="")

  def instrumented_policy(obs):
    action = policy(obs)
    info = collect_instrumentation(raw_env)
    row = format_instrumentation_row(info, env_idx=0)
    row["step"] = instrumented_policy.step  # type: ignore[attr-defined]

    nonlocal csv_writer
    if csv_file is not None:
      if csv_writer is None:
        csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(row.keys()))
        csv_writer.writeheader()
      csv_writer.writerow(row)

    if cfg.print_every and instrumented_policy.step % cfg.print_every == 0:  # type: ignore[attr-defined]
      _print_summary(instrumented_policy.step, row)  # type: ignore[attr-defined]

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

  wrapped.close()


def main() -> None:
  args = tyro.cli(CableTestConfig, args=sys.argv[1:])
  run(args)


if __name__ == "__main__":
  main()
