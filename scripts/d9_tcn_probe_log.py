"""D9 Stage 3: log /lowstate-like proprioception from the wire sim at physics rate.

Runs the walking policy into the wire with recovery switched off (the entry trigger is
delayed forever), and records per physics substep (400 Hz): joint q/dq/torque, base
orientation, base velocities, foot ground-contact force, and the per-leg wire-contact
flags (the oracle label). Output is one .npz per seed, read by d9_tcn_probe_eval.py.

Nothing in the recovery code is touched; the hook wraps one method on the env instance.

Usage:
    uv run python scripts/d9_tcn_probe_log.py --seed 0 --out logs/d9_tcn_probe/raw
    uv run python scripts/d9_tcn_probe_log.py --seed 0 --wire-height 0.5   # no-contact control
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_1_front_pair_wire as d61  # noqa: E402
from d6_1_front_pair_wire import PAIRS, D61Config  # noqa: E402

from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402

LEGS = PAIRS["all"]

_orig_env_cfg = d61.go2_wire_recovery_residual_env_cfg


def _env_cfg_with_foot_sensor(*a, **k):
  """Same env cfg plus a foot-ground contact sensor (the recovery env has none)."""
  cfg = _orig_env_cfg(*a, **k)
  foot = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=tuple(f"{leg}_foot_collision" for leg in LEGS),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), foot)
  return cfg


d61.go2_wire_recovery_residual_env_cfg = _env_cfg_with_foot_sensor


@dataclass
class ProbeConfig:
  seed: int = 0
  steps: int = 320
  """Control steps (0.02 s each) to run."""
  wire_height: float = 0.10
  """0.10 = the usual wire. Use 0.5 for a no-contact (negative) control."""
  out: str = "logs/d9_tcn_probe/raw"


def main() -> None:
  import tyro

  p = tyro.cli(ProbeConfig)
  cfg = D61Config(
    pair="all",
    start_x_range=(-0.485, -0.35),
    seed=p.seed,
    # Recovery never starts: the walking policy alone meets the wire.
    entry_delay_steps=10**9,
    wire_height=p.wire_height,
  )
  env = d61.build_env(cfg, p.seed, "cpu")
  term = env.action_manager.get_term("joint_pos")
  entity = term._entity
  sensors = env.scene.sensors
  feet = sensors["feet_ground_contact"]
  wire_sensors = {leg: sensors[f"wire_contact_{leg}"] for leg in LEGS}
  env.reset(seed=p.seed)

  rec: dict[str, list] = {
    k: [] for k in ("q", "dq", "tau", "quat", "angv_b", "linv_w", "foot", "wire")
  }

  orig = env.metrics_manager.compute_substep

  def hooked(*a, **k):
    d = entity.data
    rec["q"].append(d.joint_pos[0].cpu().numpy().copy())
    rec["dq"].append(d.joint_vel[0].cpu().numpy().copy())
    rec["tau"].append(d.qfrc_actuator[0].cpu().numpy().copy())
    rec["quat"].append(d.root_link_quat_w[0].cpu().numpy().copy())
    rec["angv_b"].append(d.root_link_ang_vel_b[0].cpu().numpy().copy())
    rec["linv_w"].append(d.root_link_lin_vel_w[0].cpu().numpy().copy())
    f = feet.data.force[0]
    rec["foot"].append(f.norm(dim=-1).reshape(-1).cpu().numpy().copy())
    rec["wire"].append(
      np.array([bool((wire_sensors[leg].data.found[0] > 0).any()) for leg in LEGS])
    )
    return orig(*a, **k)

  env.metrics_manager.compute_substep = hooked  # type: ignore[method-assign]

  zero = torch.zeros(env.action_manager.action.shape, device="cpu")
  for _ in range(p.steps):
    env.step(zero)

  out = Path(p.out)
  out.mkdir(parents=True, exist_ok=True)
  tag = f"seed{p.seed}_h{int(round(p.wire_height * 100)):02d}"
  np.savez_compressed(
    out / f"{tag}.npz",
    **{k: np.stack(v) for k, v in rec.items()},
    dt=env.physics_dt,
    joint_names=np.array(entity.joint_names),
    foot_names=np.array(list(feet.primary_names)),
    wire_legs=np.array(LEGS),
  )
  w = np.stack(rec["wire"])
  first = {
    leg: (int(np.argmax(w[:, i])) if w[:, i].any() else -1)
    for i, leg in enumerate(LEGS)
  }
  print(
    f"{tag}: {w.shape[0]} substeps dt={env.physics_dt} first_contact_substep={first}"
  )
  env.close()


if __name__ == "__main__":
  main()
