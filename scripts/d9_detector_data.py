"""D9 Stage 2, step 1: record proprioception + oracle wire-contact labels from the sim.

Runs the existing ``d6_1_front_pair_wire.py`` episode (through ``d9_loose_wire_recovery.py``'s
wire swap, so the taut original wire, the loose-to-tight wire and the recovery all work
unchanged) and records, per 50 Hz control step, features pooled over the 8 physics substeps
of what a real robot's /lowstate has, plus the per-leg wire-contact flag (the label).

Features per step (``feature_names`` in the output): mean over substeps of joint q, dq and
torque (12 each, order FR FL RR RL x hip thigh calf), max |torque| (12), foot-ground force
(4, FR FL RR RL), roll, pitch, gyro (3), accelerometer = body-frame specific force (3).
Labels: ``contact`` [T, 4] (FR FL RR RL) = the leg's calf touched the wire in any substep.
Also saved: per-episode overlap check (wire passing through a leg) and metadata.

Usage (walk into a taut wire at 8 cm, recovery off, 140 steps):
    uv run python scripts/d9_detector_data.py --original-wire --kind walk \
      --pair all --start-x-range -0.485 -0.35 --seed 200 --wire-height 0.08 \
      --entry-delay-steps 1000000000 --max-steps 140 --out-dir logs/d9_detector/scratch
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d9_loose_wire_recovery as R  # noqa: E402  (applies the wire-swap patches)

from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402

d61 = R.d61
LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")


@dataclass
class DataConfig(R.LooseConfig):
  kind: str = "walk"
  """Label for the episode (walk / loose / recovery / nowire); stored in the file."""
  data_dir: str = "logs/d9_detector/data"


CFG = DataConfig()
REC: dict = {}

_prev_cfg = d61.go2_wire_recovery_residual_env_cfg
_prev_build = d61.build_env


def _cfg_with_foot_sensor(*a, **k):
  cfg = _prev_cfg(*a, **k)
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


def feature_names() -> list[str]:
  jn = [f"{leg}_{j}" for leg in LEGS for j in JOINTS]
  return (
    [f"q_{n}" for n in jn]
    + [f"dq_{n}" for n in jn]
    + [f"tau_{n}" for n in jn]
    + [f"taumax_{n}" for n in jn]
    + [f"foot_{leg}" for leg in LEGS]
    + ["roll", "pitch", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
  )


def _build_with_recorder(cfg, seed, device):
  env = _prev_build(cfg, seed, device)
  entity = env.action_manager.get_term("joint_pos")._entity
  sensors = env.scene.sensors
  feet = sensors["feet_ground_contact"]
  wire_sensors = [sensors[f"wire_contact_{leg}"] for leg in LEGS]
  jn = list(entity.joint_names)
  j_idx = [jn.index(f"{leg}_{j}_joint") for leg in LEGS for j in JOINTS]
  foot_names = [str(n).split("_")[0] for n in feet.primary_names]
  f_idx = [foot_names.index(leg) for leg in LEGS]
  dt = env.physics_dt
  dec = env.cfg.decimation
  REC.update(X=[], Y=[], sub=[], prev_v=None)
  orig = env.metrics_manager.compute_substep

  def hooked(*a, **k):
    d = entity.data
    q = d.joint_pos[0].cpu().numpy()[j_idx]
    dq = d.joint_vel[0].cpu().numpy()[j_idx]
    tau = d.qfrc_actuator[0].cpu().numpy()[j_idx]
    foot = feet.data.force[0].norm(dim=-1).reshape(-1).cpu().numpy()[f_idx]
    w, x, y, z = d.root_link_quat_w[0].cpu().numpy()
    gyro = d.root_link_ang_vel_b[0].cpu().numpy()
    v = d.root_link_lin_vel_w[0].cpu().numpy()
    acc_w = (v - REC["prev_v"]) / dt if REC["prev_v"] is not None else np.zeros(3)
    acc_w = acc_w + np.array([0.0, 0.0, 9.81])
    REC["prev_v"] = v.copy()
    rot = np.array(
      [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
      ]
    )
    acc = rot.T @ acc_w
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    contact = [bool((s.data.found[0] > 0).any()) for s in wire_sensors]
    REC["sub"].append(
      (
        np.concatenate([q, dq, tau, np.abs(tau), foot, [roll, pitch], gyro, acc]),
        contact,
      )
    )
    if len(REC["sub"]) == dec:
      rows = np.stack([r for r, _ in REC["sub"]])
      mean = rows.mean(axis=0)
      mean[36:48] = rows[:, 36:48].max(axis=0)  # max |tau| over the substeps
      REC["X"].append(mean)
      REC["Y"].append(np.any([c for _, c in REC["sub"]], axis=0))
      REC["sub"].clear()
    return orig(*a, **k)

  env.metrics_manager.compute_substep = hooked  # type: ignore[method-assign]
  return env


d61.go2_wire_recovery_residual_env_cfg = _cfg_with_foot_sensor
d61.build_env = _build_with_recorder


def main() -> None:
  global CFG
  loose, rest = tyro.cli(DataConfig, args=sys.argv[1:], return_unknown_args=True)
  CFG = loose
  R.LOOSE = loose  # the wire-swap runner reads its own copy
  cfg = tyro.cli(d61.D61Config, args=rest)
  if cfg.seeds:
    raise SystemExit("use --seed N (one process per seed)")
  tag = f"{loose.kind}_s{cfg.seed}_h{round(cfg.wire_height * 100)}"
  if cfg.out_dir:
    cfg = dataclasses.replace(cfg, out_dir=f"{cfg.out_dir}/{tag}")
  label, detail, events = d61.run(cfg, cfg.seed, quiet=True)
  X = np.stack(REC["X"])
  Y = np.stack(REC["Y"])
  out = Path(loose.data_dir)
  out.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out / f"{tag}.npz",
    X=X.astype(np.float32),
    contact=Y,
    feature_names=np.array(feature_names()),
    legs=np.array(LEGS),
    kind=loose.kind,
    seed=cfg.seed,
    wire_height=cfg.wire_height,
    slack_cm=-1.0 if loose.original_wire else loose.slack_cm,
    tighten_time=loose.tighten_time,
    anchor_solref=loose.anchor_solref,
    verdict=label,
    max_overlap_mm=max(float(R.STATE["worst"].max()) * 1000, 0.0),
    deep_overlap_substeps=int(R.STATE["deep"]),
    step_dt=0.02,
  )
  first = {
    leg: (int(np.argmax(Y[:, i])) if Y[:, i].any() else -1)
    for i, leg in enumerate(LEGS)
  }
  print(
    f"{tag}: T={len(X)} verdict={label} first_contact_step={first} "
    f"max_overlap_mm={max(float(R.STATE['worst'].max()) * 1000, 0):.1f} "
    f"deep={R.STATE['deep']}",
    flush=True,
  )


if __name__ == "__main__":
  main()
