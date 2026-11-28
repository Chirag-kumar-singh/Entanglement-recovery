"""D9 Stage 3: run the go2-entanglement-tcn ONNX detector on logged sim proprioception.

Reads the .npz files from d9_tcn_probe_log.py (400 Hz physics rate), resamples them to the
detector's 500 Hz, builds /lowstate-shaped samples and pushes them through the detector's
own EntanglementEngine (same code and config as the robot node). Reports, per run, when
each leg first touched the wire (oracle) and when the detector alarmed.

The detector is the user's own project and is imported, not modified.

Usage:
    uv run python scripts/d9_tcn_probe_eval.py --raw logs/d9_tcn_probe/raw
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PKG = Path(
  "/home/chirag/Downloads/go2-entanglement-tcn-main/robot_package/src/entanglement_detector"
)
sys.path.insert(0, str(PKG))
from entanglement_detector.engine import EntanglementEngine  # noqa: E402

LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")
FOOT_ORDER = ("FL", "FR", "RL", "RR")  # order of the detector's foot_* channels
HZ = 500
TEMPERATURE = 5.91804997319397  # models/calibration.json


@dataclass
class EvalConfig:
  raw: str = "logs/d9_tcn_probe/raw"
  out: str = "logs/d9_tcn_probe"
  foot_scale: float = 1.0
  """Multiplier on the simulated foot force (N). The real channel is a raw sensor value,
  roughly the same size as N at standing load (assumption, not calibrated)."""
  threshold: float = 0.9999
  """Raw P(entangled) alarm threshold (the shipped config uses 0.9999)."""
  stationary_gate: bool = True
  """The detector's Stop/Lock gate. It clears the buffer when joints are nearly still,
  which is also what a robot pressed against the wire looks like. False = disabled."""
  plots: bool = True

  @property
  def tag(self) -> str:
    return (
      f"foot{self.foot_scale:g}_thr{self.threshold:g}_"
      f"{'gate' if self.stationary_gate else 'nogate'}"
    )


def make_engine(cfg: EvalConfig) -> EntanglementEngine:
  m = PKG / "models"
  return EntanglementEngine(
    model_path=str(m / "entanglement_tcn.onnx"),
    normalize_path=str(m / "normalize.json"),
    intensity_calib_path=str(m / "intensity_calib.json"),
    temperature=TEMPERATURE,
    # config/config.yaml
    detection_threshold=cfg.threshold,
    debounce_ms=75,
    leg_thresholds={"FR": 0.5, "FL": 0.5, "RR": 0.9, "RL": 0.5},
    intensity_blend=0.5,
    num_threads=1,
    stationary_dq_thresh=0.30 if cfg.stationary_gate else 0.0,
    stationary_min_ms=100,
    stabilize_ms=300,
  )


def to_lowstate_samples(d, foot_scale: float):
  """npz -> (t500, list of sample dicts), channels named like the training CSVs."""
  dt = float(d["dt"])
  n = d["q"].shape[0]
  t_src = np.arange(n) * dt
  t = np.arange(0.0, t_src[-1], 1.0 / HZ)

  def rs(x):
    return np.stack([np.interp(t, t_src, x[:, i]) for i in range(x.shape[1])], axis=1)

  jn = [str(s) for s in d["joint_names"]]
  q, dq, tau = rs(d["q"]), rs(d["dq"]), rs(d["tau"])
  fn = [str(s).split("_")[0] for s in d["foot_names"]]
  foot = rs(d["foot"]) * foot_scale

  quat = d["quat"]  # w, x, y, z
  w, x, y, z = quat.T
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  # Accelerometer = specific force in the body frame: R^T (a_world + g).
  a_w = np.gradient(d["linv_w"], dt, axis=0) + np.array([0.0, 0.0, 9.81])
  rot = np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ],
    axis=1,
  )  # [n, 3, 3] body->world
  acc_b = np.einsum("nji,nj->ni", rot, a_w)
  imu = np.concatenate(
    [roll[:, None], pitch[:, None], d["angv_b"], acc_b], axis=1
  )  # roll, pitch, gyro xyz, acc xyz
  imu = rs(imu)

  samples = []
  for k in range(len(t)):
    s: dict[str, float] = {}
    for leg in LEGS:
      for j in JOINTS:
        i = jn.index(f"{leg}_{j}_joint")
        s[f"{leg}_{j}_q"] = float(q[k, i])
        s[f"{leg}_{j}_dq"] = float(dq[k, i])
        s[f"{leg}_{j}_tau"] = float(tau[k, i])
    for leg in FOOT_ORDER:
      s[f"foot_{leg}"] = float(foot[k, fn.index(leg)])
    for name, v in zip(
      ("roll", "pitch", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"),
      imu[k],
      strict=True,
    ):
      s[name] = float(v)
    samples.append(s)
  return t, samples


def run_one(path: Path, cfg: EvalConfig) -> dict:
  d = np.load(path)
  t, samples = to_lowstate_samples(d, cfg.foot_scale)
  eng = make_engine(cfg)
  p_raw = np.full(len(t), np.nan)
  alarm = np.zeros(len(t), bool)
  alarm_leg = [""] * len(t)
  p_legs = np.full((len(t), 4), np.nan)
  for k, s in enumerate(samples):
    out = eng.push(s)
    if out is None:
      continue
    p_raw[k] = out["p_bin_raw"]
    alarm[k] = out["entangled"]
    alarm_leg[k] = out["alarm_leg"] or ""
    p_legs[k] = [out["p_legs"][leg] for leg in LEGS]

  dt = float(d["dt"])
  wire = d["wire"]
  wire_legs = [str(s) for s in d["wire_legs"]]
  first_c = {
    leg: (float(np.argmax(wire[:, i]) * dt) if wire[:, i].any() else None)
    for i, leg in enumerate(wire_legs)
  }
  touched = [v for v in first_c.values() if v is not None]
  t_contact = min(touched) if touched else None
  t_alarm = float(t[np.argmax(alarm)]) if alarm.any() else None
  return dict(
    name=path.stem,
    t=t,
    p_raw=p_raw,
    alarm=alarm,
    alarm_leg=alarm_leg,
    p_legs=p_legs,
    first_contact=first_c,
    t_contact=t_contact,
    t_alarm=t_alarm,
    wire=wire,
    dt=dt,
  )


def summarize(r: dict, thr: float) -> str:
  tc, ta = r["t_contact"], r["t_alarm"]
  p = r["p_raw"]
  valid = ~np.isnan(p)
  pre = valid & (r["t"] < (tc if tc is not None else 1e9))
  fire_pre = float(np.mean(p[pre] >= thr)) * 100 if pre.any() else float("nan")
  leg = next((r["alarm_leg"][k] for k in np.flatnonzero(r["alarm"])[:1]), "")
  lat = "-" if (tc is None or ta is None) else f"{(ta - tc) * 1000:+.0f}"
  fc = {k: (None if v is None else round(v, 2)) for k, v in r["first_contact"].items()}
  return (
    f"{r['name']:14} contact={'none' if tc is None else f'{tc:.2f}s'} "
    f"alarm={'none' if ta is None else f'{ta:.2f}s'} latency_ms={lat} "
    f"alarm_leg={leg or '-'} fire%pre-contact={fire_pre:.1f} "
    f"max_p_raw={np.nanmax(p):.4f} first_contact_by_leg={fc}"
  )


def plot(results: list[dict], out: Path, cfg: EvalConfig) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  out.mkdir(parents=True, exist_ok=True)
  n = len(results)
  cols = 2
  rows = (n + 1) // 2
  fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
  for ax, r in zip(axes.flat, results, strict=False):
    ax.plot(r["t"], r["p_raw"], lw=0.8, label="P(entangled) raw")
    ax.axhline(cfg.threshold, color="gray", ls=":", lw=0.8)
    for leg, tc in r["first_contact"].items():
      if tc is not None:
        ax.axvline(tc, color="red", alpha=0.5, lw=0.8)
        ax.text(tc, 1.02, leg, color="red", fontsize=7)
    if r["t_alarm"] is not None:
      ax.axvline(r["t_alarm"], color="green", lw=1.2)
    ax.set_title(r["name"], fontsize=8)
    ax.set_ylim(-0.02, 1.1)
  for ax in list(axes.flat)[n:]:
    ax.axis("off")
  fig.suptitle("TCN P(entangled) on sim logs. red = oracle wire contact, green = alarm")
  fig.tight_layout()
  fig.savefig(out / f"tcn_on_sim_{cfg.tag}.png", dpi=110)


def main() -> None:
  import tyro

  cfg = tyro.cli(EvalConfig)
  files = sorted(
    Path(cfg.raw).glob("*.npz"), key=lambda p: (p.stem.split("_")[1], p.stem)
  )
  results = []
  lines = []
  for f in files:
    r = run_one(f, cfg)
    results.append(r)
    lines.append(summarize(r, cfg.threshold))
    print(lines[-1], flush=True)
  out = Path(cfg.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / f"summary_{cfg.tag}.txt").write_text("\n".join(lines) + "\n")
  if cfg.plots:
    plot(results, out / "plots", cfg)


if __name__ == "__main__":
  main()
