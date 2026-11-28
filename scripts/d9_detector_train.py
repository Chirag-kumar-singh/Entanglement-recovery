"""D9 Stage 2, step 2: train and evaluate a sim detector on the recorded episodes.

Input: the .npz episodes from ``d9_detector_data.py`` (50 Hz features, per-leg wire-contact
labels). Detector: a causal 1D CNN over the last ``--window`` control steps that predicts, for
every step, whether each leg's calf is touching the wire. A torque-threshold detector in the
style of the paper (hip/thigh/calf torque above normal) is evaluated the same way as the
baseline; its thresholds are calibrated on our sim data, not copied.

Split: by episode, test = seed % 4 == 0 (so no window or episode is shared). Episodes with a
deep wire/leg overlap are dropped. Metrics, per event = (episode, leg's first contact):
latency (ms from the leg's first contact to the first alarm of that leg), miss (no alarm within
``--horizon`` steps), early alarm (an alarm of that leg before its first contact); and per
negative episode: false-alarm episodes.

Usage:
    uv run python scripts/d9_detector_train.py --data logs/d9_detector/data_v0
"""

from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from torch import nn

LEGS = ("FR", "FL", "RR", "RL")
DT_MS = 20.0


@dataclass
class TrainConfig:
  data: str = "logs/d9_detector/data_v0"
  window: int = 10
  """Control steps of history per prediction (0.2 s at 50 Hz)."""
  epochs: int = 40
  lr: float = 2e-3
  horizon: int = 25
  """Steps after a leg's first contact within which an alarm counts (0.5 s)."""
  seed: int = 0
  out: str = "logs/d9_detector/model_v0.pt"


def load(path: str) -> list[dict]:
  eps = []
  for f in sorted(glob.glob(f"{path}/*.npz")):
    d = np.load(f)
    eps.append(
      dict(
        name=f.split("/")[-1][:-4],
        X=d["X"],
        c=d["contact"].astype(np.float32),
        kind=str(d["kind"]),
        seed=int(d["seed"]),
        deep=int(d["deep_overlap_substeps"]),
        names=[str(n) for n in d["feature_names"]],
      )
    )
  return eps


def windows(X: np.ndarray, w: int) -> np.ndarray:
  """[T, F] -> [T, F, w], causal, first frame repeated before the start."""
  pad = np.concatenate([np.repeat(X[:1], w - 1, axis=0), X], axis=0)
  return np.stack([pad[i : i + w] for i in range(len(X))]).transpose(0, 2, 1)


class Net(nn.Module):
  def __init__(self, n_in: int, w: int):
    super().__init__()
    self.body = nn.Sequential(
      nn.Conv1d(n_in, 48, 3, padding=0), nn.GELU(),
      nn.Conv1d(48, 48, 3, padding=0), nn.GELU(),
      nn.Conv1d(48, 48, 3, padding=0), nn.GELU(),
    )  # fmt: skip
    self.head = nn.Linear(48 * (w - 6), 4)

  def forward(self, x):
    return self.head(self.body(x).flatten(1))


def first_contact(c: np.ndarray) -> list[int]:
  return [int(np.argmax(c[:, k])) if c[:, k].any() else -1 for k in range(4)]


def alarms(p: np.ndarray, thr: float, m: int) -> np.ndarray:
  """p [T, 4] -> bool [T, 4]: p >= thr for m consecutive steps (current included)."""
  hot = p >= thr
  run = np.zeros_like(hot, dtype=int)
  for t in range(len(hot)):
    run[t] = np.where(hot[t], (run[t - 1] if t else 0) + 1, 0)
  return run >= m


def evaluate(
  eps: list[dict], scores: list[np.ndarray], thr: float, m: int, horizon: int
):
  lat, miss, early, n_ev = [], 0, 0, 0
  fa_eps, n_neg = 0, 0
  any_lat, any_miss, any_early, any_lead, n_any = [], 0, 0, [], 0
  grp = {"front": [], "rear": []}
  grp_miss = {"front": 0, "rear": 0}
  wrong_leg = 0
  for e, p in zip(eps, scores, strict=True):
    a = alarms(p, thr, m)
    fc = first_contact(e["c"])
    for k in range(4):
      if fc[k] < 0:
        continue
      n_ev += 1
      g = "front" if k < 2 else "rear"
      if a[: fc[k], k].any():
        early += 1
      hit = np.flatnonzero(a[fc[k] : fc[k] + horizon, k])
      if len(hit):
        lat.append(float(hit[0]) * DT_MS)
        grp[g].append(float(hit[0]) * DT_MS)
      else:
        miss += 1
        grp_miss[g] += 1
    touched = [f for f in fc if f >= 0]
    if touched:
      # Trigger view: the first alarm of ANY leg against the first contact of ANY leg.
      t0 = min(touched)
      a_any = a.any(axis=1)
      n_any += 1
      if a_any[:t0].any():
        any_early += 1
        any_lead.append((t0 - int(np.argmax(a_any))) * DT_MS)
      else:
        hit = np.flatnonzero(a_any[t0 : t0 + horizon])
        if len(hit):
          any_lat.append(float(hit[0]) * DT_MS)
          ts = t0 + int(np.argmax(a_any[t0:]))  # step of the first alarm
          named = np.flatnonzero(a[ts])
          wrong_leg += int(not any(e["c"][ts, k] for k in named))
        else:
          any_miss += 1
    if not any(f >= 0 for f in fc):
      n_neg += 1
      fa_eps += int(a.any())
  return dict(
    n_events=n_ev,
    median_ms=float(np.median(lat)) if lat else float("nan"),
    p90_ms=float(np.percentile(lat, 90)) if lat else float("nan"),
    miss=miss,
    early=early,
    front_median_ms=float(np.median(grp["front"])) if grp["front"] else float("nan"),
    front_miss=grp_miss["front"],
    rear_median_ms=float(np.median(grp["rear"])) if grp["rear"] else float("nan"),
    rear_miss=grp_miss["rear"],
    trigger_eps=n_any,
    trigger_median_ms=float(np.median(any_lat)) if any_lat else float("nan"),
    trigger_miss=any_miss,
    trigger_early=any_early,
    trigger_early_lead_ms=float(np.median(any_lead)) if any_lead else float("nan"),
    first_alarm_wrong_leg=wrong_leg,
    neg_eps=n_neg,
    false_alarm_eps=fa_eps,
  )


def torque_scores(eps: list[dict], ref: dict) -> list[np.ndarray]:
  """Per-leg score = largest z-score of any joint's max |torque| vs the normal-walking
  distribution (the paper's idea: torque above normal), shape [T, 4]."""
  names = eps[0]["names"]
  idx = [
    [names.index(f"taumax_{leg}_{j}") for j in ("hip", "thigh", "calf")] for leg in LEGS
  ]
  out = []
  for e in eps:
    z = (e["X"][:, [i for leg in idx for i in leg]] - ref["mu"]) / ref["sd"]
    out.append(z.reshape(len(z), 4, 3).max(axis=2))
  return out


def main() -> None:
  cfg = tyro.cli(TrainConfig)
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  eps = load(cfg.data)
  dropped = [e["name"] for e in eps if e["deep"] > 0]
  eps = [e for e in eps if e["deep"] == 0]
  print(
    f"{len(eps)} episodes ({len(dropped)} dropped for deep wire/leg overlap: {dropped})"
  )
  test = [e for e in eps if e["seed"] % 4 == 0]
  train = [e for e in eps if e["seed"] % 4 != 0]
  for nm, s in (("train", train), ("test", test)):
    kinds = {k: sum(e["kind"] == k for e in s) for k in sorted({e["kind"] for e in s})}
    print(f"{nm}: {len(s)} episodes {kinds}, steps {sum(len(e['X']) for e in s)}, "
          f"contact steps {int(sum(e['c'].sum() for e in s))}")  # fmt: skip

  allX = np.concatenate([e["X"] for e in train])
  mu, sd = allX.mean(0), allX.std(0) + 1e-6
  w = cfg.window

  def prep(s):
    xs = [windows((e["X"] - mu) / sd, w) for e in s]
    return xs

  xtr = torch.tensor(np.concatenate(prep(train)), dtype=torch.float32)
  ytr = torch.tensor(np.concatenate([e["c"] for e in train]), dtype=torch.float32)
  pos_w = ((1 - ytr.mean(0)) / ytr.mean(0).clamp_min(1e-3)).clamp(max=20.0)
  net = Net(xtr.shape[1], w)
  opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
  lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
  n = len(xtr)
  for ep in range(cfg.epochs):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, 256):
      b = perm[i : i + 256]
      opt.zero_grad()
      loss = lossf(net(xtr[b]), ytr[b])
      loss.backward()
      opt.step()
      tot += float(loss) * len(b)
    if ep % 10 == 9 or ep == 0:
      print(f"epoch {ep + 1:3d} loss {tot / n:.4f}")
  torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, window=w), cfg.out)

  def predict(s):
    net.eval()
    with torch.no_grad():
      return [
        torch.sigmoid(net(torch.tensor(x, dtype=torch.float32))).numpy()
        for x in prep(s)
      ]

  # Torque baseline reference: frames with no contact on any leg before the first contact.
  ref_rows = []
  names = train[0]["names"]
  cols = [
    names.index(f"taumax_{leg}_{j}") for leg in LEGS for j in ("hip", "thigh", "calf")
  ]
  for e in train:
    fc = [f for f in first_contact(e["c"]) if f >= 0]
    end = min(fc) if fc else len(e["X"])
    ref_rows.append(e["X"][:end][:, cols])
  ref_rows = np.concatenate(ref_rows)
  ref = dict(mu=ref_rows.mean(0), sd=ref_rows.std(0) + 1e-6)

  for label, s in (("TRAIN", train), ("TEST", test)):
    print(f"\n=== {label}: sim-trained CNN (window {w} steps) ===")
    sc = predict(s)
    for thr in (0.5, 0.7, 0.9):
      for m in (1, 2):
        r = evaluate(s, sc, thr, m, cfg.horizon)
        print(f"thr {thr} m {m}: {r}")
    print(f"=== {label}: torque-threshold baseline (z-score of max |torque|) ===")
    sc = torque_scores(s, ref)
    for thr in (3.0, 5.0, 8.0):
      for m in (1, 2):
        r = evaluate(s, sc, thr, m, cfg.horizon)
        print(f"z {thr} m {m}: {r}")


if __name__ == "__main__":
  main()
