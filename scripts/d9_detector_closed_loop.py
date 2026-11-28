"""D9 Stage 2 step 3: run the sim-trained detector inside the recovery episode.

Wraps ``d9_detector_data.py`` (same episode, wire swap and feature recorder). After every
control step the newest feature row goes through the trained causal CNN
(``d9_detector_train.py``); per leg, ``p >= --thr`` for ``--m`` consecutive steps is an alarm.

Two modes:
  default   log only. Alarms are recorded next to the oracle contact and a geometric
            per-part gap to the wire (thigh / calf1 / calf2 / foot of each leg), so early
            alarms can be matched against what actually touched the wire. Run it with
            recovery off (``--entry-delay-steps 1000000000``).
  --act     the alarm replaces the contact sensor as each leg's recovery entry trigger
            (``ResidualRecoveryAction.external_trigger``). The release logic still reads
            the oracle contact sensor.

Own flags are stripped here; everything else goes to ``d9_detector_data.py``.

Usage (closed loop, best four-leg recipe, one seed):
    uv run python scripts/d9_detector_closed_loop.py --act --seed 10 --kind closed \
      --pair all --start-x-range -0.485 -0.35 --original-wire <recipe flags>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d9_detector_data as D  # noqa: E402  (applies the wire-swap + recorder patches)
from d9_detector_train import Net, windows  # noqa: E402
from d9_loose_wire_lib import OverlapMonitor  # noqa: E402

from mjlab.tasks.recovery.mdp.actions import LEG_NAMES  # noqa: E402

d61 = D.d61
LEGS = D.LEGS


@dataclass
class ClosedConfig:
  model: str = "logs/d9_detector/model_v1.pt"
  thr: float = 0.9
  m: int = 1
  """Consecutive 50 Hz steps with p >= thr that make an alarm."""
  act: bool = False
  """Use the alarm as the recovery entry trigger (otherwise log only)."""
  act_legs: tuple[str, ...] = ("FR", "FL", "RR", "RL")
  """Legs whose entry the alarm controls with --act; the others keep the oracle trigger."""
  no_early: bool = False
  """Diagnostic (uses the oracle): ignore a leg's alarm until its oracle contact has
  happened, which removes early entries and false alarms."""
  touch_mm: float = 3.0
  """A part counts as touching the wire when its gap is below this many mm."""


CC = ClosedConfig()
LOG: dict = {}
_prev_build = d61.build_env


def _build_with_detector(cfg, seed, device):
  env = _prev_build(cfg, seed, device)
  ck = torch.load(CC.model, weights_only=False)
  w = int(ck["window"])
  mu, sd = np.asarray(ck["mu"]), np.asarray(ck["sd"])
  net = Net(len(mu), w)
  net.load_state_dict(ck["state"])
  net.eval()
  term = env.action_manager.get_term("joint_pos")
  mon = OverlapMonitor(env)
  LOG.update(
    p=[],
    alarm=[],
    gap=[],
    names=mon.names,
    run=np.zeros(4, int),
    seen=np.zeros(4, bool),
  )
  # Detector legs are ordered FR FL RR RL, the controller's terms FL FR RL RR: map by name.
  leg_term = {LEG_NAMES[t.cfg.affected_leg_idx]: t for t in term.legs}
  acted = [i for i, name in enumerate(LEGS) if name in CC.act_legs]
  if CC.act:
    for i in acted:
      leg_term[LEGS[i]].external_trigger = torch.zeros(env.num_envs, dtype=torch.bool)
  rows = D.REC
  orig_step = env.step

  def step(action):
    out = orig_step(action)
    if len(rows["X"]) > len(LOG["p"]):
      x = (np.stack(rows["X"][-w:]) - mu) / sd
      x = windows(x, w)[-1:]
      with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(x, dtype=torch.float32)))[0].numpy()
      hot = p >= CC.thr
      LOG["run"] = np.where(hot, LOG["run"] + 1, 0)
      alarm = LOG["run"] >= CC.m
      LOG["p"].append(p)
      LOG["alarm"].append(alarm.copy())
      LOG["gap"].append(mon.gap())
      if CC.no_early:
        LOG["seen"] |= np.asarray(rows["Y"][-1], bool)
        alarm = alarm & LOG["seen"]
      if CC.act:
        for i in acted:
          leg_term[LEGS[i]].external_trigger = torch.tensor([bool(alarm[i])])
    return out

  env.step = step  # type: ignore[method-assign]
  return env


d61.build_env = _build_with_detector


def summarize() -> str:
  p = np.array(LOG["p"])
  a = np.array(LOG["alarm"])
  gap = np.array(LOG["gap"]) * 1000
  names = LOG["names"]
  touch = gap < CC.touch_mm
  y = np.stack(D.REC["Y"])[: len(a)]
  fa = int(np.argmax(a.any(1))) if a.any() else -1
  leg_a = LEGS[int(np.argmax(a[fa]))] if fa >= 0 else "-"
  fc = int(np.argmax(y.any(1))) if y.any() else -1
  ft = int(np.argmax(touch.any(1))) if touch.any() else -1
  part = names[int(np.argmax(touch[ft]))] if ft >= 0 else "-"
  per_leg = {
    leg: int(np.argmax(a[:, i])) if a[:, i].any() else -1 for i, leg in enumerate(LEGS)
  }
  oracle = {
    leg: int(np.argmax(y[:, i])) if y[:, i].any() else -1 for i, leg in enumerate(LEGS)
  }
  print(f"DETECT per_leg_alarm={per_leg} per_leg_oracle={oracle}")
  if fc >= 0:
    g = {n: round(float(v), 1) for n, v in zip(names, gap[fc], strict=True)}
    print(f"DETECT gaps_mm_at_first_contact={g}")
  if fa >= 0:
    j = int(np.argmin(gap[fa]))
    print(
      f"DETECT at_alarm: lead_steps={fc - fa} closest={names[j]}:{gap[fa, j]:.1f}mm "
      f"descent_of_closest_since_alarm-10={gap[max(fa - 10, 0), j] - gap[fa, j]:.1f}mm"
    )
  print(f"DETECT min_gap_mm={gap.min():.1f} at {names[int(np.argmin(gap.min(0)))]}")
  return (
    f"DETECT first_alarm={fa}({leg_a}) first_oracle_contact={fc} "
    f"first_any_part_touch={ft}({part}) max_p={p.max():.3f}"
  )


def main() -> None:
  global CC
  CC, rest = tyro.cli(ClosedConfig, args=sys.argv[1:], return_unknown_args=True)
  sys.argv = [sys.argv[0], *rest]
  try:
    D.main()
  finally:
    if LOG.get("p"):
      print(summarize(), flush=True)


if __name__ == "__main__":
  main()
