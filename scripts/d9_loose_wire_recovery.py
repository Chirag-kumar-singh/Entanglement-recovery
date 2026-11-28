"""D9 step 2: the four-leg recovery (d6_1_front_pair_wire.py) on a loose-to-tight wire.

Runs the existing script unchanged, with the wire swapped for one that starts loose
(sagging, ``--slack-cm`` of extra length) and is pulled straight by a mocap anchor
``--tighten-delay`` s after the first leg touches it, over ``--tighten-time`` s. The tight
wire sits at ``--wire-height`` (default 0.10 m, the height every earlier result used).
The recovery trigger is still the contact sensor, so this tests the recovery itself, not
a detector. A geometric check counts robot/wire overlap (wire passing through a leg).

Our flags come first, everything else goes to ``d6_1_front_pair_wire.py`` (use ``--seed``,
one process per seed; ``--debug`` opens the viewer). Example (best four-leg recipe):

    uv run python scripts/d9_loose_wire_recovery.py --slack-cm 3 --tighten-time 2 \
      --pair all --start-x-range -0.485 -0.35 --seed 0 --out-dir logs/x \
      --max-attempts FL 2 FR 2 RL 2 RR 2 --max-recovery-steps 96 --td-window 17 19 \
      --serialize-gap-steps 12 --support-gate-z 0.04 --step-gate FL 6 FR 6 \
      --crossing-fallback 10 --new-legs FL FR --support-lift-thigh=-0.2 \
      --support-lift-calf=0.3 --support-lift-ramp-steps 12 --step-gate-timeout 30 \
      --support-lift-trigger FL FR
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_1_front_pair_wire as d61  # noqa: E402
from d9_loose_wire_lib import (  # noqa: E402
  OverlapMonitor,
  build_nodes,
  make_loose_wire_spec_fn,
)

from mjlab.tasks.recovery.env_cfg_wire import DEFAULT_WIRE_CFG  # noqa: E402

LEGS = d61.PAIRS["all"]


@dataclass
class LooseConfig:
  slack_cm: float = 3.0
  """Extra chain length over the 0.5 m span, in cm (0 = a straight wire)."""
  tighten_delay: float = 0.2
  """Seconds after the first wire contact before the anchor starts to move."""
  tighten_time: float = 2.0
  """Seconds the anchor takes to straighten the wire. <= 0 = never tighten."""
  anchor_solref: float = 0.005
  """End-anchor constraint time constant (s); 0.02 = the original wire."""
  overtravel_cm: float = 0.0
  """How far past straight the anchor ends (cm)."""
  original_wire: bool = False
  """Keep the original wire (no swap, no tightening); only the overlap check is added."""


LOOSE = LooseConfig()
STATE: dict = {}

_orig_cfg = d61.go2_wire_recovery_residual_env_cfg
_orig_build = d61.build_env


def _cfg_with_loose_wire(*a, **k):
  cfg = _orig_cfg(*a, **k)
  if LOOSE.original_wire:
    return cfg
  height = k["wire_cfg"].anchor_start[2]
  nodes = build_nodes(
    0.5, height, LOOSE.slack_cm / 100.0, DEFAULT_WIRE_CFG.num_segments
  )
  wire = dataclasses.replace(DEFAULT_WIRE_CFG, eq_solref=(LOOSE.anchor_solref, 1.0))
  cfg.scene.spec_fn = make_loose_wire_spec_fn(wire, nodes)
  STATE["nodes"] = nodes
  STATE["height"] = height
  return cfg


def _build_with_hook(cfg, seed, device):
  env = _orig_build(cfg, seed, device)
  nodes = STATE.get("nodes")
  tighten = not LOOSE.original_wire and LOOSE.tighten_time > 0
  if tighten:
    y_tight = float(
      nodes[0, 1]
      + np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()
      + LOOSE.overtravel_cm / 100.0
    )
  sensors = env.scene.sensors
  wire_sensors = [sensors[f"wire_contact_{leg}"] for leg in LEGS]
  mocap = wp.to_torch(env.sim.wp_data.mocap_pos) if tighten else None  # shares memory
  mon = OverlapMonitor(env)
  dt = env.physics_dt
  STATE.update(
    sub=0,
    t_first=None,
    worst=np.zeros(len(mon.names)),
    deep=0,
    wire_r=mon.wire_r,
    names=mon.names,
  )
  orig = env.metrics_manager.compute_substep

  def hooked(*a, **k):
    sub = STATE["sub"]
    if STATE["t_first"] is None and any(
      bool((s.data.found[0] > 0).any()) for s in wire_sensors
    ):
      STATE["t_first"] = sub * dt
    if tighten and STATE["t_first"] is not None:
      t0 = STATE["t_first"] + LOOSE.tighten_delay
      frac = min(max((sub * dt - t0) / LOOSE.tighten_time, 0.0), 1.0)
      mocap[0, 0, 1] = float(nodes[-1, 1] + frac * (y_tight - nodes[-1, 1]))
    gap = mon.gap()
    STATE["worst"] = np.maximum(STATE["worst"], -gap)
    STATE["deep"] += int((gap.min() < -mon.wire_r))
    STATE["sub"] += 1
    return orig(*a, **k)

  env.metrics_manager.compute_substep = hooked  # type: ignore[method-assign]
  return env


d61.go2_wire_recovery_residual_env_cfg = _cfg_with_loose_wire
d61.build_env = _build_with_hook


def main() -> None:
  global LOOSE
  loose, rest = tyro.cli(LooseConfig, args=sys.argv[1:], return_unknown_args=True)
  LOOSE = loose
  cfg = tyro.cli(d61.D61Config, args=rest)
  if cfg.seeds:
    raise SystemExit("use --seed N (one process per seed)")
  if cfg.out_dir:
    cfg = dataclasses.replace(cfg, out_dir=f"{cfg.out_dir}/seed{cfg.seed}")
  torch.set_grad_enabled(False)
  label, detail, events = d61.run(cfg, cfg.seed, quiet=not cfg.debug)
  worst = float(STATE["worst"].max()) * 1000
  part = STATE["names"][int(STATE["worst"].argmax())]
  print(
    f"seed={cfg.seed} {label} {detail} {events} | "
    f"{'original wire' if LOOSE.original_wire else f'loose slack={LOOSE.slack_cm:g}'} "
    f"T={LOOSE.tighten_time:g} first_contact={STATE['t_first']} "
    f"max_overlap_mm={max(worst, 0.0):.1f}({part}) deep_overlap_substeps={STATE['deep']}",
    flush=True,
  )


if __name__ == "__main__":
  main()
