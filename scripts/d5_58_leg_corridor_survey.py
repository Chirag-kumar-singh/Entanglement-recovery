"""D5.58a -- Per-leg corridor survey (measurement, not an experiment).

Purpose: measure, for ALL FOUR legs simultaneously, where each leg's
`*_calf2_collision` geom (the geom the recovery task's `wire_contact`
sensor keys on) and each leg's foot site actually travel while the robot
walks forward under the frozen walking policy -- so that the per-leg
wire placements in `scripts/per_leg_wire.py` are MEASURED rather than
guessed, exactly as `ISOLATED_FL_WIRE_CFG` was derived for FL.

To get an UNPERTURBED reference walk, the wire is present in the scene
(so the geometry/reference frame is identical to the real runs) but is
contact-excluded from all four legs, so it never trips the robot and the
FSM never enters RECOVERY. This script therefore makes no claim about
recovery behaviour -- it only reports kinematics.

Outputs (under `--out-dir`):
  trajectory.csv -- per-step x/y/z of every leg's calf2 geom and foot
                    site, plus base pose.
  summary.txt    -- per-leg lateral (y) corridor and the step at which
                    that leg's calf2 first crosses the wire plane x=0.

Nothing in D5.3 or in the production configs is modified.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from per_leg_wire import (  # noqa: E402
  LEG_NAMES,
  PER_LEG_SPAWN_Y,
  PER_LEG_WIRE_CFG,
  make_leg_isolating_spec_fn,
)

from mjlab.tasks.recovery.env_cfg_residual import (  # noqa: E402
  go2_wire_recovery_residual_env_cfg,
)


@dataclass
class SurveyConfig:
  seed: int = 0
  device: str | None = None
  max_steps: int = 220
  survey_leg: str = "FL"
  """Which leg's wire placement to put in the scene for this survey. The
  wire is contact-excluded from EVERY leg regardless, so this only
  changes where the (inert) wire sits; all four legs are measured."""
  out_dir: str = "logs/d5_58_leg_corridor_survey"


def build_survey_env(cfg: SurveyConfig, device: str):
  wire = PER_LEG_WIRE_CFG[cfg.survey_leg]
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=LEG_NAMES.index(cfg.survey_leg),
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=wire,
    spawn_y_override=PER_LEG_SPAWN_Y,
  )
  env_cfg.seed = cfg.seed
  env_cfg.sim.device = device
  # Exclude ALL legs (target included) -> fully unperturbed reference walk.
  env_cfg.scene.spec_fn = make_leg_isolating_spec_fn(
    env_cfg.scene.spec_fn, cfg.survey_leg, extra_excluded_legs=(cfg.survey_leg,)
  )

  from mjlab.envs import ManagerBasedRlEnv

  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def run(cfg: SurveyConfig) -> Path:
  device = cfg.device or "cpu"
  env = build_survey_env(cfg, device)
  act = env.action_manager._terms["joint_pos"]
  entity = act._entity

  calf_idx = {}
  foot_idx = {}
  site_names = list(entity.site_names)
  for leg in LEG_NAMES:
    ids, _ = entity.find_geoms(f"{leg}_calf2_collision")
    assert len(ids) == 1, f"ambiguous calf geom for {leg}"
    calf_idx[leg] = ids[0]
    foot_idx[leg] = site_names.index(leg)

  wire = PER_LEG_WIRE_CFG[cfg.survey_leg]
  wire_plane_x = wire.anchor_start[0]

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env.reset(seed=cfg.seed)
  rows = []
  for step in range(cfg.max_steps):
    env.step(torch.zeros(env.action_manager.action.shape, device=device))
    base_pos = entity.data.root_link_pos_w[0]
    row = {
      "step": step,
      "phase": int(act.phase[0]),
      "base_x": float(base_pos[0]),
      "base_y": float(base_pos[1]),
      "base_z": float(base_pos[2]),
    }
    for leg in LEG_NAMES:
      cp = entity.data.geom_pos_w[0, calf_idx[leg], :]
      fp = entity.data.site_pos_w[0, foot_idx[leg], :]
      row[f"{leg}_calf_x"] = float(cp[0])
      row[f"{leg}_calf_y"] = float(cp[1])
      row[f"{leg}_calf_z"] = float(cp[2])
      row[f"{leg}_foot_x"] = float(fp[0])
      row[f"{leg}_foot_y"] = float(fp[1])
      row[f"{leg}_foot_z"] = float(fp[2])
    rows.append(row)

  env.close()

  csv_path = out_dir / "trajectory.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  lines = [
    f"survey_leg={cfg.survey_leg} seed={cfg.seed} steps={cfg.max_steps}",
    f"wire plane x={wire_plane_x:.3f} span y="
    f"[{wire.anchor_start[1]:.3f}, {wire.anchor_end[1]:.3f}] z="
    f"{wire.anchor_start[2]:.3f}",
    "",
  ]
  for leg in LEG_NAMES:
    cross = next((r["step"] for r in rows if r[f"{leg}_calf_x"] >= wire_plane_x), None)
    # Lateral corridor measured in the +-0.15 m x-window around the wire
    # plane (i.e. while that leg is actually in a position to snag it).
    near = [r for r in rows if abs(r[f"{leg}_calf_x"] - wire_plane_x) <= 0.15]
    ys = [r[f"{leg}_calf_y"] for r in near] or [float("nan")]
    zs = [r[f"{leg}_calf_z"] for r in near] or [float("nan")]
    lines.append(
      f"{leg}: calf crosses x={wire_plane_x:.2f} at step={cross}; "
      f"near-plane calf y in [{min(ys):.4f}, {max(ys):.4f}], "
      f"calf z in [{min(zs):.4f}, {max(zs):.4f}]"
    )
  summary = "\n".join(lines)
  (out_dir / "summary.txt").write_text(summary + "\n")
  print(summary)
  print(f"[D5.58a] wrote {csv_path}")
  return csv_path


def main() -> None:
  import tyro

  run(tyro.cli(SurveyConfig))


if __name__ == "__main__":
  main()
