#!/usr/bin/env python
"""D5.40 annotated playback artifacts for baseline vs gradual transition.

This reuses the exact D5.40 seed/config/reset logic and the same
`rgb_array` render path, but draws evaluator-derived annotations on each
frame so the motion can be inspected without relying on the native
viewer.

The annotations are derived from the live recovery metrics and geometry,
not from image inference: signed forward distance, height above the wire
reference, phase, and Stage 1/Stage 2 latch state all come directly from
`ResidualRecoveryAction` and the evaluator’s coordinate convention.

Outputs go under ``logs/d5_40/annotated_playback/``.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from d5_40_gradual_thigh_transition_experiment import D540Config, RunSpec

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.tasks.recovery.mdp.actions import (
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
  ResidualRecoveryAction,
)
from mjlab.utils.torch import configure_torch_backends

PHASE_NAMES = {
  PHASE_NORMAL_WALK: "NORMAL_WALK",
  PHASE_RECOVERY: "RECOVERY",
  PHASE_POST_RECOVERY: "POST_RECOVERY",
}

REFERENCE_CSV_DIR = Path("logs/d5_40/gradual_thigh_transition_experiment")


def _build_env(run: RunSpec, device: str) -> ManagerBasedRlEnv:
  env_cfg = go2_wire_recovery_residual_env_cfg(
    affected_leg_idx=0,
    num_envs=1,
    recovery_mode="hybrid",
    policy_device=device,
    play=True,
    wire_cfg=ISOLATED_FL_WIRE_CFG,
    spawn_y_override=ISOLATED_FL_WIRE_SPAWN_Y,
    phase_aware_thresholds=False,
    max_recovery_steps=48,
    experimental_two_phase_hybrid_recovery=run.experimental_two_phase,
    experimental_two_phase_lift_confirm_steps=run.lift_confirm_steps,
    experimental_two_phase_transition_steps=run.transition_steps,
  )
  env_cfg.seed = run.seed
  return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")


@dataclass(frozen=True)
class PlaybackConfig:
  device: str = "cpu"
  max_steps: int = 300
  fps: float = 50.0
  out_dir: str = "logs/d5_40/annotated_playback"
  runs: tuple[RunSpec, ...] = D540Config().runs


def _render_frame(env: ManagerBasedRlEnv) -> np.ndarray | None:
  frame = env.render()
  if frame is None:
    return None
  if frame.dtype != np.uint8:
    frame = np.clip(frame, 0, 255).astype(np.uint8)
  return frame


def _verify_video(path: Path) -> dict[str, Any]:
  probe = subprocess.run(
    [
      "ffprobe",
      "-v",
      "error",
      "-show_format",
      "-show_streams",
      "-of",
      "json",
      str(path),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(probe.stdout)
  stream = next(
    (item for item in payload.get("streams", []) if item.get("codec_type") == "video"),
    None,
  )
  if stream is None:
    raise RuntimeError(f"No video stream found in {path}")
  return {
    "codec_name": stream.get("codec_name"),
    "width": int(stream.get("width", 0)),
    "height": int(stream.get("height", 0)),
    "duration": float(
      stream.get("duration", payload.get("format", {}).get("duration", 0.0))
    ),
    "nb_frames": stream.get("nb_frames"),
  }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
  if font_path.exists():
    return ImageFont.truetype(str(font_path), size=size)
  return ImageFont.load_default()


def _draw_badge(draw: ImageDraw.ImageDraw, xy, text: str, fill, font) -> None:
  x, y = xy
  pad_x, pad_y = 10, 5
  bbox = draw.textbbox((0, 0), text, font=font)
  width = bbox[2] - bbox[0] + 2 * pad_x
  height = bbox[3] - bbox[1] + 2 * pad_y
  draw.rounded_rectangle([x, y, x + width, y + height], radius=10, fill=fill)
  draw.text((x + pad_x, y + pad_y - 1), text, fill=(255, 255, 255), font=font)


def _draw_schematic(
  draw: ImageDraw.ImageDraw,
  box: tuple[int, int, int, int],
  row: dict[str, Any],
  x_limits: tuple[float, float],
  y_limits: tuple[float, float],
) -> None:
  left, top, right, bottom = box
  draw.rounded_rectangle(box, radius=18, fill=(18, 22, 30), outline=(70, 80, 100), width=2)

  margin_x = 50
  margin_y = 30
  plot_left = left + margin_x
  plot_right = right - 24
  plot_top = top + margin_y
  plot_bottom = bottom - 58

  x_min, x_max = x_limits
  y_min, y_max = y_limits
  if math.isclose(x_min, x_max):
    x_max = x_min + 1.0
  if math.isclose(y_min, y_max):
    y_max = y_min + 1.0

  def sx(value: float) -> float:
    return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

  def sy(value: float) -> float:
    return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

  grid_color = (50, 60, 75)
  for frac in np.linspace(0.0, 1.0, 5):
    x = plot_left + frac * (plot_right - plot_left)
    draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
  for frac in np.linspace(0.0, 1.0, 4):
    y = plot_top + frac * (plot_bottom - plot_top)
    draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)

  wire_x = sx(0.0)
  wire_y = sy(0.0)
  draw.line((wire_x, plot_top, wire_x, plot_bottom), fill=(245, 196, 0), width=4)
  draw.line((plot_left, wire_y, plot_right, wire_y), fill=(245, 196, 0), width=2)
  draw.text((wire_x + 6, plot_top + 2), "wire plane fd=0", fill=(245, 196, 0), font=_font(16))
  draw.text((plot_left + 4, wire_y + 4), "wire height 0 m", fill=(245, 196, 0), font=_font(16))

  def draw_point(x_val: float, y_val: float, color, label: str) -> None:
    px = sx(x_val)
    py = sy(y_val)
    r = 7
    draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(255, 255, 255), width=2)
    draw.text((px + 10, py - 14), label, fill=color, font=_font(16))

  draw_point(
    float(row["calf_forward_dist_m"]),
    float(row["calf_height_above_wire_m"]),
    (255, 116, 116),
    "calf",
  )
  draw_point(
    float(row["foot_forward_dist_m"]),
    float(row["foot_height_above_wire_m"]),
    (127, 191, 255),
    "toe",
  )

  def tick_label(value: float) -> str:
    return f"{value:+.2f}"

  for value in np.linspace(x_min, x_max, 4):
    px = sx(float(value))
    draw.line((px, plot_bottom, px, plot_bottom + 6), fill=(200, 200, 200), width=1)
    label = tick_label(float(value))
    bbox = draw.textbbox((0, 0), label, font=_font(14))
    draw.text((px - (bbox[2] - bbox[0]) / 2, plot_bottom + 8), label, fill=(200, 200, 200), font=_font(14))
  for value in np.linspace(y_min, y_max, 4):
    py = sy(float(value))
    draw.line((plot_left - 6, py, plot_left, py), fill=(200, 200, 200), width=1)
    label = f"{value:.2f}"
    bbox = draw.textbbox((0, 0), label, font=_font(14))
    draw.text((plot_left - 12 - (bbox[2] - bbox[0]), py - 8), label, fill=(200, 200, 200), font=_font(14))

  draw.text((left + 20, bottom - 48), "signed forward distance (m)  -> far side +", fill=(220, 220, 220), font=_font(16))
  draw.text((left + 20, bottom - 28), "height above wire (m)", fill=(220, 220, 220), font=_font(16))


def _annotate_frame(frame: np.ndarray, row: dict[str, Any], x_limits: tuple[float, float], y_limits: tuple[float, float]) -> np.ndarray:
  image = Image.fromarray(frame)
  base_w, base_h = image.size
  panel_w = 520
  canvas = Image.new("RGB", (base_w + panel_w, base_h), (9, 12, 18))
  canvas.paste(image, (0, 0))
  draw = ImageDraw.Draw(canvas)

  title_font = _font(26)
  body_font = _font(18)
  small_font = _font(16)

  panel_left = base_w
  draw.rectangle((panel_left, 0, base_w + panel_w, base_h), fill=(12, 16, 24))
  draw.line((panel_left, 0, panel_left, base_h), fill=(80, 92, 110), width=2)

  draw.text((panel_left + 18, 14), f"Seed {row['seed']}", fill=(255, 255, 255), font=title_font)
  draw.text((panel_left + 18, 48), str(row["variant"]).replace("_", " ").upper(), fill=(190, 220, 255), font=title_font)

  if row.get("frame_kind") == "reset":
    time_label = "reset state"
  else:
    time_label = (
      f"step {int(row['step'])}   |   t = {float(row['step']) * 0.02:.2f} s"
      f"   |   recovery t = {float(row['recovery_elapsed']) * 0.02:.2f} s"
    )
  draw.text((panel_left + 18, 90), time_label, fill=(230, 230, 230), font=body_font)
  draw.text((panel_left + 18, 116), f"phase: {row['phase']}", fill=(230, 230, 230), font=body_font)
  draw.text(
    (panel_left + 18, 142),
    f"exit: {row['recovery_exit_reason']}   |   outcome: {row['observation_outcome']}",
    fill=(230, 230, 230),
    font=body_font,
  )

  badge_y = 176
  phase_fill = {
    "NORMAL_WALK": (82, 82, 82),
    "RECOVERY": (208, 138, 14),
    "POST_RECOVERY": (38, 128, 64),
  }.get(str(row["phase"]), (82, 82, 82))
  _draw_badge(draw, (panel_left + 18, badge_y), f"phase {row['phase']}", phase_fill, body_font)
  stage1_fill = (38, 128, 64) if row["calf_clearance_achieved"] else (90, 95, 105)
  stage2_fill = (38, 128, 64) if row["foot_crossed_achieved"] else (90, 95, 105)
  cross_fill = (38, 128, 64) if row["crossing_achieved"] else (90, 95, 105)
  _draw_badge(draw, (panel_left + 18, badge_y + 42), f"stage 1 {'latched' if row['calf_clearance_achieved'] else 'pending'}", stage1_fill, small_font)
  _draw_badge(draw, (panel_left + 18, badge_y + 84), f"stage 2 {'latched' if row['foot_crossed_achieved'] else 'pending'}", stage2_fill, small_font)
  _draw_badge(draw, (panel_left + 18, badge_y + 126), f"crossing {'latched' if row['crossing_achieved'] else 'pending'}", cross_fill, small_font)

  info_lines = [
    f"calf fd = {float(row['calf_forward_dist_m']):+.3f} m",
    f"calf h  = {float(row['calf_height_above_wire_m']):+.3f} m",
    f"toe fd  = {float(row['foot_forward_dist_m']):+.3f} m",
    f"toe h   = {float(row['foot_height_above_wire_m']):+.3f} m",
    f"two-phase active = {bool(row['two_phase_advance_active'])}",
    f"switch step = {int(row['two_phase_switch_step'])}",
  ]
  text_y = 364
  for line in info_lines:
    draw.text((panel_left + 18, text_y), line, fill=(230, 230, 230), font=body_font)
    text_y += 26

  draw.text((panel_left + 18, 520), "Evaluator convention:", fill=(245, 196, 0), font=body_font)
  convention = [
    "forward distance uses the robot's latched",
    "heading at RECOVERY entry; positive = far side.",
  ]
  for idx, line in enumerate(convention):
    draw.text((panel_left + 18, 546 + 22 * idx), line, fill=(230, 230, 230), font=small_font)

  schematic_box = (panel_left + 18, 602 - 248, panel_left + panel_w - 18, 602)
  _draw_schematic(draw, schematic_box, row, x_limits=x_limits, y_limits=y_limits)

  marker_x = 18
  draw.line((marker_x, 8, marker_x, base_h - 8), fill=(245, 196, 0), width=3)
  draw.rounded_rectangle(
    (26, 10, 235, 46),
    radius=8,
    fill=(16, 18, 25),
    outline=(245, 196, 0),
    width=2,
  )
  draw.text((36, 16), "wire plane x=0", fill=(245, 196, 0), font=small_font)

  return np.asarray(canvas)


def _load_reference_rows(run: RunSpec) -> list[dict[str, Any]]:
  csv_path = REFERENCE_CSV_DIR / f"{run.label}.csv"
  if not csv_path.exists():
    raise FileNotFoundError(
      f"Missing evaluator CSV for {run.label}: {csv_path}. Run the D5.40 experiment first."
    )

  with csv_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    rows: list[dict[str, Any]] = []
    for raw_row in reader:
      row: dict[str, Any] = dict(raw_row)
      for key in ("step", "recovery_elapsed", "two_phase_lift_counter", "two_phase_switch_step"):
        if key in row and row[key] != "":
          row[key] = int(row[key])
      for key in (
        "calf_forward_dist_m",
        "calf_height_above_wire_m",
        "foot_forward_dist_m",
        "foot_height_above_wire_m",
        "effective_clearance_margin_m",
        "base_forward_vel_body_x",
      ):
        if key in row and row[key] != "":
          row[key] = float(row[key])
      for key in (
        "newly_entangled",
        "newly_recovered",
        "newly_succeeded",
        "calf_clearance_achieved",
        "foot_crossed_achieved",
        "crossing_achieved",
        "release_achieved",
        "recovery_failed_timeout",
        "two_phase_advance_active",
      ):
        if key in row and row[key] != "":
          row[key] = row[key] == "True"
      rows.append(row)
  return rows


def _sample_indices(num_rows: int) -> list[int]:
  raw_indices = {0, num_rows - 1}
  for fraction in (0.2, 0.4, 0.6, 0.8):
    raw_indices.add(min(num_rows - 1, max(0, int(round(fraction * (num_rows - 1))))))
  return sorted(raw_indices)


def _compare_rows(actual: dict[str, Any], expected: dict[str, Any], *, atol: float = 1e-6) -> None:
  keys_to_check = (
    "step",
    "phase",
    "recovery_elapsed",
    "recovery_offset",
    "post_recovery_elapsed",
    "calf_clearance_achieved",
    "foot_crossed_achieved",
    "crossing_achieved",
    "release_achieved",
    "calf_forward_dist_m",
    "calf_height_above_wire_m",
    "foot_forward_dist_m",
    "foot_height_above_wire_m",
    "effective_clearance_margin_m",
    "base_forward_vel_body_x",
    "two_phase_advance_active",
    "two_phase_lift_counter",
    "two_phase_switch_step",
  )
  for key in keys_to_check:
    if key not in actual or key not in expected:
      continue
    left = actual[key]
    right = expected[key]
    if isinstance(left, float) or isinstance(right, float):
      if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=0.0):
        raise AssertionError(f"{key} mismatch: actual={left!r} expected={right!r}")
    else:
      if left != right:
        raise AssertionError(f"{key} mismatch: actual={left!r} expected={right!r}")


def _verify_against_reference_csv(
  run: RunSpec, rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
  sample_indices = _sample_indices(len(rows))
  checks: list[dict[str, Any]] = []
  for sample_idx in sample_indices:
    actual = rows[sample_idx]
    expected = reference_rows[sample_idx]
    _compare_rows(actual, expected)
    checks.append(
      {
        "sample_index": sample_idx,
        "step": int(actual["step"]),
        "calf_forward_dist_m": float(actual["calf_forward_dist_m"]),
        "calf_height_above_wire_m": float(actual["calf_height_above_wire_m"]),
        "foot_forward_dist_m": float(actual["foot_forward_dist_m"]),
        "foot_height_above_wire_m": float(actual["foot_height_above_wire_m"]),
      }
    )
  print(f"[verify] {run.label}: matched {len(sample_indices)} sampled CSV rows")
  return checks


def _record_run(
  run: RunSpec, device: str, max_steps: int, fps: float, out_dir: Path
) -> dict[str, Any]:
  env = _build_env(run, device)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  entity = act._entity
  env.reset(seed=run.seed)

  rows: list[dict[str, Any]] = []
  frames: list[np.ndarray] = []
  entered_recovery = False
  post_recovery_entered = False
  recovery_exit_reason = "not_entered"
  observation_outcome = "horizon_exhausted"

  try:
    initial = _render_frame(env)
    if initial is not None:
      rows.append(
        {
          "frame_kind": "reset",
          "step": -1,
          "phase": PHASE_NAMES[int(act.phase[0].item())],
          "recovery_elapsed": int(act.recovery_elapsed[0].item()),
          "recovery_exit_reason": recovery_exit_reason,
          "observation_outcome": observation_outcome,
          "calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
          "calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
          "foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
          "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
          "calf_clearance_achieved": bool(act.calf_clearance_achieved[0].item()),
          "foot_crossed_achieved": bool(act.foot_crossed_achieved[0].item()),
          "crossing_achieved": bool(act.crossing_achieved[0].item()),
          "release_achieved": bool(act.release_achieved[0].item()),
          "two_phase_advance_active": bool(act.hybrid_two_phase_advance_active[0].item()),
          "two_phase_switch_step": int(act.two_phase_switch_step[0].item()),
          "two_phase_lift_counter": int(act.two_phase_lift_counter[0].item()),
          "base_forward_vel_body_x": float(entity.data.root_link_lin_vel_b[0, 0].item()),
          "variant": "gradual_transition" if run.experimental_two_phase else "baseline",
          "seed": run.seed,
        }
      )
      frames.append(initial)

    for step in range(1, max_steps + 1):
      actions = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
      _obs, _rew, terminateds, truncateds, _info = env.step(actions)

      phase = int(act.phase[0].item())
      entered_recovery = (
        entered_recovery or phase == PHASE_RECOVERY or bool(act.newly_entangled[0].item())
      )
      post_recovery_entered = (
        post_recovery_entered
        or phase == PHASE_POST_RECOVERY
        or bool(act.newly_recovered[0].item())
      )
      if bool(act.newly_recovered[0].item()):
        recovery_exit_reason = (
          "crossing_release"
          if bool(act.crossing_achieved[0].item()) and bool(act.release_achieved[0].item())
          else "cap_timeout"
        )

      row = {
        "frame_kind": "post_step",
        "step": step - 1,
        "phase": PHASE_NAMES[phase],
        "recovery_elapsed": int(act.recovery_elapsed[0].item()),
        "recovery_exit_reason": recovery_exit_reason,
        "observation_outcome": observation_outcome,
        "calf_forward_dist_m": float(act.calf_forward_dist[0].item()),
        "calf_height_above_wire_m": float(act.calf_height_above_wire[0].item()),
        "foot_forward_dist_m": float(act.foot_forward_dist[0].item()),
        "foot_height_above_wire_m": float(act.foot_height_above_wire[0].item()),
        "calf_clearance_achieved": bool(act.calf_clearance_achieved[0].item()),
        "foot_crossed_achieved": bool(act.foot_crossed_achieved[0].item()),
        "crossing_achieved": bool(act.crossing_achieved[0].item()),
        "release_achieved": bool(act.release_achieved[0].item()),
        "two_phase_advance_active": bool(act.hybrid_two_phase_advance_active[0].item()),
        "two_phase_switch_step": int(act.two_phase_switch_step[0].item()),
        "two_phase_lift_counter": int(act.two_phase_lift_counter[0].item()),
        "base_forward_vel_body_x": float(entity.data.root_link_lin_vel_b[0, 0].item()),
        "variant": "gradual_transition" if run.experimental_two_phase else "baseline",
        "seed": run.seed,
      }
      rows.append(row)

      maybe_frame = _render_frame(env)
      if maybe_frame is not None:
        frames.append(maybe_frame)

      if bool(act.newly_succeeded[0].item()):
        observation_outcome = "script_success"
      if bool(terminateds[0].item()) or bool(truncateds[0].item()):
        if bool(terminateds[0].item()):
          observation_outcome = "terminated"
        elif bool(truncateds[0].item()):
          observation_outcome = "truncated"
        break
  finally:
    env.close()

  csv_path = out_dir / f"{run.label}_playback_rows.csv"
  with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  reference_rows = _load_reference_rows(run)
  playback_rows = rows[1:]
  if len(reference_rows) < len(playback_rows):
    raise RuntimeError(
      f"Reference CSV for {run.label} is shorter than playback rows: {len(reference_rows)} < {len(playback_rows)}"
    )
  sample_checks = _verify_against_reference_csv(run, playback_rows, reference_rows)

  x_vals = [
    float(row["calf_forward_dist_m"]) for row in rows
  ] + [float(row["foot_forward_dist_m"]) for row in rows]
  y_vals = [
    float(row["calf_height_above_wire_m"]) for row in rows
  ] + [float(row["foot_height_above_wire_m"]) for row in rows]
  x_min = min(x_vals + [-0.08]) - 0.04
  x_max = max(x_vals + [0.28]) + 0.04
  y_min = min(y_vals + [0.0]) - 0.02
  y_max = max(y_vals + [0.12]) + 0.03

  annotated_frames = [
    _annotate_frame(frame, row, (x_min, x_max), (y_min, y_max))
    for frame, row in zip(frames, rows, strict=False)
  ]

  video_path = out_dir / f"{run.label}_annotated.mp4"
  media.write_video(str(video_path), annotated_frames, fps=fps)
  video_info = _verify_video(video_path)

  return {
    "label": run.label,
    "seed": run.seed,
    "variant": "gradual_transition" if run.experimental_two_phase else "baseline",
    "experimental_two_phase": run.experimental_two_phase,
    "entered_recovery": entered_recovery,
    "post_recovery_entered": post_recovery_entered,
    "recovery_exit_reason": recovery_exit_reason,
    "observation_outcome": observation_outcome,
    "frame_count": len(annotated_frames),
    "csv_path": str(csv_path),
    "reference_csv_path": str(REFERENCE_CSV_DIR / f"{run.label}.csv"),
    "sample_checks": sample_checks,
    "video_path": str(video_path),
    "video_info": video_info,
  }


def main() -> None:
  configure_torch_backends()
  cfg = PlaybackConfig()
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summaries = [_record_run(run, cfg.device, cfg.max_steps, cfg.fps, out_dir) for run in cfg.runs]
  manifest = out_dir / "manifest.json"
  manifest.write_text(json.dumps(summaries, indent=2, sort_keys=True))

  print(f"Wrote D5.40 annotated playback artifacts to {out_dir}")
  for summary in summaries:
    info = summary["video_info"]
    print(
      f"{summary['label']}: {summary['video_path']} "
      f"({info['width']}x{info['height']}, {info['duration']:.3f}s, frames={summary['frame_count']})"
    )


if __name__ == "__main__":
  main()