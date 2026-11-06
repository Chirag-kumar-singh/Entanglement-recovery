#!/usr/bin/env python
"""D5.40 bounded playback artifacts for baseline vs gradual transition.

This reuses the exact seed/config/reset logic from the D5.40 evaluator,
but records playable MP4s instead of relying on the native viewer.

Outputs go under ``logs/d5_40/playback_artifacts/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from d5_40_annotated_playback import (  # noqa: F401
  D540Config,
  PlaybackConfig as AnnotatedPlaybackConfig,
  RunSpec,
  _record_run,
  configure_torch_backends,
)


@dataclass(frozen=True)
class PlaybackConfig(AnnotatedPlaybackConfig):
  out_dir: str = "logs/d5_40/playback_artifacts"


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
  raise SystemExit

import mediapy as media
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from d5_40_gradual_thigh_transition_experiment import (  # noqa: E402
  D540Config,
  RunSpec,
)

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
  go2_wire_recovery_residual_env_cfg,
)
from mjlab.tasks.recovery.mdp.actions import (  # noqa: E402
  PHASE_NORMAL_WALK,
  PHASE_POST_RECOVERY,
  PHASE_RECOVERY,
  ResidualRecoveryAction,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

PHASE_NAMES = {
  PHASE_NORMAL_WALK: "NORMAL_WALK",
  PHASE_RECOVERY: "RECOVERY",
  PHASE_POST_RECOVERY: "POST_RECOVERY",
}


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
  out_dir: str = "logs/d5_40/playback_artifacts"
  runs: tuple[RunSpec, ...] = D540Config().runs


def _render_frame(env) -> np.ndarray | None:
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
    "duration": float(stream.get("duration", payload.get("format", {}).get("duration", 0.0))),
    "nb_frames": stream.get("nb_frames"),
  }


def _record_run(run: RunSpec, device: str, max_steps: int, fps: float, out_dir: Path) -> dict[str, Any]:
  env = _build_env(run, device)
  act = cast(ResidualRecoveryAction, env.action_manager.get_term("joint_pos"))
  env.reset(seed=run.seed)

  frames: list[np.ndarray] = []
  maybe_frame = _render_frame(env)
  if maybe_frame is not None:
    frames.append(maybe_frame)

  entered_recovery = False
  post_recovery_entered = False
  recovery_exit_reason = "not_entered"
  observation_outcome = "horizon_exhausted"

  try:
    for _step in range(max_steps):
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

  video_path = out_dir / f"{run.label}.mp4"
  media.write_video(str(video_path), frames, fps=fps)
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
    "frame_count": len(frames),
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

  print(f"Wrote D5.40 playback artifacts to {out_dir}")
  for summary in summaries:
    info = summary["video_info"]
    print(
      f"{summary['label']}: {summary['video_path']} "
      f"({info['width']}x{info['height']}, {info['duration']:.3f}s, frames={summary['frame_count']})"
    )


if __name__ == "__main__":
  main()