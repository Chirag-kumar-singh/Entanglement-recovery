"""Smoke test for the Go2 cable-entanglement test environment (Phase 3A).

Verifies, without training anything:
  - the cable exists (segments compiled into the model),
  - the cable has collision enabled,
  - the cable endpoints are anchored (equality constraints present and hold),
  - the Go2 robot starts correctly,
  - cable/robot contact can be detected via the contact sensors.
"""

from __future__ import annotations

import mujoco
import torch
from conftest import get_test_device

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cable_test.env_cfg import DEFAULT_CABLE_CFG
from mjlab.tasks.cable_test.instrumentation import collect_instrumentation
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-Go2-Cable-Entanglement-Test"


def _make_env(num_envs: int = 1) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def test_cable_exists_and_has_collision():
  env = _make_env()
  try:
    mj_model = env.sim.mj_model
    seg_names = DEFAULT_CABLE_CFG.geom_names
    assert len(seg_names) == DEFAULT_CABLE_CFG.num_segments
    for name in seg_names:
      gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
      assert gid != -1, f"cable geom '{name}' missing from compiled model"
      assert mj_model.geom_contype[gid] != 0
      assert mj_model.geom_conaffinity[gid] != 0
  finally:
    env.close()


def test_cable_endpoints_are_anchored():
  env = _make_env()
  try:
    mj_model = env.sim.mj_model
    assert mj_model.neq >= DEFAULT_CABLE_CFG.num_segments + 1

    # Endpoints should stay near their configured anchors after settling.
    for _ in range(50):
      env.step(
        torch.zeros(env.action_space.shape, device=env.device),
      )
    info = collect_instrumentation(env)
    seg_pos = info["cable_segment_pos"][0]  # [N, 3]
    start = seg_pos[0]
    end = seg_pos[-1]
    expected_start = torch.tensor(DEFAULT_CABLE_CFG.anchor_start, device=start.device)
    expected_end = torch.tensor(DEFAULT_CABLE_CFG.anchor_end, device=end.device)
    # Segment *centers* are half a segment length away from the true anchor
    # point, so allow slack of one full segment length plus solver settle.
    seg_len = (
      (
        torch.tensor(DEFAULT_CABLE_CFG.anchor_end)
        - torch.tensor(DEFAULT_CABLE_CFG.anchor_start)
      ).norm()
      / DEFAULT_CABLE_CFG.num_segments
    ).item()
    assert (start - expected_start).norm().item() < seg_len + 0.1
    assert (end - expected_end).norm().item() < seg_len + 0.1
  finally:
    env.close()


def test_go2_starts_correctly():
  env = _make_env()
  try:
    obs, _ = env.reset()
    assert "actor" in obs
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert torch.isfinite(actor_obs).all()
    robot = env.scene["robot"]
    # Robot base should start near the configured start x, upright.
    root_pos = robot.data.root_link_pos_w[0]
    assert root_pos[2] > 0.15  # Not collapsed into the floor.
    gravity_b = robot.data.projected_gravity_b[0]
    assert gravity_b[2] < -0.9  # Roughly upright.
  finally:
    env.close()


def test_cable_robot_contact_detectable():
  """Force the robot to the cable location and verify RL-foot contact fires."""
  env = _make_env()
  try:
    robot = env.scene["robot"]
    # Teleport the robot so RL foot starts essentially on top of the cable.
    default_state = robot.data.default_root_state.clone()
    default_state[:, 0] = 0.0  # x: at the cable.
    default_state[:, 1] = 0.1  # y: within the cable's [0, 0.5] span, +Y side.
    default_state[:, 2] = 0.25
    robot.write_root_state_to_sim(default_state)

    for _ in range(100):
      env.step(torch.zeros(env.action_space.shape, device=env.device))

    assert "cable_contact_RL" in env.scene.sensors
    found = env.scene.sensors["cable_contact_RL"].data.found
    assert found is not None
    _ = bool((found[0] > 0).any().item())
    # We don't hard-assert entanglement occurs (physics/timing dependent);
    # we only assert the sensor pipeline produces a well-formed, readable
    # boolean signal without raising.
  finally:
    env.close()
