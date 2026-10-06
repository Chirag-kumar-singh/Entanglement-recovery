"""Ball-joint articulated wire/cable, for use as the entanglement obstacle
in recovery training.

Mechanically this is a single MuJoCo kinematic tree (parent/child bodies
connected by ``mjJNT_BALL`` joints), not the older free-jointed/soft-
equality-chain cable in :mod:`mjlab.tasks.cable_test.cable`. Only ONE
``mjEQ_CONNECT`` equality constraint is used in the whole wire, pinning
the free tip to a fixed world anchor so the wire spans/stays roughly taut
across the path -- the ball joints themselves are an exact (not soft)
kinematic constraint between segments.

This exact formulation (geometry, contact params, anti-tunneling margin,
timestep-appropriate parameters) was validated in a multi-stage read-only
diagnostic (see chat history / test2-test4 scripts under
``/tmp/cable_ball_test``) against the existing trained Go2 walking
policy: stable (no NaN/Inf), no tunneling, and the FL calf makes real,
measurably-loaded contact with the wire. Do not change these defaults
without re-running that diagnostic.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco

BALL_CABLE_SEG_PREFIX = "wire_seg"


@dataclasses.dataclass(frozen=True)
class BallCableCfg:
  """Configuration for the ball-joint articulated wire."""

  anchor_start: tuple[float, float, float] = (0.0, 0.0, 0.10)
  anchor_end: tuple[float, float, float] = (0.0, 0.5, 0.10)

  radius: float = 0.006  # 6mm radius (12mm diameter).
  aspect_ratio: float = 2.5  # segment_length / diameter.

  # Contact parameters (matches the validated diagnostic configuration).
  condim: int = 4
  friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)
  geom_solref: tuple[float, float] = (0.01, 1.0)
  geom_solimp: tuple[float, float, float, float, float] = (0.99, 0.99, 0.01, 0.5, 2.0)
  priority: int = 1
  # Anti-tunneling margin: generous contact-detection buffer, validated
  # against the wire's observed ~10 m/s peak segment velocity at a
  # 0.0025s physics timestep. Do not shrink without re-verifying no
  # tunneling occurs (see diagnostic history).
  margin: float = 0.015

  # Tip-pin equality constraint (the only soft constraint in the wire).
  eq_solref: tuple[float, float] = (0.02, 1.0)
  eq_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)

  rgba: tuple[float, float, float, float] = (0.1, 0.9, 0.9, 1.0)

  contype: int = 1
  conaffinity: int = 1

  @property
  def num_segments(self) -> int:
    total_span = math.dist(self.anchor_start, self.anchor_end)
    seg_len_target = self.aspect_ratio * (2 * self.radius)
    return max(2, round(total_span / seg_len_target))

  @property
  def segment_len(self) -> float:
    total_span = math.dist(self.anchor_start, self.anchor_end)
    return total_span / self.num_segments

  @property
  def mass_per_segment(self) -> float:
    # Scaled by capsule volume ratio vs. the validated 10mm-radius,
    # 0.05m-long, 0.02kg baseline segment (volume ~ radius^2 * length).
    volume_ratio = (self.radius / 0.01) ** 2 * (self.segment_len / 0.05)
    return 0.02 * volume_ratio

  @property
  def joint_damping(self) -> float:
    return 0.05 * (self.radius / 0.01)

  @property
  def joint_armature(self) -> float:
    return 0.00005 * (self.radius / 0.01)

  def segment_names(self, env_id: int = 0) -> tuple[str, ...]:
    prefix = f"{BALL_CABLE_SEG_PREFIX}{env_id}_"
    return tuple(f"{prefix}{i}" for i in range(self.num_segments))

  def geom_names(self, env_id: int = 0) -> tuple[str, ...]:
    return tuple(f"{n}_geom" for n in self.segment_names(env_id))


def _add_single_ball_cable(
  spec: mujoco.MjSpec,
  cfg: BallCableCfg,
  env_origin_xy: tuple[float, float],
  env_id: int,
) -> None:
  """Add one copy of the ball-joint wire, offset by an environment's grid
  origin (x, y only, matching the terrain's own per-env grid layout)."""
  import numpy as np

  ox, oy = env_origin_xy
  p0 = np.array(cfg.anchor_start) + np.array([ox, oy, 0.0])
  p1 = np.array(cfg.anchor_end) + np.array([ox, oy, 0.0])
  n = cfg.num_segments
  seg_vec = (p1 - p0) / n

  prefix = f"{BALL_CABLE_SEG_PREFIX}{env_id}_"
  root_name = f"{prefix}0"
  root_body = spec.worldbody.add_body(name=root_name, pos=tuple(p0))
  root_body.add_geom(
    name=f"{root_name}_geom",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=(cfg.radius, 0.0, 0.0),
    fromto=(0.0, 0.0, 0.0, *seg_vec),
    rgba=cfg.rgba,
    contype=cfg.contype,
    conaffinity=cfg.conaffinity,
    mass=cfg.mass_per_segment,
    condim=cfg.condim,
    friction=cfg.friction,
    solref=cfg.geom_solref,
    solimp=cfg.geom_solimp,
    priority=cfg.priority,
    margin=cfg.margin,
  )

  prev_body = root_body
  prev_name = root_name
  for i in range(1, n):
    name = f"{prefix}{i}"
    body = prev_body.add_body(name=name, pos=tuple(seg_vec))
    body.add_joint(
      type=mujoco.mjtJoint.mjJNT_BALL,
      damping=cfg.joint_damping,
      armature=cfg.joint_armature,
    )
    body.add_geom(
      name=f"{name}_geom",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=(cfg.radius, 0.0, 0.0),
      fromto=(0.0, 0.0, 0.0, *seg_vec),
      rgba=cfg.rgba,
      contype=cfg.contype,
      conaffinity=cfg.conaffinity,
      mass=cfg.mass_per_segment,
      condim=cfg.condim,
      friction=cfg.friction,
      solref=cfg.geom_solref,
      solimp=cfg.geom_solimp,
      priority=cfg.priority,
      margin=cfg.margin,
    )
    spec.add_exclude(bodyname1=prev_name, bodyname2=name)
    prev_body = body
    prev_name = name

  eq = spec.add_equality(
    type=mujoco.mjtEq.mjEQ_CONNECT,
    name1=prev_name,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name=f"wire_anchor_end_{env_id}",
    solref=cfg.eq_solref,
    solimp=cfg.eq_solimp,
  )
  eq.data[0:3] = tuple(seg_vec)


def build_ball_cable_spec_fn(
  cfg: BallCableCfg,
  env_spacing: float = 2.0,
):
  """Return a ``spec_fn`` adding ``num_envs`` independent wire copies, one
  per environment grid cell (same grid layout as
  ``TerrainEntity._compute_env_origins_grid`` for ``terrain_type="plane"``),
  matching the pattern used by the rigid-rod recovery env
  (:mod:`mjlab.tasks.recovery.env_cfg`).

  The returned ``spec_fn`` has signature ``(spec, num_envs) -> None``,
  matching ``SceneCfg.spec_fn``'s calling convention: ``Scene.__init__``
  invokes it with the scene's authoritative, final ``num_envs`` (resolved
  after any CLI/eval overrides to ``SceneCfg.num_envs``), NOT a value
  captured when this factory was called. Always use the ``num_envs``
  argument passed in at call time -- do not close over a value captured
  here, since the caller's `num_envs`-at-config-construction-time may be
  stale by the time the scene is actually built (e.g. deep-copied/
  reconstructed by a task registry or tyro CLI parsing)."""

  def _grid_origins(n: int, spacing: float) -> list[tuple[float, float]]:
    num_rows = math.ceil(n / int(math.isqrt(n)) if n > 0 else 1)
    num_cols = math.ceil(n / num_rows)
    origins = []
    for idx in range(n):
      ii = idx // num_cols
      jj = idx % num_cols
      ox = -(ii - (num_rows - 1) / 2) * spacing
      oy = (jj - (num_cols - 1) / 2) * spacing
      origins.append((ox, oy))
    return origins

  def _add_wires(spec: mujoco.MjSpec, num_envs: int) -> None:
    origins = _grid_origins(num_envs, env_spacing)
    for env_id, origin_xy in enumerate(origins):
      _add_single_ball_cable(spec, cfg, origin_xy, env_id)

  return _add_wires
