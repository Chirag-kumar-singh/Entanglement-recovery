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

  def segment_names(self) -> tuple[str, ...]:
    prefix = f"{BALL_CABLE_SEG_PREFIX}_"
    return tuple(f"{prefix}{i}" for i in range(self.num_segments))

  def geom_names(self) -> tuple[str, ...]:
    return tuple(f"{n}_geom" for n in self.segment_names())


def _add_single_ball_cable(
  spec: mujoco.MjSpec,
  cfg: BallCableCfg,
) -> None:
  """Add the ONE ball-joint wire copy that belongs in the shared MjSpec
  template.

  mujoco_warp's ``nworld=num_envs`` batching (invoked once, by
  ``Simulation._finish_init`` via ``mjwarp.put_data``) is solely
  responsible for turning this single wire -- like the single Go2 entity
  added by ``Scene._add_entities`` -- into ``num_envs`` independent,
  non-interacting physics copies. Do NOT loop over ``num_envs`` here: a
  per-env Python loop baking N wire copies into ONE spec combined with
  mujoco_warp's OWN per-world replication produces N wires PER world
  (O(num_envs^2) total wire bodies/contacts), not N wires total -- this
  was a real, previously-shipped bug (see git history / chat record) that
  made GPU memory and step time scale quadratically in num_envs.
  """
  import numpy as np

  p0 = np.array(cfg.anchor_start)
  p1 = np.array(cfg.anchor_end)
  n = cfg.num_segments
  seg_vec = (p1 - p0) / n

  prefix = f"{BALL_CABLE_SEG_PREFIX}_"
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
    name="wire_anchor_end",
    solref=cfg.eq_solref,
    solimp=cfg.eq_solimp,
  )
  eq.data[0:3] = tuple(seg_vec)


def build_ball_cable_spec_fn(cfg: BallCableCfg):
  """Return a ``spec_fn`` adding exactly ONE ball-joint wire copy to the
  shared ``MjSpec`` template, matching the single-instance pattern used
  for every other scene element (the Go2 robot entity, the ground-plane
  terrain geom, etc.): mujoco_warp's own ``nworld=num_envs`` batching
  (applied once, at ``Simulation`` construction) is what turns this one
  spec-level wire into ``num_envs`` independent per-world copies -- NOT
  anything done here.

  Physical per-env variation (e.g. the robot spawning at a different
  grid cell per world via ``env.scene.env_origins``) is irrelevant to the
  wire: since the robot's own reset pose for THIS task uses
  ``env_spacing=0.0`` (all robots reset to the same local (x, y), each in
  its own independent, non-interacting ``nworld`` physics copy), the
  single wire -- fixed at ``cfg.anchor_start``/``cfg.anchor_end`` in every
  world identically -- lines up with every world's robot without needing
  any per-env offset of its own.

  The returned ``spec_fn`` has signature ``(spec, num_envs) -> None`` to
  match ``SceneCfg.spec_fn``'s calling convention, but ``num_envs`` is
  unused: exactly one wire is added regardless of how many environments
  will ultimately be simulated."""

  def _add_wire(spec: mujoco.MjSpec, num_envs: int) -> None:
    del num_envs  # Unused: one wire in the spec, replicated by nworld.
    _add_single_ball_cable(spec, cfg)

  return _add_wire
