"""Simple articulated cable (thin rope) built from capsule segments.

This module is used only by the Go2 cable-entanglement *test* environment
(:mod:`mjlab.tasks.cable_test`). It has nothing to do with the existing
recovery task or the Go2 walking tasks and does not modify them.

The cable is intentionally simple and stable rather than a realistic rope
solver:

- ``num_segments`` free-floating capsule bodies (each with a free joint) are
  laid out along the world Y axis.
- Consecutive segments are pinned together at their shared endpoint with a
  ``mjEQ_CONNECT`` equality constraint.
- The two end segments are additionally pinned to fixed points in the world
  with a body-to-world ``mjEQ_CONNECT`` equality constraint, which is what
  "anchored endpoints" means here: the two extreme points of the cable are
  constrained to stay at fixed world coordinates, even though the segments
  themselves are free bodies that can swing/sag under gravity and contact
  forces.
"""

from __future__ import annotations

import dataclasses

import mujoco

CABLE_BODY_PREFIX = "cable_seg"
CABLE_GEOM_PREFIX = "cable_seg"


@dataclasses.dataclass(frozen=True)
class CableCfg:
  """Configuration for the articulated test cable."""

  # World-frame endpoints of the cable (anchored).
  anchor_start: tuple[float, float, float] = (0.0, 0.0, 0.015)
  anchor_end: tuple[float, float, float] = (0.0, 0.5, 0.015)

  num_segments: int = 10
  radius: float = 0.01

  rgba: tuple[float, float, float, float] = (0.9, 0.75, 0.1, 1.0)

  # Collision bitmasks. Must overlap with the robot's collision geoms
  # (Go2 FULL_COLLISION uses contype=1, conaffinity=0) and with the ground
  # plane so the cable can rest on the floor and be stepped on.
  contype: int = 1
  conaffinity: int = 1

  mass_per_segment: float = 0.02

  # Stiffness of the connect-equality constraints linking segments together
  # and to the world anchors.
  #
  # Step 3B stability pass: the original (0.002, 1.0) solref was too stiff
  # relative to the segments' tiny mass/inertia and produced a NaN blow-up
  # once the cable's free oscillation touched the terrain (see diagnostic
  # report). Softened to (0.02, 1.0) -- MuJoCo's own default time-constant
  # magnitude -- as the first, isolated change to test.
  eq_solref: tuple[float, float] = (0.02, 1.0)
  eq_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)

  # Step 3B: modest linear+angular damping on each segment's free joint.
  # Previously 0 (fully undamped), which let any small perturbation ring
  # indefinitely through the chain. A small value adds dissipation without
  # noticeably changing the cable's rest shape/height.
  joint_damping: float = 0.05

  # Step 3B: disable collision between adjacent (link-connected) segments.
  # Consecutive segments share an endpoint by construction, so without this
  # exclusion they permanently register a (self-)contact at that shared
  # point, which fights the connect-equality constraint at the same
  # location and destabilizes the solver. Robot-cable and cable-terrain
  # collisions are untouched -- only cable-vs-adjacent-cable is excluded.
  exclude_adjacent_self_collision: bool = True

  @property
  def segment_names(self) -> tuple[str, ...]:
    return tuple(f"{CABLE_BODY_PREFIX}{i}" for i in range(self.num_segments))

  @property
  def geom_names(self) -> tuple[str, ...]:
    return tuple(f"{CABLE_GEOM_PREFIX}{i}" for i in range(self.num_segments))


def build_cable_spec_fn(cable_cfg: CableCfg):
  """Return a ``spec_fn`` post-processor that adds the cable to a spec.

  The returned callable takes an already-built :class:`mujoco.MjSpec` and
  mutates it in place by adding the cable bodies, joints, geoms and equality
  constraints. This is meant to be assigned to ``SceneCfg.spec_fn``.
  """

  def _add_cable(spec: mujoco.MjSpec, num_envs: int) -> None:
    del num_envs  # Unused: single cable, replicated by native nworld.
    start = mujoco.mju_zero3 if False else None  # no-op, keeps linters calm
    del start

    n = cable_cfg.num_segments
    p0 = cable_cfg.anchor_start
    p1 = cable_cfg.anchor_end
    seg_vec = tuple(p1[i] - p0[i] for i in range(3))
    seg_len = tuple(seg_vec[i] / n for i in range(3))
    total_len = sum(v * v for v in seg_vec) ** 0.5
    half_len = (total_len / n) / 2.0

    prev_body = None
    prev_local_end = None  # local anchor point (end) on previous segment.

    for i in range(n):
      center = tuple(p0[k] + seg_len[k] * (i + 0.5) for k in range(3))
      name = f"{CABLE_BODY_PREFIX}{i}"
      body = spec.worldbody.add_body(name=name, pos=center)
      body.add_joint(
        type=mujoco.mjtJoint.mjJNT_FREE,
        damping=cable_cfg.joint_damping,
      )
      # Local capsule axis: segment direction expressed in the body's local
      # frame (== world frame, since the body has no rotation offset).
      dir_local = tuple((seg_len[k] / 2.0) for k in range(3))
      fromto = (
        -dir_local[0],
        -dir_local[1],
        -dir_local[2],
        dir_local[0],
        dir_local[1],
        dir_local[2],
      )
      body.add_geom(
        name=f"{CABLE_GEOM_PREFIX}{i}",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(cable_cfg.radius, half_len, 0.0),
        fromto=fromto,
        rgba=cable_cfg.rgba,
        contype=cable_cfg.contype,
        conaffinity=cable_cfg.conaffinity,
        mass=cable_cfg.mass_per_segment,
      )

      local_start = tuple(-dir_local[k] for k in range(3))
      local_end = tuple(dir_local[k] for k in range(3))

      if prev_body is None:
        # First segment: pin its start point to the fixed world anchor.
        eq = spec.add_equality(
          type=mujoco.mjtEq.mjEQ_CONNECT,
          name1=name,
          objtype=mujoco.mjtObj.mjOBJ_BODY,
          name="cable_anchor_start",
          solref=cable_cfg.eq_solref,
          solimp=cable_cfg.eq_solimp,
        )
        eq.data[0:3] = local_start
      else:
        # Pin this segment's start to the previous segment's end.
        eq = spec.add_equality(
          type=mujoco.mjtEq.mjEQ_CONNECT,
          name1=prev_body,
          name2=name,
          objtype=mujoco.mjtObj.mjOBJ_BODY,
          name=f"cable_link_{i - 1}_{i}",
          solref=cable_cfg.eq_solref,
          solimp=cable_cfg.eq_solimp,
        )
        eq.data[0:3] = prev_local_end
        eq.data[3:6] = local_start

      # Step 3B: exclude collision between this segment and the previous one
      # (they share an endpoint via the connect-equality above, so letting
      # them also collide double-counts that contact and fights the
      # equality constraint at the exact same point).
      if cable_cfg.exclude_adjacent_self_collision and i > 0:
        spec.add_exclude(
          bodyname1=f"{CABLE_BODY_PREFIX}{i - 1}",
          bodyname2=name,
        )

      prev_body = name
      prev_local_end = local_end

    # Last segment: pin its end point to the fixed world anchor.
    assert prev_body is not None
    eq = spec.add_equality(
      type=mujoco.mjtEq.mjEQ_CONNECT,
      name1=prev_body,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      name="cable_anchor_end",
      solref=cable_cfg.eq_solref,
      solimp=cable_cfg.eq_solimp,
    )
    eq.data[0:3] = prev_local_end

  return _add_cable
