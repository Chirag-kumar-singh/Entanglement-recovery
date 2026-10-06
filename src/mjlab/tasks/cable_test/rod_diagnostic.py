"""TEMPORARY diagnostic module: single rigid capsule "rod" variant of the
articulated test cable.

This module is a CONTROL experiment only. It does not modify, import-and-
mutate, or replace anything in ``cable.py``, ``env_cfg.py``, or any other
existing source file. It provides an independent ``spec_fn`` that adds a
single static (no joint, no equality constraint) rigid capsule body to the
world, matching the existing test cable's:

- center position / orientation (same anchor endpoints, so the same line
  segment in space)
- total length (endpoint-to-endpoint distance is identical)
- radius (0.01 m)
- center height (0.10 m)
- collision masks (contype/conaffinity)

Unlike the articulated cable (``cable.py``), the rod has:

- exactly one body and one geom (no per-segment bodies)
- no joint at all (the body is welded to the world by omission of a joint,
  i.e. it is a static/fixed body)
- no ``mjEQ_CONNECT`` equality constraints
- no cable dynamics whatsoever

This lets a diagnostic run isolate "does the robot behave differently
against a rigid obstacle of the same shape/size?" from cable-specific
(free-jointed, equality-constrained) dynamics.
"""

from __future__ import annotations

import dataclasses

import mujoco

ROD_BODY_NAME = "rod_diagnostic"
ROD_GEOM_NAME = "rod_diagnostic"


@dataclasses.dataclass(frozen=True)
class RodCfg:
  """Configuration for the temporary rigid-rod diagnostic obstacle."""

  # Same endpoints as DEFAULT_CABLE_CFG in env_cfg.py: same center position,
  # orientation, total length, and center height (0.10 m).
  anchor_start: tuple[float, float, float] = (0.0, 0.0, 0.10)
  anchor_end: tuple[float, float, float] = (0.0, 0.5, 0.10)

  radius: float = 0.01

  rgba: tuple[float, float, float, float] = (0.1, 0.8, 0.95, 1.0)

  # Same collision masks as the test cable (CableCfg defaults).
  contype: int = 1
  conaffinity: int = 1

  # Arbitrary (body is static / has no joint, so mass never affects
  # dynamics); kept only so the geom has a well-defined mass property.
  mass: float = 0.2

  @property
  def geom_names(self) -> tuple[str, ...]:
    return (ROD_GEOM_NAME,)

  @property
  def body_names(self) -> tuple[str, ...]:
    return (ROD_BODY_NAME,)


def build_rod_spec_fn(rod_cfg: RodCfg):
  """Return a ``spec_fn`` post-processor that adds the rigid rod to a spec.

  Mirrors :func:`mjlab.tasks.cable_test.cable.build_cable_spec_fn`'s
  interface (a callable that mutates an already-built ``mujoco.MjSpec`` in
  place) so it can be dropped into ``SceneCfg.spec_fn`` as a drop-in
  replacement for a single diagnostic run.
  """

  def _add_rod(spec: mujoco.MjSpec) -> None:
    p0 = rod_cfg.anchor_start
    p1 = rod_cfg.anchor_end
    center = tuple((p0[k] + p1[k]) / 2.0 for k in range(3))
    half_vec = tuple((p1[k] - p0[k]) / 2.0 for k in range(3))

    # Static body: no joint is added, so it is welded to the world at a
    # fixed pose -- a single rigid body/geom, not a free-jointed segment.
    body = spec.worldbody.add_body(name=ROD_BODY_NAME, pos=center)
    fromto = (
      -half_vec[0],
      -half_vec[1],
      -half_vec[2],
      half_vec[0],
      half_vec[1],
      half_vec[2],
    )
    body.add_geom(
      name=ROD_GEOM_NAME,
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=(rod_cfg.radius, 0.0, 0.0),
      fromto=fromto,
      rgba=rod_cfg.rgba,
      contype=rod_cfg.contype,
      conaffinity=rod_cfg.conaffinity,
      mass=rod_cfg.mass,
    )

  return _add_rod
