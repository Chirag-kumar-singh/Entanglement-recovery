"""Loose-wire builder shared by the D9 loose-to-tight scripts (no side effects on import).

``build_nodes`` lays a chain of capsules along a sagging parabola, ``make_loose_wire_spec_fn``
turns the nodes into an MjSpec wire (same body/geom names and parameters as the original
ball-joint wire) whose end is pinned to a mocap anchor that a script can move at run time.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from mjlab.tasks.cable_test.ball_cable import BALL_CABLE_SEG_PREFIX, BallCableCfg

MOCAP_NAME = "wire_anchor_mocap"


def sag_for_surplus(span: float, surplus: float) -> float:
  """Parabola sag (m) of a chain with `surplus` extra length over `span`."""
  return math.sqrt(3.0 * span * surplus / 8.0)


def build_nodes(span: float, z: float, surplus: float, n: int) -> np.ndarray:
  """n+1 nodes along a sagging parabola from (0,0,z) to (0,span,z).

  Equal arc-length spacing; the parabola's sag is solved so its arc length is
  span + surplus. Nodes lie in the plane x = 0.
  """
  target = span + surplus
  lo, hi = 0.0, 0.5
  u = np.linspace(0.0, 1.0, 4001)

  def arc(s: float) -> float:
    zz = z - 4.0 * s * u * (1.0 - u)
    return float(np.sum(np.hypot(np.diff(u * span), np.diff(zz))))

  for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if arc(mid) < target else (lo, mid)
  s = 0.5 * (lo + hi)
  zz = z - 4.0 * s * u * (1.0 - u)
  yy = u * span
  seg = np.hypot(np.diff(yy), np.diff(zz))
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  targets = np.linspace(0.0, cum[-1], n + 1)
  ny = np.interp(targets, cum, yy)
  nz = np.interp(targets, cum, zz)
  return np.stack([np.zeros(n + 1), ny, nz], axis=1)


def make_loose_wire_spec_fn(cfg: BallCableCfg, nodes: np.ndarray):
  """Like ``build_ball_cable_spec_fn`` but along `nodes`, end pinned to a mocap body.

  Same body and geom names (``wire_seg_i`` / ``wire_seg_i_geom``) and parameters, so the
  contact sensors and wire-geometry helpers work unchanged.
  """
  n = len(nodes) - 1
  vecs = np.diff(nodes, axis=0)
  prefix = f"{BALL_CABLE_SEG_PREFIX}_"

  def geom_kw(name: str, vec: np.ndarray) -> dict:
    return dict(
      name=f"{name}_geom",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=(cfg.radius, 0.0, 0.0),
      fromto=(0.0, 0.0, 0.0, *vec),
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

  def add_wire(spec: mujoco.MjSpec, num_envs: int) -> None:
    del num_envs
    root = spec.worldbody.add_body(name=f"{prefix}0", pos=tuple(nodes[0]))
    root.add_geom(**geom_kw(f"{prefix}0", vecs[0]))
    prev, prev_name = root, f"{prefix}0"
    for i in range(1, n):
      name = f"{prefix}{i}"
      body = prev.add_body(name=name, pos=tuple(vecs[i - 1]))
      body.add_joint(
        type=mujoco.mjtJoint.mjJNT_BALL,
        damping=cfg.joint_damping,
        armature=cfg.joint_armature,
      )
      body.add_geom(**geom_kw(name, vecs[i]))
      spec.add_exclude(bodyname1=prev_name, bodyname2=name)
      prev, prev_name = body, name
    spec.worldbody.add_body(name=MOCAP_NAME, pos=tuple(nodes[-1]), mocap=True)
    eq = spec.add_equality(
      type=mujoco.mjtEq.mjEQ_CONNECT,
      name1=prev_name,
      name2=MOCAP_NAME,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      name="wire_anchor_end",
      solref=cfg.eq_solref,
      solimp=cfg.eq_solimp,
    )
    eq.data[0:3] = tuple(vecs[-1])
    eq.data[3:6] = (0.0, 0.0, 0.0)

  return add_wire


def seg_seg_dist(p1, q1, p2, q2):
  """Min distance between segments p1q1 and p2q2 (broadcasting, shape [..., 3])."""
  d1, d2, r = q1 - p1, q2 - p2, p1 - p2
  a = np.maximum((d1 * d1).sum(-1), 1e-12)
  e = np.maximum((d2 * d2).sum(-1), 1e-12)
  f, c, b = (d2 * r).sum(-1), (d1 * r).sum(-1), (d1 * d2).sum(-1)
  den = a * e - b * b
  s = np.where(
    den > 1e-12, np.clip((b * f - c * e) / np.maximum(den, 1e-12), 0, 1), 0.0
  )
  t = (b * s + f) / e
  s = np.where(
    t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s)
  )
  t = np.clip(t, 0, 1)
  return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)


class OverlapMonitor:
  """Smallest surface gap between robot leg parts and the wire, from geometry.

  Independent of the contact solver. Negative = overlap; deeper than the wire radius means
  the wire centre line is inside the part (it went through). Cylinders are treated as
  capsules and the thigh box as a capsule of its larger half-thickness, so this is
  conservative near flat ends. Same checks as ``d9_loose_wire_probe_log.py``.
  """

  def __init__(self, env, legs=("FR", "FL", "RR", "RL"), n_seg=17):
    m = env.sim.mj_model
    self.sd = env.sim.data

    def gid(name: str) -> int:
      g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
      assert g >= 0, name
      return g

    self.wire_gids = np.array(
      [gid(f"{BALL_CABLE_SEG_PREFIX}_{i}_geom") for i in range(n_seg)]
    )
    self.wire_half = m.geom_size[self.wire_gids, 1]
    self.wire_r = float(m.geom_size[self.wire_gids[0], 0])
    gids, kind, half, rad, names = [], [], [], [], []
    for leg in legs:
      for part in ("thigh", "calf1", "calf2", "foot"):
        g = gid(f"robot/{leg}_{part}_collision")
        size = m.geom_size[g]
        typ = mujoco.mjtGeom(m.geom_type[g])
        if typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
          k, h, r = 2, float(size[1]), float(size[0])
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
          k, h, r = 0, float(size[0]), float(max(size[1], size[2]))
        else:
          k, h, r = 2, 0.0, float(size[0])
        gids.append(g)
        kind.append(k)
        half.append(h)
        rad.append(r)
        names.append(f"{leg}_{part}")
    self.gids = np.array(gids)
    self.kind = np.array(kind)
    self.half = np.array(half)
    self.rad = np.array(rad)
    self.names = names

  def gap(self) -> np.ndarray:
    """Per part: smallest gap (m) to any wire segment, at the current state."""
    xpos = self.sd.geom_xpos[0].cpu().numpy()
    xmat = self.sd.geom_xmat[0].cpu().numpy()
    w_ax = xmat[self.wire_gids][:, :, 2]
    w_mid = xpos[self.wire_gids]
    w_p = w_mid - w_ax * self.wire_half[:, None]
    w_q = w_mid + w_ax * self.wire_half[:, None]
    r_mid = xpos[self.gids]
    r_mat = xmat[self.gids]
    axis = np.where((self.kind == 0)[:, None], r_mat[:, :, 0], r_mat[:, :, 2])
    r_p = r_mid - axis * self.half[:, None]
    r_q = r_mid + axis * self.half[:, None]
    dist = seg_seg_dist(
      r_p[:, None, :], r_q[:, None, :], w_p[None, :, :], w_q[None, :, :]
    )
    return (dist - self.rad[:, None] - self.wire_r).min(axis=1)
