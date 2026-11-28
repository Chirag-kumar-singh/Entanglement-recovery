"""D9: loose-to-tight wire probe. Logs /lowstate-like data plus wire-integrity checks.

The wire starts LOOSE and tightens slowly after the first leg touches it. The robot's
recovery is off (entry trigger delayed forever), so the walking policy walks into the
wire and stays pressed against it, as in ``d9_tcn_probe_log.py``. The output .npz has the
same keys as that script (so ``d9_tcn_probe_eval.py`` reads it unchanged) plus checks that
the robot does not pass through the wire:

* ``gap``: per physics substep, for 16 robot parts (thigh, calf1, calf2, foot per leg),
  the smallest surface gap to any wire segment (negative = overlap), computed from the
  geometry, independently of the contact solver. A deep overlap (more than the wire
  radius) means the wire centreline is inside the part, i.e. the part went through.
* ``chain_gap``: largest distance between consecutive wire segments' joint points (the
  chain is kinematic, so this should stay ~0) and ``anchor_lag``: distance from the last
  segment end to its anchor.
* ``wire_mid``: wire segment midpoints at every control step, to see loose -> tight.

How loose / tightening: the chain is built with extra length ``slack_cm`` and a sag
shape; its end is pinned to a mocap anchor. After ``tighten_delay`` s from the first
contact the anchor moves along the wire's axis over ``tighten_time`` s until the chain is
straight. The tight state is at ``wire_height`` (0.10 m, the value of every earlier
result); the loose state hangs below it.

Nothing existing is modified: the wire builder is swapped in on the env cfg inside this
script only.

Usage:
    uv run python scripts/d9_loose_wire_probe_log.py --seed 0 --slack-cm 1.0 --tighten-time 2.0
    add --viewer to watch it live (see D6_MULTI_LEG_RECOVERY_SUMMARY.md, D9 loose wire)
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_1_front_pair_wire as d61  # noqa: E402
from d6_1_front_pair_wire import PAIRS, D61Config  # noqa: E402
from d9_loose_wire_lib import (  # noqa: E402
  MOCAP_NAME,  # noqa: F401
  build_nodes,
  make_loose_wire_spec_fn,
  sag_for_surplus,
)

from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.cable_test.ball_cable import BALL_CABLE_SEG_PREFIX  # noqa: E402
from mjlab.tasks.recovery.env_cfg_wire import DEFAULT_WIRE_CFG  # noqa: E402

LEGS = PAIRS["all"]
PARTS = ("thigh", "calf1", "calf2", "foot")

# Set from ProbeConfig before the env is built (the cfg factory below reads it).
_WIRE: dict = {}


_orig_env_cfg = d61.go2_wire_recovery_residual_env_cfg


def _env_cfg_loose(*a, **k):
  """Same env cfg + a foot-ground contact sensor + the loose wire."""
  cfg = _orig_env_cfg(*a, **k)
  foot = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=tuple(f"{leg}_foot_collision" for leg in LEGS),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), foot)
  cfg.scene.spec_fn = make_loose_wire_spec_fn(_WIRE["cfg"], _WIRE["nodes"])
  return cfg


d61.go2_wire_recovery_residual_env_cfg = _env_cfg_loose


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


@dataclass
class ProbeConfig:
  seed: int = 0
  steps: int = 400
  """Control steps (0.02 s each)."""
  wire_height: float = 0.10
  """Height of the TIGHT wire (m)."""
  slack_cm: float = 1.0
  """Extra chain length over the 0.5 m span, in cm. 0.5 -> ~3 cm sag, 1 -> ~4 cm."""
  tighten_delay: float = 0.2
  """Seconds after the first wire contact before the anchor starts to move."""
  tighten_time: float = 2.0
  """Seconds the anchor takes to straighten the wire. <= 0 = never tighten."""
  anchor_solref: float = 0.02
  """Time constant (s) of the end-anchor constraint. 0.02 = the original wire. Smaller
  is stiffer (the solver needs at least 2 physics steps = 0.005)."""
  overtravel_cm: float = 0.0
  """How far past straight the anchor ends (cm), so the tight wire is under tension."""
  out: str = "logs/d9_loose_wire/raw"
  viewer: bool = False
  """Open the MuJoCo viewer and print overlap events live (see the module docstring).
  In the viewer, tick 'Geom groups > 3' to see the collision shapes."""


def main() -> None:
  import tyro

  p = tyro.cli(ProbeConfig)
  span = 0.5
  n = DEFAULT_WIRE_CFG.num_segments
  nodes = build_nodes(span, p.wire_height, p.slack_cm / 100.0, n)
  link = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
  _WIRE["nodes"] = nodes
  _WIRE["cfg"] = dataclasses.replace(DEFAULT_WIRE_CFG, eq_solref=(p.anchor_solref, 1.0))
  chain_len = float(link.sum())
  y_tight = float(nodes[0, 1] + chain_len + p.overtravel_cm / 100.0)
  sag = sag_for_surplus(span, p.slack_cm / 100.0)

  cfg = D61Config(
    pair="all",
    start_x_range=(-0.485, -0.35),
    seed=p.seed,
    entry_delay_steps=10**9,
    wire_height=p.wire_height,
  )
  env = d61.build_env(cfg, p.seed, "cpu")
  term = env.action_manager.get_term("joint_pos")
  entity = term._entity
  m = env.sim.mj_model
  sd = env.sim.data
  sensors = env.scene.sensors
  feet = sensors["feet_ground_contact"]
  wire_sensors = {leg: sensors[f"wire_contact_{leg}"] for leg in LEGS}
  assert m.nmocap == 1, m.nmocap
  mocap = wp.to_torch(env.sim.wp_data.mocap_pos)  # [nworld, 1, 3], shares memory
  env.reset(seed=p.seed)
  mocap[0, 0, :] = torch.tensor(nodes[-1], dtype=mocap.dtype)

  def gid(name: str) -> int:
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert g >= 0, name
    return g

  wire_gids = np.array([gid(f"{BALL_CABLE_SEG_PREFIX}_{i}_geom") for i in range(n)])
  wire_half = m.geom_size[wire_gids, 1]
  wire_r = float(m.geom_size[wire_gids[0], 0])
  part_gids, part_kind, part_half, part_r, part_names = [], [], [], [], []
  for leg in LEGS:
    for part in PARTS:
      g = gid(f"robot/{leg}_{part}_collision")
      size = m.geom_size[g]
      typ = mujoco.mjtGeom(m.geom_type[g])
      if typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
        kind, half, r = 2, float(size[1]), float(size[0])  # axis = local z
      elif typ == mujoco.mjtGeom.mjGEOM_BOX:
        kind, half, r = (
          0,
          float(size[0]),
          float(max(size[1], size[2])),
        )  # axis = local x
      else:
        kind, half, r = 2, 0.0, float(size[0])  # sphere
      part_gids.append(g)
      part_kind.append(kind)
      part_half.append(half)
      part_r.append(r)
      part_names.append(f"{leg}_{part}")
  part_gids = np.array(part_gids)
  part_kind = np.array(part_kind)
  part_half = np.array(part_half)
  part_r = np.array(part_r)

  rec: dict[str, list] = {
    k: []
    for k in (
      "q", "dq", "tau", "quat", "angv_b", "linv_w", "foot", "wire", "gap",
      "chain_gap", "anchor_lag", "anchor_y",
    )
  }  # fmt: skip
  wire_mid: list[np.ndarray] = []
  state = {"sub": 0, "t_first": None}
  dt = env.physics_dt
  orig = env.metrics_manager.compute_substep

  def hooked(*a, **k):
    d = entity.data
    rec["q"].append(d.joint_pos[0].cpu().numpy().copy())
    rec["dq"].append(d.joint_vel[0].cpu().numpy().copy())
    rec["tau"].append(d.qfrc_actuator[0].cpu().numpy().copy())
    rec["quat"].append(d.root_link_quat_w[0].cpu().numpy().copy())
    rec["angv_b"].append(d.root_link_ang_vel_b[0].cpu().numpy().copy())
    rec["linv_w"].append(d.root_link_lin_vel_w[0].cpu().numpy().copy())
    rec["foot"].append(feet.data.force[0].norm(dim=-1).reshape(-1).cpu().numpy().copy())
    contact = np.array(
      [bool((wire_sensors[leg].data.found[0] > 0).any()) for leg in LEGS]
    )
    rec["wire"].append(contact)
    sub = state["sub"]
    if state["t_first"] is None and contact.any():
      state["t_first"] = sub * dt

    # Tighten: move the end anchor along y until the chain is straight.
    y = float(nodes[-1, 1])
    if p.tighten_time > 0 and state["t_first"] is not None:
      t0 = state["t_first"] + p.tighten_delay
      frac = min(max((sub * dt - t0) / p.tighten_time, 0.0), 1.0)
      y = float(nodes[-1, 1] + frac * (y_tight - nodes[-1, 1]))
      z = float(nodes[-1, 2] + frac * (p.wire_height - nodes[-1, 2]))
      mocap[0, 0, 1] = y
      mocap[0, 0, 2] = z
    rec["anchor_y"].append(y)

    xpos = sd.geom_xpos[0].cpu().numpy()
    xmat = sd.geom_xmat[0].cpu().numpy()
    w_ax = xmat[wire_gids][:, :, 2]
    w_ax = w_ax * np.where(w_ax[:, 1:2] < 0, -1.0, 1.0)  # orient start -> end (+y)
    w_mid = xpos[wire_gids]
    w_p, w_q = w_mid - w_ax * wire_half[:, None], w_mid + w_ax * wire_half[:, None]
    # chain integrity: end of segment k vs start of segment k+1, and last end vs anchor.
    rec["chain_gap"].append(float(np.linalg.norm(w_q[:-1] - w_p[1:], axis=1).max()))
    rec["anchor_lag"].append(float(np.linalg.norm(w_q[-1] - mocap[0, 0].numpy())))
    # robot parts as segments
    r_mid = xpos[part_gids]
    r_mat = xmat[part_gids]
    axis = np.where(
      (part_kind == 0)[:, None], r_mat[:, :, 0], r_mat[:, :, 2]
    )  # local x for the thigh box, local z otherwise
    r_p, r_q = r_mid - axis * part_half[:, None], r_mid + axis * part_half[:, None]
    dist = seg_seg_dist(
      r_p[:, None, :], r_q[:, None, :], w_p[None, :, :], w_q[None, :, :]
    )  # [parts, segs]
    gap_now = (dist - part_r[:, None] - wire_r).min(axis=1)
    rec["gap"].append(gap_now)
    if p.viewer:
      # Live report: first time a part's gap goes below -6 mm (wire centre inside the part),
      # then at most every 0.5 s while it stays there.
      for i in np.flatnonzero(gap_now < -wire_r):
        if sub * dt - state.setdefault("last", {}).get(i, -9.0) >= 0.5:
          state["last"][i] = sub * dt
          print(
            f"[OVERLAP] t={sub * dt:5.2f}s {part_names[i]:9} gap={gap_now[i] * 1000:6.1f} mm "
            f"(wire centre {-gap_now[i] * 1000 - wire_r * 1000:.1f} mm inside the surface)",
            flush=True,
          )
      if state["t_first"] is not None and "told" not in state:
        state["told"] = True
        print(f"[CONTACT] t={sub * dt:5.2f}s first wire contact", flush=True)
    if sub % env.cfg.decimation == 0:
      wire_mid.append(w_mid.copy())
    state["sub"] += 1
    return orig(*a, **k)

  env.metrics_manager.compute_substep = hooked  # type: ignore[method-assign]
  zero = torch.zeros(env.action_manager.action.shape, device="cpu")
  if p.viewer:
    from mjlab.viewer import NativeMujocoViewer

    counter = {"step": 0}

    def viewer_policy(_obs):
      counter["step"] += 1
      if counter["step"] > p.steps:
        raise SystemExit(0)
      return zero

    try:
      NativeMujocoViewer(env, viewer_policy).run()
    except SystemExit:
      pass
  else:
    for _ in range(p.steps):
      env.step(zero)
  if not rec["gap"]:
    return

  out = Path(p.out)
  out.mkdir(parents=True, exist_ok=True)
  tag = f"s{p.seed}_slack{p.slack_cm:g}_T{p.tighten_time:g}"
  if p.anchor_solref != 0.02 or p.wire_height != 0.10 or p.overtravel_cm:
    tag += f"_h{p.wire_height * 100:g}_sr{p.anchor_solref:g}_ot{p.overtravel_cm:g}"
  np.savez_compressed(
    out / f"{tag}.npz",
    **{k: np.stack(v) for k, v in rec.items()},
    wire_mid=np.stack(wire_mid),
    dt=dt,
    joint_names=np.array(entity.joint_names),
    foot_names=np.array(list(feet.primary_names)),
    wire_legs=np.array(LEGS),
    part_names=np.array(part_names),
    wire_radius=wire_r,
    t_first_contact=-1.0 if state["t_first"] is None else state["t_first"],
    sag=sag,
    chain_len=chain_len,
  )
  gap = np.stack(rec["gap"])
  w = np.stack(rec["wire"])
  deep = int((gap.min(axis=1) < -wire_r).sum())
  print(
    f"{tag}: first_contact={state['t_first']} sag={sag * 100:.1f}cm "
    f"max_overlap_mm={max(0.0, -gap.min()) * 1000:.1f} deep_overlap_substeps={deep} "
    f"max_chain_gap_mm={max(rec['chain_gap']) * 1000:.2f} "
    f"max_anchor_lag_mm={max(rec['anchor_lag']) * 1000:.1f} "
    f"contact_legs={[leg for i, leg in enumerate(LEGS) if w[:, i].any()]}"
  )
  env.close()


if __name__ == "__main__":
  main()
