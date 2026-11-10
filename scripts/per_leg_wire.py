"""Shared helpers for per-leg (FL/FR/RL/RR) wire-entanglement isolation.

Background
----------
D5.3 (`scripts/d5_3_extended_recovery_window.py`) established a working
single-leg entanglement/recovery setup for the FRONT-LEFT leg only. Its
isolation mechanism is `ISOLATED_FL_WIRE_CFG` in
`mjlab.tasks.recovery.env_cfg_residual`: a wire shortened to span only
`y in [0.30, 0.50]`, paired with `spawn_y_override=0.25`, so the wire
covers FL's lateral swing corridor while staying laterally clear of FR's.

That LATERAL-ONLY trick cannot generalize to all four legs, because the
Go2's left legs (FL, RL) share essentially the SAME lateral corridor
(hip y offset +0.0465 + thigh y offset +0.0955 = +0.142 from the base
centerline for BOTH FL and RL; mirrored to -0.142 for FR and RR -- see
`asset_zoo/robots/unitree_go2/xmls/go2.xml`). A wire narrow enough to
miss FR is still squarely in RL's path, and since the robot walks toward
+X the FRONT leg always reaches the wire first and would trip instead of
the intended rear leg.

Isolation used here
-------------------
Two complementary mechanisms, both applied per target leg:

1. LATERAL placement (same idea as D5.3's FL wire, mirrored for the
   right-side legs): the wire spans only the target leg's side of the
   robot. This keeps the setup as close as possible to the validated FL
   geometry rather than inventing a new one.
2. CONTACT EXCLUSION of the three NON-target legs: MjSpec body-pair
   excludes between every `wire_seg_*` body and each non-target leg's
   hip/thigh/calf bodies. This is the same `spec.add_exclude` primitive
   D4.8/D5.3 already use for their `robot/FL_hip <-> wire_seg_*`
   ablation, just applied to whole legs. It is what makes a REAR-leg
   entanglement reachable at all: the front leg on that side passes
   through the wire without interacting, so the wire is first (and only)
   engaged by the target leg, exactly as the task statement requires
   ("make environment such for each leg that wire interacts with that
   leg only").

Nothing in the target leg's own interaction with the wire is excluded or
altered: its hip, thigh and calf all collide with the wire exactly as
FL's do in D5.3. The base is likewise left un-excluded (it sits far
above the 0.10 m wire and never contacts it in the FL baseline).

For FL this module reproduces D5.3's environment exactly: the FL wire
config/spawn are imported from `env_cfg_residual`, and for FL the
non-target exclusion set is {FR, RL, RR} -- none of which touch the wire
in the D5.3 baseline anyway (that was the whole point of
`ISOLATED_FL_WIRE_CFG`), so FL results are expected to match D5.3's.
That expectation is checked by `scripts/d5_58_per_leg_parity_check.py`.

Nothing in this module modifies D5.3, the production env_cfg defaults,
any checkpoint, gain, or threshold.
"""

from __future__ import annotations

import mujoco

from mjlab.tasks.cable_test.ball_cable import BallCableCfg
from mjlab.tasks.recovery.env_cfg_residual import (
  ISOLATED_FL_WIRE_CFG,
  ISOLATED_FL_WIRE_SPAWN_Y,
)

LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_IDX = {name: i for i, name in enumerate(LEG_NAMES)}

# Robot spawn lateral centre. Deliberately the SAME value for all four
# legs (and identical to D5.3's `ISOLATED_FL_WIRE_SPAWN_Y`), so that the
# robot's own spawn pose -- and therefore every leg's absolute lateral
# corridor -- is byte-identical across the four per-leg environments.
# Only the wire's own extent differs between legs.
PER_LEG_SPAWN_Y = ISOLATED_FL_WIRE_SPAWN_Y

# Mirror of `ISOLATED_FL_WIRE_CFG` about the spawn centreline y=0.25:
# [0.30, 0.50] reflects to [0.00, 0.20]. Measured right-leg calf
# corridors (`scripts/d5_58_leg_corridor_survey.py`, seed 0, within
# +-0.15 m of the wire plane) are FR y in [0.132, 0.151] and RR y in
# [0.090, 0.107] -- both comfortably inside this span -- while the span's
# nearest point (y=0.20) stays >=0.13 m clear of the left legs'
# corridors (FL y in [0.368, 0.388], RL y in [0.330, 0.348]).
ISOLATED_RIGHT_WIRE_CFG = BallCableCfg(
  anchor_start=(0.0, 0.00, 0.10),
  anchor_end=(0.0, 0.20, 0.10),
  radius=ISOLATED_FL_WIRE_CFG.radius,
)

# Per-leg wire geometry. Left legs (FL, RL) reuse D5.3's validated FL
# wire verbatim (it covers both left corridors: FL y in [0.368, 0.388]
# and RL y in [0.330, 0.348] both lie inside y in [0.30, 0.50]). Right
# legs (FR, RR) use its mirror image. Front vs. rear legs on the same
# side share the same wire -- they are separated by the
# contact-exclusion mechanism, not by geometry (see module docstring).
PER_LEG_WIRE_CFG: dict[str, BallCableCfg] = {
  "FL": ISOLATED_FL_WIRE_CFG,
  "RL": ISOLATED_FL_WIRE_CFG,
  "FR": ISOLATED_RIGHT_WIRE_CFG,
  "RR": ISOLATED_RIGHT_WIRE_CFG,
}

# Bodies making up one leg, in the robot entity's namespace.
_LEG_BODY_SUFFIXES = ("hip", "thigh", "calf")


def leg_body_names(leg: str) -> tuple[str, ...]:
  """Compiled body names of `leg`'s three links, entity-prefixed."""
  return tuple(f"robot/{leg}_{part}" for part in _LEG_BODY_SUFFIXES)


def wire_segment_body_names(spec: mujoco.MjSpec) -> list[str]:
  """All `wire_seg_*` body names currently present in `spec`.

  Must be called AFTER the wire's own spec_fn has run, otherwise the
  bodies do not exist yet (same ordering requirement D5.3's ablation
  spec_fn asserts on).
  """
  return [b.name for b in spec.bodies if b.name.startswith("wire_seg_")]


def exclude_legs_from_wire(
  spec: mujoco.MjSpec, legs: tuple[str, ...], verbose: bool = True
) -> int:
  """Add body-pair contact excludes between every wire segment and every
  body of each leg in `legs`.

  Returns the number of exclude pairs added. Raises if the wire bodies
  are absent (spec_fn composition-order error).
  """
  seg_bodies = wire_segment_body_names(spec)
  if not seg_bodies:
    raise RuntimeError(
      "No wire_seg_* bodies found in spec -- the wire spec_fn must run "
      "BEFORE leg exclusions are added (composition order is wrong)."
    )
  added = 0
  for leg in legs:
    for body in leg_body_names(leg):
      for seg in seg_bodies:
        spec.add_exclude(bodyname1=body, bodyname2=seg)
        added += 1
  if verbose:
    print(
      f"[per_leg_wire] excluded legs {legs} from {len(seg_bodies)} wire "
      f"segments ({added} body pairs)"
    )
  return added


def make_leg_isolating_spec_fn(
  base_spec_fn,
  target_leg: str,
  verbose: bool = True,
  extra_excluded_legs: tuple[str, ...] = (),
):
  """Compose `base_spec_fn` (the wire builder) with exclusions that make
  the wire interact with `target_leg` ONLY.

  `extra_excluded_legs` additionally excludes the target leg itself (pass
  `(target_leg,)`) -- used by the corridor survey to obtain a completely
  unperturbed walk.
  """
  if target_leg not in LEG_IDX:
    raise ValueError(f"Unknown leg {target_leg!r}; expected one of {LEG_NAMES}")
  excluded = tuple(leg for leg in LEG_NAMES if leg != target_leg)
  excluded = excluded + tuple(leg for leg in extra_excluded_legs if leg not in excluded)

  def spec_fn(spec: mujoco.MjSpec, num_envs: int) -> None:
    if base_spec_fn is not None:
      base_spec_fn(spec, num_envs)
    exclude_legs_from_wire(spec, excluded, verbose=verbose)

  return spec_fn
