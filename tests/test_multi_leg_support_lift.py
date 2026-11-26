"""Unit test for the multi-leg support lift (D7).

`_apply_support_lift` only needs a few attributes, so it is called on a stub instead of
building the wire environment and the frozen walking policy.
"""

from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.recovery.mdp.actions import LEG_NAMES, WALK_JOINT_ORDER
from mjlab.tasks.recovery.mdp.multi_leg import MultiLegRecoveryAction

THIGH, CALF, RAMP = -0.2, 0.3, 4


def _stub(busy_legs: tuple[str, ...], trigger: tuple[str, ...]) -> SimpleNamespace:
  legs = [
    SimpleNamespace(
      cfg=SimpleNamespace(affected_leg_idx=i),
      name=name,
      _joint_pos_action=SimpleNamespace(_scale=torch.ones(1, 12)),
    )
    for i, name in enumerate(LEG_NAMES)
  ]
  return SimpleNamespace(
    legs=legs,
    cfg=SimpleNamespace(
      support_lift_trigger=trigger,
      support_lift_ramp_steps=RAMP,
      support_lift_thigh=THIGH,
      support_lift_calf=CALF,
    ),
    lift_level=torch.zeros(len(LEG_NAMES), 1),
    leg_busy=lambda leg: torch.tensor([leg.name in busy_legs]),
  )


def _apply(stub: SimpleNamespace, calls: int = 1) -> torch.Tensor:
  final = torch.zeros(1, 12)
  for _ in range(calls):
    final = MultiLegRecoveryAction._apply_support_lift(stub, torch.zeros(1, 12))  # ty: ignore
  return final


def _offset(final: torch.Tensor, leg: str, part: str) -> float:
  return float(final[0, WALK_JOINT_ORDER.index(f"{leg}_{part}_joint")])


def test_lift_ramps_on_standing_legs_only() -> None:
  stub = _stub(busy_legs=("FL",), trigger=("FL", "FR"))
  one = _apply(stub)
  assert _offset(one, "FR", "thigh") == pytest.approx(THIGH / RAMP)
  assert _offset(one, "RL", "calf") == pytest.approx(CALF / RAMP)
  assert _offset(one, "FL", "thigh") == 0.0
  full = _apply(stub, calls=RAMP - 1)
  assert _offset(full, "RR", "thigh") == pytest.approx(THIGH)
  assert _offset(full, "RR", "calf") == pytest.approx(CALF)
  assert _offset(full, "FL", "calf") == 0.0


def test_trigger_ignores_rear_steps() -> None:
  stub = _stub(busy_legs=("RL",), trigger=("FL", "FR"))
  assert torch.count_nonzero(_apply(stub, calls=RAMP)) == 0


def test_empty_trigger_lifts_for_any_leg() -> None:
  stub = _stub(busy_legs=("RL",), trigger=())
  full = _apply(stub, calls=RAMP)
  assert _offset(full, "FL", "thigh") == pytest.approx(THIGH)
  assert _offset(full, "RL", "thigh") == 0.0
