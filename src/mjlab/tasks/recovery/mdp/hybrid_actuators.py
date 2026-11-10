"""Named auxiliary actuators for the D3 hybrid recovery controller.

The stock ``BuiltinVelocityActuatorCfg``/``BuiltinMotorActuatorCfg`` classes
hardcode the compiled MuJoCo actuator name to the target joint's name (see
``create_velocity_actuator``/``create_motor_actuator`` in
``mjlab.utils.spec``). Go2's existing walking actuators (``GO2_ACTUATOR_HIP/
THIGH/CALF``) ALREADY use the bare joint name as their actuator name (e.g. the
position actuator on "FL_thigh_joint" is itself named "FL_thigh_joint"). Since
a single MjSpec cannot have two actuators with the same name, adding a second,
auxiliary actuator on the SAME joint (needed for the D3 hybrid controller --
one joint, two actuators, only one active at a time via runtime gain
toggling) requires giving the auxiliary actuator a distinct name.

These two tiny subclasses change nothing about the parent classes' control
law or ``compute()`` (inherited unchanged -- see ``BuiltinVelocityActuator``/
``BuiltinMotorActuator`` in ``mjlab.actuator.builtin_actuator``); they only
override ``edit_spec`` to pass an explicit, distinct ``actuator_name`` through
to the underlying ``create_velocity_actuator``/raw MuJoCo actuator
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator,
  BuiltinMotorActuatorCfg,
  BuiltinVelocityActuator,
  BuiltinVelocityActuatorCfg,
)
from mjlab.utils.spec import apply_target_overrides, create_velocity_actuator


@dataclass(kw_only=True)
class NamedVelocityActuatorCfg(BuiltinVelocityActuatorCfg):
  """``BuiltinVelocityActuatorCfg`` with an explicit, distinct actuator name."""

  actuator_name: str

  def build(self, entity, target_ids: list[int], target_names: list[str]):
    return NamedVelocityActuator(self, entity, target_ids, target_names)


class NamedVelocityActuator(BuiltinVelocityActuator):
  cfg: NamedVelocityActuatorCfg

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    assert len(target_names) == 1, (
      "NamedVelocityActuatorCfg is intended for single-joint recovery "
      f"auxiliary actuators only, got {len(target_names)} targets."
    )
    for target_name in target_names:
      actuator = create_velocity_actuator(
        spec,
        target_name,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
        actuator_name=self.cfg.actuator_name,
      )
      self._mjs_actuators.append(actuator)


@dataclass(kw_only=True)
class NamedMotorActuatorCfg(BuiltinMotorActuatorCfg):
  """``BuiltinMotorActuatorCfg`` with an explicit, distinct actuator name."""

  actuator_name: str

  def build(self, entity, target_ids: list[int], target_names: list[str]):
    return NamedMotorActuator(self, entity, target_ids, target_names)


class NamedMotorActuator(BuiltinMotorActuator):
  cfg: NamedMotorActuatorCfg

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    assert len(target_names) == 1, (
      "NamedMotorActuatorCfg is intended for single-joint recovery "
      f"auxiliary actuators only, got {len(target_names)} targets."
    )
    cfg = self.cfg
    for target_name in target_names:
      # Mirrors mjlab.utils.spec.create_motor_actuator exactly, except for
      # the actuator name (create_motor_actuator hardcodes name=joint_name,
      # which collides with the existing position actuator on this joint).
      actuator = spec.add_actuator(name=cfg.actuator_name, target=target_name)
      actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
      actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
      actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
      actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
      actuator.gear[0] = cfg.gear
      actuator.forcelimited = True
      actuator.forcerange[:] = np.array([-cfg.effort_limit, cfg.effort_limit])
      actuator.ctrllimited = True
      actuator.ctrlrange[:] = np.array([-cfg.effort_limit, cfg.effort_limit])
      apply_target_overrides(
        spec,
        target_name,
        TransmissionType.JOINT,
        armature=cfg.armature,
        frictionloss=cfg.frictionloss,
        viscous_damping=cfg.viscous_damping,
      )
      self._mjs_actuators.append(actuator)


# `BuiltinActuatorGroup.process` (mjlab.actuator.builtin_group) classifies
# builtin actuators into batched index groups keyed by EXACT class identity
# (`type(act)`, not `isinstance`), then looks up which EntityData target
# tensor each group reads from via `_TARGET_TENSOR_MAP`. Since
# `NamedVelocityActuator`/`NamedMotorActuator` are subclasses (not instances
# of the exact base classes), they would otherwise fall through to a
# `KeyError` at the per-step `apply_controls()` call. They behave IDENTICALLY
# to their parents in every other respect (same `compute()`, same target
# tensor semantics), so registering them under the same target attribute is
# correct and not a special case -- it only affects which Python class
# Python groups ctrl writes by, not the physics.
from mjlab.actuator.builtin_group import _TARGET_TENSOR_MAP  # noqa: E402

_TARGET_TENSOR_MAP[(NamedVelocityActuator, TransmissionType.JOINT)] = "joint_vel_target"
_TARGET_TENSOR_MAP[(NamedMotorActuator, TransmissionType.JOINT)] = "joint_effort_target"
