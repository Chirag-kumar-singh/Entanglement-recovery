# Go2 Entanglement Recovery - Quick Start

## Phase 1: Oracle Recovery (✅ COMPLETED)

### Task ID
```
Mjlab-Go2-Entanglement-Recovery-Oracle
```

### Files Created (8 files, 714 lines)
- `src/mjlab/tasks/recovery/__init__.py` - Task registration
- `src/mjlab/tasks/recovery/env_cfg.py` - Environment config
- `src/mjlab/tasks/recovery/rl_cfg.py` - RL algorithm config
- `src/mjlab/tasks/recovery/mdp/__init__.py` - MDP exports
- `src/mjlab/tasks/recovery/mdp/observations.py` - Oracle observation (one-hot)
- `src/mjlab/tasks/recovery/mdp/rewards.py` - Recovery reward functions
- `src/mjlab/tasks/recovery/mdp/terminations.py` - Success/failure conditions
- `tests/test_recovery_env.py` - Smoke tests (10 tests, all passing ✓)

### Observation Space
- **Dimensions**: 49 (actor) + 49 (critic)
- **Components**:
  - Affected leg oracle: 4-dim one-hot [0,0,1,0] for RL
  - Base linear velocity: 3-dim
  - Base angular velocity: 3-dim
  - Projected gravity: 3-dim (orientation proxy)
  - Joint positions: 12-dim
  - Joint velocities: 12-dim
  - Previous actions: 12-dim

### Action Space
- **Dimensions**: 12 (Go2 has 12 DOF)
- **Type**: Joint position targets
- **Scaling**: Per-joint from GO2_ACTION_SCALE

### Reward Terms (5 total)
| Name | Weight | Purpose |
|------|--------|---------|
| clearance_from_obstacle | +2.0 | PRIMARY: distance from obstacle |
| body_stability | -0.2 | Penalize tipping |
| joint_torque | -0.05 | Smooth effort |
| joint_vel | -0.1 | Smooth motion |
| action_rate | -0.05 | Reduce jerkiness |

### Termination Conditions
1. **Robot fell** (tilt > 70°) → episode ends
2. **Timeout** (10 seconds) → episode ends normally

### Configuration Parameters (Customizable)
```python
go2_entanglement_recovery_env_cfg(
    affected_leg_idx=2,              # 0=FL, 1=FR, 2=RL, 3=RR
    affected_leg_joint="RL_calf_joint",
    obstacle_pos=(0.5, 0.0, 0.3),    # [x, y, z] meters
    obstacle_radius=0.05,            # meters
    play=False,                       # True for clean playback
)
```

---

## Quick Commands

### 1. Verify (Smoke Test)
```bash
cd /home/arc01/mjlab
uv run python tests/test_recovery_env.py
```
**Expected**: "All smoke tests passed! ✓"

### 2. Train (PPO)
```bash
cd /home/arc01/mjlab
CUDA_VISIBLE_DEVICES="" uv run train Mjlab-Go2-Entanglement-Recovery-Oracle --env.scene.num-envs 256
```
**Duration**: ~1-2 hours on CPU (adjust num-envs for your hardware)

### 3. Play (Visualize)
```bash
cd /home/arc01/mjlab
uv run play Mjlab-Go2-Entanglement-Recovery-Oracle --checkpoint-file logs/rsl_rl/go2_recovery_oracle/2026-09-18_XX-XX-XX/model_5000.pt
```

### 4. Train Different Leg (Example: FR)
```bash
# Modify training script to use:
from mjlab.tasks.recovery.env_cfg import go2_entanglement_recovery_env_cfg
cfg = go2_entanglement_recovery_env_cfg(
    affected_leg_idx=1,  # FR leg
    affected_leg_joint="FR_calf_joint",
)
```

---

## Network Architecture

**Actor & Critic**:
```
Input (49) → 256 → 128 → 64 → Output (12 or 1)
Activation: ELU
Normalization: Yes
```

**Smaller than velocity task** for faster convergence on recovery task.

---

## PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 5e-4 | Lower → stable recovery maneuver |
| Gamma | 0.99 | Future rewards matter |
| Lambda | 0.95 | Standard GAE |
| Clip Param | 0.2 | Standard PPO |
| Entropy Coeff | 0.01 | Light exploration |
| Init Std | 0.5 | Lower for precise movements |
| Max Iterations | 5000 | Shorter for testing |
| Save Interval | 100 | Checkpoint every 100 steps |

---

## Success Criteria

**Policy has learned recovery if**:
1. Average clearance distance > obstacle_radius + 0.15m
2. Success rate > 80% across legs (FL, FR, RL, RR)
3. Episodes where robot falls < 5%
4. Convergence by ~2000-3000 iterations

---

## Phase 2 Preview (TCN Integration)

When ready to add perception:
1. Replace `affected_leg_oracle` in observations with TCN output
2. Add TCN network to intake proprioceptive history
3. Keep everything else identical
4. Fine-tune on same recovery task

---

## Existing Velocity Task (Unmodified)

✅ **Mjlab-Velocity-Flat-Unitree-Go2** - Still works
✅ **Mjlab-Velocity-Rough-Unitree-Go2** - Still works

No changes made to existing Go2 velocity task.

---

## Debugging Tips

**Issue**: Environment creation slow
→ Normal on CPU; use `num_envs=1` for testing

**Issue**: Policy not improving
→ Check observation noise levels, reward weights, learning rate

**Issue**: Robot keeps falling
→ Increase body_stability weight; decrease other weights

**Issue**: Policy doesn't clear obstacle
→ Increase clearance_from_obstacle weight (currently 2.0)

---

## Files Summary

```
src/mjlab/tasks/recovery/
├── __init__.py                    13 lines    Registration
├── env_cfg.py                    214 lines    Environment
├── rl_cfg.py                      50 lines    PPO config
└── mdp/
    ├── __init__.py                20 lines    Exports
    ├── observations.py            37 lines    Oracle one-hot
    ├── rewards.py                 85 lines    Reward functions
    └── terminations.py           102 lines    Success/failure logic

tests/
└── test_recovery_env.py          193 lines    Smoke tests

Docs/
├── RECOVERY_TASK_REPORT.md       380 lines   Full report
└── RECOVERY_QUICK_START.md       This file
```

**Total**: 714 lines of implementation code

---

## Next Actions

1. **Verify**: Run smoke test to confirm environment works
2. **Train**: Launch PPO training (1-2 hours)
3. **Evaluate**: Check success rate and convergence
4. **Visualize**: Play trained policy
5. **Document Results**: Save metrics and checkpoints
6. **Prepare Phase 2**: Design TCN integration

---

## Document References

- **Full Report**: [RECOVERY_TASK_REPORT.md](RECOVERY_TASK_REPORT.md)
- **Policy Guide**: [POLICY_GUIDE.md](POLICY_GUIDE.md)
- **Environment Explanation**: [ENV_CFGS_EXPLANATION.md](ENV_CFGS_EXPLANATION.md)

---

Ready for Phase 1 training and validation! 🚀
