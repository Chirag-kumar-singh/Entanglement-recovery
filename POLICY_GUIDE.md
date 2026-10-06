# Writing Your Own RL Policies in mjlab

This guide explains how the Go2 velocity policy was built and how to create your own custom policies.

## Architecture Overview

An RL policy in mjlab has **3 main components**:

```
┌─────────────────────────────────────────────────────────────┐
│  Task Registration (register_mjlab_task)                    │
│  ├─ task_id: Unique identifier (e.g., "Mjlab-Velocity-Flat")│
│  ├─ env_cfg: Environment configuration (observations, rewards)
│  ├─ play_env_cfg: Modified config for playing (no training) │
│  ├─ rl_cfg: PPO algorithm hyperparameters                  │
│  └─ runner_cls: Which RL algorithm to use                  │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Environment Config (env_cfgs.py)                           │
│  ├─ Observations: What the agent sees (state info)          │
│  ├─ Actions: What the agent can do                          │
│  ├─ Rewards: How to score good behavior                     │
│  ├─ Terminations: When episodes end                         │
│  ├─ Commands: What targets/goals to track                   │
│  └─ Events: Randomization and reset procedures             │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  RL Config (rl_cfg.py)                                      │
│  ├─ Actor/Critic network architecture (layers, activations)│
│  ├─ PPO algorithm settings (learning rate, gamma, etc.)     │
│  ├─ Training loop parameters (iterations, batch sizes)      │
│  └─ Checkpoint and logging settings                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. ENVIRONMENT CONFIGURATION (env_cfgs.py)

This is where you define the RL problem: what to observe, what to reward, what to do.

### 1.1 Start with a Base Config

Instead of building from scratch, inherit from an existing template:

```python
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

def my_robot_velocity_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Start with the base velocity task
    cfg = make_velocity_env_cfg()
    
    # Customize below...
    return cfg
```

The `make_velocity_env_cfg()` already has:
- ✅ Observations (linear velocity, angular velocity, joint positions, etc.)
- ✅ Actions (joint position control)
- ✅ Rewards (velocity tracking, energy, upright penalty, etc.)
- ✅ Commands (random velocity targets)
- ✅ Terminations (falling, leaving terrain bounds)

### 1.2 Set Your Robot

```python
def my_robot_velocity_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    
    # Replace the default robot with yours
    from mjlab.asset_zoo.robots import get_my_robot_cfg
    cfg.scene.entities = {"robot": get_my_robot_cfg()}
    
    return cfg
```

The robot config specifies:
- MJCF file location
- Actuator names and ranges
- Collision geometries
- Mass/inertia properties

### 1.3 Customize Observations

Observations are what the neural network sees. The **actor** network gets noisy observations, the **critic** gets clean ones (privileged info).

**For the actor** (what the policy network sees):
```python
actor_terms = {
    "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        noise=Unoise(n_min=-0.5, n_max=0.5),  # Add realistic sensor noise
    ),
    "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},  # Misaligned sensors in training
        noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
    ),
}
```

**For the critic** (what the value function sees—truth for stability):
```python
critic_terms = {
    **actor_terms,
    # Critic also gets privileged information:
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),  # No noise
    "foot_contact": ObservationTermCfg(
        func=mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    ),
}
```

**Useful observation functions** (in `mdp` module):
- `builtin_sensor`: Read from sensors (IMU, joint encoders, etc.)
- `joint_pos_rel`: Relative joint positions (normalized by range)
- `joint_vel_rel`: Relative joint velocities
- `projected_gravity`: Gravity vector in robot frame (orientation info)
- `generated_commands`: The current velocity target
- `last_action`: Previous action (helps with smoothness)
- `foot_contact`: Which feet are touching ground
- `foot_air_time`: How long each foot has been in air
- `foot_height`: Height of each foot above terrain

### 1.4 Customize Actions

Actions are what the agent controls. Typically joint position targets:

```python
actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),  # Control all actuators
        scale=0.5,  # Action range: [-0.5*limit, +0.5*limit] rad
        use_default_offset=True,  # Add default pose offset
    )
}
```

The `scale` parameter is crucial:
- Too small → agent can't move
- Too large → jerky, unstable movements
- Go2: We use `GO2_ACTION_SCALE` (tuned per-joint from reference)

**Other action types:**
- `JointVelocityActionCfg`: Direct velocity commands
- `JointTorqueActionCfg`: Direct torque commands

### 1.5 Customize Rewards

Rewards shape the behavior. The agent learns to maximize cumulative reward.

```python
rewards: dict[str, RewardTermCfg] = {
    "tracking_lin_vel_x": RewardTermCfg(
        func=mdp.tracking_lin_vel_xy,
        weight=1.0,
        params={"axis": "x"},  # Track x-velocity
    ),
    "tracking_lin_vel_y": RewardTermCfg(
        func=mdp.tracking_lin_vel_xy,
        weight=0.5,  # Less important than x
        params={"axis": "y"},
    ),
    "tracking_ang_vel_z": RewardTermCfg(
        func=mdp.tracking_ang_vel_z,
        weight=0.8,
    ),
    "action_smoothness": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.005,  # Penalty (negative weight)
    ),
    "joint_acc": RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-1e-5,  # Small penalty
    ),
}
```

**Common reward functions:**
- `tracking_lin_vel_xy`: How well we match linear velocity command
- `tracking_ang_vel_z`: How well we match angular velocity command
- `action_rate_l2`: Penalize jerky actions (smooth motion)
- `joint_acc_l2`: Penalize joint acceleration (reduce wear)
- `upright`: Penalize tilting (stay upright)
- `torque_l2`: Penalize high joint torques
- `energy`: Penalize high energy consumption

**Weight meanings:**
- `weight=1.0` → Encourage this (maximize reward)
- `weight=-0.01` → Discourage this (minimize it)
- `weight=0.0` → Ignore this

### 1.6 Customize Terminations

When does the episode end?

```python
terminations: dict[str, TerminationTermCfg] = {
    "is_falling": TerminationTermCfg(
        func=mdp.falling,
        params={"velocity_threshold": 3.0},  # If body drops too fast
    ),
    "illegal_contact": TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_name": "nonfoot_ground_touch",
            "force_threshold": 10.0,  # Body hit ground too hard
        },
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
        func=mdp.out_of_terrain_bounds,
        params={"distance_threshold": 3.0},
    ),
}
```

### 1.7 Customize Commands (Goals)

What targets should the agent track?

```python
commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),  # Change command every 3-8 sec
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),   # Random x velocity in [-1, 1] m/s
            lin_vel_y=(-1.0, 1.0),   # Random y velocity in [-1, 1] m/s
            ang_vel_z=(-0.5, 0.5),   # Random yaw rate in [-0.5, 0.5] rad/s
        ),
    )
}
```

During curriculum learning, these ranges expand:
- Start: `lin_vel_x=(-0.5, 0.5)` → easier
- End: `lin_vel_x=(-2.0, 3.0)` → harder

### 1.8 Customize Events (Randomization)

Events are triggered at reset or during episode for domain randomization:

```python
events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",  # Happens at episode start
        params={
            "pose_range": {
                "x": (-0.5, 0.5),  # Random spawn position
                "y": (-0.5, 0.5),
                "z": (0.01, 0.05),
            },
        },
    ),
    "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),  # Push every 1-3 seconds
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),  # Random push direction/magnitude
                "y": (-0.5, 0.5),
            },
        },
    ),
    "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geoms),
            "ranges": (0.3, 1.2),  # Random friction coefficient
            "shared_random": True,  # All foot geoms get same friction
        },
    ),
}
```

### 1.9 Go2-Specific Customizations

Here's what we did for Go2:

```python
def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    
    # 1. Set the robot
    cfg.scene.entities = {"robot": get_go2_robot_cfg()}
    
    # 2. Add Go2-specific sensors (terrain scan, foot height)
    # 3. Tune action scale for Go2's joint ranges
    cfg.actions["joint_pos"].scale = GO2_ACTION_SCALE
    
    # 4. Customize rewards for Go2's morphology
    cfg.rewards["pose"].params["std_standing"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,    # Hip: small penalty
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,   # Thigh: medium
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,   # Calf: larger
    }
    
    # 5. Customize terminations
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 10.0},
    )
    
    # 6. Set viewer camera
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.5
    
    return cfg
```

---

## 2. RL CONFIGURATION (rl_cfg.py)

This defines the PPO algorithm and neural network architecture.

### 2.1 Network Architecture

Both actor and critic are MLPs (multi-layer perceptrons):

```python
actor=RslRlModelCfg(
    hidden_dims=(512, 256, 128),  # 3 hidden layers
    activation="elu",              # ELU activation (vs. ReLU, tanh, etc.)
    obs_normalization=True,        # Normalize input observations
    distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,           # Action std dev (how exploratory)
        "std_type": "scalar",      # One std dev for all actions
    },
),
```

**What this means:**
- Input: All observations concatenated (e.g., 48 floats)
- Layer 1: 48 → 512 neurons, apply ELU activation
- Layer 2: 512 → 256 neurons, apply ELU activation
- Layer 3: 256 → 128 neurons, apply ELU activation
- Output: 128 → 12 neurons (action dimension for Go2)

**Architecture choices:**
- `hidden_dims=(512, 256, 128)`: Larger = more capacity but slower. Start here.
- `hidden_dims=(256, 256)`: Faster, sufficient for simpler tasks
- `hidden_dims=(1024, 512, 256)`: Overkill, very slow
- `activation="elu"`: Smooth gradients, helps training
- `activation="relu"`: Faster but sometimes unstable
- `init_std=1.0`: Start with high exploration; shrink during training
- `init_std=0.1`: Lower exploration, converge faster

### 2.2 PPO Algorithm Settings

```python
algorithm=RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,            # Weight of critic loss
    use_clipped_value_loss=True,    # Prevent value from changing too fast
    clip_param=0.2,                 # PPO clipping: prevent large policy updates
    entropy_coef=0.01,              # Encourage exploration
    num_learning_epochs=5,          # How many times to iterate over batch
    num_mini_batches=4,             # Divide batch into N smaller batches
    learning_rate=1.0e-3,           # How fast to update weights
    schedule="adaptive",            # Adapt LR based on KL divergence
    gamma=0.99,                     # Discount factor (value of future rewards)
    lam=0.95,                       # GAE lambda (balance bias/variance)
    desired_kl=0.01,                # Stop if KL divergence gets too high
    max_grad_norm=1.0,              # Gradient clipping for stability
),
```

**Key hyperparameters:**

| Param | Meaning | Typical Range |
|-------|---------|---------------|
| `learning_rate` | Step size for weight updates | 1e-5 to 1e-2 |
| `gamma` | Discount factor (0.99=far future matters) | 0.90 to 0.999 |
| `lam` | GAE parameter (0.95=balance of N-step and infinite-step) | 0.90 to 0.99 |
| `clip_param` | PPO clipping threshold | 0.1 to 0.3 |
| `entropy_coef` | Exploration bonus | 0.0 to 0.1 |
| `num_learning_epochs` | Passes over batch per update | 3 to 10 |

**Tuning guide:**
- Too high LR → training diverges (NaN)
- Too low LR → training stalls
- Increase entropy → more exploration
- Decrease entropy → faster convergence
- Increase gamma → plan further ahead
- Decrease gamma → focus on immediate rewards

### 2.3 Training Loop Parameters

```python
experiment_name="go2_velocity",     # Checkpoint folder name
save_interval=100,                  # Save checkpoint every 100 iterations
num_steps_per_env=24,               # Collect 24 steps per env per update
max_iterations=10_001,              # Total training iterations
```

**What these mean:**
- `num_steps_per_env=24`: Parallel envs collect trajectories → PPO updates weights → repeat
  - Each iteration: `4096 envs × 24 steps = 98,304 transitions`
  - At 10Hz control: ~2.4 seconds of simulation per iteration
- `max_iterations=10_001`: Total training time ≈ 7 hours on 4096 envs
- `save_interval=100`: Checkpoints at iterations 0, 100, 200, ..., 10000

---

## 3. TASK REGISTRATION

Connect everything together:

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import unitree_go2_flat_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Unitree-Go2",  # Unique name
    env_cfg=unitree_go2_flat_env_cfg(),          # Training environment
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),  # Evaluation (no noise, longer episodes)
    rl_cfg=unitree_go2_ppo_runner_cfg(),         # PPO settings
    runner_cls=VelocityOnPolicyRunner,           # Which runner to use
)
```

The `play_env_cfg` is identical but with:
- ✅ No observation noise
- ✅ No domain randomization
- ✅ Infinite episode length (no terminations except falling)
- ✅ Fixed terrain (no curriculum)

---

## 4. COMPLETE EXAMPLE: Writing Your Own Policy

Let's create a custom policy for a new robot (e.g., "MyRobot"):

### Step 1: Robot Config

```python
# src/mjlab/asset_zoo/robots/my_robot/my_robot_constants.py

MY_ROBOT_ACTION_SCALE = {
    "hip_joint": 0.5,
    "knee_joint": 0.7,
}

def get_my_robot_cfg() -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        rigid_body_material=RigidBodyPropertiesCfg.RigidBodyMaterialCfg(...),
        ...
    )
```

### Step 2: Environment Config

```python
# src/mjlab/tasks/velocity/config/my_robot/env_cfgs.py

from mjlab.asset_zoo.robots import get_my_robot_cfg, MY_ROBOT_ACTION_SCALE
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

def my_robot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    
    # 1. Set robot
    cfg.scene.entities = {"robot": get_my_robot_cfg()}
    
    # 2. Customize action scale
    cfg.actions["joint_pos"].scale = MY_ROBOT_ACTION_SCALE
    
    # 3. Customize rewards (adjust weights based on morphology)
    cfg.rewards["tracking_lin_vel_x"].weight = 1.0
    cfg.rewards["tracking_ang_vel_z"].weight = 0.8
    cfg.rewards["action_smoothness"].weight = -0.01
    
    # 4. Customize observations if needed
    # (Usually the base config is fine)
    
    # 5. Customize commands if needed
    if play:
        cfg.commands["twist"].ranges.lin_vel_x = (-0.5, 1.0)
    
    return cfg
```

### Step 3: RL Config

```python
# src/mjlab/tasks/velocity/config/my_robot/rl_cfg.py

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg

def my_robot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1.0e-3,
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            entropy_coef=0.01,
        ),
        experiment_name="my_robot_velocity",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=5000,  # Adjust based on complexity
    )
```

### Step 4: Register Task

```python
# src/mjlab/tasks/velocity/config/my_robot/__init__.py

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import my_robot_env_cfg
from .rl_cfg import my_robot_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MyRobot",
    env_cfg=my_robot_env_cfg(),
    play_env_cfg=my_robot_env_cfg(play=True),
    rl_cfg=my_robot_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

### Step 5: Train & Play

```bash
# Train
CUDA_VISIBLE_DEVICES="" uv run train Mjlab-Velocity-Flat-MyRobot --env.scene.num-envs 4096 --agent.logger tensorboard

# Play
uv run play Mjlab-Velocity-Flat-MyRobot --checkpoint-file logs/rsl_rl/my_robot_velocity/2026-09-17_13-00-16/model_5000.pt
```

---

## 5. KEY TUNING PARAMETERS

When your policy trains poorly, adjust these:

| Problem | Solution |
|---------|----------|
| Agent falls immediately | Increase `upright` reward weight; decrease `tracking` weights |
| Jerky, unstable motion | Increase `action_smoothness` weight or `clip_param` |
| Doesn't follow velocity command | Increase `tracking_lin_vel` weight; ensure sensor noise isn't too high |
| Training diverges (NaN loss) | Decrease learning rate; increase clip_param; add gradient clipping |
| Too slow convergence | Increase learning rate; increase entropy_coef for more exploration |
| Agent overshoots commands | Increase damping (action scale); decrease learning rate |
| Poor generalization in play | Increase domain randomization (friction, mass, etc.) |

---

## 6. Common Pitfalls

❌ **Too much sensor noise** → Agent can't learn
```python
# Bad
"joint_pos": ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(n_min=-1.0, n_max=1.0),  # Way too high
)
```

✅ **Right amount** → Realistic sim-to-real gap
```python
"joint_pos": ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),  # ±0.01 rad ≈ ±0.6°
)
```

❌ **Conflicting rewards**
```python
# Bad: Both encourage and penalize motion
"high_speed": RewardTermCfg(func=..., weight=1.0),
"energy": RewardTermCfg(func=..., weight=-10.0),  # Conflicts!
```

✅ **Balanced rewards**
```python
"tracking": RewardTermCfg(func=..., weight=1.0),
"smoothness": RewardTermCfg(func=..., weight=-0.01),  # Small penalty
```

---

## Summary Checklist

When creating a new policy, fill in:

- [ ] Robot MJCF file + constants (action scale, body names)
- [ ] Base env config from template
- [ ] Set robot in scene
- [ ] Tune action scale
- [ ] Pick observations (actor sees noisy, critic sees true)
- [ ] Pick rewards and weights
- [ ] Pick termination conditions
- [ ] Set command ranges
- [ ] Domain randomization events
- [ ] RL hyperparameters (network architecture, learning rate)
- [ ] Training loop parameters (iterations, save interval)
- [ ] Register task with unique ID
- [ ] Train: `uv run train <task_id> --env.scene.num-envs 4096`
- [ ] Play: `uv run play <task_id> --checkpoint-file logs/.../model_10000.pt`

Good luck! 🚀
