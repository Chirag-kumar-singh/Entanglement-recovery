# Detailed Breakdown: Go2 Flat Environment Configuration

This document explains **exactly** what came from the template vs. what was customized, and **why** the Go2 walks in random directions.

---

## Part 1: The Template (`make_velocity_env_cfg()`)

The base velocity task configuration defines the **entire RL problem framework** with sensible defaults for any quadruped robot. Think of it as a reusable blueprint.

### What's in the Template (Ready-to-Use)

```
make_velocity_env_cfg() includes:
├─ Observations (vision of the agent)
│  ├─ Actor: noisy sensors (realistic)
│  ├─ Critic: clean sensors (privileged)
│  └─ Height scan (terrain height map)
│
├─ Actions (joint position control)
│  └─ Control all joints with configurable scale
│
├─ Commands (velocity targets)
│  └─ Random twist commands (vx, vy, wz)
│
├─ Events (randomization)
│  ├─ Reset position/orientation
│  ├─ Push robot randomly
│  ├─ Randomize friction
│  ├─ Randomize joint biases
│  └─ Randomize center of mass
│
├─ Rewards (12 reward functions)
│  ├─ Track linear velocity → MAIN GOAL
│  ├─ Track angular velocity → MAIN GOAL
│  ├─ Stay upright
│  ├─ Joint posture
│  ├─ Action smoothness (penalize jerky)
│  ├─ Foot clearance
│  ├─ Foot slip (penalize sliding)
│  ├─ Air time (encourage dynamic gait)
│  └─ ... and more
│
├─ Terminations (when episode ends)
│  ├─ Fell over (70° tilt)
│  ├─ Timeout (20 seconds)
│  └─ Left terrain bounds
│
└─ Curriculum (difficulty increases over time)
   ├─ Terrain gets harder
   └─ Velocity targets get faster
```

### Key Template Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `resampling_time_range=(3.0, 8.0)` | Per command | Change target every 3-8 seconds |
| `lin_vel_x=(-1.0, 1.0)` | Initial | Move forward/backward in range |
| `lin_vel_y=(-1.0, 1.0)` | Initial | Strafe left/right in range |
| `ang_vel_z=(-0.5, 0.5)` | Initial | Rotate CCW/CW in range |
| `episode_length_s=20.0` | Duration | Each episode lasts 20 seconds |
| `heading_command=True` | Enabled | Agent can rotate independently |
| `rel_forward_envs=0.2` | Probability | 20% of time: forward-only movement |
| `rel_heading_envs=0.3` | Probability | 30% of time: rotation-only movement |
| `rel_standing_envs=0.1` | Probability | 10% of time: stay still |

---

## Part 2: Go2-Specific Customizations

Now we modify this template for Go2's specific needs.

### Step 1: `unitree_go2_rough_env_cfg()` - Rough Terrain Version

This function takes the template and customizes it for Go2:

```python
cfg = make_velocity_env_cfg()  # Start with template
```

#### Customization 1: Physics Parameters

```python
cfg.sim.mujoco.ccd_iterations = 500        # More collision detection iterations
cfg.sim.contact_sensor_maxmatch = 500      # Track more simultaneous contacts
```

**Why?** Rough terrain has many obstacles → need more collision checks → avoid missing contacts.

#### Customization 2: Set the Robot

```python
cfg.scene.entities = {"robot": get_go2_robot_cfg()}
```

**What this does:**
- Replace the generic placeholder robot with Go2's actual MJCF model
- Go2 MJCF includes:
  - 4 legs (FR/FL/RR/RL = Front-Right, Front-Left, Rear-Right, Rear-Left)
  - Each leg has 3 joints (hip, thigh, calf)
  - Total: 12 DOF
  - Specific masses, inertias, joint limits

#### Customization 3: Update Sensor Frames

The template has generic sensors that need robot-specific configuration:

```python
for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
        sensor.frame.name = "base_link"  # Mount raycast on Go2's body center
```

**Why?** The template doesn't know where to attach sensors. Go2 has a specific body link called `"base_link"`.

#### Customization 4: Add Go2-Specific Sensors

The template only has basic terrain sensors. Go2 needs more:

```python
# Define foot geometry names for Go2
foot_names = ("FR", "FL", "RR", "RL")
site_names = ("FR", "FL", "RR", "RL")
geom_names = tuple(f"{name}_foot_collision" for name in foot_names)
```

**What this does:**
- Maps logical foot names (FR, FL, etc.) to Go2's MJCF geometry names
- Go2's MJCF defines geometry like `"FR_foot_collision"`, `"FL_foot_collision"`, etc.

**Add foot height sensor:**

```python
sensor.frame = tuple(
    ObjRef(type="site", name=site_name, entity="robot") for site_name in site_names
)
sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
```

**Why?** Measure height of each foot → can tell if foot is in air or on ground → enables complex gaits.

**Add contact sensors:**

```python
feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,  # How long foot has been in air
)
```

**Why?** Detect when feet touch ground → measure contact forces → reward "soft landings" and penalize "stumbles".

#### Customization 5: Update Terrain Curriculum

```python
if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True
```

**Why?** Start on easy flat ground → gradually get rougher terrain.

#### Customization 6: Set Action Scale

```python
joint_pos_action = cfg.actions["joint_pos"]
joint_pos_action.scale = GO2_ACTION_SCALE
```

**What is `GO2_ACTION_SCALE`?** From Go2 constants:

```python
GO2_ACTION_SCALE = {
    "hip": 0.5,       # Hip joints: ±0.5 rad = ±28.6°
    "thigh": 0.7,     # Thigh joints: ±0.7 rad = ±40°
    "calf": 0.7,      # Calf joints: ±0.7 rad = ±40°
}
```

**Why different scales?** Each joint has different torque limits and range of motion. This is learned from the reference implementation.

#### Customization 7: Customize Rewards for Go2

The template rewards exist but use default parameters. Go2 needs specific tuning:

```python
cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,    # Relaxed hip when standing
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,   # Medium thigh penalty
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,   # Stricter calf penalty
}
cfg.rewards["pose"].params["std_walking"] = {
    # When moving fast, allow more joint movement
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
}
```

**Why?** Go2 has a quadruped morphology:
- When standing still → joints should be close to default → small reward variance
- When walking/running → joints move more → large reward variance

#### Customization 8: Set Illegal Contact Termination

```python
cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
)
```

**Why?** Detect if body (non-foot) touches ground hard → robot fell → terminate episode.

#### Customization 9: Play Mode

When `play=True` (evaluation, not training):

```python
if play:
    cfg.episode_length_s = int(1e9)  # Nearly infinite
    cfg.observations["actor"].enable_corruption = False  # Remove noise
    cfg.events.pop("push_robot", None)  # Don't push during demo
    cfg.terminations.pop("out_of_terrain_bounds", None)  # Let it roam
    cfg.curriculum = {}  # Use final, hardest difficulty
```

### Step 2: `unitree_go2_flat_env_cfg()` - Flat Terrain Version

This further customizes the rough version for flat ground:

```python
cfg = unitree_go2_rough_env_cfg(play=play)  # Start with rough version
```

#### What's Different for Flat?

**1. Reduce simulation complexity:**

```python
cfg.sim.njmax = 300                          # Fewer constraint slots
cfg.sim.mujoco.ccd_iterations = 50           # Fewer collision checks
cfg.sim.contact_sensor_maxmatch = 64         # Fewer contact slots
```

**Why?** Flat ground has no obstacles → fewer collisions → can use cheaper settings → train faster.

**2. Change terrain to flat plane:**

```python
cfg.scene.terrain.terrain_type = "plane"     # No roughness
cfg.scene.terrain.terrain_generator = None   # No generation
```

**3. Remove terrain-related observations:**

```python
cfg.scene.sensors = tuple(
    sensor for sensor in (...) if sensor.name != "terrain_scan"
)
del cfg.observations["actor"].terms["height_scan"]
del cfg.observations["critic"].terms["height_scan"]
```

**Why?** No terrain = no need to scan it.

**4. Remove terrain curriculum:**

```python
cfg.curriculum.pop("terrain_levels", None)  # No terrain difficulty increase
```

**Why?** Only velocity curriculum on flat terrain.

**5. In play mode, reduce velocity ranges:**

```python
if play:
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)   # Reduce for stable demo
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
```

**Why?** During demo, keep motion reasonable and visible.

---

## Part 3: WHY GO2 WALKS IN RANDOM DIRECTIONS

This is defined by the **velocity command system**. Let me explain:

### The Command System

From the template:

```python
commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,      # 10% → vx=vy=wz=0 (stand still)
        rel_heading_envs=0.3,       # 30% → vx=vy=0, random wz (rotate in place)
        rel_forward_envs=0.2,       # 20% → vy=0, random vx (go forward/back)
        heading_command=True,       # Allow rotation
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
        ),
    )
}
```

### What This Means

**Every 3-8 seconds**, a new random command is sampled:

```
Time: 0s   → Sample: vx = 0.5, vy = -0.3, wz = 0.2
            ↓ Go2 receives command and tries to follow it
Time: 3-8s → Sample: vx = -0.7, vy = 0.1, wz = -0.4
            ↓ New direction! Go2 re-orients
Time: 6-16s → Sample: vx = 0.0, vy = 0.0, wz = 0.0
            ↓ Stand still command
```

### The Probability Distribution

```
100% of commands split as:
│
├─ 10% Standing:    vx=0, vy=0, wz=0
├─ 30% Heading:     vx=0, vy=0, wz=random(-0.5, 0.5)
├─ 20% Forward:     vx=random(-1, 1), vy=0, wz=0
└─ 40% General:     vx=random(-1, 1), vy=random(-1, 1), wz=random(-0.5, 0.5)
```

### Curriculum: Velocity Gets Harder Over Time

From the template curriculum:

```python
"command_vel": CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
        "velocity_stages": [
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
            {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
    },
),
```

**What this means:**

| Training Iteration | Step Count | lin_vel_x Range | Meaning |
|-------------------|-----------|-----------------|---------|
| 0-5000 | 0 to 120K | (-1.0, 1.0) | Easy: small, slow movements |
| 5000-10000 | 120K to 240K | (-1.5, 2.0) | Medium: faster forward, backward |
| 10000+ | 240K+ | (-2.0, 3.0) | Hard: very fast movements |

**Timeline:**
- **Iterations 0-5000**: Go2 learns basic walking with small velocity targets
- **Iterations 5000-10000**: Speed increases to ~2 m/s forward
- **Iterations 10000+**: Final stage: 3 m/s forward, 2 m/s backward

### Why Random Directions = More Robust

If we only trained on forward movement:
- ❌ Go2 would only learn to move forward
- ❌ Can't handle sideways commands
- ❌ Can't rotate

By **randomizing directions** every 3-8 seconds:
- ✅ Go2 learns forward, backward, and sideways equally
- ✅ Go2 learns rotation
- ✅ Policy generalizes to any velocity command

**During play mode**, we keep changing commands so the demo shows:
```
Go2 walks forward → stops → rotates → walks backward → turns sideways
→ combines forward + rotation → stands still → repeat
```

---

## Part 4: Observation Details (Why Noisy Actor vs. Clean Critic)

### Actor Network (Policy) - Gets Noisy Observations

From the template:

```python
actor_terms = {
    "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        noise=Unoise(n_min=-0.5, n_max=0.5),  # Noisy velocity reading
    ),
    "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},  # Misaligned joint sensors
        noise=Unoise(n_min=-0.01, n_max=0.01),  # ±0.01 rad error
    ),
    ...
}
```

**Why noisy?**
- Real robots have sensor noise (gyro drift, encoder quantization)
- If we train on perfect sensors → policy breaks on real robot
- Adding realistic noise during training → policy handles real noise

### Critic Network (Value Function) - Gets Clean Observations

```python
critic_terms = {
    **actor_terms,
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),  # No noise!
    "foot_contact": ObservationTermCfg(...),  # Privileged info
    ...
}
```

**Why clean?**
- Value function only estimates "how good is this state?"
- Doesn't directly control → noise doesn't hurt it
- Clean observations → stable value estimates → better training
- Critic uses **privileged information** (foot contact, air time) that won't be available on real robot

---

## Part 5: Summary Table

### Template vs. Go2-Rough Customizations

| Aspect | Template | Go2-Rough | Why Change |
|--------|----------|-----------|-----------|
| Robot | Generic | Go2 MJCF | Need specific morphology |
| Physics iterations | Std | 500 CCD | Rough terrain needs more checks |
| Sensors | Terrain scan + height | + foot contacts | Detect ground contact |
| Action scale | 0.5 (default) | GO2_ACTION_SCALE | Go2-specific joint limits |
| Reward tuning | Generic | Go2-specific | Quadruped joint patterns |
| Terrain | Rough generator | Rough with curriculum | Difficulty progression |

### Go2-Rough vs. Go2-Flat Customizations

| Aspect | Rough | Flat | Why Change |
|--------|-------|------|-----------|
| Terrain | Generator (random) | Plane (simple) | Flat easier to learn |
| Physics iterations | 500 CCD | 50 CCD | Fewer obstacles |
| Height scan obs | Yes | No | No terrain to scan |
| Terrain curriculum | Yes | No | Flat doesn't need it |
| Training cost | Slower | Faster | Flat is simpler |

### Random Walk Mechanism

| Component | Value | Effect |
|-----------|-------|--------|
| Resampling interval | 3-8 seconds | Change goal every few seconds |
| Forward movement | 20% probability | Sometimes just go forward |
| Rotation | 30% probability | Sometimes just rotate |
| Standing | 10% probability | Sometimes stop |
| General | 40% probability | Random combination |
| Velocity curriculum | 3 stages | Gradually increase difficulty |
| lin_vel_x range | (-1 to +3) m/s | From slow to fast |

---

## Key Takeaway

The Go2 flat configuration is built in **layers**:

```
make_velocity_env_cfg()          [Generic velocity task template]
    ↓
unitree_go2_rough_env_cfg()      [Add Go2-specific sensors, morphology]
    ↓
unitree_go2_flat_env_cfg()       [Simplify for flat terrain]
    ↓
[Training with random velocity commands every 3-8 seconds]
    ↓
[Go2 learns to walk in ANY direction]
```

The **random directions** come from the command system, not the policy itself. The curriculum ensures Go2 starts with easy targets and gradually learns harder tasks.
