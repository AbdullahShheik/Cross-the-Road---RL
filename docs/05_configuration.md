# ⚙️ Advanced Configuration Guide

This guide shows you how to customize the simulation, training, and agent behavior to create your own variations of the crossroad environment.

## 🎯 Configuration Overview

All settings are stored in `config.py`, organized into logical sections:

```python
config.py
├── ENVIRONMENT_CONFIG    # 3D world settings
├── CAR_CONFIG           # Vehicle behavior  
├── PEDESTRIAN_CONFIG    # Agent properties
├── RL_CONFIG           # Learning parameters
└── SIMULATION_CONFIG   # Performance settings
```

## 🌍 Environment Customization

### 🛣️ Road Layout Settings

```python
ENVIRONMENT_CONFIG = {
    # Intersection dimensions
    'intersection_size': 6,      # Size of central crossing area
    'road_length': 12,           # Length of each road segment
    'road_width': 2.5,           # Width of each road
    
    # Visual settings
    'grass_color': [0.2, 0.6, 0.2, 1],     # Green grass
    'road_color': [0.1, 0.1, 0.1, 1],      # Dark asphalt
    'crosswalk_color': [1, 1, 1, 1],       # White stripes
}
```

#### Customization Examples:

**Larger Intersection:**
```python
'intersection_size': 10,    # Bigger crossing area
'road_length': 20,          # Longer approach roads
```

**Different Colors:**
```python
'grass_color': [0.8, 0.7, 0.5, 1],    # Desert sand
'road_color': [0.3, 0.2, 0.2, 1],     # Brick road
```

### 📹 Camera Settings

```python
ENVIRONMENT_CONFIG = {
    'camera_distance': 18,       # How far back camera sits
    'camera_yaw': 45,           # Rotation angle
    'camera_pitch': -35,        # Looking down angle
    'camera_target': [0, 0, 0], # What camera focuses on
}
```

**Better Overview:**
```python
'camera_distance': 25,    # Further back
'camera_pitch': -45,      # Look down more
```

## 🚗 Vehicle Behavior Tuning

### 🚙 Traffic Density and Speed

```python
CAR_CONFIG = {
    'num_cars': 8,                        # Total vehicles
    'car_speed_range': [0.05, 0.1],       # Min/max speeds
    'rule_breaker_probability': 0.25,      # 25% break rules
    
    # Traffic light compliance
    'stop_distance': 4.5,                 # When to stop for lights
}
```

#### Traffic Scenarios:

**Heavy Traffic:**
```python
'num_cars': 12,
'car_speed_range': [0.03, 0.07],    # Slower speeds
'rule_breaker_probability': 0.4,     # More chaos
```

**Light Traffic:**
```python
'num_cars': 4,
'car_speed_range': [0.08, 0.15],    # Faster speeds  
'rule_breaker_probability': 0.1,     # More orderly
```

**Aggressive Drivers:**
```python
'rule_breaker_probability': 0.6,     # 60% ignore lights!
'stop_distance': 2.0,               # Stop later
```

### 🎨 Car Appearance

```python
CAR_CONFIG = {
    'car_colors': [
        [1, 0, 0, 1],       # Red
        [0, 0, 1, 1],       # Blue
        # Add more colors...
    ],
    'car_dimensions': [0.8, 0.4, 0.2],    # Length, width, height
}
```

## 🚦 Traffic Light Timing

```python
ENVIRONMENT_CONFIG = {
    'signal_change_interval': 240,    # Frames (4 seconds at 60fps)
    'yellow_light_duration': 40,     # Frames (0.67 seconds)
}
```

**Faster Traffic:**
```python
'signal_change_interval': 180,    # 3 seconds green
'yellow_light_duration': 30,     # 0.5 seconds yellow
```

**Slower, More Careful:**
```python
'signal_change_interval': 360,    # 6 seconds green  
'yellow_light_duration': 60,     # 1 second yellow
```

## 🚶 Pedestrian Behavior

### 🎯 Navigation Waypoints

```python
PEDESTRIAN_CONFIG = {
    'roundabout_waypoints': [
        [-2.5, 3.5, 0.6],   # Start
        [0, 3.5, 0.6],      # North center
        [2.5, 3.5, 0.6],    # North end
        # ... continue around
    ],
    'waypoint_tolerance': 0.8,      # How close = "reached"
}
```

#### Custom Routes:

**Simple Crossing (Original):**
```python
'roundabout_waypoints': [
    [-2.5, 3.5, 0.6],    # Start
    [2.5, 3.5, 0.6],     # End
],
```

**Figure-8 Pattern:**
```python
'roundabout_waypoints': [
    [-2.5, 3.5, 0.6],    # Start north
    [0, 0, 0.6],         # Center
    [2.5, -3.5, 0.6],    # South east
    [0, 0, 0.6],         # Center again
    [-2.5, 3.5, 0.6],    # Back to start
],
```

**Random Exploration:**
```python
'navigation_mode': 'random',  # Instead of 'roundabout'
```

### 🏃 Movement Settings

```python
PEDESTRIAN_CONFIG = {
    'movement_speed': 0.05,           # Base speed
    'safe_distance_from_car': 3.0,   # Safety margin
    'body_radius': 0.3,              # Collision size
}
```

**Faster Pedestrian:**
```python
'movement_speed': 0.08,           # 60% faster
```

**More Cautious:**
```python
'safe_distance_from_car': 5.0,   # Wider safety margin
```

## 🧠 Learning Parameters

### 🎯 Reward System Tuning

```python
RL_CONFIG = {
    'reward_structure': {
        'reach_waypoint': 50,           # Waypoint bonus
        'reach_final_target': 200,      # Completion bonus  
        'collision_penalty': -100,      # Crash penalty
        'time_penalty': -0.05,          # Step penalty
        'progress_reward': 2,           # Movement bonus
        'traffic_awareness_bonus': 5,   # Smart timing
    },
}
```

#### Reward Tuning Examples:

**Encourage Speed:**
```python
'time_penalty': -0.1,            # Higher step penalty
'progress_reward': 5,            # More movement reward
```

**Encourage Safety:**
```python
'collision_penalty': -200,       # Harsher crash penalty  
'traffic_awareness_bonus': 10,   # Bigger safety bonus
'near_miss_penalty': -10,        # Penalty for close calls
```

**Encourage Exploration:**
```python
'exploration_reward': 5,         # Bonus for new areas
'waypoint_tolerance': 0.5,       # Must be more precise
```

### 🔬 Training Hyperparameters

```python
RL_CONFIG = {
    'learning_rate': 0.0005,         # How fast to learn
    'epsilon_decay': 0.995,          # Exploration decrease rate
    'batch_size': 128,               # Training batch size
    'replay_buffer_size': 50000,     # Experience memory
    'target_update_frequency': 200,   # Target net updates
}
```

#### Performance Tuning:

**Faster Learning (Less Stable):**
```python
'learning_rate': 0.001,          # Higher learning rate
'target_update_frequency': 100,   # More frequent updates
```

**More Stable (Slower):**
```python  
'learning_rate': 0.0001,         # Lower learning rate
'batch_size': 256,               # Larger batches
'epsilon_decay': 0.999,          # Slower exploration decay
```

**Better Sample Efficiency:**
```python
'replay_buffer_size': 100000,    # More experience storage
'min_replay_size': 5000,         # More warmup experience
```

### 🏗️ Neural Network Architecture

```python
RL_CONFIG = {
    'hidden_layers': [256, 256, 128],    # Network size
    'state_space_size': 39,              # Input dimensions
    'action_space_size': 5,              # Output actions
}
```

**Larger Network (More Capacity):**
```python
'hidden_layers': [512, 512, 256, 128],
```

**Smaller Network (Faster):**
```python
'hidden_layers': [128, 64],
```

## ⚡ Performance Optimization

### 🖥️ Simulation Performance

```python
SIMULATION_CONFIG = {
    'fps': 60,                           # Target framerate
    'physics_timestep': 1/240,           # Physics accuracy
    'physics_solver_iterations': 50,     # Physics quality
    'enable_shadows': True,              # Visual quality
}
```

#### Performance Modes:

**Maximum Performance:**
```python
'fps': 30,                           # Lower framerate
'physics_solver_iterations': 20,     # Faster physics  
'enable_shadows': False,             # No shadows
'enable_anti_aliasing': False,       # No smoothing
```

**Maximum Quality:**
```python
'fps': 120,                          # High framerate
'physics_solver_iterations': 100,    # Accurate physics
'shadow_map_size': 2048,            # High-res shadows
'enable_anti_aliasing': True,        # Smooth graphics
```

### 🧠 Training Performance

```python
# GPU settings (if available)
RL_CONFIG = {
    'device': 'cuda',               # Use GPU
    'batch_size': 256,              # Larger batches for GPU
}

# CPU optimization
RL_CONFIG = {
    'device': 'cpu',                # Use CPU
    'batch_size': 64,               # Smaller batches
    'train_frequency': 8,           # Train less often
}
```

## 🎨 Custom Scenarios

### Scenario 1: Rush Hour Traffic
```python
CAR_CONFIG['num_cars'] = 16
CAR_CONFIG['car_speed_range'] = [0.02, 0.05]
CAR_CONFIG['rule_breaker_probability'] = 0.5
ENVIRONMENT_CONFIG['signal_change_interval'] = 120  # Shorter lights
```

### Scenario 2: Sunday Morning  
```python
CAR_CONFIG['num_cars'] = 2
CAR_CONFIG['car_speed_range'] = [0.08, 0.12] 
CAR_CONFIG['rule_breaker_probability'] = 0.05
ENVIRONMENT_CONFIG['signal_change_interval'] = 480  # Longer lights
```

### Scenario 3: School Zone
```python
CAR_CONFIG['car_speed_range'] = [0.02, 0.04]  # Very slow
PEDESTRIAN_CONFIG['safe_distance_from_car'] = 2.0  # Less cautious
RL_CONFIG['collision_penalty'] = -500  # Much worse penalty
```

### Scenario 4: Highway On-Ramp
```python
CAR_CONFIG['num_cars'] = 6
CAR_CONFIG['car_speed_range'] = [0.15, 0.25]  # Very fast
PEDESTRIAN_CONFIG['movement_speed'] = 0.08  # Faster walking
RL_CONFIG['time_penalty'] = -0.2  # Encourage speed
```

## 🔧 Configuration Tips

### 🧪 Testing Changes

1. **Start small**: Change one parameter at a time
2. **Test quickly**: Use 50-episode training runs
3. **Compare results**: Keep notes on performance
4. **Backup configs**: Save working configurations

### 📊 Monitoring Impact

Watch these metrics when changing config:
- **Success rate**: % of completed roundabouts
- **Training time**: Episodes to reach good performance  
- **Stability**: Consistency across runs
- **Realism**: Does behavior look natural?

### ⚠️ Common Pitfalls

**Don't make rewards too sparse:**
```python
# BAD: Only reward final success
'reach_waypoint': 0
'progress_reward': 0

# GOOD: Provide intermediate rewards
'reach_waypoint': 50  
'progress_reward': 2
```

**Don't make environment too hard:**
```python
# BAD: Impossible scenario
'num_cars': 50
'rule_breaker_probability': 1.0  

# GOOD: Challenging but fair
'num_cars': 12
'rule_breaker_probability': 0.4
```

## 📝 Configuration Templates

Save these as different config files:

### beginner_config.py
```python
# Easy learning environment
CAR_CONFIG['num_cars'] = 4
CAR_CONFIG['rule_breaker_probability'] = 0.1
RL_CONFIG['collision_penalty'] = -50
```

### expert_config.py  
```python
# Challenging environment  
CAR_CONFIG['num_cars'] = 12
CAR_CONFIG['rule_breaker_probability'] = 0.4
RL_CONFIG['collision_penalty'] = -200
```

### speed_config.py
```python
# Fast training  
SIMULATION_CONFIG['enable_shadows'] = False
RL_CONFIG['batch_size'] = 256
RL_CONFIG['train_frequency'] = 2
```

## 🚀 Next Steps

After customizing configuration:

1. **Test changes**: Run short training sessions
2. **Document results**: Keep track of what works
3. **Share findings**: Help others with your discoveries  
4. **Iterate**: Continue improving based on results

**Need help debugging? → [Troubleshooting Guide](06_troubleshooting.md)**

**Ready to experiment? Start with small changes and work your way up!**