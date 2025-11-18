"""
Configuration file for the Crossroad RL Environment
Improved settings for realistic Karachi-style traffic simulation
"""

# Environment Settings
ENVIRONMENT_CONFIG = {
    # Camera settings
    'camera_distance': 18,
    'camera_yaw': 45,
    'camera_pitch': -35,
    'camera_target': [0, 0, 0],
    
    # Road dimensions
    'intersection_size': 6,
    'road_length': 12,
    'road_width': 2.5,
    
    # Traffic light timing (in frames, assuming 60 FPS)
    'signal_change_interval': 240,  # 4 seconds green light
    'yellow_light_duration': 40,    # ~0.67 seconds yellow
    
    # Environment colors
    'grass_color': [0.2, 0.6, 0.2, 1],
    'road_color': [0.1, 0.1, 0.1, 1],
    'intersection_color': [0.15, 0.15, 0.15, 1],
    'crosswalk_color': [1, 1, 1, 1],
    'yellow_line_color': [1, 1, 0, 1],
}

# Car Settings
CAR_CONFIG = {
    'num_cars': 8,  # Increased for more traffic
    'car_speed_range': [0.05, 0.1],  # Slightly faster, more realistic
    'rule_breaker_probability': 0.25,  # 25% chance to break traffic rules (Karachi-style!)
    
    'car_colors': [
        [1, 0, 0, 1],       # Red
        [0, 0, 1, 1],       # Blue
        [0, 1, 0, 1],       # Green
        [1, 1, 0, 1],       # Yellow
        [1, 0, 1, 1],       # Purple
        [0, 1, 1, 1],       # Cyan
        [0.8, 0.4, 0, 1],   # Orange
        [0.5, 0.8, 0.3, 1], # Lime
    ],
    
    'car_dimensions': [0.8, 0.4, 0.2],
    'wheel_radius': 0.2,
    
    # Lane positions (offset from center)
    'lane_offsets': {
        'north': -1.2,   # Left lane for northbound
        'south': 1.2,    # Right lane for southbound
        'east': -1.2,    # Left lane for eastbound
        'west': 1.2,     # Right lane for westbound
    },
    
    # Stop distance from intersection center
    'stop_distance': 4.5,
}

# Pedestrian Settings
PEDESTRIAN_CONFIG = {
    'start_position': [-2.5, 3.5, 0.6],
    'target_position': [2.5, 3.5, 0.6],
    'body_radius': 0.3,
    'body_height': 1.2,
    'head_radius': 0.2,
    'skin_color': [0.8, 0.6, 0.4, 1],
    
    # Movement parameters for RL
    'movement_speed': 0.05,
    'safe_distance_from_car': 3.0,  # Minimum safe distance from cars
}

# Environmental Objects
ENVIRONMENT_OBJECTS = {
    'trees': {
        'positions': [
            [-8, 8], [-8, -8], [8, 8], [8, -8],
            [-6, 12], [6, 12], [-6, -12], [6, -12],
        ],
        'trunk_color': [0.4, 0.2, 0.1, 1],
        'leaves_color': [0.1, 0.5, 0.1, 1],
    },
    'buildings': [
        {'position': [-15, 15, 3], 'dimensions': [3, 3, 6], 'color': [0.7, 0.7, 0.9, 1]},
        {'position': [15, 15, 3], 'dimensions': [2, 4, 6], 'color': [0.8, 0.6, 0.6, 1]},
        {'position': [-15, -15, 3], 'dimensions': [4, 2, 6], 'color': [0.6, 0.6, 0.8, 1]},
        {'position': [15, -15, 3], 'dimensions': [3, 3, 6], 'color': [0.8, 0.8, 0.6, 1]},
    ],
    'street_lamps': {
        'positions': [
            [-2.5, 8], [2.5, 8], [-2.5, -8], [2.5, -8],
            [8, -2.5], [8, 2.5], [-8, -2.5], [-8, 2.5]
        ],
        'pole_color': [0.3, 0.3, 0.3, 1],
        'light_color': [1, 1, 0.8, 1],
    },
    'fire_hydrants': {
        'positions': [[-3, 6], [3, 6], [-3, -6], [3, -6]],
        'color': [1, 0, 0, 1],
    }
}

# Traffic Light Settings
TRAFFIC_LIGHT_CONFIG = {
    'positions': [
        [-4, 4, "north"],
        [4, -4, "south"],
        [4, 4, "east"],
        [-4, -4, "west"]
    ],
    'pole_color': [0.3, 0.3, 0.3, 1],
    'box_color': [0.2, 0.2, 0.2, 1],
    
    # Enhanced light colors for better visibility
    'light_colors': {
        'red_on': [1.0, 0.0, 0.0, 1],      # Bright red
        'red_off': [0.2, 0.0, 0.0, 1],     # Dim red
        'yellow_on': [1.0, 1.0, 0.0, 1],   # Bright yellow
        'yellow_off': [0.2, 0.2, 0.0, 1],  # Dim yellow
        'green_on': [0.0, 1.0, 0.0, 1],    # Bright green
        'green_off': [0.0, 0.2, 0.0, 1],   # Dim green
    },
    
    'light_radius': 0.15,  # Slightly larger for visibility
}

# RL Training Settings
RL_CONFIG = {
    # State space: [ped_x, ped_y, ped_vx, ped_vy, target_x, target_y, 
    #               4 traffic lights (N,S,E,W), 8 cars (x,y for each)]
    'state_space_size': 22,  # Updated for 8 cars
    'action_space_size': 5,  # 0=stay, 1=forward, 2=back, 3=left, 4=right
    
    'max_episode_steps': 1000,
    
    'reward_structure': {
        'reach_target': 100,
        'collision_penalty': -100,
        'time_penalty': -0.1,
        'safe_crossing_bonus': 50,
        'near_miss_penalty': -5,  # Penalty for getting too close to cars
        'progress_reward': 1,      # Small reward for moving toward goal
    },
    
    # DQN Hyperparameters
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'replay_buffer_size': 10000,
    'target_update_frequency': 100,
    'train_frequency': 4,
    'min_replay_size': 1000,
    
    # Neural Network Architecture
    'hidden_layers': [128, 128, 64],
}

# Simulation Settings
SIMULATION_CONFIG = {
    'fps': 60,
    'gui_mode': True,
    'enable_shadows': True,
    'physics_timestep': 1/240,  # More accurate physics
}