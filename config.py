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
    
    # Navigation behavior
    'navigation_mode': 'sequential_cross',  # 'sequential_cross' or 'roundabout'
    'sequential_cross_phases': [
        {
            'name': 'north_cross',
            'target': [2.5, 3.5, 0.6],
            'completion_zone': {'x': [1.5, 4.5], 'y': [2.0, 4.5]},  # Expanded zone
            'reward': 120
        },
        {
            'name': 'east_cross',
            'target': [2.5, -3.5, 0.6],
            'completion_zone': {'x': [1.5, 4.5], 'y': [-4.5, -2.0]},  # Expanded zone
            'reward': 140
        },
        {
            'name': 'south_cross',
            'target': [-2.5, -3.5, 0.6],
            'completion_zone': {'x': [-4.5, -1.5], 'y': [-4.5, -2.0]},  # Expanded zone
            'reward': 160
        },
        {
            'name': 'west_cross',
            'target': [-2.5, 3.5, 0.6],
            'completion_zone': {'x': [-4.5, -1.5], 'y': [2.0, 4.5]},  # Expanded zone
            'reward': 200
        },
    ],
    'roundabout_waypoints': [
        [-2.5, 3.5, 0.6],   # Start position (North side)
        [2.5, 3.5, 0.6],    # East side (cross north crosswalk)
        [2.5, -3.5, 0.6],   # South side (cross east crosswalk)
        [-2.5, -3.5, 0.6],  # West side (cross south crosswalk)  
        [-2.5, 3.5, 0.6],   # Back to start (cross west crosswalk - complete loop)
    ],
    'waypoint_tolerance': 1.5,  # Distance to consider waypoint reached (increased from 1.0)
    'require_sequential_waypoints': True,  # Must reach waypoints in order
    'center_idle_zone': {'x': [-1.2, 1.2], 'y': [-1.2, 1.2]},
    'center_idle_penalty': -0.6,
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
    # Enhanced state space for roundabout navigation
    # [ped_x, ped_y, ped_vx, ped_vy, current_target_x, current_target_y, waypoint_index,
    #  4 traffic lights (N,S,E,W), 8 cars (rel_x,rel_y,vx,vy for each)]
    'state_space_size': 43,  # 4 + 3 + 4 + 32 = 43 total dimensions
    'action_space_size': 5,  # 0=stay, 1=forward, 2=back, 3=left, 4=right
    
    'max_episode_steps': 2000,  # Longer episodes for roundabout navigation
    
    'reward_structure': {
        'reach_waypoint': 150,         # Increased from 100 - makes waypoint completion more valuable
        'reach_final_target': 1000,    # Increased from 500 - huge bonus for completing all 4
        'collision_penalty': -80,      # Further reduced from -100 - less discouraging
        'time_penalty': -0.005,        # Further reduced - less time pressure
        'safe_crossing_bonus': 50,     # Bonus for crossing safely
        'near_miss_penalty': -3,       # Further reduced from -5 - less discouraging
        'progress_reward': 15,         # Increased from 10 - stronger incentive to move forward
        'wrong_direction_penalty': -0.5, # Reduced from -1 - less harsh
        'traffic_awareness_bonus': 15, # Increased from 10 - more reward for smart behavior
        'sequential_bonus': 75,        # Increased from 50 - more reward for sequential progress
        'movement_bonus': 0.5,         # Small bonus for taking any movement action (encourages exploration)
        'survival_bonus': 0.02,        # Reduced survival bonus (was 0.1) - prevents it from dominating
        'midway_reward': 30,           # Reward for crossing center line (midway checkpoint)
        'proximity_bonus_scale': 50,   # Bonus for getting very close to waypoint (50% distance = 25 bonus)
    },
    
    # DQN Hyperparameters - optimized for better roundabout learning
    'learning_rate': 0.0005,  # Slightly reduced for stability
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,      # Keep some exploration even late in training
    'epsilon_decay': 0.9995,  # Slower decay so agent explores longer
    'batch_size': 128,  # Increased for more stable learning
    'replay_buffer_size': 50000,  # Increased buffer for better experience diversity
    'target_update_frequency': 200,  # Less frequent updates for stability
    'train_frequency': 4,
    'min_replay_size': 2000,  # More initial experience before training
    
    # Neural Network Architecture - optimized  
    'hidden_layers': [256, 256, 128],  # Larger network for complex roundabout behavior
}

# Simulation Settings
SIMULATION_CONFIG = {
    'fps': 60,
    'gui_mode': True,
    'enable_shadows': True,
    'physics_timestep': 1/240,  # More accurate physics
    
    # Performance optimizations
    'enable_graphics_optimization': True,
    'max_render_fps': 30,  # Limit rendering FPS for performance
    'physics_solver_iterations': 50,  # Reduced for better performance
    'collision_margin': 0.001,  # Smaller margin for better performance
    
    # Visual quality settings
    'enable_anti_aliasing': False,  # Disable for performance
    'shadow_map_size': 512,       # Reduced shadow quality for performance
    'enable_depth_buffer': False, # Disable for performance
    'enable_rgb_preview': False,  # Disable unnecessary previews
}