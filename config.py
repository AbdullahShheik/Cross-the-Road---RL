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
    'waypoint_tolerance': 1.5,  # Increased from 1.0 for easier waypoint detection
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
    # Enhanced state space for roundabout navigation with collision awareness
    # [ped_x, ped_y, target_x, target_y, rel1_x, rel1_y, rel2_x, rel2_y, 
    #  ns_green, ew_green, min_car_distance, is_in_danger]
    'state_space_size': 12,
    'action_space_size': 5,
    
    'max_episode_steps': 1500,  # Increased from 1000 - give more time to find safe crossing windows
    
    'reward_structure': {
        'reach_waypoint': 80,                # Reward for each waypoint reached
        'reach_final_target': 400,           # Reward for completing a full round
        'round_bonus': 200,                  # Bonus when finishing a round but more rounds remain
        'collision_penalty': -100,           # Penalty for collisions
        'time_penalty': -0.001,              # Small per-step time penalty

        'safe_crossing_bonus_far': 0.10,     # Per-step bonus when far from cars
        'safe_crossing_bonus_mod': 0.05,     # Per-step bonus when moderately far

        'progress_reward': 8.0,              # Multiplier for progress toward current waypoint
        'wrong_direction_penalty': -0.01,    # Small penalty for moving away from target when safe
        'movement_bonus': 0.01,              # Tiny reward for taking an action (prevents standing still)
        
        'zebra_on_bonus': 0.5,                # Reward for staying on a zebra crossing
        'zebra_survival_bonus': 0.02,         # Small extra while on zebra
        'zebra_off_penalty': -0.20,           # Penalty when off-zebra in center
        'zebra_idle_penalty': -0.50,          # Extra penalty for idling off-zebra
        'zebra_progress_multiplier': 6.0,     # Multiplier for progress toward nearest zebra
    },
    
    # DQN Hyperparameters - balanced to encourage active exploration + movement
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,  # Increased from 0.01 - keep exploring even late
    'epsilon_decay': 0.988,  # Slower decay = MORE exploration (was 0.992)
    'batch_size': 256,
    'replay_buffer_size': 100000,
    'target_update_frequency': 100,
    'train_frequency': 2,
    'min_replay_size': 500,  # Reduced from 1000 - start learning MUCH earlier
    
    # Neural Network Architecture
    'hidden_layers': [256, 256, 128],
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