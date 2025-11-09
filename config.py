"""
Configuration file for the Crossroad RL Environment
Modify these parameters to customize the simulation
"""

# Environment Settings
ENVIRONMENT_CONFIG = {
    # Camera settings
    'camera_distance': 15,
    'camera_yaw': 45,
    'camera_pitch': -30,
    'camera_target': [0, 0, 0],
    
    # Road dimensions
    'intersection_size': 6,    # Size of main intersection
    'road_length': 12,         # Length of roads extending from intersection
    'road_width': 2.5,         # Width of each road
    
    # Traffic light timing
    'signal_change_interval': 180,  # frames (3 seconds at 60 FPS)
    
    # Environment colors
    'grass_color': [0.2, 0.6, 0.2, 1],
    'road_color': [0.1, 0.1, 0.1, 1],
    'intersection_color': [0.15, 0.15, 0.15, 1],
    'crosswalk_color': [1, 1, 1, 1],
    'yellow_line_color': [1, 1, 0, 1],
}

# Car Settings
CAR_CONFIG = {
    'num_cars': 6,
    'car_speed_range': [0.03, 0.08],  # Min and max speed
    'car_colors': [
        [1, 0, 0, 1],    # Red
        [0, 0, 1, 1],    # Blue  
        [0, 1, 0, 1],    # Green
        [1, 1, 0, 1],    # Yellow
        [1, 0, 1, 1],    # Purple
        [0, 1, 1, 1],    # Cyan
    ],
    'car_dimensions': [0.8, 0.4, 0.2],  # Length, width, height
    'wheel_radius': 0.2,
}

# Pedestrian Settings  
PEDESTRIAN_CONFIG = {
    'start_position': [-2.5, 3.5, 0.6],
    'target_position': [2.5, 3.5, 0.6],
    'body_radius': 0.3,
    'body_height': 1.2,
    'head_radius': 0.2,
    'skin_color': [0.8, 0.6, 0.4, 1],
}

# Environmental Objects
ENVIRONMENT_OBJECTS = {
    'trees': {
        'positions': [
            [-8, 8], [-8, -8], [8, 8], [8, -8],      # Corner trees
            [-6, 12], [6, 12], [-6, -12], [6, -12],  # Additional trees
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
            [-2.5, 8], [2.5, 8], [-2.5, -8], [2.5, -8],    # North-South road
            [8, -2.5], [8, 2.5], [-8, -2.5], [-8, 2.5]     # East-West road
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
    'light_colors': {
        'red_on': [0.8, 0, 0, 1],
        'red_off': [0.2, 0, 0, 1],
        'yellow_on': [0.8, 0.8, 0, 1], 
        'yellow_off': [0.2, 0.2, 0, 1],
        'green_on': [0, 0.8, 0, 1],
        'green_off': [0, 0.2, 0, 1],
    }
}

# RL Training Settings
RL_CONFIG = {
    'state_space_size': 16,  # Observation space dimension (4 agent + 2 target + 4 traffic lights + 6 cars)
    'action_space_size': 5,  # 0=stay, 1=forward, 2=back, 3=left, 4=right
    'max_episode_steps': 1000,
    'reward_structure': {
        'reach_target': 100,
        'collision_penalty': -100,
        'time_penalty': -0.1,
        'safe_crossing_bonus': 50,
    },
    # DQN Hyperparameters
    'learning_rate': 0.001,
    'gamma': 0.99,  # Discount factor
    'epsilon_start': 1.0,  # Initial exploration rate
    'epsilon_end': 0.01,  # Final exploration rate
    'epsilon_decay': 0.995,  # Exploration decay rate
    'batch_size': 64,
    'replay_buffer_size': 10000,
    'target_update_frequency': 100,  # Update target network every N steps
    'train_frequency': 4,  # Train every N steps
    'min_replay_size': 1000,  # Minimum experiences before training
    # Neural Network Architecture
    'hidden_layers': [128, 128],  # Hidden layer sizes
}