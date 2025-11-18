# gym_crossroad_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
from intersection_env import CrossroadEnvironment as BaseCrossroad 
from config import PEDESTRIAN_CONFIG, RL_CONFIG
import math

class EnhancedCrossroadGymEnv(gym.Env):
    """
    Enhanced Gym wrapper with better reward shaping and state representation.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, gui=False, max_steps=None, reward_shaping=True):
        super().__init__()
        self.gui = gui
        self._max_steps = max_steps or RL_CONFIG.get('max_episode_steps', 1000)
        self.reward_shaping = reward_shaping
        
        self._p_connection_mode = p.GUI if gui else p.DIRECT
        self._init_pybullet()

        # Action space: 0=stay, 1=forward, 2=back, 3=left, 4=right
        self.action_space = spaces.Discrete(RL_CONFIG.get('action_space_size', 5))

        # Enhanced observation space
        obs_len = 4 + 2 + 4 + 3 * 4  # agent(4) + target(2) + 4 tl + four nearest cars
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(obs_len,), dtype=np.float32
        )

        self.step_count = 0
        self.last_distance_to_target = None
        self.collision_detected = False
        
        # Statistics
        self.episode_stats = {
            'collisions': 0,
            'successes': 0,
            'timeouts': 0,
            'total_reward': 0
        }

    def _init_pybullet(self):
        """Initialize PyBullet environment."""
        try:
            p.disconnect()
        except Exception:
            pass
        
        self.base_env = BaseCrossroad(connection_mode=self._p_connection_mode)
        self.physics_client = self.base_env.physicsClient
        
        self.ped_body_id = self.base_env.pedestrian['torso']
        self.target_pos = PEDESTRIAN_CONFIG.get('target_position', [2.5, 3.5, 0.6])
        self.start_pos = PEDESTRIAN_CONFIG.get('start_position', [-2.5, 3.5, 0.6])

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.np_random = np.random
        return [seed]

    def _get_obs(self):
        """Get enhanced observation with more car information."""
        # Agent state
        pos, orn = p.getBasePositionAndOrientation(self.ped_body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.ped_body_id)
        agent_x, agent_y = pos[0], pos[1]
        agent_vx, agent_vy = lin_vel[0], lin_vel[1]

        # Target position
        target_x, target_y = self.target_pos[0], self.target_pos[1]

        # Traffic lights
        mapping = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0}
        tl_n = mapping.get(self.base_env.signal_states.get("north", "RED"), 0.0)
        tl_s = mapping.get(self.base_env.signal_states.get("south", "RED"), 0.0)
        tl_e = mapping.get(self.base_env.signal_states.get("east", "RED"), 0.0)
        tl_w = mapping.get(self.base_env.signal_states.get("west", "RED"), 0.0)

        # Get information about nearest 4 cars (only active cars)
        cars = []
        for car in self.base_env.cars:
            # Skip removed cars
            if car['id'] in self.base_env.removed_cars:
                continue
                
            try:
                pos_c, _ = p.getBasePositionAndOrientation(car['body'])
                # For kinematic cars, calculate speed from position changes
                rel_dx = pos_c[0] - agent_x
                rel_dy = pos_c[1] - agent_y
                speed = car['speed']  # Use stored speed
                dist = math.hypot(rel_dx, rel_dy)
                cars.append((dist, rel_dx, rel_dy, speed))
            except Exception:
                # Car might have been removed, skip it
                continue
        
        cars.sort(key=lambda x: x[0])
        
        # Pad to 4 cars
        nearest = cars[:4]
        while len(nearest) < 4:
            nearest.append((999.0, 0.0, 0.0, 0.0))

        obs = np.array([
            agent_x, agent_y, agent_vx, agent_vy,
            target_x, target_y,
            tl_n, tl_s, tl_e, tl_w,
            nearest[0][1], nearest[0][2], nearest[0][3],
            nearest[1][1], nearest[1][2], nearest[1][3],
            nearest[2][1], nearest[2][2], nearest[2][3],
            nearest[3][1], nearest[3][2], nearest[3][3]
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self, action, info):
        """Calculate reward with dense reward shaping."""
        reward = 0.0
        
        # Get current position
        agent_pos, _ = p.getBasePositionAndOrientation(self.ped_body_id)
        dist_to_target = math.hypot(
            agent_pos[0] - self.target_pos[0],
            agent_pos[1] - self.target_pos[1]
        )
        
        # Check for collision
        if self._check_collision():
            reward += RL_CONFIG['reward_structure'].get('collision_penalty', -100)
            info['collision'] = True
            self.episode_stats['collisions'] += 1
            return reward, True, info
        
        # Check if reached target
        if dist_to_target < 0.5:
            reward += RL_CONFIG['reward_structure'].get('reach_target', 100)
            info['success'] = True
            self.episode_stats['successes'] += 1
            return reward, True, info
        
        # Dense reward shaping (if enabled)
        if self.reward_shaping:
            # 1. Progress toward target
            if self.last_distance_to_target is not None:
                progress = self.last_distance_to_target - dist_to_target
                reward += progress * 5.0  # Reward for moving closer
            
            # 2. Penalty for being too close to cars
            min_car_dist = self._get_min_car_distance()
            if min_car_dist < 2.0:
                reward -= (2.0 - min_car_dist) * 0.5
            
            # 3. Small penalty for staying still
            if action == 0:
                reward -= 0.05
            
            # 4. Reward for being in crosswalk area
            if abs(agent_pos[1] - 3.5) < 1.0 and abs(agent_pos[0]) < 3.0:
                reward += 0.1
        
        # Base time penalty
        reward += RL_CONFIG['reward_structure'].get('time_penalty', -0.1)
        
        self.last_distance_to_target = dist_to_target
        
        return reward, False, info

    def _check_collision(self):
        """Check for collision with any car."""
        for car in self.base_env.cars:
            contacts = p.getContactPoints(
                bodyA=self.ped_body_id, 
                bodyB=car['body']
            )
            if len(contacts) > 0:
                return True
        return False

    def _get_min_car_distance(self):
        """Get distance to nearest car."""
        agent_pos, _ = p.getBasePositionAndOrientation(self.ped_body_id)
        min_dist = float('inf')
        
        for car in self.base_env.cars:
            car_pos, _ = p.getBasePositionAndOrientation(car['body'])
            dist = math.hypot(
                car_pos[0] - agent_pos[0],
                car_pos[1] - agent_pos[1]
            )
            min_dist = min(min_dist, dist)
        
        return min_dist

    def step(self, action):
        """Execute action and return next state."""
        # Map action to movement
        step_size = 0.15
        dx, dy = 0.0, 0.0
        
        if action == 1:    # forward: +y
            dy = step_size
        elif action == 2:  # back: -y
            dy = -step_size
        elif action == 3:  # left: -x
            dx = -step_size
        elif action == 4:  # right: +x
            dx = step_size
        
        # Move pedestrian
        pos, orn = p.getBasePositionAndOrientation(self.ped_body_id)
        new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
        
        # Boundary constraints (keep pedestrian in reasonable area)
        new_pos[0] = np.clip(new_pos[0], -5.0, 5.0)
        new_pos[1] = np.clip(new_pos[1], -5.0, 5.0)
        
        p.resetBasePositionAndOrientation(self.ped_body_id, new_pos, orn)
        
        # Move head
        if 'head' in self.base_env.pedestrian:
            head_pos = [new_pos[0], new_pos[1], new_pos[2] + 0.8]
            p.resetBasePositionAndOrientation(
                self.base_env.pedestrian['head'], head_pos, orn
            )

        # Update environment
        self.base_env.update_traffic_lights()
        self.base_env.update_cars()
        p.stepSimulation()

        # Calculate reward and check termination
        info = {}
        reward, done, info = self._calculate_reward(action, info)
        
        self.step_count += 1
        
        # Check timeout
        if self.step_count >= self._max_steps:
            done = True
            info['timeout'] = True
            self.episode_stats['timeouts'] += 1

        obs = self._get_obs()
        self.episode_stats['total_reward'] += reward
        
        return obs, float(reward), done, False, info

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        # Disconnect and recreate
        try:
            p.disconnect()
        except Exception:
            pass
        
        self._init_pybullet()
        self.step_count = 0
        self.last_distance_to_target = math.hypot(
            self.start_pos[0] - self.target_pos[0],
            self.start_pos[1] - self.target_pos[1]
        )
        self.collision_detected = False
        
        # Reset pedestrian position
        p.resetBasePositionAndOrientation(
            self.ped_body_id, self.start_pos, [0, 0, 0, 1]
        )
        
        if 'head' in self.base_env.pedestrian:
            head_pos = [
                self.start_pos[0], 
                self.start_pos[1], 
                self.start_pos[2] + 0.8
            ]
            p.resetBasePositionAndOrientation(
                self.base_env.pedestrian['head'], head_pos, [0, 0, 0, 1]
            )
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def render(self, mode="human"):
        """Render the environment."""
        if self.gui:
            time.sleep(1/60)

    def close(self):
        """Clean up resources."""
        try:
            p.disconnect()
        except Exception:
            pass
    
    def get_statistics(self):
        """Get episode statistics."""
        return self.episode_stats.copy()


# Keep backward compatibility
CrossroadGymEnv = EnhancedCrossroadGymEnv