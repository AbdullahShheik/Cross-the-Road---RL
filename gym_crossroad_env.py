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

class CrossroadGymEnv(gym.Env):
    """
    Enhanced Gym wrapper with roundabout navigation and better reward shaping.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, gui=False, max_steps=None, reward_shaping=True, target_rounds=1):
        super().__init__()
        self.gui = gui
        self._max_steps = max_steps or RL_CONFIG.get('max_episode_steps', 2000)
        self.reward_shaping = reward_shaping
        self.target_rounds = target_rounds
        
        self._p_connection_mode = p.GUI if gui else p.DIRECT
        self.navigation_mode = PEDESTRIAN_CONFIG.get('navigation_mode', 'roundabout')
        self.sequential_phases = PEDESTRIAN_CONFIG.get('sequential_cross_phases', [])
        self.center_idle_zone = PEDESTRIAN_CONFIG.get(
            'center_idle_zone',
            {'x': [-1.0, 1.0], 'y': [-1.0, 1.0]}
        )
        self.center_idle_penalty = PEDESTRIAN_CONFIG.get('center_idle_penalty', -0.5)
        self._configure_navigation_plan()
        self._init_pybullet()

        # Action space: 0=stay, 1=forward, 2=back, 3=left, 4=right
        self.action_space = spaces.Discrete(RL_CONFIG.get('action_space_size', 5))

        # Enhanced observation space for roundabout navigation
        # State: [ped_x, ped_y, target_x, target_y, rel1_x, rel1_y, rel2_x, rel2_y, 
        #         ns_green, ew_green, min_car_distance, is_in_danger]
        obs_len = 12  # Increased from 10 to include collision awareness
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(obs_len,), dtype=np.float32
        )

        self.step_count = 0
        self.last_distance_to_target = None
        self.collision_detected = False
        
        # Roundabout/Sequential navigation state
        self.current_waypoint_index = 0
        self.waypoints_reached = 0
        self.total_waypoints_reached = 0
        self.rounds_completed = 0
        
        # Progress tracking
        self.last_progress_distance = None
        self.steps_since_progress = 0
        
        # Statistics
        self.episode_stats = {
            'collisions': 0,
            'successes': 0,
            'timeouts': 0,
            'total_reward': 0,
            'waypoints_reached': 0,
            'full_loops_completed': 0,
            'sequential_completions': 0,
            'target_rounds': self.target_rounds,
            'rounds_completed': 0
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
        self.start_pos = PEDESTRIAN_CONFIG.get('start_position', [-2.5, 3.5, 0.6])
        self._reset_navigation_state()

    def _configure_navigation_plan(self):
        """Prepare the waypoint plan based on configured navigation mode."""
        if self.navigation_mode == 'sequential_cross' and self.sequential_phases:
            self.navigation_plan = self.sequential_phases
            self.waypoints = [phase['target'] for phase in self.navigation_plan]
        else:
            self.navigation_plan = []
            self.waypoints = PEDESTRIAN_CONFIG.get(
                'roundabout_waypoints',
                [PEDESTRIAN_CONFIG.get('start_position', [-2.5, 3.5, 0.6])]
            )

    def _reset_navigation_state(self):
        """Reset indices and targets for navigation."""
        if not getattr(self, 'waypoints', None):
            self._configure_navigation_plan()
        self.current_waypoint_index = 0
        self.waypoints_reached = 0
        self.rounds_completed = 0
        if self.waypoints:
            self.current_target_pos = self.waypoints[0]
        else:
            self.current_target_pos = PEDESTRIAN_CONFIG.get('start_position', [-2.5, 3.5, 0.6])
        self.last_progress_distance = math.hypot(
            self.start_pos[0] - self.current_target_pos[0],
            self.start_pos[1] - self.current_target_pos[1]
        )
        self.steps_since_progress = 0
        self.center_crossed_this_waypoint = False

    def _get_current_phase(self):
        """Return metadata for the current sequential phase, if available."""
        if (
            self.navigation_mode == 'sequential_cross'
            and self.navigation_plan
            and 0 <= self.current_waypoint_index < len(self.navigation_plan)
        ):
            return self.navigation_plan[self.current_waypoint_index]
        return None

    def _position_in_zone(self, zone, position):
        """Check if a position lies within a rectangular zone."""
        if not zone:
            return False
        x_min, x_max = zone.get('x', (-float('inf'), float('inf')))
        y_min, y_max = zone.get('y', (-float('inf'), float('inf')))
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.np_random = np.random
        return [seed]

    def _get_obs(self):
        # ---- 1. Pedestrian position ----
        ped_pos, _ = p.getBasePositionAndOrientation(self.ped_body_id)
        ped_x, ped_y = ped_pos[0], ped_pos[1]

        # ---- 2. Current target (next waypoint) ----
        target_x, target_y = self.current_target_pos[0], self.current_target_pos[1]

        # ---- 3. Nearest two cars (relative positions) ----
        car_positions = []
        min_car_distance = 100.0  # Large default value

        for car in self.base_env.cars:
            car_pos, _ = p.getBasePositionAndOrientation(car['body'])
            rel_x = car_pos[0] - ped_x
            rel_y = car_pos[1] - ped_y
            dist = math.sqrt(rel_x**2 + rel_y**2)
            car_positions.append((rel_x, rel_y, dist))
            min_car_distance = min(min_car_distance, dist)

        # sort by distance
        car_positions.sort(key=lambda c: c[2])

        # take 2 nearest cars
        if len(car_positions) >= 2:
            (rel1_x, rel1_y, _), (rel2_x, rel2_y, _) = car_positions[:2]
        elif len(car_positions) == 1:
            rel1_x, rel1_y, _ = car_positions[0]
            rel2_x, rel2_y = (10.0, 10.0) # placeholder far away
        else:
            rel1_x, rel1_y = (10.0, 10.0)
            rel2_x, rel2_y = (10.0, 10.0)

        # ---- 4. Traffic light state encoding ----
        # Encode as 1 (green) / 0 (not green)
        ns_green = 1 if (
            self.base_env.signal_states['north'] == "GREEN" or
            self.base_env.signal_states['south'] == "GREEN"
        ) else 0

        ew_green = 1 if (
            self.base_env.signal_states['east'] == "GREEN" or
            self.base_env.signal_states['west'] == "GREEN"
        ) else 0

        # ---- 5. Collision awareness ----
        # Danger flag: 1 if car is within 2.0 units (very close), 0 otherwise
        is_in_danger = 1.0 if min_car_distance < 2.0 else 0.0

        # ---- FINAL STATE VECTOR (12 dims) ----
        state = [
            ped_x, ped_y,
            target_x, target_y,
            rel1_x, rel1_y,
            rel2_x, rel2_y,
            ns_green, ew_green,
            min_car_distance,  # Normalized distance to nearest car
            is_in_danger       # Binary danger indicator
        ]

        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, action, info):
        """
        Balanced reward function: encourages movement AND safety.
        Avoids over-penalizing collisions which causes learned passivity.
        """
        reward = 0.0
        done = False

        # Get current position and car distance
        agent_pos, _ = p.getBasePositionAndOrientation(self.ped_body_id)
        agent_x, agent_y = agent_pos[0], agent_pos[1]
        min_car_distance = self._get_min_car_distance()

        # ---- 1. COLLISION CHECK ----
        if self._check_collision():
            reward = -100  # Moderate penalty (was -300, which caused passivity)
            info["collision"] = True
            return reward, True, info

        # ---- 2. PROGRESS TOWARD WAYPOINT (PRIMARY GOAL) ----
        dist_to_target = math.hypot(
            agent_x - self.current_target_pos[0],
            agent_y - self.current_target_pos[1]
        )

        # Measure progress compared to previous step
        if self.last_distance_to_target is not None:
            prev_dist = self.last_distance_to_target
            progress = prev_dist - dist_to_target
            
            # Strong positive reward for moving toward target
            # This is the PRIMARY learning signal
            reward += progress * 8.0  # Increased multiplier (was 5.0)
            
            # Penalty for moving away from target (slight)
            if progress < 0 and min_car_distance > 2.5:
                # Only penalize bad direction if NOT near a car
                reward -= 0.01
        else:
            self.last_distance_to_target = dist_to_target

        # Update stored distance
        self.last_distance_to_target = dist_to_target

        # ---- 3. SAFETY BONUS (SECONDARY) ----
        # Small bonus for being cautious, but don't over-reward passivity
        if min_car_distance > 4.0:
            reward += 0.1  # Reduced from 0.5
        elif min_car_distance > 2.5:
            reward += 0.05  # Reduced from 0.2

        # ---- 4. WAYPOINT REACHED ----
        tolerance = PEDESTRIAN_CONFIG.get('waypoint_tolerance', 1.5)
        reached = dist_to_target < tolerance

        if reached:
            # Significant reward for reaching waypoint
            reward += 80  # Reduced from 100 (relative to progress signal)
            info["waypoint_reached"] = self.current_waypoint_index

            self.current_waypoint_index += 1
            self.waypoints_reached += 1

            # If all waypoints completed → Check if more rounds needed
            if self.current_waypoint_index >= len(self.waypoints):
                self.rounds_completed += 1
                self.episode_stats['rounds_completed'] = self.rounds_completed
                
                # Calculate total waypoints completed across all rounds
                total_waypoints_completed = self.rounds_completed * len(self.waypoints)
                
                # Check if target rounds completed
                if self.rounds_completed >= self.target_rounds:
                    reward += 400  # Large bonus for completing all target rounds
                    info["success"] = True
                    info["waypoints_completed"] = total_waypoints_completed
                    info["rounds_completed"] = self.rounds_completed
                    info["target_rounds_reached"] = True
                    print(f"🎉 Agent completed {self.rounds_completed}/{self.target_rounds} rounds successfully!")
                    done = True
                else:
                    # More rounds needed - reset waypoints but continue episode
                    reward += 200  # Moderate bonus for completing one round
                    info["round_completed"] = self.rounds_completed
                    info["waypoints_completed"] = total_waypoints_completed
                    info["rounds_completed"] = self.rounds_completed
                    print(f"🔄 Round {self.rounds_completed} completed! Starting round {self.rounds_completed + 1}/{self.target_rounds}")
                    
                    # Reset waypoint tracking for next round
                    self.current_waypoint_index = 0
                    self.current_target_pos = self.waypoints[0]
                    
                    # Reset progress measurement for new round
                    self.last_distance_to_target = math.hypot(
                        agent_x - self.current_target_pos[0],
                        agent_y - self.current_target_pos[1]
                    )
            else:
                # Move to next waypoint
                self.current_target_pos = self.waypoints[self.current_waypoint_index]

            # Reset progress measurement
            self.last_distance_to_target = math.hypot(
                agent_x - self.current_target_pos[0],
                agent_y - self.current_target_pos[1]
            )

        # ---- 5. MINIMAL TIME PENALTY ----
        # Very small penalty for wasting time (encourages efficiency)
        reward -= 0.001  # Reduced from 0.005

        return reward, done, info

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
        step_size = 0.30  # Increased from 0.15 to allow faster navigation
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
        
        # STRICT: Force agent to stay on zebra crossings - no wandering in center!
        zebra_half_width = 1.8
        zebra_band_half = 1.2
        intersection_center_size = 3.5  # Size of center intersection
        
        # Check if agent is in the center intersection area
        in_center = abs(new_pos[0]) < intersection_center_size and abs(new_pos[1]) < intersection_center_size
        
        # if in_center:
            # In center intersection - must stay on one of the zebra crossing paths
            # Check which crossing path the agent is closest to
            # dist_to_north = abs(new_pos[1] - 3.5)
            # dist_to_south = abs(new_pos[1] + 3.5)
            # dist_to_east = abs(new_pos[0] - 3.5)
            # dist_to_west = abs(new_pos[0] + 3.5)
            
            # min_dist = min(dist_to_north, dist_to_south, dist_to_east, dist_to_west)
            
            # # Force agent onto the nearest zebra crossing path
            # if min_dist == dist_to_north or min_dist == dist_to_south:
            #     # On North/South path - constrain x to zebra width
            #     new_pos[0] = np.clip(new_pos[0], -zebra_half_width, zebra_half_width)
            # elif min_dist == dist_to_east or min_dist == dist_to_west:
            #     # On East/West path - constrain y to zebra width
            #     new_pos[1] = np.clip(new_pos[1], -zebra_half_width, zebra_half_width)
        
        # # North/South zebra crossings - constrain x when in crossing band
        # if (3.5 - zebra_band_half) <= new_pos[1] <= (3.5 + zebra_band_half):
        #     # In north crosswalk - must stay within zebra width
        #     new_pos[0] = np.clip(new_pos[0], -zebra_half_width, zebra_half_width)
        # elif (-3.5 - zebra_band_half) <= new_pos[1] <= (-3.5 + zebra_band_half):
        #     # In south crosswalk - must stay within zebra width
        #     new_pos[0] = np.clip(new_pos[0], -zebra_half_width, zebra_half_width)
        
        # # East/West zebra crossings - constrain y when in crossing band
        # if (3.5 - zebra_band_half) <= new_pos[0] <= (3.5 + zebra_band_half):
        #     # In east crosswalk - must stay within zebra width
        #     new_pos[1] = np.clip(new_pos[1], -zebra_half_width, zebra_half_width)
        # elif (-3.5 - zebra_band_half) <= new_pos[0] <= (-3.5 + zebra_band_half):
        #     # In west crosswalk - must stay within zebra width
        #     new_pos[1] = np.clip(new_pos[1], -zebra_half_width, zebra_half_width)
        
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
        info = {
            'waypoints_completed': self.rounds_completed * len(self.waypoints) + self.current_waypoint_index,
            'current_waypoint': self.current_waypoint_index,
            'total_waypoints': len(self.waypoints),
            'rounds_completed': self.rounds_completed,
            'target_rounds': self.target_rounds
        }
        
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
        self._reset_navigation_state()
        
        self.last_distance_to_target = math.hypot(
            self.start_pos[0] - self.current_target_pos[0],
            self.start_pos[1] - self.current_target_pos[1]
        )
        self.collision_detected = False
        self.center_crossed_this_waypoint = False
        
        # Reset pedestrian position to start
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
        info = {
            'current_waypoint': self.current_waypoint_index, 
            'waypoints_reached': self.waypoints_reached,
            'target_position': self.current_target_pos,
            'rounds_completed': self.rounds_completed,
            'target_rounds': self.target_rounds
        }
        
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
        stats = self.episode_stats.copy()
        stats.update({
            'current_waypoint': self.current_waypoint_index,
            'waypoints_reached_current_episode': self.waypoints_reached,
            'completion_percentage': (self.waypoints_reached / len(self.waypoints)) * 100,
            'rounds_completed': self.rounds_completed,
            'target_rounds': self.target_rounds,
            'rounds_completion_percentage': (self.rounds_completed / self.target_rounds) * 100
        })
        return stats

