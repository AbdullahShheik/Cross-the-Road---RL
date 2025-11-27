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

    def __init__(self, gui=False, max_steps=None, reward_shaping=True):
        super().__init__()
        self.gui = gui
        self._max_steps = max_steps or RL_CONFIG.get('max_episode_steps', 2000)
        self.reward_shaping = reward_shaping
        
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
        obs_len = RL_CONFIG.get('state_space_size', 43)
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
            'sequential_completions': 0
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
        if self.waypoints:
            self.current_target_pos = self.waypoints[0]
        else:
            self.current_target_pos = self.start_pos
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
        """Get enhanced observation with roundabout navigation information."""
        # Agent state
        pos, orn = p.getBasePositionAndOrientation(self.ped_body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.ped_body_id)
        agent_x, agent_y = pos[0], pos[1]
        agent_vx, agent_vy = lin_vel[0], lin_vel[1]

        # Current target position (waypoint)
        target_x, target_y = self.current_target_pos[0], self.current_target_pos[1]
        waypoint_idx = float(self.current_waypoint_index) / len(self.waypoints)

        # Traffic lights
        mapping = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0}
        tl_n = mapping.get(self.base_env.signal_states.get("north", "RED"), 0.0)
        tl_s = mapping.get(self.base_env.signal_states.get("south", "RED"), 0.0)
        tl_e = mapping.get(self.base_env.signal_states.get("east", "RED"), 0.0)
        tl_w = mapping.get(self.base_env.signal_states.get("west", "RED"), 0.0)

        # Get information about nearest 8 cars with velocity info
        cars = []
        for car in self.base_env.cars:
            # Skip removed cars
            if car['id'] in self.base_env.removed_cars:
                continue
                
            try:
                pos_c, _ = p.getBasePositionAndOrientation(car['body'])
                # Calculate relative position and velocity
                rel_dx = pos_c[0] - agent_x
                rel_dy = pos_c[1] - agent_y
                
                # Estimate velocity based on car direction and speed
                direction = car['direction']
                speed = car['speed']
                if direction == "north":
                    car_vx, car_vy = 0, speed
                elif direction == "south":
                    car_vx, car_vy = 0, -speed
                elif direction == "east":
                    car_vx, car_vy = speed, 0
                elif direction == "west":
                    car_vx, car_vy = -speed, 0
                else:
                    car_vx, car_vy = 0, 0
                
                dist = math.hypot(rel_dx, rel_dy)
                cars.append((dist, rel_dx, rel_dy, car_vx, car_vy))
            except Exception:
                # Car might have been removed, skip it
                continue
        
        cars.sort(key=lambda x: x[0])
        
        # Pad to 8 cars (4 features each: rel_x, rel_y, vx, vy)
        nearest = cars[:8]
        while len(nearest) < 8:
            nearest.append((999.0, 0.0, 0.0, 0.0, 0.0))

        obs = np.array([
            agent_x, agent_y, agent_vx, agent_vy,     # Agent state: 4
            target_x, target_y, waypoint_idx,         # Target/waypoint info: 3  
            tl_n, tl_s, tl_e, tl_w,                  # Traffic lights: 4
            # 8 cars × 4 features each = 32
            nearest[0][1], nearest[0][2], nearest[0][3], nearest[0][4],  # Car 1
            nearest[1][1], nearest[1][2], nearest[1][3], nearest[1][4],  # Car 2
            nearest[2][1], nearest[2][2], nearest[2][3], nearest[2][4],  # Car 3
            nearest[3][1], nearest[3][2], nearest[3][3], nearest[3][4],  # Car 4
            nearest[4][1], nearest[4][2], nearest[4][3], nearest[4][4],  # Car 5
            nearest[5][1], nearest[5][2], nearest[5][3], nearest[5][4],  # Car 6
            nearest[6][1], nearest[6][2], nearest[6][3], nearest[6][4],  # Car 7
            nearest[7][1], nearest[7][2], nearest[7][3], nearest[7][4],  # Car 8
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self, action, info):
        """Calculate reward with enhanced roundabout navigation."""
        reward = 0.0
        
        # Get current position
        agent_pos, _ = p.getBasePositionAndOrientation(self.ped_body_id)
        agent_x, agent_y = agent_pos[0], agent_pos[1]
        dist_to_current_target = math.hypot(
            agent_pos[0] - self.current_target_pos[0],
            agent_pos[1] - self.current_target_pos[1]
        )
        
        # Check for collision
        if self._check_collision():
            reward += RL_CONFIG['reward_structure'].get('collision_penalty', -200)
            info['collision'] = True
            self.episode_stats['collisions'] += 1
            return reward, True, info
        
        # Check if reached current waypoint/phase
        current_phase = self._get_current_phase()
        tolerance = PEDESTRIAN_CONFIG.get('waypoint_tolerance', 1.0)
        if current_phase and 'tolerance' in current_phase:
            tolerance = current_phase['tolerance']
        
        completion_zone = current_phase.get('completion_zone') if current_phase else None
        reached_waypoint = False
        if completion_zone and self._position_in_zone(completion_zone, agent_pos):
            reached_waypoint = True
        elif dist_to_current_target < tolerance:
            reached_waypoint = True
        
        if reached_waypoint:
            print("WAYPOINT REACHED:", current_phase['name'], "at", agent_pos)
            idx_before_advance = self.current_waypoint_index
            phase_reward = RL_CONFIG['reward_structure'].get('reach_waypoint', 100)
            if current_phase and 'reward' in current_phase:
                phase_reward = current_phase['reward']
            
            if idx_before_advance == self.waypoints_reached:
                phase_reward += RL_CONFIG['reward_structure'].get('sequential_bonus', 50)
                info['sequential_waypoint'] = True
                self.episode_stats['sequential_completions'] += 1
            
            reward += phase_reward
            self.waypoints_reached += 1
            self.total_waypoints_reached += 1
            self.episode_stats['waypoints_reached'] += 1
            info['waypoint_reached'] = idx_before_advance
            info['waypoints_completed'] = self.waypoints_reached
            
            if self.navigation_mode == 'sequential_cross' and self.waypoints:
                # Success only when ALL waypoints have been reached (explicit check)
                if self.waypoints_reached >= len(self.waypoints):
                    reward += RL_CONFIG['reward_structure'].get('reach_final_target', 500)
                    info['success'] = True
                    info['full_loop_completed'] = True
                    info['waypoints_completed'] = self.waypoints_reached
                    self.episode_stats['successes'] += 1
                    self.episode_stats['full_loops_completed'] += 1
                    return reward, True, info
                
                # Advance to next waypoint
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    self.current_target_pos = self.waypoints[self.current_waypoint_index]
                else:
                    # Should not reach here if success check above works, but just in case
                    reward += RL_CONFIG['reward_structure'].get('reach_final_target', 500)
                    info['success'] = True
                    info['full_loop_completed'] = True
                    info['waypoints_completed'] = self.waypoints_reached
                    self.episode_stats['successes'] += 1
                    self.episode_stats['full_loops_completed'] += 1
                    return reward, True, info
            else:
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                self.current_target_pos = self.waypoints[self.current_waypoint_index]
                
                if self.current_waypoint_index == 0 and self.waypoints_reached >= len(self.waypoints):
                    reward += RL_CONFIG['reward_structure'].get('reach_final_target', 500)
                    info['success'] = True
                    info['full_loop_completed'] = True
                    self.episode_stats['successes'] += 1
                    self.episode_stats['full_loops_completed'] += 1
                    return reward, True, info
            
            # Reset progress tracking for new waypoint
            new_target_dist = math.hypot(
                agent_pos[0] - self.current_target_pos[0],
                agent_pos[1] - self.current_target_pos[1]
            )
            self.last_progress_distance = new_target_dist
            dist_to_current_target = new_target_dist
            self.steps_since_progress = 0
            # Reset center crossing flag for next waypoint
            self.center_crossed_this_waypoint = False
        
        # Progress reward - strong reward for moving toward current waypoint
        if self.last_progress_distance is not None:
            progress = self.last_progress_distance - dist_to_current_target
            if progress > 0:
                # Moving toward waypoint
                reward += progress * RL_CONFIG['reward_structure'].get('progress_reward', 5)
                self.steps_since_progress = 0
            else:
                # Moving away from waypoint
                reward += progress * RL_CONFIG['reward_structure'].get('wrong_direction_penalty', -2)
                self.steps_since_progress += 1
        
        # Proximity bonus - extra reward for getting very close to waypoint
        if current_phase and dist_to_current_target > 0:
            initial_dist = self.last_progress_distance if self.last_progress_distance else dist_to_current_target
            if initial_dist > 0:
                progress_ratio = 1.0 - (dist_to_current_target / initial_dist)
                if progress_ratio > 0.5:  # More than 50% progress toward waypoint
                    proximity_bonus = RL_CONFIG['reward_structure'].get('proximity_bonus_scale', 50) * progress_ratio
                    reward += proximity_bonus
        
        # Midway reward - reward for crossing the center line (midway checkpoint)
        if not hasattr(self, 'center_crossed_this_waypoint'):
            self.center_crossed_this_waypoint = False
        
        center_x, center_y = 0.0, 0.0
        if current_phase:
            target = current_phase['target']
            start = self.start_pos if self.current_waypoint_index == 0 else self.waypoints[self.current_waypoint_index - 1]
            # Check if agent has crossed the center between start and target
            if self.current_waypoint_index == 0:
                start_x, start_y = self.start_pos[0], self.start_pos[1]
            else:
                prev_target = self.waypoints[self.current_waypoint_index - 1]
                start_x, start_y = prev_target[0], prev_target[1]
            target_x, target_y = target[0], target[1]
            
            # Check if we're crossing center (between start and target)
            mid_x = (start_x + target_x) / 2
            mid_y = (start_y + target_y) / 2
            
            # If agent is near center and hasn't been rewarded yet
            if abs(agent_x - center_x) < 1.0 and abs(agent_y - center_y) < 1.0:
                if not self.center_crossed_this_waypoint:
                    reward += RL_CONFIG['reward_structure'].get('midway_reward', 30)
                    self.center_crossed_this_waypoint = True
        
        # Penalty for staying in one place too long
        if self.steps_since_progress > 50:
            reward -= 1.0

        # STRICT Crosswalk discipline: heavy penalty for being off zebra when crossing
        # North/South zebra: centered at y = ±3.5, zebra spans roughly |x| <= 1.8
        zebra_half_width_x = 1.8
        zebra_band_half_y = 1.2  # from create_crosswalks halfExtents

        # North crosswalk band (y around +3.5)
        if (3.5 - zebra_band_half_y) <= agent_y <= (3.5 + zebra_band_half_y):
            if abs(agent_x) > zebra_half_width_x:
                # Scale penalty by distance from zebra center - much stronger!
                distance_off = abs(agent_x) - zebra_half_width_x
                reward -= 15.0 * (1.0 + distance_off)  # -15 to -30+ penalty

        # South crosswalk band (y around -3.5)
        if (-3.5 - zebra_band_half_y) <= agent_y <= (-3.5 + zebra_band_half_y):
            if abs(agent_x) > zebra_half_width_x:
                distance_off = abs(agent_x) - zebra_half_width_x
                reward -= 15.0 * (1.0 + distance_off)

        # East/West zebra: centered at x = ±3.5, zebra spans roughly |y| <= 1.8
        zebra_half_width_y = 1.8
        zebra_band_half_x = 1.2

        # East crosswalk band (x around +3.5)
        if (3.5 - zebra_band_half_x) <= agent_x <= (3.5 + zebra_band_half_x):
            if abs(agent_y) > zebra_half_width_y:
                distance_off = abs(agent_y) - zebra_half_width_y
                reward -= 15.0 * (1.0 + distance_off)

        # West crosswalk band (x around -3.5)
        if (-3.5 - zebra_band_half_x) <= agent_x <= (-3.5 + zebra_band_half_x):
            if abs(agent_y) > zebra_half_width_y:
                distance_off = abs(agent_y) - zebra_half_width_y
                reward -= 15.0 * (1.0 + distance_off)
        
        # Safety rewards and penalties
        min_car_dist = self._get_min_car_distance()
        if min_car_dist < 1.5:
            penalty = (1.5 - min_car_dist) * RL_CONFIG['reward_structure'].get('near_miss_penalty', -10)
            reward += penalty
        elif min_car_dist > 3.0:  # Bonus for maintaining safe distance
            reward += 0.5
        
        # Traffic light awareness bonus
        in_intersection = abs(agent_x) < 3.0 and abs(agent_y) < 3.0
        
        if in_intersection:
            # Check if pedestrian is crossing during safe signal
            crossing_ns = abs(agent_y) > abs(agent_x)  # More N-S movement
            
            if crossing_ns:
                # Check North-South signals
                ns_green = any(self.base_env.signal_states[d] == "GREEN" for d in ['north', 'south'])
                if ns_green:
                    reward += RL_CONFIG['reward_structure'].get('traffic_awareness_bonus', 10)
            else:
                # Check East-West signals  
                ew_green = any(self.base_env.signal_states[d] == "GREEN" for d in ['east', 'west'])
                if ew_green:
                    reward += RL_CONFIG['reward_structure'].get('traffic_awareness_bonus', 10)
        
        # Idle penalty only when loitering in the center of the intersection
        if action == 0 and self._position_in_zone(self.center_idle_zone, agent_pos):
            reward += self.center_idle_penalty
        
        # Small bonus for taking movement actions (encourages exploration)
        if action != 0:  # Any movement action
            reward += RL_CONFIG['reward_structure'].get('movement_bonus', 0.5)
        
        # Survival bonus - small reward for each step survived (encourages avoiding collisions)
        reward += RL_CONFIG['reward_structure'].get('survival_bonus', 0.1)
        
        # Very small time penalty
        reward += RL_CONFIG['reward_structure'].get('time_penalty', -0.005)
        
        self.last_distance_to_target = dist_to_current_target
        self.last_progress_distance = dist_to_current_target
        
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
        
        # STRICT: Force agent to stay on zebra crossings - no wandering in center!
        zebra_half_width = 1.8
        zebra_band_half = 1.2
        intersection_center_size = 3.5  # Size of center intersection
        
        # Check if agent is in the center intersection area
        in_center = abs(new_pos[0]) < intersection_center_size and abs(new_pos[1]) < intersection_center_size
        
        if in_center:
            # In center intersection - must stay on one of the zebra crossing paths
            # Check which crossing path the agent is closest to
            dist_to_north = abs(new_pos[1] - 3.5)
            dist_to_south = abs(new_pos[1] + 3.5)
            dist_to_east = abs(new_pos[0] - 3.5)
            dist_to_west = abs(new_pos[0] + 3.5)
            
            min_dist = min(dist_to_north, dist_to_south, dist_to_east, dist_to_west)
            
            # Force agent onto the nearest zebra crossing path
            if min_dist == dist_to_north or min_dist == dist_to_south:
                # On North/South path - constrain x to zebra width
                new_pos[0] = np.clip(new_pos[0], -zebra_half_width, zebra_half_width)
            elif min_dist == dist_to_east or min_dist == dist_to_west:
                # On East/West path - constrain y to zebra width
                new_pos[1] = np.clip(new_pos[1], -zebra_half_width, zebra_half_width)
        
        # North/South zebra crossings - constrain x when in crossing band
        if (3.5 - zebra_band_half) <= new_pos[1] <= (3.5 + zebra_band_half):
            # In north crosswalk - must stay within zebra width
            new_pos[0] = np.clip(new_pos[0], -zebra_half_width, zebra_half_width)
        elif (-3.5 - zebra_band_half) <= new_pos[1] <= (-3.5 + zebra_band_half):
            # In south crosswalk - must stay within zebra width
            new_pos[0] = np.clip(new_pos[0], -zebra_half_width, zebra_half_width)
        
        # East/West zebra crossings - constrain y when in crossing band
        if (3.5 - zebra_band_half) <= new_pos[0] <= (3.5 + zebra_band_half):
            # In east crosswalk - must stay within zebra width
            new_pos[1] = np.clip(new_pos[1], -zebra_half_width, zebra_half_width)
        elif (-3.5 - zebra_band_half) <= new_pos[0] <= (-3.5 + zebra_band_half):
            # In west crosswalk - must stay within zebra width
            new_pos[1] = np.clip(new_pos[1], -zebra_half_width, zebra_half_width)
        
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
            'waypoints_completed': self.waypoints_reached,
            'current_waypoint': self.current_waypoint_index,
            'total_waypoints': len(self.waypoints)
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
            'target_position': self.current_target_pos
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
            'completion_percentage': (self.waypoints_reached / len(self.waypoints)) * 100
        })
        return stats

