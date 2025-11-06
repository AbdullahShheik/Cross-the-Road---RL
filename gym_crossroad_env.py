# gym_crossroad_env.py
import gymnasium as gym
from gym import spaces
import numpy as np
import pybullet as p
import time
import random
from intersection_env import CrossroadEnvironment as BaseCrossroad  # your file
from config import ENVIRONMENT_CONFIG, CAR_CONFIG, PEDESTRIAN_CONFIG, ENVIRONMENT_OBJECTS, TRAFFIC_LIGHT_CONFIG, RL_CONFIG
import math

class CrossroadGymEnv(gym.Env):
    """
    Gym wrapper around your CrossroadEnvironment (intersection_env.py).
    Discrete actions: 0=stay, 1=forward, 2=back, 3=left, 4=right
    Observation: compact numeric vector (see below)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False, max_steps=None):
        super().__init__()
        self.gui = gui
        self._max_steps = max_steps or RL_CONFIG.get('max_episode_steps', 1000)

        # Create base environment in DIRECT for training; if gui=True use GUI
        self._p_connection_mode = p.GUI if gui else p.DIRECT

        # We'll instantiate BaseCrossroad ourselves but modify it to use DIRECT/GUI
        # Create a thin new instance of your environment which uses pybullet.
        # Here I call BaseCrossroad but we will modify how it connects inside BaseCrossroad's __init__:
        # To keep it simple, we re-create parts of the scene in a lightweight way:
        self._init_pybullet()

        # Action space: discrete 5
        self.action_space = spaces.Discrete(RL_CONFIG.get('action_space_size', 5))

        # Observation: fixed-length vector
        # Format: [agent_x, agent_y, agent_vx, agent_vy, target_x, target_y,
        #          tl_north_state, tl_south_state, tl_east_state, tl_west_state,
        #          nearest1_dx, nearest1_dy, nearest1_speed, nearest2_dx, nearest2_dy, nearest2_speed]
        obs_len = 4 + 2 + 4 + 3 * 2  # agent(4) + target(2) + 4 tl + two nearest cars (3 each)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(obs_len,), dtype=np.float32)

        self.step_count = 0
        self.seed()
        self._last_collision = False

    def _init_pybullet(self):
        # connect pybullet
        if hasattr(p, 'disconnect_all'):
            try:
                p.disconnect()
            except Exception:
                pass
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath("pybullet_data")
        p.setGravity(0, 0, -9.8)
        # Use parts of your existing env code to set up scene.
        # For simplicity, instantiate your existing class but override its pybullet connection.
        # If BaseCrossroad always connects in GUI, modify it to accept a mode parameter (recommended).
        # Here we'll import and create it (assumes BaseCrossroad uses p.connect at top).
        # If needed, adapt BaseCrossroad to accept 'connection_mode' parameter.
        self.base_env = BaseCrossroad(connection_mode=self._p_connection_mode)
        # Now make pedestrian controllable: store body id
        self.ped_body_id = self.base_env.pedestrian['torso']
        self.target_pos = PEDESTRIAN_CONFIG.get('target_position', [2.5, 3.5, 0.6])

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _get_obs(self):
        # Agent state
        pos, orn = p.getBasePositionAndOrientation(self.ped_body_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.ped_body_id)
        agent_x, agent_y = pos[0], pos[1]
        agent_vx, agent_vy = lin_vel[0], lin_vel[1]

        # Target
        target_x, target_y = self.base_env.pedestrian['position'][0], self.base_env.pedestrian['position'][1]

        # Traffic lights (map GREEN/YELLOW/RED -> numeric)
        mapping = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0}
        tl_n = mapping.get(self.base_env.signal_states.get("north", "RED"), 0.0)
        tl_s = mapping.get(self.base_env.signal_states.get("south", "RED"), 0.0)
        tl_e = mapping.get(self.base_env.signal_states.get("east", "RED"), 0.0)
        tl_w = mapping.get(self.base_env.signal_states.get("west", "RED"), 0.0)

        # Nearest two cars (relative pos and speed)
        cars = []
        for car in self.base_env.cars:
            pos_c, _ = p.getBasePositionAndOrientation(car['body'])
            vel_c, _ = p.getBaseVelocity(car['body'])
            rel_dx = pos_c[0] - agent_x
            rel_dy = pos_c[1] - agent_y
            speed = math.hypot(vel_c[0], vel_c[1])
            dist = math.hypot(rel_dx, rel_dy)
            cars.append((dist, rel_dx, rel_dy, speed))
        cars.sort(key=lambda x: x[0])
        # pad if less than 2 cars
        nearest = cars[:2]
        while len(nearest) < 2:
            nearest.append((999.0, 0.0, 0.0, 0.0))

        obs = np.array([
            agent_x, agent_y, agent_vx, agent_vy,
            target_x, target_y,
            tl_n, tl_s, tl_e, tl_w,
            nearest[0][1], nearest[0][2], nearest[0][3],
            nearest[1][1], nearest[1][2], nearest[1][3]
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """
        Apply simple discrete actions by moving the pedestrian a small step in world frame.
        Note: For smoother physics you can apply forces or set velocities instead.
        """
        # Map action to delta position
        step_size = 0.15  # meters per step (tune)
        dx = 0.0
        dy = 0.0
        if action == 1:  # forward: +y
            dy = step_size
        elif action == 2:  # back
            dy = -step_size
        elif action == 3:  # left
            dx = -step_size
        elif action == 4:  # right
            dx = step_size
        # action==0 -> stay

        # Move the pedestrian
        pos, orn = p.getBasePositionAndOrientation(self.ped_body_id)
        new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
        p.resetBasePositionAndOrientation(self.ped_body_id, new_pos, orn)

        # Step the base env: update cars/traffic lights etc. (reuse your update functions)
        self.base_env.update_traffic_lights()
        self.base_env.update_cars()
        p.stepSimulation()

        # Reward and termination checks
        reward = 0.0
        done = False
        info = {}

        # Collision detection: if pedestrian has contact with any car
        contact_points = []
        for car in self.base_env.cars:
            cps = p.getContactPoints(bodyA=self.ped_body_id, bodyB=car['body'])
            if len(cps) > 0:
                contact_points.extend(cps)
        if len(contact_points) > 0:
            reward += RL_CONFIG['reward_structure'].get('collision_penalty', -100)
            done = True
            self._last_collision = True
        else:
            # time penalty
            reward += RL_CONFIG['reward_structure'].get('time_penalty', -0.1)
            self._last_collision = False

            # check reach target (distance threshold)
            agent_pos, _ = p.getBasePositionAndOrientation(self.ped_body_id)
            dist_to_target = math.hypot(agent_pos[0] - self.base_env.pedestrian['position'][0],
                                        agent_pos[1] - self.base_env.pedestrian['position'][1])
            if dist_to_target < 0.5:
                reward += RL_CONFIG['reward_structure'].get('reach_target', 100)
                done = True
                info['success'] = True

        self.step_count += 1
        if self.step_count >= self._max_steps:
            done = True
            info['timeout'] = True

        obs = self._get_obs()
        return obs, float(reward), done, info

    def reset(self):
        # Reset pybullet world and re-create the base environment
        try:
            p.resetSimulation()
        except Exception:
            pass
        self._init_pybullet()
        self.step_count = 0
        self._last_collision = False
        # place pedestrian at start (use config)
        start = self.base_env.pedestrian['position']
        p.resetBasePositionAndOrientation(self.ped_body_id, start, [0, 0, 0, 1])
        # return initial observation
        return self._get_obs()

    def render(self, mode="human"):
        # If GUI connected, pybullet GUI already shows scene. For headless, could return an image (not implemented).
        if self.gui:
            time.sleep(1/60)
        else:
            pass

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass
