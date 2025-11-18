import pybullet as p
import pybullet_data
import time
import math
import random
import numpy as np
from config import ENVIRONMENT_CONFIG, CAR_CONFIG, PEDESTRIAN_CONFIG, ENVIRONMENT_OBJECTS, TRAFFIC_LIGHT_CONFIG

class CrossroadEnvironment:
    def __init__(self, connection_mode=p.GUI):
        # Connect to PyBullet GUI
        self.physicsClient = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Enhanced lighting and camera
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        
        p.resetDebugVisualizerCamera(
            cameraDistance=18,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Initialize scene components
        self.cars = []
        self.traffic_lights = []
        self.pedestrian = None
        
        # Traffic light states
        self.signal_states = {"north": "RED", "south": "RED", "east": "GREEN", "west": "GREEN"}
        self.signal_change_timer = 0
        self.signal_change_interval = 240  # Increased for better observation (4 seconds)
        self.yellow_light_duration = 40  # Yellow light duration
        self.in_yellow_phase = False
        self.yellow_timer = 0
        
        self.setup_environment()
        self.create_traffic_infrastructure()
        self.spawn_cars()
        self.create_pedestrian()
    
    def _update_light_visuals(self):
        """Update traffic light visuals based on current state"""
        colors = TRAFFIC_LIGHT_CONFIG['light_colors']
        
        for light_set in self.traffic_lights:
            direction = light_set['direction']
            state = self.signal_states[direction]
            red_light, yellow_light, green_light = light_set['lights']
            
            # Update light colors based on state
            if state == "RED":
                p.changeVisualShape(red_light, -1, rgbaColor=colors['red_on'])
                p.changeVisualShape(yellow_light, -1, rgbaColor=colors['yellow_off'])
                p.changeVisualShape(green_light, -1, rgbaColor=colors['green_off'])
            elif state == "YELLOW":
                p.changeVisualShape(red_light, -1, rgbaColor=colors['red_off'])
                p.changeVisualShape(yellow_light, -1, rgbaColor=colors['yellow_on'])
                p.changeVisualShape(green_light, -1, rgbaColor=colors['green_off'])
            else:  # GREEN
                p.changeVisualShape(red_light, -1, rgbaColor=colors['red_off'])
                p.changeVisualShape(yellow_light, -1, rgbaColor=colors['yellow_off'])
                p.changeVisualShape(green_light, -1, rgbaColor=colors['green_on'])

    def setup_environment(self):
        """Create the base environment with roads and grass"""
        # Create large grass base
        grass_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[25, 25, 0.1],
            rgbaColor=[0.2, 0.6, 0.2, 1]
        )
        grass_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[25, 25, 0.1])
        self.grass = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=grass_collision,
            baseVisualShapeIndex=grass_visual,
            basePosition=[0, 0, -0.1]
        )
        
        # Create main intersection base
        intersection_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[6, 6, 0.05],
            rgbaColor=[0.15, 0.15, 0.15, 1]
        )
        intersection_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[6, 6, 0.05])
        self.intersection = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=intersection_collision,
            baseVisualShapeIndex=intersection_visual,
            basePosition=[0, 0, 0.05]
        )
        
        # Create four main roads
        road_positions = [
            [0, 12, 0.05],   # North road
            [0, -12, 0.05],  # South road
            [12, 0, 0.05],   # East road
            [-12, 0, 0.05]   # West road
        ]
        
        road_dimensions = [
            [2.5, 6, 0.05],  # North/South roads
            [2.5, 6, 0.05],
            [6, 2.5, 0.05],  # East/West roads
            [6, 2.5, 0.05]
        ]
        
        for i, (pos, dims) in enumerate(zip(road_positions, road_dimensions)):
            road_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=dims,
                rgbaColor=[0.1, 0.1, 0.1, 1]
            )
            road_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=road_collision,
                baseVisualShapeIndex=road_visual,
                basePosition=pos
            )
        
        self.create_road_markings()
        self.create_crosswalks()

    def create_road_markings(self):
        """Create yellow center lines and lane dividers"""
        # Center line for north-south road
        for y in range(-18, 19, 2):
            if abs(y) > 6:
                line_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.05, 0.5, 0.01],
                    rgbaColor=[1, 1, 0, 1]
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=line_visual,
                    basePosition=[0, y, 0.11]
                )
        
        # Center line for east-west road
        for x in range(-18, 19, 2):
            if abs(x) > 6:
                line_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.5, 0.05, 0.01],
                    rgbaColor=[1, 1, 0, 1]
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=line_visual,
                    basePosition=[x, 0, 0.11]
                )

    def create_crosswalks(self):
        """Create zebra crossings at intersection"""
        crosswalk_positions = [
            [0, 3.5, "horizontal"],   # North crosswalk
            [0, -3.5, "horizontal"],  # South crosswalk
            [3.5, 0, "vertical"],     # East crosswalk
            [-3.5, 0, "vertical"]     # West crosswalk
        ]
        
        for x_center, y_center, orientation in crosswalk_positions:
            for i in range(-4, 5):
                if orientation == "horizontal":
                    pos = [x_center + i * 0.4, y_center, 0.12]
                    half_extents = [0.15, 1.2, 0.01]
                else:
                    pos = [x_center, y_center + i * 0.4, 0.12]
                    half_extents = [1.2, 0.15, 0.01]
                
                stripe_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=[1, 1, 1, 1]
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=stripe_visual,
                    basePosition=pos
                )

    def create_traffic_infrastructure(self):
        """Create traffic lights and poles with better visibility"""
        traffic_light_positions = [
            [-4, 4, "north"],
            [4, -4, "south"],
            [4, 4, "east"],
            [-4, -4, "west"]
        ]
        
        for x, y, direction in traffic_light_positions:
            # Create pole
            pole_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.1,
                length=4,
                rgbaColor=[0.3, 0.3, 0.3, 1]
            )
            pole_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.1,
                height=4
            )
            pole = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=pole_collision,
                baseVisualShapeIndex=pole_visual,
                basePosition=[x, y, 2]
            )
            
            # Rotate light box
            yaw_map = {"north": 180, "south": 0, "east": 90, "west": -90}
            yaw = math.radians(yaw_map[direction])
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            
            # Create back panel of traffic light box (behind the lights)
            back_panel_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.25, 0.05, 0.7],
                rgbaColor=[0.2, 0.2, 0.2, 1]
            )
            back_panel = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=back_panel_visual,
                basePosition=[x, y, 4.2],
                baseOrientation=quat
            )
            
            # Create side panels for structural support
            side_offset = 0.15
            for side in [-1, 1]:
                side_panel_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.05, 0.15, 0.7],
                    rgbaColor=[0.2, 0.2, 0.2, 1]
                )
                # Calculate side panel position relative to direction
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                side_x = x + side * side_offset * sin_yaw
                side_y = y - side * side_offset * cos_yaw
                
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=side_panel_visual,
                    basePosition=[side_x, side_y, 4.2],
                    baseOrientation=quat
                )
            
            # Create top and bottom panels
            top_bottom_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.25, 0.15, 0.05],
                rgbaColor=[0.2, 0.2, 0.2, 1]
            )
            for z_offset in [0.65, -0.65]:
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=top_bottom_visual,
                    basePosition=[x, y, 4.2 + z_offset],
                    baseOrientation=quat
                )
            
            # Create individual lights (red, yellow, green) - positioned in front of back panel
            lights = []
            colors = TRAFFIC_LIGHT_CONFIG['light_colors']
            initial_colors = [colors['red_off'], colors['yellow_off'], colors['green_off']]
            
            # Calculate offset to place lights in front of the box
            light_offset = 0.2  # Distance in front of back panel
            
            for i, color in enumerate(initial_colors):
                light_visual = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.18,  # Larger for better visibility
                    rgbaColor=color
                )
                
                # Calculate light position (in front of the box, facing traffic)
                light_z = 4.6 - i * 0.4
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                light_x = x + light_offset * cos_yaw
                light_y = y + light_offset * sin_yaw
                
                light = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=light_visual,
                    basePosition=[light_x, light_y, light_z]
                )
                lights.append(light)
            
            self.traffic_lights.append({
                'direction': direction,
                'position': [x, y],
                'lights': lights,
                'pole': pole,
                'box': back_panel  # Store back panel reference
            })
        
        # Initialize light visuals
        self._update_light_visuals()

    def spawn_cars(self):
        """Spawn cars on different lanes with better spacing"""
        car_spawn_data = [
            # [x, y, z, orientation, lane_direction, color, rule_breaker]
            [-1.2, -15, 0.3, 1.57, "north", [1, 0, 0, 1], random.random() < 0.25],      # Red car
            [-1.2, -10, 0.3, 1.57, "north", [0.8, 0.4, 0, 1], random.random() < 0.25],  # Orange car
            
            [1.2, 15, 0.3, -1.57, "south", [0, 0, 1, 1], random.random() < 0.25],       # Blue car
            [1.2, 10, 0.3, -1.57, "south", [0, 0.6, 0.8, 1], random.random() < 0.25],   # Teal car
            
            [15, 1.2, 0.3, 3.14, "west", [0, 1, 0, 1], random.random() < 0.25],         # Green car
            [10, 1.2, 0.3, 3.14, "west", [0.5, 0.8, 0.3, 1], random.random() < 0.25],   # Lime car
            
            [-15, -1.2, 0.3, 0, "east", [1, 1, 0, 1], random.random() < 0.25],          # Yellow car
            [-10, -1.2, 0.3, 0, "east", [1, 0, 1, 1], random.random() < 0.25],          # Purple car
        ]
        
        for spawn_data in car_spawn_data:
            x, y, z, orientation, direction, color, rule_breaker = spawn_data
            car = self.create_car([x, y, z], orientation, color, direction, rule_breaker)
            self.cars.append(car)

    def create_car(self, position, orientation, color, direction, rule_breaker=False):
        """Create a single car with realistic appearance"""
        # Create car body
        car_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.8, 0.4, 0.2],
            rgbaColor=color
        )
        car_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.8, 0.4, 0.2]
        )
        
        quat = p.getQuaternionFromEuler([0, 0, orientation])
        car_body = p.createMultiBody(
            baseMass=1000,
            baseCollisionShapeIndex=car_collision,
            baseVisualShapeIndex=car_visual,
            basePosition=position,
            baseOrientation=quat
        )
        
        # Add wheels
        wheel_positions = [
            [0.6, 0.3, -0.1],
            [0.6, -0.3, -0.1],
            [-0.6, 0.3, -0.1],
            [-0.6, -0.3, -0.1]
        ]
        
        wheels = []
        for wheel_pos in wheel_positions:
            wheel_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.2,
                length=0.1,
                rgbaColor=[0.1, 0.1, 0.1, 1]
            )
            world_wheel_pos = np.array(position) + np.array(wheel_pos)
            wheel = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=wheel_visual,
                basePosition=world_wheel_pos.tolist(),
                baseOrientation=quat
            )
            wheels.append(wheel)
        
        return {
            'body': car_body,
            'wheels': wheels,
            'direction': direction,
            'speed': random.uniform(0.05, 0.1),
            'position': position,
            'orientation': orientation,
            'stopped': False,
            'rule_breaker': rule_breaker,  # 25% chance to run red lights
            'waiting_time': 0  # Track how long car has been waiting
        }

    def create_pedestrian(self):
        """Create a pedestrian at the crosswalk"""
        # Create torso
        torso_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=1.2,
            rgbaColor=[0.8, 0.6, 0.4, 1]
        )
        torso_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            height=1.2
        )
        
        # Create head
        head_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.2,
            rgbaColor=[0.8, 0.6, 0.4, 1]
        )
        head_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.2
        )
        
        pedestrian_pos = [-2.5, 3.5, 0.6]
        head_pos = [-2.5, 3.5, 1.4]
        
        self.pedestrian = {
            'torso': p.createMultiBody(
                baseMass=70,
                baseCollisionShapeIndex=torso_collision,
                baseVisualShapeIndex=torso_visual,
                basePosition=pedestrian_pos
            ),
            'head': p.createMultiBody(
                baseMass=5,
                baseCollisionShapeIndex=head_collision,
                baseVisualShapeIndex=head_visual,
                basePosition=head_pos
            ),
            'position': pedestrian_pos
        }

    def update_traffic_lights(self):
        """Update traffic light states with proper N-S and E-W alternation"""
        self.signal_change_timer += 1
        
        # Handle yellow light phase
        if self.in_yellow_phase:
            self.yellow_timer += 1
            if self.yellow_timer >= self.yellow_light_duration:
                # Switch to opposite direction - CLEAR N-S vs E-W pattern
                self.in_yellow_phase = False
                self.yellow_timer = 0
                
                # Toggle between N-S GREEN and E-W GREEN
                if self.signal_states["north"] == "YELLOW":
                    # Was N-S green, now switch to E-W green
                    self.signal_states["north"] = "RED"
                    self.signal_states["south"] = "RED"
                    self.signal_states["east"] = "GREEN"
                    self.signal_states["west"] = "GREEN"
                else:
                    # Was E-W green, now switch to N-S green
                    self.signal_states["north"] = "GREEN"
                    self.signal_states["south"] = "GREEN"
                    self.signal_states["east"] = "RED"
                    self.signal_states["west"] = "RED"
                
                self._update_light_visuals()
        
        # Check if it's time to change signals
        elif self.signal_change_timer >= self.signal_change_interval:
            self.signal_change_timer = 0
            self.in_yellow_phase = True
            
            # Set current green direction to yellow
            if self.signal_states["north"] == "GREEN":
                # N-S is green, set to yellow
                self.signal_states["north"] = "YELLOW"
                self.signal_states["south"] = "YELLOW"
                # E-W stays RED
                self.signal_states["east"] = "RED"
                self.signal_states["west"] = "RED"
            else:
                # E-W is green, set to yellow
                self.signal_states["east"] = "YELLOW"
                self.signal_states["west"] = "YELLOW"
                # N-S stays RED
                self.signal_states["north"] = "RED"
                self.signal_states["south"] = "RED"
            
            self._update_light_visuals()

    def check_car_ahead(self, car, check_distance=4.0):
        """Check if there's a car ahead in the same lane - improved collision detection"""
        current_pos, _ = p.getBasePositionAndOrientation(car['body'])
        direction = car['direction']
        
        min_distance = float('inf')
        car_detected = False
        
        for other_car in self.cars:
            if other_car['body'] == car['body']:
                continue
            
            if other_car['direction'] != direction:
                continue
            
            other_pos, _ = p.getBasePositionAndOrientation(other_car['body'])
            
            # Check if other car is ahead in the same lane with better collision detection
            if direction == "north":
                if (other_pos[1] > current_pos[1] and 
                    abs(other_pos[0] - current_pos[0]) < 1.0):  # Same lane tolerance
                    distance = other_pos[1] - current_pos[1]
                    if distance < check_distance:
                        car_detected = True
                        min_distance = min(min_distance, distance)
                    
            elif direction == "south":
                if (other_pos[1] < current_pos[1] and 
                    abs(other_pos[0] - current_pos[0]) < 1.0):
                    distance = current_pos[1] - other_pos[1]
                    if distance < check_distance:
                        car_detected = True
                        min_distance = min(min_distance, distance)
                    
            elif direction == "east":
                if (other_pos[0] > current_pos[0] and 
                    abs(other_pos[1] - current_pos[1]) < 1.0):
                    distance = other_pos[0] - current_pos[0]
                    if distance < check_distance:
                        car_detected = True
                        min_distance = min(min_distance, distance)
                    
            elif direction == "west":
                if (other_pos[0] < current_pos[0] and 
                    abs(other_pos[1] - current_pos[1]) < 1.0):
                    distance = current_pos[0] - other_pos[0]
                    if distance < check_distance:
                        car_detected = True
                        min_distance = min(min_distance, distance)
        
        return car_detected, min_distance if car_detected else check_distance

    def update_cars(self):
        """Update car positions with robust collision avoidance and rule breaking"""
        for car in self.cars:
            direction = car['direction']
            current_pos, current_orn = p.getBasePositionAndOrientation(car['body'])
            
            # Check if there's a car directly ahead
            car_ahead, distance_to_car = self.check_car_ahead(car, check_distance=4.0)
            
            # Check traffic light state
            signal_state = self.signal_states[direction]
            should_stop = False
            can_slow_move = False
            
            # PRIORITY 1: Collision avoidance - ALWAYS stop if car is too close
            if car_ahead and distance_to_car < 2.2:  # Critical safety distance
                should_stop = True
            
            # PRIORITY 2: Traffic light rules (only if no immediate collision risk)
            elif signal_state == "RED" or signal_state == "YELLOW":
                stop_distance = 4.5
                
                if direction == "north" and -stop_distance < current_pos[1] < -2.5:
                    if not car['rule_breaker']:
                        should_stop = True
                    else:
                        can_slow_move = True
                        
                elif direction == "south" and 2.5 < current_pos[1] < stop_distance:
                    if not car['rule_breaker']:
                        should_stop = True
                    else:
                        can_slow_move = True
                        
                elif direction == "east" and -stop_distance < current_pos[0] < -2.5:
                    if not car['rule_breaker']:
                        should_stop = True
                    else:
                        can_slow_move = True
                        
                elif direction == "west" and 2.5 < current_pos[0] < stop_distance:
                    if not car['rule_breaker']:
                        should_stop = True
                    else:
                        can_slow_move = True
            
            # Adaptive speed based on distance to car ahead
            if car_ahead and distance_to_car < 3.5:
                # Slow down proportionally to distance
                speed_factor = max(0.2, (distance_to_car - 2.0) / 1.5)
                car['speed'] = car['speed'] * speed_factor
            
            # Update car position
            if should_stop:
                car['waiting_time'] += 1
                # Car stays in place
                
            else:
                car['waiting_time'] = 0
                
                # Determine speed
                if can_slow_move:
                    speed = 0.03  # Rule breakers move slowly through red
                else:
                    speed = random.uniform(0.05, 0.1)  # Normal speed
                
                # Calculate new position
                if direction == "north":
                    new_pos = [current_pos[0], current_pos[1] + speed, current_pos[2]]
                elif direction == "south":
                    new_pos = [current_pos[0], current_pos[1] - speed, current_pos[2]]
                elif direction == "east":
                    new_pos = [current_pos[0] + speed, current_pos[1], current_pos[2]]
                elif direction == "west":
                    new_pos = [current_pos[0] - speed, current_pos[1], current_pos[2]]
                
                # Reset car position if it goes too far
                if abs(new_pos[0]) > 20 or abs(new_pos[1]) > 20:
                    if direction == "north":
                        new_pos = [current_pos[0], -15, current_pos[2]]
                    elif direction == "south":
                        new_pos = [current_pos[0], 15, current_pos[2]]
                    elif direction == "east":
                        new_pos = [-15, current_pos[1], current_pos[2]]
                    elif direction == "west":
                        new_pos = [15, current_pos[1], current_pos[2]]
                    
                    # Reassign rule-breaker status (25% probability)
                    car['rule_breaker'] = random.random() < 0.25
                    car['speed'] = random.uniform(0.05, 0.1)
                
                # Apply the new position
                p.resetBasePositionAndOrientation(car['body'], new_pos, current_orn)
                
                # Keep car upright (prevent flipping)
                p.resetBaseVelocity(car['body'], 
                                   linearVelocity=[0, 0, 0],
                                   angularVelocity=[0, 0, 0])
                
                # Update wheel positions
                for i, wheel in enumerate(car['wheels']):
                    wheel_offset = [
                        [0.6, 0.3, -0.1],
                        [0.6, -0.3, -0.1],
                        [-0.6, 0.3, -0.1],
                        [-0.6, -0.3, -0.1]
                    ][i]
                    
                    cos_theta = math.cos(car['orientation'])
                    sin_theta = math.sin(car['orientation'])
                    rotated_offset = [
                        wheel_offset[0] * cos_theta - wheel_offset[1] * sin_theta,
                        wheel_offset[0] * sin_theta + wheel_offset[1] * cos_theta,
                        wheel_offset[2]
                    ]
                    
                    wheel_pos = [
                        new_pos[0] + rotated_offset[0],
                        new_pos[1] + rotated_offset[1],
                        new_pos[2] + rotated_offset[2]
                    ]
                    p.resetBasePositionAndOrientation(wheel, wheel_pos, current_orn)

    def run_simulation(self):
        """Main simulation loop"""
        print("🚦 Starting Enhanced Crossroad Simulation...")
        print("Features: N-S/E-W traffic coordination, collision-free cars, 25% rule breakers")
        print("Traffic Pattern: North-South ↔️ East-West alternating")
        print("Press Ctrl+C to stop\n")
        
        step = 0
        try:
            while True:
                # Update simulation components
                self.update_traffic_lights()
                self.update_cars()
                
                # Step physics simulation
                p.stepSimulation()
                time.sleep(1/60)  # 60 FPS
                
                step += 1
                
                # Print status every few seconds
                if step % 240 == 0:
                    rule_breakers = sum(1 for car in self.cars if car['rule_breaker'])
                    ns_state = self.signal_states['north']
                    ew_state = self.signal_states['east']
                    print(f"⏱️  Step {step}: N-S: {ns_state:6s} | E-W: {ew_state:6s} | Rule breakers: {rule_breakers}/{len(self.cars)}")
                    
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")
        finally:
            p.disconnect()
            print("✅ Simulation ended successfully")


if __name__ == "__main__":
    env = CrossroadEnvironment()
    env.run_simulation()