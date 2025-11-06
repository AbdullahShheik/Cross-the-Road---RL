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
            cameraDistance=15, 
            cameraYaw=45, 
            cameraPitch=-30, 
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Initialize scene components
        self.cars = []
        self.traffic_lights = []
        self.pedestrian = None
        self.signal_states = {"north": "RED", "south": "RED", "east": "GREEN", "west": "GREEN"}
        self.signal_change_timer = 0
        self.signal_change_interval = 180  # frames
        
        self.setup_environment()
        self.create_traffic_infrastructure()
        self.spawn_cars()
        self.create_pedestrian()
        self.add_environmental_objects()

    def setup_environment(self):
        """Create the base environment with roads and grass"""
        # Create large grass base
        grass_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[25, 25, 0.1],
            rgbaColor=[0.2, 0.6, 0.2, 1]  # Green grass
        )
        grass_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[25, 25, 0.1])
        self.grass = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=grass_collision,
            baseVisualShapeIndex=grass_visual,
            basePosition=[0, 0, -0.1]
        )

        
        # Create main intersection base (asphalt)
        intersection_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[6, 6, 0.05],
            rgbaColor=[0.15, 0.15, 0.15, 1]  # Dark asphalt
        )
        intersection_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[6, 6, 0.05])
        self.intersection = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=intersection_collision,
            baseVisualShapeIndex=intersection_visual,
            basePosition=[0, 0, 0.05]
        )

        # Create four main roads extending from intersection
        road_positions = [
            [0, 12, 0.05],   # North road
            [0, -12, 0.05],  # South road
            [12, 0, 0.05],   # East road
            [-12, 0, 0.05]   # West road
        ]
        
        road_dimensions = [
            [2.5, 6, 0.05],   # North/South roads
            [2.5, 6, 0.05],   # North/South roads  
            [6, 2.5, 0.05],   # East/West roads
            [6, 2.5, 0.05]    # East/West roads
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

        # Add road markings
        self.create_road_markings()
        self.create_crosswalks()

    def create_road_markings(self):
        """Create yellow center lines and lane dividers"""
        # Center line for north-south road
        for y in range(-18, 19, 2):
            if abs(y) > 6:  # Skip intersection area
                line_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.05, 0.5, 0.01],
                    rgbaColor=[1, 1, 0, 1]  # Yellow
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=line_visual,
                    basePosition=[0, y, 0.11]
                )

        # Center line for east-west road
        for x in range(-18, 19, 2):
            if abs(x) > 6:  # Skip intersection area
                line_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.5, 0.05, 0.01],
                    rgbaColor=[1, 1, 0, 1]  # Yellow
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
                    rgbaColor=[1, 1, 1, 1]  # White stripes
                )
                p.createMultiBody(
                    baseMass=0, 
                    baseVisualShapeIndex=stripe_visual, 
                    basePosition=pos
                )

    def create_traffic_infrastructure(self):
        """Create traffic lights and poles"""
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
                rgbaColor=[0.3, 0.3, 0.3, 1]  # Gray pole
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

            # Create traffic light box
            light_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.2, 0.2, 0.6],
                rgbaColor=[0.2, 0.2, 0.2, 1]  # Dark gray box
            )
            light_collision = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=[0.2, 0.2, 0.6]
            )
            light_box = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=light_collision,
                baseVisualShapeIndex=light_visual,
                basePosition=[x, y, 4.2]
            )

            # Create individual lights (red, yellow, green)
            lights = []
            colors = [[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]]  # Red, Yellow, Green
            for i, color in enumerate(colors):
                light_visual = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.12,
                    rgbaColor=color
                )
                light = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=light_visual,
                    basePosition=[x, y, 4.6 - i * 0.3]
                )
                lights.append(light)
            
            self.traffic_lights.append({
                'direction': direction,
                'position': [x, y],
                'lights': lights,
                'pole': pole,
                'box': light_box
            })

    def spawn_cars(self):
        """Spawn cars on different lanes"""
        car_spawn_data = [
            # [x, y, z, orientation, lane_direction, color]
            [0, -15, 0.3, 1.57, "north", [1, 0, 0, 1]],      # Red car going north
            [0, 15, 0.3, -1.57, "south", [0, 0, 1, 1]],      # Blue car going south
            [15, 0, 0.3, 3.14, "west", [0, 1, 0, 1]],        # Green car going west
            [-15, 0, 0.3, 0, "east", [1, 1, 0, 1]],          # Yellow car going east
            [1.5, -10, 0.3, 1.57, "north", [1, 0, 1, 1]],    # Purple car going north
            [-1.5, 10, 0.3, -1.57, "south", [0, 1, 1, 1]],   # Cyan car going south
        ]

        for spawn_data in car_spawn_data:
            x, y, z, orientation, direction, color = spawn_data
            car = self.create_car([x, y, z], orientation, color, direction)
            self.cars.append(car)

    def create_car(self, position, orientation, color, direction):
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
        
        # Calculate quaternion from orientation angle
        quat = p.getQuaternionFromEuler([0, 0, orientation])
        
        car_body = p.createMultiBody(
            baseMass=1000,
            baseCollisionShapeIndex=car_collision,
            baseVisualShapeIndex=car_visual,
            basePosition=position,
            baseOrientation=quat
        )

        # Add wheels (visual only for now)
        wheel_positions = [
            [0.6, 0.3, -0.1], [0.6, -0.3, -0.1],   # Front wheels
            [-0.6, 0.3, -0.1], [-0.6, -0.3, -0.1]  # Rear wheels
        ]
        
        wheels = []
        for wheel_pos in wheel_positions:
            wheel_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.2,
                length=0.1,
                rgbaColor=[0.1, 0.1, 0.1, 1]  # Black wheels
            )
            # Transform wheel position relative to car
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
            'speed': random.uniform(0.03, 0.08),
            'position': position,
            'orientation': orientation,
            'stopped': False
        }

    def create_pedestrian(self):
        """Create a static pedestrian"""
        # Create pedestrian body (cylinder for torso)
        torso_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=1.2,
            rgbaColor=[0.8, 0.6, 0.4, 1]  # Skin color
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
            rgbaColor=[0.8, 0.6, 0.4, 1]  # Skin color
        )
        head_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.2
        )

        # Position pedestrian at crosswalk
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

    def add_environmental_objects(self):
        """Add trees, buildings, and other environmental objects"""
        # Add trees around the intersection
        tree_positions = [
            [-8, 8], [-8, -8], [8, 8], [8, -8],  # Corner trees
            [-6, 12], [6, 12], [-6, -12], [6, -12],  # Additional trees
        ]

        for x, y in tree_positions:
            self.create_tree([x, y, 0])

        # Add simple buildings
        building_data = [
            [[-15, 15, 3], [3, 3, 6], [0.7, 0.7, 0.9, 1]],   # Light gray building
            [[15, 15, 3], [2, 4, 6], [0.8, 0.6, 0.6, 1]],    # Light red building
            [[-15, -15, 3], [4, 2, 6], [0.6, 0.6, 0.8, 1]],  # Light blue building
            [[15, -15, 3], [3, 3, 6], [0.8, 0.8, 0.6, 1]],   # Light yellow building
        ]

        for position, dimensions, color in building_data:
            self.create_building(position, dimensions, color)

        # Add some decorative elements
        self.create_fire_hydrants()
        self.create_street_lamps()

    def create_tree(self, position):
        """Create a simple tree"""
        # Tree trunk
        trunk_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=3,
            rgbaColor=[0.4, 0.2, 0.1, 1]  # Brown trunk
        )
        trunk_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            height=3
        )
        trunk = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=trunk_collision,
            baseVisualShapeIndex=trunk_visual,
            basePosition=[position[0], position[1], 1.5]
        )

        # Tree foliage
        foliage_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=1.5,
            rgbaColor=[0.1, 0.5, 0.1, 1]  # Green leaves
        )
        foliage = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=foliage_visual,
            basePosition=[position[0], position[1], 4]
        )

    def create_building(self, position, dimensions, color):
        """Create a simple building"""
        building_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=dimensions,
            rgbaColor=color
        )
        building_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=dimensions
        )
        building = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=building_collision,
            baseVisualShapeIndex=building_visual,
            basePosition=position
        )

    def create_fire_hydrants(self):
        """Add fire hydrants near intersections"""
        hydrant_positions = [
            [-3, 6], [3, 6], [-3, -6], [3, -6]
        ]
        
        for x, y in hydrant_positions:
            hydrant_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.15,
                length=0.8,
                rgbaColor=[1, 0, 0, 1]  # Red hydrant
            )
            hydrant_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.15,
                height=0.8
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=hydrant_collision,
                baseVisualShapeIndex=hydrant_visual,
                basePosition=[x, y, 0.4]
            )

    def create_street_lamps(self):
        """Add street lamps along the roads"""
        lamp_positions = [
            [-2.5, 8], [2.5, 8], [-2.5, -8], [2.5, -8],  # North-South road
            [8, -2.5], [8, 2.5], [-8, -2.5], [-8, 2.5]   # East-West road
        ]

        for x, y in lamp_positions:
            # Lamp pole
            pole_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.08,
                length=3,
                rgbaColor=[0.3, 0.3, 0.3, 1]  # Gray pole
            )
            pole_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.08,
                height=3
            )
            pole = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=pole_collision,
                baseVisualShapeIndex=pole_visual,
                basePosition=[x, y, 1.5]
            )

            # Lamp head
            lamp_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.2,
                rgbaColor=[1, 1, 0.8, 1]  # Warm white light
            )
            lamp = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=lamp_visual,
                basePosition=[x, y, 3.2]
            )

    def update_traffic_lights(self):
        """Update traffic light states"""
        self.signal_change_timer += 1
        
        if self.signal_change_timer >= self.signal_change_interval:
            self.signal_change_timer = 0
            
            # Toggle light states
            if self.signal_states["north"] == "RED":
                self.signal_states["north"] = "GREEN"
                self.signal_states["south"] = "GREEN"
                self.signal_states["east"] = "RED"
                self.signal_states["west"] = "RED"
            else:
                self.signal_states["north"] = "RED"
                self.signal_states["south"] = "RED"
                self.signal_states["east"] = "GREEN"
                self.signal_states["west"] = "GREEN"

        # Update visual lights
        for traffic_light in self.traffic_lights:
            direction = traffic_light['direction']
            lights = traffic_light['lights']
            state = self.signal_states[direction]
            
            # Reset all lights to dim
            for i, light in enumerate(lights):
                if i == 0:  # Red light
                    color = [0.8, 0, 0, 1] if state == "RED" else [0.2, 0, 0, 1]
                elif i == 1:  # Yellow light  
                    color = [0.8, 0.8, 0, 1] if state == "YELLOW" else [0.2, 0.2, 0, 1]
                else:  # Green light
                    color = [0, 0.8, 0, 1] if state == "GREEN" else [0, 0.2, 0, 1]
                
                p.changeVisualShape(light, -1, rgbaColor=color)

    def update_cars(self):
        """Update car positions based on traffic lights"""
        for car in self.cars:
            direction = car['direction']
            current_pos, current_orn = p.getBasePositionAndOrientation(car['body'])
            
            # Check if car should stop at red light
            signal_state = self.signal_states[direction]
            should_stop = False
            
            if signal_state == "RED":
                # Check if car is approaching intersection
                if direction == "north" and current_pos[1] < -2 and current_pos[1] > -5:
                    should_stop = True
                elif direction == "south" and current_pos[1] > 2 and current_pos[1] < 5:
                    should_stop = True
                elif direction == "east" and current_pos[0] < -2 and current_pos[0] > -5:
                    should_stop = True
                elif direction == "west" and current_pos[0] > 2 and current_pos[0] < 5:
                    should_stop = True

            if not should_stop:
                # Move car forward
                speed = car['speed']
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

                p.resetBasePositionAndOrientation(car['body'], new_pos, current_orn)
                
                # Update wheel positions
                for i, wheel in enumerate(car['wheels']):
                    wheel_offset = [
                        [0.6, 0.3, -0.1], [0.6, -0.3, -0.1],
                        [-0.6, 0.3, -0.1], [-0.6, -0.3, -0.1]
                    ][i]
                    
                    # Rotate wheel offset based on car orientation
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
        print("🚦 Starting Crossroad Simulation...")
        print("Features: Multi-lane traffic, traffic lights, pedestrian, environmental objects")
        
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
                if step % 180 == 0:
                    print(f"Step {step}: North/South lights: {self.signal_states['north']}, "
                          f"East/West lights: {self.signal_states['east']}")
                
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")
        finally:
            p.disconnect()
            print("✅ Simulation ended successfully")

# Create and run the environment
if __name__ == "__main__":
    env = CrossroadEnvironment()
    env.run_simulation()
