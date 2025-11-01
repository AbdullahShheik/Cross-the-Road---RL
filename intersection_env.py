import pybullet as p
import pybullet_data
import time

# --- Connect to PyBullet GUI ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Better lighting and camera angle
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

# Base plane
plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0], useMaximalCoordinates=True)

# --- Create intersection base (darker asphalt) ---
road_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[4, 4, 0.01],
    rgbaColor=[0.1, 0.1, 0.1, 1]
)
road_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 4, 0.01])
road = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=road_collision,
    baseVisualShapeIndex=road_visual,
    basePosition=[0, 0, 0.01]
)

# --- Create vertical road ---
vertical_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[1, 4, 0.02],
    rgbaColor=[0.03, 0.03, 0.03, 1]
)
vertical_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 4, 0.02])
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=vertical_collision,
    baseVisualShapeIndex=vertical_visual,
    basePosition=[0, 0, 0.02]
)

# --- Create horizontal road ---
horizontal_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[4, 1, 0.02],
    rgbaColor=[0.03, 0.03, 0.03, 1]
)
horizontal_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, 1, 0.02])
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=horizontal_collision,
    baseVisualShapeIndex=horizontal_visual,
    basePosition=[0, 0, 0.02]
)

# --- Add zebra crossings (bright white & thicker) ---
def create_crosswalk(x_center, y_center, orientation="horizontal"):
    for i in range(-2, 3):
        if orientation == "horizontal":
            pos = [x_center + i * 0.35, y_center, 0.03]
            half = [0.12, 0.55, 0.015]
        else:
            pos = [x_center, y_center + i * 0.35, 0.03]
            half = [0.55, 0.12, 0.015]
        stripe_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[1, 1, 1]
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=stripe_visual, basePosition=pos)

# Four zebra crossings (intersection)
create_crosswalk(0, 1.3, "horizontal")
create_crosswalk(0, -1.3, "horizontal")
create_crosswalk(1.3, 0, "vertical")
create_crosswalk(-1.3, 0, "vertical")

# --- Traffic light pole ---
def create_signal(x, y, color):
    color_map = {"RED": [1, 0, 0, 1], "GREEN": [0, 1, 0, 1]}
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.3], rgbaColor=color_map[color]
    )
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.3])
    return p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, 0.5]
    )

signal_box = create_signal(-1.8, 1.8, "RED")

# --- Agents ---
pedestrian = p.loadURDF("sphere2.urdf", [-2, -1.3, 0.1], globalScaling=0.25)
car = p.loadURDF("r2d2.urdf", [0, -3, 0.2])

# --- Simulation Loop ---
signal_state = "RED"
change_interval = 5
step = 0

while step < 100:
    time.sleep(0.15)
    step += 1

    # Change signal
    if step % change_interval == 0:
        signal_state = "GREEN" if signal_state == "RED" else "RED"
        p.removeBody(signal_box)
        signal_box = create_signal(-1.8, 1.8, signal_state)

    # Move car vertically only if green
    car_pos, _ = p.getBasePositionAndOrientation(car)
    if signal_state == "GREEN":
        p.resetBasePositionAndOrientation(car, [car_pos[0], car_pos[1] + 0.1, car_pos[2]], [0, 0, 0, 1])

    # Move pedestrian horizontally
    ped_pos, _ = p.getBasePositionAndOrientation(pedestrian)
    p.resetBasePositionAndOrientation(pedestrian, [ped_pos[0] + 0.05, ped_pos[1], ped_pos[2]], [0, 0, 0, 1])

    print(f"[{step}s] Light: {signal_state}, Car Y={car_pos[1]:.2f}, Ped X={ped_pos[0]:.2f}")

p.disconnect()
print("✅ Simulation finished successfully.")
