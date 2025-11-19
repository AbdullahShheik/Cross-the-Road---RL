# 🌍 Understanding the Simulation

This guide explains how the 3D crossroad simulation works and what you'll see when running the project.

## 🎭 The Virtual World

### 🏗️ Environment Layout

The simulation creates a realistic crossroad intersection with:

```
         North Road
             ↑
             │
West Road ← [🚦] → East Road  
             │
             ↓
         South Road
```

### Key Components:

#### 🛣️ **Roads and Intersection**
- **4 Roads**: North, South, East, West extending from center
- **Central Intersection**: 6x6 meter crossing area
- **Crosswalks**: Zebra-striped pedestrian crossings on each side
- **Lane Markings**: Yellow center lines and road boundaries

#### 🚗 **Vehicles (8 Cars)**
- **Realistic Movement**: Cars follow traffic rules (mostly!)
- **Different Colors**: Red, Blue, Green, Yellow, Purple, Cyan, Orange, Lime
- **Smart Behavior**: Stop at red lights, avoid collisions
- **Rule Breakers**: 25% of cars occasionally run red lights (like real traffic!)
- **Continuous Flow**: Cars respawn when they exit the simulation area

#### 🚦 **Traffic Light System**
- **4 Traffic Lights**: One for each direction
- **Coordinated Timing**: North-South vs East-West alternation
- **5-Second Cycles**: Green for 5 seconds, then 0.5 seconds yellow
- **Visual Feedback**: Bright colors show current state clearly

#### 🚶 **The AI Pedestrian**
- **Brown Cylinder**: The agent learning to cross
- **Head and Body**: Realistic human proportions
- **Smart Movement**: Learns to avoid cars and use traffic signals

## 🎯 The Agent's Mission

### Simple vs Enhanced Behavior

**Before (Simple Crossing):**
- Agent only crossed from west to east
- Single target: reach the other side
- Limited learning about traffic patterns

**After (Roundabout Navigation):**
- Agent navigates in a square around the intersection
- Multiple waypoints to visit in sequence
- Complex traffic awareness and timing

### 📍 Waypoint System

The agent follows these waypoints in order:

1. **Start**: West side of north crosswalk `[-2.5, 3.5]`
2. **North Center**: Middle of north crosswalk `[0, 3.5]`
3. **East Side**: End of north crosswalk `[2.5, 3.5]`
4. **East Center**: Middle of east crosswalk `[2.5, 0]`
5. **South Side**: Start of south crosswalk `[2.5, -3.5]`
6. **South Center**: Middle of south crosswalk `[0, -3.5]`
7. **West Side**: End of south crosswalk `[-2.5, -3.5]`
8. **West Center**: Middle of west crosswalk `[-2.5, 0]`
9. **Complete Loop**: Back to start

## 🧠 What the Agent Observes

The agent's "vision" includes 39 different measurements:

### Agent State (4 values):
- Current X position
- Current Y position  
- X velocity (speed sideways)
- Y velocity (speed forward/backward)

### Target Information (3 values):
- Target X position (current waypoint)
- Target Y position (current waypoint)
- Progress through waypoint sequence

### Traffic Lights (4 values):
- North signal state (0=Red, 0.5=Yellow, 1=Green)
- South signal state
- East signal state  
- West signal state

### Nearby Cars (32 values - 8 cars × 4 each):
For each of the 8 nearest cars:
- Relative X position (distance horizontally)
- Relative Y position (distance vertically)
- Car's X velocity
- Car's Y velocity

## 🎁 Reward System

The agent learns through rewards and penalties:

### 🏆 **Positive Rewards:**
- **+50 points**: Reaching each waypoint
- **+200 points**: Completing full roundabout loop
- **+2 points**: Moving toward current target
- **+5 points**: Smart traffic light timing
- **+0.2 points**: Exploring different areas
- **+0.1 points**: Maintaining safe distance from cars

### ⚠️ **Penalties:**
- **-100 points**: Collision with car (episode ends)
- **-3 points**: Getting too close to cars
- **-0.1 points**: Staying still (encourages movement)
- **-0.05 points**: Each time step (encourages efficiency)

## 🎮 Controls During Simulation

When watching the simulation, you can:

### Camera Controls:
- **Mouse drag**: Rotate view around intersection
- **Mouse wheel**: Zoom in/out
- **Right-click drag**: Pan camera
- **Middle-click**: Reset camera view

### During Training:
- **Ctrl+C**: Stop training early
- **Close window**: End simulation

### During Evaluation:
- **Watch the agent**: See how it navigates
- **Console output**: Shows success/failure for each episode

## 📊 Visual Indicators

### Traffic Light Colors:
- 🔴 **Red**: Stop (cars should wait)
- 🟡 **Yellow**: Caution (about to change)
- 🟢 **Green**: Go (cars can proceed)

### Car Behaviors:
- **Stopped cars**: Waiting at red lights
- **Slow-moving cars**: Rule-breakers sneaking through
- **Normal speed**: Following traffic rules

### Agent Behaviors:
- **Quick movements**: Confident crossing
- **Hesitation**: Learning safe timing
- **Smart positioning**: Using crosswalks effectively

## 🔍 Performance Metrics

During simulation, watch for:

### Training Progress:
- **Episode Score**: Higher scores = better performance
- **Success Rate**: Percentage of completed loops
- **Collision Rate**: How often agent gets hit
- **Training Time**: How long to reach good performance

### Agent Intelligence:
- **Traffic Awareness**: Does it wait for green lights?
- **Path Efficiency**: Does it take direct routes?
- **Safety**: Does it maintain distance from cars?
- **Adaptability**: Can it handle rule-breaking cars?

## 🎯 What Makes a Good Agent?

A well-trained agent will:

1. **Navigate the full roundabout** consistently
2. **Avoid all car collisions** 
3. **Use traffic signals intelligently**
4. **Take efficient paths** between waypoints
5. **Handle unexpected situations** (rule-breakers, etc.)

## 🚀 Next Steps

Now that you understand the simulation:

1. **Run a test**: `python launcher.py test`
2. **Watch car patterns**: Notice how traffic flows
3. **Observe agent behavior**: See how it learns
4. **Ready to train**: Continue to [Training Guide](03_training.md)

**Want to customize the simulation? Check the [Configuration Guide](05_configuration.md)**