# Road Crossing RL

A reinforcement learning project that trains a pedestrian agent to safely navigate a busy traffic intersection using **Deep Q-Learning (DQN)**.

The project combines a custom **PyBullet traffic simulation**, a **Gymnasium-compatible environment**, and a **PyTorch DQN agent**. The agent learns when and where to move by observing nearby vehicles, traffic signals, waypoint targets, and collision risk while balancing progress and safety.

## Architecture

The project is split into three main layers:

```text
                    ┌──────────────────────┐
                    │      DQN Agent       │
                    │      PyTorch         │
                    │                      │
                    │  Q-Network           │
                    │  Target Network      │
                    │  ε-greedy policy     │
                    └──────────┬───────────┘
                               │
                        state / action
                               │
                    ┌──────────▼───────────┐
                    │  Gymnasium RL Layer  │
                    │ CrossroadGymEnv      │
                    │                      │
                    │ observations         │
                    │ reward shaping       │
                    │ waypoint tracking    │
                    │ episode termination  │
                    └──────────┬───────────┘
                               │
                         simulation state
                               │
                    ┌──────────▼───────────┐
                    │ PyBullet Simulation  │
                    │ CrossroadEnvironment │
                    │                      │
                    │ cars & traffic       │
                    │ traffic lights       │
                    │ pedestrian           │
                    │ collision physics    │
                    └──────────────────────┘
```

### 1. Simulation Layer — `intersection_env.py`

`CrossroadEnvironment` owns the physical world. It builds and updates the intersection, pedestrian, cars, traffic lights, road infrastructure, and PyBullet physics.

Traffic is dynamic rather than completely deterministic: vehicles move through different directions, respond to coordinated traffic signals, and some vehicles can behave as rule breakers.

### 2. RL Environment — `gym_crossroad_env.py`

`CrossroadGymEnv` wraps the simulation behind the standard Gymnasium `reset()` / `step()` interface.

The agent has **5 discrete actions**:

```text
0 → stay
1 → forward
2 → backward
3 → left
4 → right
```

The observation is a **12-dimensional state vector**:

```text
[
  pedestrian_x,
  pedestrian_y,
  target_x,
  target_y,
  nearest_car_1_relative_x,
  nearest_car_1_relative_y,
  nearest_car_2_relative_x,
  nearest_car_2_relative_y,
  north_south_green,
  east_west_green,
  minimum_car_distance,
  danger_flag
]
```

This gives the policy information about both **navigation and immediate collision risk** instead of requiring it to infer danger only from sparse collision events.

Navigation is waypoint-based and supports sequential crossings and multiple rounds around the intersection.

### 3. DQN Agent — `dqn_agent.py`

The policy is learned using a Deep Q-Network implemented in PyTorch.

The default network is:

```text
12-dimensional state
        ↓
      256
        ↓
      256
        ↓
      128
        ↓
5 Q-values (one per action)
```

The agent maintains two networks:

* **Local Q-network** — optimized during training
* **Target Q-network** — periodically synchronized to provide more stable TD targets

For each sampled transition, the target is computed as:

```text
Q_target = reward + γ · max Q_target(next_state)
```

and the local network minimizes the MSE between this target and the predicted Q-value for the selected action.

Action selection uses an **epsilon-greedy policy**, allowing the agent to explore early in training and increasingly exploit its learned policy later.

## Experience Replay

`replay_buffer.py` stores transitions in the form:

```text
(state, action, reward, next_state, done)
```

Training uses randomly sampled mini-batches rather than learning only from consecutive transitions. This reduces temporal correlation between updates and allows previous experiences to be reused.

The current configuration uses a replay capacity of **100,000 transitions** and begins learning once a minimum amount of experience has been collected.

## Reward Design

The reward function is deliberately shaped around more than simply reaching the destination.

The agent receives learning signals for:

* progress toward the current waypoint
* reaching waypoints and completing rounds
* maintaining safe distance from vehicles
* using zebra crossings
* continuing useful movement
* avoiding unnecessary delays

Collisions receive a strong negative reward, while movement away from the goal, idling in unsafe areas, and leaving zebra crossings around the intersection are discouraged.

This design addresses an important RL failure mode: if collision avoidance dominates the objective too heavily, the easiest learned policy can become **doing nothing**. The reward structure therefore balances safety with positive progress.

## Project Structure

```text
road-crossing-rl/
├── config.py                 # Environment + DQN configuration
├── dqn_agent.py              # Q-network and DQN learning logic
├── evaluate_agent.py         # Trained-policy evaluation
├── gym_crossroad_env.py      # Gymnasium RL interface + rewards
├── intersection_env.py       # PyBullet traffic simulation
├── launcher.py               # Interactive/CLI entry point
├── replay_buffer.py          # Experience replay memory
├── requirements.txt
├── test_multiple_rounds.py   # Multi-round environment tests
├── train_agent.py            # Training loop and checkpoints
└── docs/                     # Training notes and experiments
```

## Installation

Python 3.9+ is recommended.

```bash
git clone https://github.com/hassan-31x/road-crossing-rl.git
cd road-crossing-rl

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

The main dependencies are **PyTorch**, **Gymnasium**, **PyBullet**, **NumPy**, and **Matplotlib**.

## Running the Project

The launcher provides a simple interface for testing the environment, training an agent, or running a trained model:

```bash
python launcher.py
```

It can also be used directly from the command line:

```bash
python launcher.py test
python launcher.py train
python launcher.py run
```

### Train directly

```bash
python train_agent.py --episodes 500 --no-plot
```

Training can run without the PyBullet GUI for better performance.

### Evaluate a trained model

```bash
python evaluate_agent.py --model <path-to-model> --episodes 10
```

For faster headless evaluation:

```bash
python evaluate_agent.py --model <path-to-model> --episodes 10 --no-gui
```

Evaluation reports episode scores, steps, success rate, collision rate, and waypoint/round progress.

## Configuration

Most simulation and learning parameters are centralized in `config.py`, including:

```text
Environment geometry
Traffic behavior
Pedestrian navigation
Reward coefficients
DQN hyperparameters
Replay buffer size
Exploration schedule
Network architecture
Simulation performance
```

This keeps experiments reproducible without scattering tuning constants across the training and environment code.

Some important DQN defaults are:

```text
Learning rate        0.0005
Discount factor      0.99
Batch size           256
Replay capacity      100,000
Target update        every 100 steps
Hidden layers        256 → 256 → 128
```

## Training Flow

At a high level, each training step follows:

```text
Environment observation
        ↓
ε-greedy action selection
        ↓
PyBullet simulation step
        ↓
next state + shaped reward
        ↓
store transition in replay buffer
        ↓
sample mini-batch
        ↓
compute TD targets
        ↓
optimize local Q-network
        ↓
periodically sync target network
```

This separation between simulation, Gymnasium interface, replay memory, and policy learning makes it possible to modify traffic dynamics or reward design without rewriting the DQN implementation.

## Documentation

The `docs/` directory contains notes from the experimentation and tuning process, including collision-penalty analysis, reward changes, training guidance, and investigations into persistent collision behavior.

These documents capture the iterations behind the current state representation, exploration strategy, and reward shaping rather than treating the final hyperparameters as arbitrary constants.

## Goal

The project is an experiment in applying **Deep Reinforcement Learning to safety-aware navigation** in a dynamic environment.

Rather than solving a static shortest-path problem, the pedestrian must learn a policy that accounts for changing traffic signals, moving vehicles, collision risk, crossing geometry, and long-term waypoint progress.
