# Cross-the-Road RL Project

A reinforcement learning project where an agent learns to safely cross a busy intersection using Deep Q-Network (DQN).

## Features

- 🚦 Realistic 3D crossroad simulation with PyBullet
- 🚗 Multiple cars with traffic light system
- 🚶 Pedestrian agent controlled by RL
- 🤖 DQN (Deep Q-Network) agent implementation
- 📊 Training progress tracking and visualization
- 🎯 Evaluation and testing scripts

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Cross-the-Road---RL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── intersection_env.py      # Base crossroad environment
├── gym_crossroad_env.py     # Gymnasium wrapper for RL
├── dqn_agent.py            # DQN agent implementation
├── replay_buffer.py        # Experience replay buffer
├── train_agent.py          # Training script
├── evaluate_agent.py       # Evaluation script
├── config.py               # Configuration settings
├── launcher.py             # Launcher for different modes
└── requirements.txt        # Python dependencies
```

## Usage

### 1. Run Basic Simulation (without RL)

```bash
python intersection_env.py
```

Or use the launcher:
```bash
python launcher.py basic
```

### 2. Train the DQN Agent

Train the agent (without GUI for faster training):
```bash
python train_agent.py --episodes 1000 --max-steps 1000
```

Train with GUI (slower but visual):
```bash
python train_agent.py --episodes 1000 --gui
```

Training options:
- `--episodes`: Number of training episodes (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--gui`: Show GUI during training (slower)
- `--save-dir`: Directory to save models (default: 'models')
- `--save-frequency`: Save model every N episodes (default: 100)
- `--no-plot`: Disable training plots

### 3. Evaluate Trained Agent

Evaluate a trained model:
```bash
python evaluate_agent.py --model models/dqn_model_final.pth --episodes 10
```

Evaluation options:
- `--model`: Path to saved model (required)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--no-gui`: Disable GUI for faster evaluation
- `--max-steps`: Maximum steps per episode (default: 1000)

### 4. RL Demo Mode

```bash
python launcher.py rl-demo
```

## Environment Details

### Observation Space
16-dimensional vector containing:
- Agent position (x, y) and velocity (vx, vy): 4 dims
- Target position (x, y): 2 dims
- Traffic light states (north, south, east, west): 4 dims
- Nearest 2 cars (relative position and speed): 6 dims

### Action Space
5 discrete actions:
- 0: Stay
- 1: Move forward
- 2: Move backward
- 3: Move left
- 4: Move right

### Reward Structure
- Reach target: +100
- Collision: -100
- Time penalty: -0.1 per step
- Safe crossing bonus: +50

## Configuration

Edit `config.py` to customize:
- Environment settings (road size, traffic light timing)
- RL hyperparameters (learning rate, gamma, epsilon decay)
- Neural network architecture
- Reward structure

## Training Tips

1. **Start without GUI**: Training is much faster without GUI
2. **Monitor progress**: Check the training plot in the `models/` directory
3. **Adjust hyperparameters**: If training is unstable, try:
   - Lower learning rate (e.g., 0.0001)
   - Increase replay buffer size
   - Adjust epsilon decay rate
4. **Save checkpoints**: Models are saved periodically for recovery

## Results

After training, you should see:
- Training progress plots in `models/training_progress.png`
- Saved model checkpoints in `models/`
- Evaluation metrics (success rate, collision rate, average score)

## Next Steps

- Experiment with different RL algorithms (PPO, A3C, etc.)
- Add more complex scenarios (multiple pedestrians, dynamic obstacles)
- Implement curriculum learning
- Add state normalization and feature engineering
- Try different neural network architectures

## License

[Your License Here]

## Contributors

[Your Name/Team]
