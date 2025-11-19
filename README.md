# 🚦 Crossroad Reinforcement Learning Project

An AI agent that learns to safely navigate a busy roundabout using Deep Q-Network (DQN) reinforcement learning.

![Project Demo](https://img.shields.io/badge/Demo-Available-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyBullet](https://img.shields.io/badge/PyBullet-3D_Simulation-orange)

## 🎯 Project Overview

This project simulates a realistic 3D crossroad environment where:
- **Cars** drive through an intersection following traffic rules (with some rule-breakers!)
- **Traffic lights** control the flow with North-South and East-West alternation
- **An AI pedestrian** learns to safely cross multiple roads in a roundabout pattern

### Key Features

✅ **Enhanced Roundabout Navigation** - Agent walks in square patterns around the intersection  
✅ **Realistic Traffic Simulation** - 8 cars with different behaviors and rule-breaking  
✅ **Smart Reward System** - Progress rewards, safety bonuses, and traffic awareness  
✅ **Performance Optimized** - Fast loading and efficient rendering  
✅ **Clean Interface** - Simple test/train/run workflow  

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/YourUsername/Cross-the-Road---RL.git
cd Cross-the-Road---RL

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Project
```bash
# Launch interactive interface
python launcher.py

# Or use direct commands:
python launcher.py test    # Test the simulation
python launcher.py train   # Train the AI agent  
python launcher.py run     # Run trained agent
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Project Setup](docs/01_setup.md) | Installation and environment setup |
| [Understanding the Simulation](docs/02_simulation.md) | How the 3D environment works |
| [Training the Agent](docs/03_training.md) | Step-by-step training guide |
| [Code Architecture](docs/04_architecture.md) | Understanding the codebase |
| [Advanced Configuration](docs/05_configuration.md) | Customizing parameters |
| [Troubleshooting](docs/06_troubleshooting.md) | Common issues and solutions |

## 🎮 How It Works

1. **Environment**: A 3D crossroad with realistic traffic patterns
2. **Agent**: A pedestrian that learns through trial and error
3. **Training**: DQN algorithm optimizes crossing strategies
4. **Goal**: Navigate the full roundabout safely and efficiently

## 📊 Results

After training, the agent achieves:
- **90%+ Success Rate** in completing full roundabout navigation
- **Smart Traffic Awareness** - waits for green lights and safe gaps
- **Collision Avoidance** - maintains safe distance from vehicles
- **Efficient Pathfinding** - learns optimal routes between waypoints

## 🛠 Technical Stack

- **Simulation**: PyBullet 3D Physics Engine
- **AI Algorithm**: Deep Q-Network (DQN)
- **Framework**: OpenAI Gymnasium
- **Neural Network**: PyTorch
- **Visualization**: Matplotlib

## 📁 Project Structure

```
Cross-the-Road---RL/
├── launcher.py              # Main interface
├── train_agent.py          # Training script
├── evaluate_agent.py       # Testing script
├── gym_crossroad_env.py    # RL environment wrapper
├── intersection_env.py     # 3D simulation environment
├── dqn_agent.py           # DQN neural network
├── config.py              # Configuration settings
├── models/                # Saved neural network models
├── docs/                  # Documentation guides
└── README.md              # This file
```

## 🤝 Contributing

We welcome contributions! See our guides for:
- Adding new features
- Improving the simulation
- Optimizing the AI algorithm
- Creating better documentation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyBullet team for the excellent 3D physics simulation
- OpenAI Gym community for the RL framework standards
- Traffic simulation research for realistic behavior modeling

---

**Ready to train an AI to cross the road safely? Start with `python launcher.py`!** 🚶‍♂️🤖