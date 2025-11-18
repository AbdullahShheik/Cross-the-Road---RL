# Quick Start Guide

## ✅ Project Status

The project is set up and ready! Here's what's working:

### ✓ Working Components:
- ✅ Environment (Crossroad simulation)
- ✅ Gymnasium wrapper
- ✅ Configuration files
- ✅ Training scripts
- ✅ Evaluation scripts

### ⚠️ Needs Installation:
- ⚠️ PyTorch (for DQN agent training)

## 🚀 Quick Start Options

### Option 1: View the Simulation (No PyTorch needed)
```bash
python intersection_env.py
```
This will open a PyBullet GUI showing the crossroad simulation with cars, traffic lights, and a pedestrian.

### Option 2: Install PyTorch and Train Agent

**Step 1: Install PyTorch**
```bash
pip install torch
```

**Step 2: Verify Installation**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

**Step 3: Quick Training Test (50 episodes)**
```bash
python train_agent.py --episodes 50 --max-steps 500
```

**Step 4: Full Training**
```bash
python train_agent.py --episodes 1000 --max-steps 1000
```

### Option 3: Test Everything
```bash
python test_setup.py
```

## 📊 What You Should See

### When Running Simulation:
- 3D crossroad with roads and intersection
- Multiple cars moving in different directions
- Traffic lights changing colors
- Pedestrian at crosswalk
- Environmental objects (trees, buildings, street lamps)

### When Training Agent:
- Episode progress printed every 10 episodes
- Scores improving over time
- Epsilon (exploration rate) decreasing
- Models saved in `models/` directory
- Training plot saved as `models/training_progress.png`

## 🎯 Next Steps

1. **View the simulation**: Run `python intersection_env.py`
2. **Install PyTorch**: Run `pip install torch`
3. **Start training**: Run `python train_agent.py --episodes 100`
4. **Evaluate**: After training, run `python evaluate_agent.py --model models/dqn_model_final.pth`

## 💡 Tips

- **For faster training**: Don't use `--gui` flag
- **For visualization**: Use `--gui` flag (slower)
- **Monitor progress**: Check `models/training_progress.png`
- **Adjust hyperparameters**: Edit `config.py`

## 🔧 Troubleshooting

### PyTorch Installation Issues:
- Try: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Or visit: https://pytorch.org/get-started/locally/

### Environment Issues:
- Make sure pybullet is installed: `pip install pybullet`
- Check all dependencies: `pip install -r requirements.txt`

## 📁 Project Structure

```
.
├── intersection_env.py      # Base environment
├── gym_crossroad_env.py     # RL wrapper
├── dqn_agent.py            # DQN agent
├── train_agent.py          # Training script
├── evaluate_agent.py       # Evaluation script
├── config.py               # Configuration
└── models/                 # Saved models (created during training)
```

Enjoy training your agent! 🎉

