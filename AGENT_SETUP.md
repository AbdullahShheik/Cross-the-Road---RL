# DQN Agent Setup Guide

## What Has Been Created

### 1. **DQN Agent** (`dqn_agent.py`)
   - Deep Q-Network implementation using PyTorch
   - Epsilon-greedy exploration strategy
   - Target network for stable learning
   - Experience replay integration

### 2. **Replay Buffer** (`replay_buffer.py`)
   - Stores experiences (state, action, reward, next_state, done)
   - Random sampling for training
   - Fixed-size circular buffer

### 3. **Training Script** (`train_agent.py`)
   - Complete training loop
   - Progress tracking and visualization
   - Model checkpointing
   - Training statistics

### 4. **Evaluation Script** (`evaluate_agent.py`)
   - Test trained agents
   - Calculate success/collision rates
   - Performance metrics

### 5. **Updated Configuration** (`config.py`)
   - DQN hyperparameters
   - Training settings
   - Neural network architecture

## Next Steps

### Step 1: Install Dependencies
```bash
pip install torch matplotlib
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Agent

**Quick Start (Headless - Fast):**
```bash
python train_agent.py --episodes 500 --max-steps 1000
```

**With Visualization (Slower):**
```bash
python train_agent.py --episodes 500 --gui
```

**Full Training:**
```bash
python train_agent.py --episodes 2000 --max-steps 1000 --save-frequency 100
```

### Step 3: Evaluate the Trained Agent

```bash
python evaluate_agent.py --model models/dqn_model_final.pth --episodes 10
```

### Step 4: Monitor Training

- Check `models/training_progress.png` for training curves
- Models are saved in `models/` directory
- Watch console output for episode statistics

## Training Tips

1. **Start Small**: Try 100-200 episodes first to verify everything works
2. **No GUI for Training**: Much faster without visualization
3. **Monitor Epsilon**: Should decay from 1.0 to 0.01 over training
4. **Check Scores**: Average score should increase over time
5. **Adjust Hyperparameters**: If training is unstable, modify `config.py`

## Expected Training Behavior

- **Early Episodes**: High exploration (epsilon ~1.0), random actions, negative scores
- **Middle Episodes**: Learning phase, epsilon decaying, scores improving
- **Later Episodes**: Low exploration (epsilon ~0.01), consistent good performance

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch: `pip install torch`

### Issue: Training is very slow
**Solution**: Remove `--gui` flag, use `--no-gui` or don't specify GUI flag

### Issue: Agent not learning
**Solution**: 
- Check if replay buffer has enough samples (min_replay_size)
- Verify reward structure in config.py
- Try adjusting learning rate or network architecture

### Issue: Out of memory errors
**Solution**: 
- Reduce batch_size in config.py
- Reduce replay_buffer_size
- Reduce hidden layer sizes

## Hyperparameter Tuning

Key hyperparameters in `config.py`:

- **learning_rate**: Start with 0.001, try 0.0001 if unstable
- **gamma**: Discount factor, usually 0.99
- **epsilon_decay**: Controls exploration decay, try 0.995 or 0.99
- **batch_size**: Usually 32-128, depends on memory
- **hidden_layers**: Network architecture, try [64, 64] or [128, 128]

## Success Criteria

The agent is considered successful when:
- Average score over 100 episodes > 50
- Success rate > 80%
- Collision rate < 20%
- Consistent performance across multiple evaluations

## Files Created

```
dqn_agent.py          # DQN agent implementation
replay_buffer.py      # Experience replay buffer
train_agent.py        # Training script
evaluate_agent.py     # Evaluation script
AGENT_SETUP.md        # This file
README.md             # Updated project documentation
```

## What to Expect

1. **First Run**: Agent will explore randomly, low scores
2. **After 100-200 Episodes**: Should start showing some improvement
3. **After 500+ Episodes**: Should achieve decent performance
4. **After 1000+ Episodes**: Should solve the environment (if hyperparameters are good)

## Next Enhancements

- [ ] Add Double DQN for more stable learning
- [ ] Implement Dueling DQN architecture
- [ ] Add prioritized experience replay
- [ ] Implement curriculum learning
- [ ] Add state normalization
- [ ] Try different reward structures
- [ ] Add TensorBoard logging
- [ ] Implement hyperparameter tuning

Good luck with training! 🚀

