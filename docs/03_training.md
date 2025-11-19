# 🎓 Training the Agent

This comprehensive guide walks you through training your AI agent to safely navigate the crossroad roundabout.

## 🎯 Training Overview

### What is Training?
Training is the process where the AI agent:
1. **Tries different actions** in the simulation
2. **Receives rewards/penalties** based on performance  
3. **Learns patterns** about safe crossing strategies
4. **Improves over time** through experience

### How Long Does Training Take?
- **Quick test**: 50-100 episodes (5-10 minutes)
- **Good performance**: 200-500 episodes (20-50 minutes)
- **Expert level**: 1000+ episodes (1-2 hours)

*Times vary based on your computer's performance*

## 🚀 Step-by-Step Training

### Method 1: Interactive Training (Recommended for beginners)

```bash
# Launch the interactive interface
python launcher.py

# Select option 2: Train Model
# Follow the prompts
```

### Method 2: Direct Command
```bash
# Quick training (fast, no graphics)
python launcher.py train

# Or with specific parameters
python train_agent.py --episodes 500 --gui
```

### Training Parameters Explained:

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `episodes` | Number of training rounds | 500 (beginner), 1000+ (expert) |
| `gui` | Show graphics during training | `False` (faster), `True` (visual) |
| `max_steps` | Maximum steps per episode | 2000 (enough for full roundabout) |
| `save_frequency` | Save model every N episodes | 50 (regular checkpoints) |

## 📊 Understanding Training Output

### Console Output Example:
```
Episode   10 | Avg Score:  -45.23 | Score:  -67.45 | Steps:  234 | Epsilon: 0.950 | Buffer: 2340
Episode   20 | Avg Score:  -32.15 | Score:  -28.90 | Steps:  456 | Epsilon: 0.903 | Buffer: 4680
Episode   50 | Avg Score:  -15.67 | Score:   12.34 | Steps:  789 | Epsilon: 0.780 | Buffer: 10000
Episode  100 | Avg Score:   25.43 | Score:   67.89 | Steps:  567 | Epsilon: 0.608 | Buffer: 20000
Episode  200 | Avg Score:   78.92 | Score:  156.78 | Steps:  432 | Epsilon: 0.370 | Buffer: 40000
```

### What Each Column Means:

#### **Episode**
- Current training round number
- Higher = more experience

#### **Avg Score**  
- Average reward over last 100 episodes
- **Negative**: Agent struggling, getting hit or lost
- **Positive**: Agent improving, reaching waypoints
- **High positive**: Expert-level performance

#### **Score**
- Reward for this specific episode
- Shows immediate performance

#### **Steps**
- How many actions taken this episode
- **High steps**: Agent wandering or being cautious
- **Low steps**: Either very good (efficient) or very bad (quick collision)

#### **Epsilon**
- Exploration rate (0.0 to 1.0)
- **High (0.8+)**: Lots of random actions (early learning)
- **Low (0.1-)**: Mostly using learned strategy

#### **Buffer**
- Number of experiences stored for learning
- More experiences = better learning

## 📈 Training Phases

### Phase 1: Random Exploration (Episodes 1-50)
**What you'll see:**
- Agent moves randomly
- Frequent collisions
- Very negative scores (-100 to -50)
- High epsilon values (0.9+)

**This is normal!** The agent is exploring the environment.

### Phase 2: Basic Learning (Episodes 50-200)
**What you'll see:**
- Agent starts avoiding cars
- Reaches some waypoints
- Scores improving (-50 to +20)
- Epsilon decreasing (0.8 to 0.4)

### Phase 3: Strategy Development (Episodes 200-500)
**What you'll see:**
- Agent using traffic lights
- Consistent waypoint reaching
- Positive scores (+20 to +80)
- Lower epsilon (0.4 to 0.1)

### Phase 4: Optimization (Episodes 500+)
**What you'll see:**
- Smooth, efficient navigation
- High success rates
- High scores (+80 to +200)
- Very low epsilon (0.1-)

## 🎮 Training With Graphics (Recommended First Time)

Enable graphics to watch your agent learn:

```bash
python launcher.py train
# When prompted, choose 'y' for GUI
```

### What to Watch For:

#### **Early Training (First 100 episodes):**
- Agent moving erratically
- Walking into cars
- Getting stuck in corners
- Ignoring traffic lights

#### **Mid Training (Episodes 100-300):**
- Agent staying on crosswalks
- Stopping when cars are near
- Reaching some waypoints
- Starting to wait for green lights

#### **Late Training (Episodes 300+):**
- Smooth, confident movement
- Perfect traffic light timing
- Efficient path-taking
- Consistent roundabout completion

## ⚡ Training Without Graphics (Faster)

For faster training, disable graphics:

```bash
python train_agent.py --episodes 1000 --no-gui
```

**Pros:**
- 3-5x faster training
- Uses less computer resources
- Can train larger networks

**Cons:**
- Can't see agent learning
- Harder to debug problems

## 💾 Saved Models

During training, models are automatically saved:

```
models/
├── dqn_model_episode_50.pth    # Checkpoint at episode 50
├── dqn_model_episode_100.pth   # Checkpoint at episode 100
├── dqn_model_episode_150.pth   # Checkpoint at episode 150
├── dqn_model_solved.pth        # When agent reaches expert level
└── dqn_model_final.pth         # Final model at end of training
```

### Loading Different Models:
```bash
# Test a specific checkpoint
python evaluate_agent.py --model models/dqn_model_episode_200.pth
```

## 📊 Training Progress Visualization

After training, you'll get a graph showing:

### Score Progress:
- **Blue line**: Individual episode scores  
- **Orange line**: Moving average (trend)
- **Upward trend**: Agent is learning

### Exploration Rate:
- **Declining curve**: Less random, more strategic
- **Should reach near 0** by end of training

## 🔧 Training Troubleshooting

### Problem: Scores Not Improving
**Symptoms:** Avg score stays negative after 200+ episodes
**Solutions:**
- Increase training episodes
- Check for simulation errors
- Reduce learning rate in config
- Try restarting training

### Problem: Training Too Slow  
**Symptoms:** Taking very long per episode
**Solutions:**
- Disable GUI: `--no-gui`
- Reduce `max_steps` 
- Close other applications
- Use performance mode in config

### Problem: Agent Gets Stuck
**Symptoms:** Agent stops moving or moves in circles
**Solutions:**
- Check reward system in config
- Increase exploration (epsilon)
- Add movement encouragement rewards

### Problem: Frequent Crashes
**Symptoms:** Training stops with errors
**Solutions:**
- Check GPU memory usage
- Reduce batch size
- Update PyBullet
- Try CPU-only mode

## 🎯 Training Success Criteria

Your agent is well-trained when:

✅ **Average score > 50** over 100 episodes  
✅ **Success rate > 80%** for completing loops  
✅ **Collision rate < 10%**  
✅ **Consistent performance** across multiple runs  

## 🏆 Advanced Training Tips

### For Better Performance:
1. **Train longer**: 1000+ episodes for expert behavior
2. **Tune hyperparameters**: Experiment with learning rates
3. **Use curriculum learning**: Start with simpler scenarios
4. **Ensemble models**: Train multiple agents

### For Faster Training:
1. **Batch training**: Higher batch sizes
2. **Experience replay**: Larger replay buffers
3. **Target networks**: Less frequent updates
4. **GPU acceleration**: Use CUDA if available

## 🚀 Next Steps

After successful training:

1. **Test your model**: `python launcher.py run`
2. **Analyze performance**: Check success rates
3. **Experiment with settings**: Try different configurations
4. **Share your results**: See how others perform

**Ready to test your trained agent? → [Running Trained Models](04_architecture.md)**

**Want to customize training? → [Advanced Configuration](05_configuration.md)**