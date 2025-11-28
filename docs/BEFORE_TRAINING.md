# Training Checklist: Before You Start

## ✅ Verify Changes Applied

Check that these files have been modified:

- [ ] **gym_crossroad_env.py** line 217: `reward = -100` (was -300)
- [ ] **gym_crossroad_env.py** line 237: `reward += progress * 8.0` (was 5.0)
- [ ] **gym_crossroad_env.py** line 248: `reward += 0.1` and `reward += 0.05` (safety bonuses reduced)
- [ ] **config.py** line 179: `'max_episode_steps': 1500,` (was 1000)
- [ ] **config.py** line 185: `'collision_penalty': -100,` (was -300)
- [ ] **config.py** line 186: `'time_penalty': -0.001,` (was -0.005)
- [ ] **config.py** line 194: `'epsilon_end': 0.05,` (was 0.01)
- [ ] **config.py** line 195: `'epsilon_decay': 0.988,` (was 0.992)
- [ ] **config.py** line 199: `'min_replay_size': 500,` (was 1000)

## 🚀 Ready to Train

```bash
# Full training run (recommended)
python train_agent.py --episodes 1000

# Or test first
python train_agent.py --episodes 200
```

## 📊 What to Expect

### Episode 50

```
Score: -100 to 0
Waypoints: 0/4 mostly
Status: COLLISION dominant
Assessment: Still exploring
```

### Episode 150

```
Score: -50 to +50
Waypoints: 0-1/4
Status: COLLISION still dominant
Assessment: Learning starting
```

### Episode 300

```
Score: +50 to +200
Waypoints: 1-2/4
Status: Balanced COLLISION/TIMEOUT
Assessment: Good progress!
```

### Episode 500

```
Score: +200 to +400
Waypoints: 2-3/4
Status: TIMEOUT more than COLLISION
Assessment: Strategy emerging
```

### Episode 1000

```
Score: +300 to +800+
Waypoints: 3-4/4
Status: SUCCESS dominant
Assessment: Learned well!
```

## 🎯 Success Criteria

Training is **working** if by Episode 300:

- ✅ Avg100 score > -50 (improving from negative)
- ✅ Waypoints showing 1/4 regularly
- ✅ More TIMEOUT than COLLISION in status

Training **needs adjustment** if:

- ❌ Avg100 score still < -100 at episode 300
- ❌ Still 0/4 waypoints at episode 300
- ❌ 100% COLLISION status still

## 🔧 Adjustment Guide

### If Learning is Slow

```python
# In config.py:
'epsilon_decay': 0.985  # More exploration (from 0.988)
'min_replay_size': 200  # Start sooner (from 500)
```

### If Waypoints Not Improving

```python
# In gym_crossroad_env.py _calculate_reward():
reward += progress * 10.0  # Stronger signal (from 8.0)
```

### If Agent Still Seems Passive

```python
# In config.py:
'epsilon_end': 0.10  # Even more exploration (from 0.05)
'epsilon_decay': 0.980  # Much slower decay (from 0.988)
```

## 📈 Monitoring Metrics

Each episode prints:

```
Episode  500 | Score: +300 | Avg100: +250 | Waypoints: 3/4 | Status: SUCCESS
              |___________| |____________| |_____________| |________________|
              Individual   Rolling avg   Progress toward Cross result
              episode      of last 100   4 waypoints
```

**Focus on Avg100**: This shows if learning is trending up.

## ⏱️ Training Time

```
200 episodes:  ~10-15 minutes
500 episodes:  ~30 minutes
1000 episodes: ~60 minutes
2000 episodes: ~2 hours
```

## 💾 Saving Results

Models are automatically saved:

```
models/dqn_model_episode_100.pth
models/dqn_model_episode_200.pth
... every 100 episodes ...
models/dqn_model_final.pth     (best)
```

Pick the one with:

- Latest episode number from good Avg100 score
- Usually around episode 800-1000

## 🧪 Testing After Training

```bash
python evaluate_agent.py --model models/dqn_model_episode_1000.pth --episodes 10
```

Watch for:

- Low collision count
- Consistent waypoint completion
- Most episodes finishing successfully

## ❌ Troubleshooting

### Python not found

```bash
# Use full python path
python .\train_agent.py --episodes 1000
```

### GPU out of memory (if using GPU)

```python
# In train_agent.py, device handling:
device = torch.device("cpu")  # Force CPU
```

### Taking too long

```bash
# Run with fewer episodes first
python train_agent.py --episodes 200
```

---

## Final Checklist Before Start

- [ ] Files saved and edited (see above)
- [ ] Python environment ready
- [ ] Models directory exists
- [ ] Terminal in correct directory (D:\Cross-the-Road---RL-1)
- [ ] Understand what to expect (see 📊 section)
- [ ] Know how to monitor (see 📈 section)

**Ready? Run:**

```bash
python train_agent.py --episodes 1000
```

Good luck! 🎯
