# Quick Reference: Critical Changes Made

## The Issue

Your agent learned to **stand still** rather than cross (reward structure was biased toward passivity).

## The Fix (Applied)

### Reward Changes

| Metric              | Old      | New            | Why                     |
| ------------------- | -------- | -------------- | ----------------------- |
| Collision           | -300     | -100           | Passivity-inducing      |
| Progress multiplier | 5.0x     | 8.0x           | Primary learning signal |
| Safety bonus        | 0.5/step | 0.05-0.10/step | Won't reward inaction   |
| Time penalty        | -0.005   | -0.001         | Less aggressive         |
| Waypoint            | 100      | 80             | Relative importance     |
| Success             | 500      | 400            | Relative importance     |

### Learning Changes

| Parameter     | Old   | New   | Why               |
| ------------- | ----- | ----- | ----------------- |
| Episode steps | 1000  | 1500  | Time to find gaps |
| Min replay    | 1000  | 500   | Start sooner      |
| Epsilon end   | 0.01  | 0.05  | Explore longer    |
| Epsilon decay | 0.992 | 0.988 | More exploration  |

## How to Train

```bash
# Quick (10 min)
python train_agent.py --episodes 200

# Standard (45 min)
python train_agent.py --episodes 1000

# Best (2+ hours)
python train_agent.py --episodes 2000
```

## What to Watch For

✅ **Good signs**:

- Avg100 score trending upward (even if absolute scores vary)
- More "TIMEOUT" status (means no crash for 1500 steps)
- Waypoint numbers increasing: 0/4 → 1/4 → 2/4 → 3/4 → 4/4
- Collisions decreasing after episode 200

❌ **Bad signs**:

- Avg100 flat/decreasing by episode 300
- Still getting <-100 average by episode 500
- 0/4 waypoints persisting after episode 300

## Expected Progression

```
Episode 100:   Avg: -100 to 0      (learning, still crashing)
Episode 300:   Avg: 0 to +100      (movement working)
Episode 500:   Avg: +100 to +250   (waypoints appearing)
Episode 1000:  Avg: +200 to +400+  (consistent crossing)
```

## If Something Goes Wrong

### Too many collisions still?

```python
# In config.py, reduce penalty further:
'collision_penalty': -50  # Was -100
```

### No waypoint improvement?

```python
# In config.py, increase movement reward:
'epsilon_decay': 0.985  # More exploration (was 0.988)
```

### Agent too passive?

```python
# In config.py:
'min_replay_size': 100  # Start learning sooner (was 500)
'epsilon_decay': 0.980  # Much more exploration (was 0.988)
```

## Core Concept

**Old problem**:

- Collision = -300 → Need 300 reward to break even → Too risky → Don't move

**New solution**:

- Collision = -100 → Can overcome with strategic movement → Worth trying → Learn crossing

---

**TL;DR**: Fixed the reward function to encourage active crossing instead of passive avoidance. Train with 1000+ episodes, watch Avg100 score improve. Should work.
