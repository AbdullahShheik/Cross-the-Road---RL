# Action Plan: Training With Fixed Rewards

## Problem Diagnosed ✅

Your previous 500-episode run had **learned passivity**:

- The -300 collision penalty was so harsh the agent learned "don't move" is safest
- One breakthrough episode (Episode 120) showed the agent CAN learn crossing
- But then regressed to avoiding all movement risk

## Solution Implemented ✅

### Key Changes:

1. **Collision penalty**: -300 → -100 (realistic cost, not paralyzing)
2. **Progress reward**: 5.0x → 8.0x (move toward goals strongly)
3. **Safety bonus**: 0.5/step → 0.05-0.10/step (won't reward standing still)
4. **Learning start**: After 1000 steps → After 500 steps (learn sooner)
5. **Episode length**: 1000 → 1500 steps (find safe gaps)
6. **Exploration**: More aggressive, epsilon decays slower

## How to Train Now

### Quick Test (10 minutes)

```bash
python train_agent.py --episodes 200
```

### Recommended (45 minutes)

```bash
python train_agent.py --episodes 1000
```

### Best Results (2+ hours)

```bash
python train_agent.py --episodes 2000
```

## What to Expect

### Early Training (Episodes 1-150)

```
Episode   50 | Score: -150 to -100 | Waypoints: 0-1/4 | Status: COLLISION
Episode  100 | Score: -50 to 0     | Waypoints: 0-1/4 | Status: COLLISION or TIMEOUT
Episode  150 | Score: 0 to +100    | Waypoints: 1-2/4 | Status: Mixed
```

- Still lots of collisions (normal - it's exploring)
- But moving more strategically
- Waypoint completions increasing

### Mid Training (Episodes 150-400)

```
Episode  250 | Score: +50 to +200  | Waypoints: 1-2/4 | Status: COLLISION decreasing
Episode  350 | Score: +150 to +350 | Waypoints: 2-3/4 | Status: SUCCESS more common
Episode  400 | Score: +200 to +400 | Waypoints: 2-4/4 | Status: Solid progress
```

- Collisions still happen but less frequent
- Multi-waypoint sequences working
- Agent learning to wait for safe gaps

### Late Training (Episodes 400+)

```
Episode  500 | Score: +300 to +600 | Waypoints: 3-4/4 | Status: Mostly SUCCESS
Episode 1000 | Score: +400 to +800 | Waypoints: 4/4   | Status: Consistent SUCCESS
```

- Most episodes collision-free
- Full circuit completion common
- Learned optimal crossing strategy

## How to Monitor Progress

Watch for these signs:

1. **Score moving average improving**: Most important indicator
2. **Waypoints increasing**: 0/4 → 1/4 → 2/4 → 3/4 → 4/4
3. **Status changing**: COLLISION → TIMEOUT → SUCCESS
4. **Epsilon decreasing**: Shows agent shifting from explore to exploit

## If Results Don't Improve

### Still too many collisions (>50% of episodes)?

- The -100 penalty is more realistic than before
- Collisions WILL happen as agent learns
- Check that **Avg100** score is improving upward
- If still negative by episode 300, increase progress reward multiplier to 10.0

### No waypoint progress?

- Increase waypoint tolerance: 1.5 → 2.0
- Increase waypoint bonus: 80 → 150
- These should dominate over collision penalties

### Agent standing still?

- Reduce `min_replay_size` to 200 (start learning even sooner)
- Increase `epsilon_decay` to 0.985 (explore more aggressively)
- Add small bonus for any movement: +0.05 per step

## Files Changed

```
✅ gym_crossroad_env.py   - Balanced reward function
✅ config.py              - Reduced penalties, longer episodes, earlier learning
✅ COLLISION_PENALTY_FIX.md - Detailed explanation (this file explains why)
```

## Next Steps

1. Run: `python train_agent.py --episodes 1000 --no-plot`
2. Watch the output for improving Avg100 score
3. After 300-400 episodes, should see clear improvement
4. Save the best model (usually around episode 800-1000)

---

**Expected outcome**: With these balanced rewards, agent should achieve:

- **Episode 300**: Moving toward waypoints reliably
- **Episode 500**: Multi-waypoint sequences working
- **Episode 1000**: Full circuit completion 50%+ of the time
- **Episode 2000**: Full circuit completion 80%+ of the time
