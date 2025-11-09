"""
Quick test script to verify the project setup
"""

print("=" * 60)
print("Cross-the-Road RL Project - Setup Verification")
print("=" * 60)

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    import pybullet as p
    import numpy as np
    import gymnasium as gym
    print("   ✓ pybullet, numpy, gymnasium imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test 2: Environment imports
print("\n2. Testing environment imports...")
try:
    from intersection_env import CrossroadEnvironment
    from gym_crossroad_env import CrossroadGymEnv
    from config import RL_CONFIG
    print("   ✓ Environment modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    exit(1)

# Test 3: Agent imports (requires PyTorch)
print("\n3. Testing agent imports...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} imported successfully")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    
    from dqn_agent import DQNAgent
    from replay_buffer import ReplayBuffer
    print("   ✓ Agent modules imported successfully")
    agent_available = True
except ImportError as e:
    print(f"   ⚠ PyTorch not installed: {e}")
    print("   ⚠ Agent training will not work without PyTorch")
    print("   💡 Install with: pip install torch")
    agent_available = False

# Test 4: Environment creation (DIRECT mode, no GUI)
print("\n4. Testing environment creation...")
try:
    env = CrossroadGymEnv(gui=False, max_steps=100)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"   ✓ Environment created successfully")
    print(f"   ✓ State size: {state_size}")
    print(f"   ✓ Action size: {action_size}")
    
    # Test reset
    state, info = env.reset()
    print(f"   ✓ Environment reset successful")
    print(f"   ✓ Initial state shape: {state.shape}")
    
    # Test step
    action = 1  # Move forward
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Environment step successful")
    print(f"   ✓ Reward: {reward:.2f}")
    
    env.close()
    print("   ✓ Environment closed successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Agent creation (if PyTorch available)
if agent_available:
    print("\n5. Testing agent creation...")
    try:
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        print(f"   ✓ DQN Agent created successfully")
        print(f"   ✓ Device: {agent.device}")
        print(f"   ✓ Initial epsilon: {agent.epsilon:.2f}")
        
        # Test action selection
        test_state = state
        action = agent.act(test_state, training=True)
        print(f"   ✓ Action selection works: action={action}")
        
        replay_buffer = ReplayBuffer(capacity=1000)
        print(f"   ✓ Replay buffer created successfully")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("Setup Verification Summary")
print("=" * 60)
print("✓ Environment: Working")
print("✓ Gymnasium wrapper: Working")
if agent_available:
    print("✓ DQN Agent: Ready")
    print("✓ Training: Ready to start")
    print("\n🚀 You can start training with:")
    print("   python train_agent.py --episodes 100")
else:
    print("⚠ DQN Agent: PyTorch not installed")
    print("⚠ Training: Not available")
    print("\n💡 To enable training, install PyTorch:")
    print("   pip install torch")

print("\n🎮 To run the simulation:")
print("   python intersection_env.py")
print("   or")
print("   python launcher.py basic")

print("\n" + "=" * 60)

