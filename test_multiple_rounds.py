#!/usr/bin/env python3
"""
Test script to verify the multiple rounds functionality
"""
import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gym_crossroad_env import CrossroadGymEnv
import numpy as np

def test_multiple_rounds():
    """Test the environment with multiple rounds"""
    print("Testing Multiple Rounds Functionality")
    print("=" * 50)
    
    # Test different round configurations
    test_configs = [
        {"target_rounds": 1, "description": "Single Round (Original)"},
        {"target_rounds": 2, "description": "Two Rounds"},
        {"target_rounds": 3, "description": "Three Rounds"},
    ]
    
    for config in test_configs:
        target_rounds = config["target_rounds"]
        description = config["description"]
        
        print(f"\n--- Testing: {description} ---")
        
        # Create environment with no GUI for faster testing
        env = CrossroadGymEnv(gui=False, max_steps=1000, target_rounds=target_rounds)
        
        print(f"Environment created with target_rounds={target_rounds}")
        print(f"Waypoints: {len(env.waypoints)}")
        print(f"Total target waypoints: {len(env.waypoints) * target_rounds}")
        
        # Test reset
        obs, info = env.reset()
        print(f"Reset info: rounds_completed={info.get('rounds_completed', 0)}, target_rounds={info.get('target_rounds', 0)}")
        
        # Simulate some steps (just random actions for testing)
        for step in range(10):
            action = np.random.randint(0, 5)  # Random action
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                success = info.get('success', False)
                rounds_completed = info.get('rounds_completed', 0)
                waypoints_completed = info.get('waypoints_completed', 0)
                
                print(f"Episode ended at step {step + 1}")
                print(f"  Success: {success}")
                print(f"  Rounds completed: {rounds_completed}/{target_rounds}")
                print(f"  Waypoints completed: {waypoints_completed}")
                break
        
        env.close()
        print(f"Test completed for {description}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    test_multiple_rounds()