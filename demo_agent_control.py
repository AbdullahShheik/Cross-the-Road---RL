"""
Demo script showing the agent (pedestrian) being controlled
This demonstrates how the RL agent will control the pedestrian to cross the road
"""

import time
import random
from gym_crossroad_env import CrossroadGymEnv
from config import PEDESTRIAN_CONFIG
import pybullet as p
import sys


def demo_random_agent():
    """Demo with random actions (shows pedestrian moving)"""
    print("=" * 60)
    print("Agent Control Demo - Random Actions")
    print("=" * 60)
    print("Watch the pedestrian (brown cylinder) move around!")
    print("Actions: 0=stay, 1=forward, 2=back, 3=left, 4=right")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Create environment with GUI
    env = CrossroadGymEnv(gui=True, max_steps=1000)
    
    # Reset environment
    state, info = env.reset()
    start_pos = PEDESTRIAN_CONFIG['start_position']
    target_pos = PEDESTRIAN_CONFIG['target_position']
    
    print(f"\nPedestrian starting position: [{start_pos[0]:.1f}, {start_pos[1]:.1f}]")
    print(f"Target position: [{target_pos[0]:.1f}, {target_pos[1]:.1f}]")
    print("\nStarting simulation...")
    
    try:
        step_count = 0
        while step_count < 500:  # Run for 500 steps
            # Random action (this simulates what the agent will do)
            action = random.randint(0, 4)
            action_names = ["STAY", "FORWARD", "BACK", "LEFT", "RIGHT"]
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get current position
            pos, _ = p.getBasePositionAndOrientation(env.ped_body_id)
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"\nStep {step_count}:")
                print(f"  Action: {action_names[action]}")
                print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}]")
                print(f"  Reward: {reward:.2f}")
                if info.get('success'):
                    print("  ✓ Reached target!")
                if done:
                    print(f"  Episode ended: {'Success' if info.get('success') else 'Collision/Timeout'}")
            
            # Render
            env.render()
            time.sleep(0.05)  # Slow down for visualization
            
            step_count += 1
            
            if done:
                print(f"\nEpisode ended at step {step_count}")
                print("Resetting environment...")
                state, info = env.reset()
                step_count = 0
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        env.close()
        print("Demo ended")

def demo_simple_policy():
    """Demo with a simple policy (move toward target)"""
    print("=" * 60)
    print("Agent Control Demo - Simple Policy (Move Toward Target)")
    print("=" * 60)
    print("Pedestrian will try to move toward the target!")
    print("=" * 60)
    
    env = CrossroadGymEnv(gui=True, max_steps=1000)
    state, info = env.reset()
    target_pos = PEDESTRIAN_CONFIG['target_position']
    
    try:
        step_count = 0
        while step_count < 500:
            # Get current position from state
            current_x = state[0]
            current_y = state[1]
            target_x = target_pos[0]
            target_y = target_pos[1]
            
            # Simple policy: move toward target
            dx = target_x - current_x
            dy = target_y - current_y
            
            # Choose action based on direction
            if abs(dx) > abs(dy):
                action = 4 if dx > 0 else 3  # Right or Left
            else:
                action = 1 if dy > 0 else 2  # Forward or Back
            
            # Sometimes add randomness
            if random.random() < 0.1:
                action = random.randint(0, 4)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            
            # Print status
            if step_count % 30 == 0:
                import pybullet as p
                pos, _ = p.getBasePositionAndOrientation(env.ped_body_id)
                dist = ((pos[0] - target_x)**2 + (pos[1] - target_y)**2)**0.5
                print(f"Step {step_count}: Position: [{pos[0]:.2f}, {pos[1]:.2f}], Distance to target: {dist:.2f}, Reward: {reward:.2f}")
            
            env.render()
            time.sleep(0.05)
            
            step_count += 1
            
            if done:
                if info.get('success'):
                    print(f"\nSuccess! Reached target in {step_count} steps!")
                else:
                    print(f"\nEpisode ended at step {step_count}")
                print("Resetting...")
                state, info = env.reset()
                step_count = 0
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        env.close()
        print("Demo ended")

def main():
    """Main function"""
    
    print("\nChoose demo mode:")
    print("1. Random actions (shows pedestrian moving randomly)")
    print("2. Simple policy (pedestrian tries to reach target)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            demo_random_agent()
        elif choice == '2':
            demo_simple_policy()
        elif choice == '3':
            print("Exiting...")
            return
        else:
            print("Invalid choice. Running random agent demo...")
            demo_random_agent()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

