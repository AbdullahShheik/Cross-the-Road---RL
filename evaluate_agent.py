"""
Evaluate trained DQN agent on Crossroad RL Environment
"""

import argparse
import numpy as np
import time
from gym_crossroad_env import CrossroadGymEnv
from dqn_agent import DQNAgent
from config import RL_CONFIG


def evaluate_agent(model_path, num_episodes=10, gui=True, max_steps=1000):
    """
    Evaluate a trained DQN agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
        gui: Whether to show GUI
        max_steps: Maximum steps per episode
    """
    # Initialize environment
    print("Initializing environment...")
    env = CrossroadGymEnv(gui=gui, max_steps=max_steps)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent and load model
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0  # No exploration during evaluation
    agent.load(model_path)
    
    print(f"Evaluating agent for {num_episodes} episodes...")
    print("-" * 50)
    
    episode_scores = []
    episode_steps = []
    successes = 0
    collisions = 0
    
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        score = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode}/{num_episodes}")
        
        while not done and steps < max_steps:
            # Select action (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            steps += 1
            
            # Render if GUI is enabled
            if gui:
                env.render()
                time.sleep(0.05)  # Slow down for visualization
            
            # Check for success or collision
            if info.get('success', False):
                successes += 1
                print(f"  ✓ Success! Reached target in {steps} steps")
            elif done and steps < max_steps and reward < -50:
                collisions += 1
                print(f"  ✗ Collision occurred at step {steps}")
        
        episode_scores.append(score)
        episode_steps.append(steps)
        
        print(f"  Score: {score:.2f}, Steps: {steps}")
        
        if gui:
            time.sleep(1)  # Pause between episodes
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Collision Rate: {collisions}/{num_episodes} ({100*collisions/num_episodes:.1f}%)")
    print(f"Best Score: {np.max(episode_scores):.2f}")
    print(f"Worst Score: {np.min(episode_scores):.2f}")
    
    env.close()
    
    return {
        'scores': episode_scores,
        'steps': episode_steps,
        'success_rate': successes / num_episodes,
        'collision_rate': collisions / num_episodes,
        'avg_score': np.mean(episode_scores),
        'avg_steps': np.mean(episode_steps)
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate DQN Agent on Crossroad Environment')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI (faster evaluation)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        gui=not args.no_gui,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()

