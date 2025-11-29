"""
Training script for DQN agent on Crossroad RL Environment
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime
import time

from gym_crossroad_env import CrossroadGymEnv
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from config import RL_CONFIG


def train_agent(
    num_episodes=1000,
    max_steps=400,
    gui=False,
    save_dir='models',
    save_frequency=100,
    plot=True,
    target_rounds=1
):
    """
    Train DQN agent on Crossroad environment.
    
    Args:
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        gui: Whether to show GUI during training
        save_dir: Directory to save models
        save_frequency: Save model every N episodes
        plot: Whether to plot training progress
        target_rounds: Number of complete rounds per episode during training
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    print("Initializing environment...")
    env = CrossroadGymEnv(gui=gui, max_steps=max_steps, target_rounds=target_rounds)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Target rounds per episode: {target_rounds}")
    
    # Initialize agent and replay buffer
    agent = DQNAgent(state_size, action_size)
    replay_buffer = ReplayBuffer(capacity=RL_CONFIG.get('replay_buffer_size', 10000))
    batch_size = RL_CONFIG.get('batch_size', 64)
    
    # Training statistics
    scores = []
    scores_window = deque(maxlen=100)  # Last 100 scores
    epsilons = []
    episode_rewards = []
    
    episode_waypoint_history = []
    episode_collision_history = []
    episode_success_history = []
    episode_step_history = []

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Initial epsilon: {agent.epsilon:.2f}")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, info = env.reset()
        score = 0
        steps = 0
        episode_return = 0.0

        episode_waypoints = 0
        episode_collision = False
        episode_success = False
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state, training=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            if info.get("waypoints_completed"):
                episode_waypoints = info["waypoints_completed"]

            if info.get("collision"):
                episode_collision = True

            if info.get("success"):
                episode_success = True
                
            
            if gui:
                time.sleep(0.05)

            # Store experience and learn
            agent.step(state, action, reward, next_state, done, replay_buffer, batch_size)
            
            state = next_state
            score += reward
            episode_return += reward
            steps += 1
            
            if done:
                break
        
        # Record statistics
        scores.append(score)
        scores_window.append(score)
        episode_rewards.append(score)

        episode_waypoint_history.append(episode_waypoints)
        episode_collision_history.append(episode_collision)
        episode_success_history.append(episode_success)
        episode_step_history.append(steps)

        # --- Exploration schedule control ------------------------------------
        # Epsilon decays linearly from 1.0 to 0.0 over the first 50% of episodes
        # After episode 500 (50% of 1000), epsilon stays at 0 (pure exploitation)
        # exploration_fraction = 0.5  # explore for 50% of episodes, then epsilon = 0
        # eps_start = RL_CONFIG.get('epsilon_start', 1.0)
        # eps_end = 0.0  # Exploration ends completely at 50% of episodes

        # if num_episodes > 0:
        #     exploration_episodes = exploration_fraction * num_episodes
        #     if episode <= exploration_episodes:
        #         # Linear decay from eps_start to eps_end over first 50% of episodes
        #         phase_progress = episode / exploration_episodes
        #         agent.epsilon = eps_start - (eps_start - eps_end) * phase_progress
        #     else:
        #         # After 50% of episodes, epsilon stays at 0 (pure exploitation)
        #         agent.epsilon = eps_end

        epsilon_start = RL_CONFIG.get('epsilon_start', 1.0)
        epsilon_min = RL_CONFIG.get('epsilon_min', 0.05)
        epsilon_decay = RL_CONFIG.get('epsilon_decay', 0.995)

        # Update epsilon after each episode
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        epsilons.append(agent.epsilon)
        
        # Print progress with more detailed info
        if episode % 10 == 0:
            avg_score = np.mean(scores_window)
            # Get episode info for better tracking
            waypoints = episode_waypoints
            success = episode_success
            collision = episode_collision

            if success:
                status = "SUCCESS"
            elif collision:
                status = "COLLISION"
            else:
                status = "TIMEOUT"
            
            print(
                f"Episode {episode:4d} | "
                f"Score: {score:8.2f} | "
                f"Avg100: {np.mean(scores_window):8.2f} | "
                f"Steps: {steps:4d} | "
                f"Waypoints: {episode_waypoints}/4 | "
                f"Status: {status:9s} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Collision: {episode_collision} | "
                f"Success: {episode_success} | "
                f"Buffer: {len(replay_buffer):5d}"
            )
        
        # Save model periodically
        if episode % save_frequency == 0:
            model_path = os.path.join(save_dir, f'dqn_model_episode_{episode}.pth')
            agent.save(model_path)
        
        # # Track success metrics for better evaluation
        # recent_successes = sum(1 for i in range(max(0, episode-20), episode) 
        #                       if i < len(episode_rewards) and episode_rewards[i-1] > 0)
        
        # # Only consider solved if BOTH high score AND actual waypoint completion
        # if len(scores_window) >= 100:
        #     avg_score = np.mean(scores_window)
            
        #     # Count recent episodes with waypoint completions (actual success)
        #     recent_waypoint_episodes = 0
        #     recent_success_episodes = 0
        #     for i in range(max(0, episode-50), episode):
        #         if i < len(episode_rewards):
        #             # Check if this episode had any waypoints (success indicator)
        #             # We'll consider an episode successful if score > 3000 (indicates waypoint rewards)
        #             if episode_rewards[i-1] > 3000:
        #                 recent_waypoint_episodes += 1
        #             # Check for full circuit completion (very high scores)
        #             if episode_rewards[i-1] > 8000:  # Full circuit completion
        #                 recent_success_episodes += 1
            
        #     waypoint_success_rate = recent_waypoint_episodes / min(50, episode) if episode > 0 else 0
        #     full_circuit_rate = recent_success_episodes / min(50, episode) if episode > 0 else 0
            
        #     # Very high threshold - only "solved" if consistently completing full circuits
        #     if avg_score >= 8000.0 and waypoint_success_rate >= 0.6 and full_circuit_rate >= 0.3:
        #         print(f"\nEnvironment truly solved in {episode} episodes!")
        #         print(f"Average score: {avg_score:.2f}")
        #         print(f"Waypoint success rate: {waypoint_success_rate*100:.1f}%")
        #         print(f"Full circuit rate: {full_circuit_rate*100:.1f}%")
        #         print("Agent consistently completing full roundabout circuits!")
        #         model_path = os.path.join(save_dir, 'dqn_model_solved.pth')
        #         agent.save(model_path)
        #         # Continue training for even better performance
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'dqn_model_final.pth')
    agent.save(final_model_path)
    
    # Close environment
    env.close()
    
    # Plot training progress
    if plot:
        plot_training_progress(scores, epsilons, save_dir)
    
    print("\nTraining completed!")
    return agent, scores


def plot_training_progress(scores, epsilons, save_dir):
    """Plot training progress."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot scores
    ax1.plot(scores, alpha=0.6, label='Episode Score')
    if len(scores) >= 100:
        # Moving average
        window = 100
        moving_avg = []
        for i in range(len(scores)):
            if i < window:
                moving_avg.append(np.mean(scores[:i+1]))
            else:
                moving_avg.append(np.mean(scores[i-window+1:i+1]))
        ax1.plot(moving_avg, label='Moving Average (100 episodes)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Progress - Episode Scores')
    ax1.legend()
    ax1.grid(True)
    
    # Plot epsilon
    ax2.plot(epsilons, label='Epsilon (Exploration Rate)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DQN Agent on Crossroad Environment')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--gui', action='store_true', help='Show GUI during training (slower)')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--save-frequency', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--no-plot', action='store_true', help='Disable training plots')
    
    args = parser.parse_args()
    
    train_agent(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        gui=args.gui,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        plot=not args.no_plot
    )


if __name__ == '__main__':
    main()

