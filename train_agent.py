"""
Training script for DQN agent on Crossroad RL Environment
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime

from gym_crossroad_env import CrossroadGymEnv
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from config import RL_CONFIG


def train_agent(
    num_episodes=1000,
    max_steps=1000,
    gui=False,
    save_dir='models',
    save_frequency=100,
    plot=True
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
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    print("Initializing environment...")
    env = CrossroadGymEnv(gui=gui, max_steps=max_steps)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Initialize agent and replay buffer
    agent = DQNAgent(state_size, action_size)
    replay_buffer = ReplayBuffer(capacity=RL_CONFIG.get('replay_buffer_size', 10000))
    batch_size = RL_CONFIG.get('batch_size', 64)
    
    # Training statistics
    scores = []
    scores_window = deque(maxlen=100)  # Last 100 scores
    epsilons = []
    episode_rewards = []
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Initial epsilon: {agent.epsilon:.2f}")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, info = env.reset()
        score = 0
        steps = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state, training=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done, replay_buffer, batch_size)
            
            state = next_state
            score += reward
            steps += 1
            
            if done:
                break
        
        # Record statistics
        scores.append(score)
        scores_window.append(score)
        epsilons.append(agent.epsilon)
        episode_rewards.append(score)
        
        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores_window)
            print(f"Episode {episode:4d} | "
                  f"Avg Score: {avg_score:7.2f} | "
                  f"Score: {score:7.2f} | "
                  f"Steps: {steps:4d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(replay_buffer):5d}")
        
        # Save model periodically
        if episode % save_frequency == 0:
            model_path = os.path.join(save_dir, f'dqn_model_episode_{episode}.pth')
            agent.save(model_path)
        
        # Check if solved (average score > threshold)
        if len(scores_window) >= 100:
            avg_score = np.mean(scores_window)
            if avg_score >= 50.0:  # Threshold for "solving" the environment
                print(f"\nEnvironment solved in {episode} episodes!")
                print(f"Average score: {avg_score:.2f}")
                model_path = os.path.join(save_dir, 'dqn_model_solved.pth')
                agent.save(model_path)
                break
    
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
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
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

