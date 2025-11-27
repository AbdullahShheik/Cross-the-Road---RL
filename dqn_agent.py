"""
Deep Q-Network (DQN) Agent for Crossroad RL Environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from config import RL_CONFIG


class DQNNetwork(nn.Module):
    """Neural network for DQN."""
    
    def __init__(self, state_size, action_size, hidden_layers=None):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list): List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = RL_CONFIG.get('hidden_layers', [128, 128])
        
        # Build the network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class DQNAgent:
    """DQN Agent that learns to cross the road safely."""
    
    def __init__(self, state_size, action_size, device=None):
        """
        Initialize an Agent object.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            device: PyTorch device (cuda or cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = RL_CONFIG.get('learning_rate', 0.001)
        self.gamma = RL_CONFIG.get('gamma', 0.99)
        self.epsilon = RL_CONFIG.get('epsilon_start', 1.0)
        self.epsilon_min = RL_CONFIG.get('epsilon_end', 0.01)
        self.epsilon_decay = RL_CONFIG.get('epsilon_decay', 0.995)
        self.update_target_freq = RL_CONFIG.get('target_update_frequency', 100)
        
        # Q-Network (main network)
        self.qnetwork_local = DQNNetwork(state_size, action_size).to(self.device)
        # Q-Network (target network)
        self.qnetwork_target = DQNNetwork(state_size, action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Step counter for target network updates
        self.steps = 0
    
    def act(self, state, training=True):
        """
        Returns actions for given state as per current policy.
        
        Args:
            state: Current state
            training (bool): If True, use epsilon-greedy exploration
            
        Returns:
            Action index
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return np.argmax(action_values.cpu().data.numpy())
    
    def step(self, state, action, reward, next_state, done, replay_buffer, batch_size=64):
        """
        Update agent's knowledge using the sampled batch.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            replay_buffer: Replay buffer to sample from
            batch_size: Size of batch to sample
        """
        # Save experience in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Learn every update_frequency steps
        self.steps += 1
        
        # If enough samples are available in memory, get random subset and learn
        if len(replay_buffer) > RL_CONFIG.get('min_replay_size', 1000):
            if self.steps % RL_CONFIG.get('train_frequency', 4) == 0:
                experiences = replay_buffer.sample(batch_size)
                self.learn(experiences)
        
        # Update target network periodically
        if self.steps % self.update_target_freq == 0:
            self.soft_update_target_network()
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Note: epsilon (exploration rate) is now controlled per-episode from the
        # training loop in train_agent.py so that we can shape the exploration
        # schedule (e.g., ensure a minimum fraction of episodes use high exploration).
    
    def soft_update_target_network(self, tau=1.0):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            tau (float): Interpolation parameter
        """
        # Hard update (copy weights)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
    
    def save(self, filepath):
        """Save the agent's neural network."""
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the agent's neural network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.steps = checkpoint.get('steps', 0)
        print(f"Model loaded from {filepath}")

