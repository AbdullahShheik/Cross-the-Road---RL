"""
Replay Buffer for storing and sampling experiences for DQN training
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, capacity=10000):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            capacity (int): Maximum size of buffer
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        
        Args:
            batch_size (int): Size of sample batch
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=bool)
        )
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

