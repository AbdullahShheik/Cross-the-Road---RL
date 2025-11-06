from gym_crossroad_env import CrossroadGymEnv
import numpy as np
import random
import pickle

env = CrossroadGymEnv(gui=True)

# Discretize state space (if continuous)
def discretize_state(state, bins=10):
    return tuple(np.digitize(state, np.linspace(-5, 5, bins)))

# Q-table initialization
action_size = env.action_space.n
q_table = {}

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.95     # discount factor
epsilon = 1.0    # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 5000
max_steps = 200

for ep in range(episodes):
    state = discretize_state(env.reset())
    total_reward = 0
    done = False

    for step in range(max_steps):
        # ε-greedy action selection
        if random.random() < epsilon or state not in q_table:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Step environment
        next_state, reward, done, info = env.step(action)
        next_state = discretize_state(next_state)

        # Initialize states in table
        if state not in q_table:
            q_table[state] = np.zeros(action_size)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(action_size)

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep+1}: total reward {total_reward:.1f}, epsilon={epsilon:.3f}")

# Save Q-table
with open("crossroad_qtable.pkl", "wb") as f:
    pickle.dump(q_table, f)

env.close()
