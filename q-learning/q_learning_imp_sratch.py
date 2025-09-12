import gymnasium as gym
import numpy as np
import pickle
import time
env = gym.make('FrozenLake-v1')

#initializing the Q-table
n_states = 16 # 4x4 grid
n_actions = 4 # up, down, left, right
goal_state = 15 # bottom right corner is the goal[Treasure box at 15th cell]

q_table = np.zeros((n_states, n_actions))


# Hyperparameters
learning_rate = 0.8
discount_factor = 0.90
exploration_prob = 0.4
epochs = 2000

for episodes in range(epochs):
    current_state, _ = env.reset()
    done = False #done when the agent reaches the goal state/terminal state
    while not done:
        if np.random.random() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])
        next_state, reward, done, info, _ = env.step(action)

        q_table[current_state, action] = q_table[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[current_state, action])
        current_state = next_state

        # learning_rate = learning_rate * 0.99
        
    print(f"Episode {episodes} completed")
    print(f"Epochs: {epochs}")
    print(f"Reward: {reward}")
    # time.sleep(0.1)
with open('./frozen_lake_q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)