import gymnasium as gym
import time
import pickle
import numpy as np

env = gym.make('FrozenLake-v1', render_mode="human")

#Load the Q-table
with open('./frozen_lake_q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

state, _ = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, info, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(0.1)
    print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")