import numpy as np
import gym
import gym_organism

env = gym.make("Organism-v0")

env.reset() # reset environment to a new, random state

q_table = np.zeros([env.observation_space.n, env.action_space.n])


"""Training the agent"""
print("How many generations ?")
gen = int(input())

import random
from time import sleep
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
for j in range(gen):
    print("Gen : ", j+1)
    time_taken = 0
    # For plotting metrics
    all_epochs = []
    all_penalties = []
    frames = []
    for i in range(1, 100):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

            state = next_state

            epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
        time_taken += epochs
    print(f"Total time in this generation is : {time_taken}")
    print("Training finished.\n")


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)