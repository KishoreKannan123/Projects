import gymnasium as gym
import numpy as np


env = gym.make("Taxi-v4")

Q_table = np.zeros(
    (env.observation_space.n, env.action_space.n)
)

alpha = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 10_000

for episode in range(episodes):
    if(episode%100 == 0):
        print(episode)

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Exploration vs exploitation
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        # Take action
        new_state, reward, terminated, truncated, info = env.step(action)

        # Q-learning update
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q_table[new_state])

        Q_table[state, action] += (
            alpha * (target - Q_table[state, action])
        )

        state = new_state

    # Reduce exploration over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


print(Q_table)
#Eval
env = gym.make("Taxi-v4",render_mode = 'human')
for episode in range(episodes):
    if(episode%100 == 0):
        print(episode)

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # # Exploration vs exploitation
        # if np.random.random() < epsilon:
        #     action = env.action_space.sample()
        # else:
        action = np.argmax(Q_table[state])

        # Take action
        new_state, reward, terminated, truncated, info = env.step(action)

        Q_table[state, action] += (
            alpha * (target - Q_table[state, action])
        )

        state = new_state



env.close()