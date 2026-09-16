
import gymnasium as gym
import numpy as np


# --------------------------------------------------
# Environment
# --------------------------------------------------

env = gym.make("CartPole-v1")


# --------------------------------------------------
# Discretization
# --------------------------------------------------

n_bins = [10,20,10,20]

Q_table = np.zeros(
    (n_bins[0], n_bins[1], n_bins[2], n_bins[3], env.action_space.n)
)


low = np.array([
    -4.8,    # cart position
    -3.0,    # cart velocity
    -0.418,  # pole angle
    -3.5     # pole angular velocity
])

high = np.array([
     4.8,
     3.0,
     0.418,
     3.5
])


bins = [
    np.linspace(low[i], high[i], n_bins[i] + 1)
    for i in range(len(low))
]


def discretize(obs):

    state = []

    for i in range(len(obs)):

        bin_index = np.digitize(obs[i], bins[i]) - 1

        # Keep index inside 0 ... n_bins-1
        bin_index = np.clip(
            bin_index,
            0,
            n_bins[i] - 1
        )

        state.append(bin_index)

    return tuple(state)


# --------------------------------------------------
# Q-learning parameters
# --------------------------------------------------

alpha = 0.1
gamma = 0.99 #cartpole needs to be balanced for longer time - therefore future rewards matter

epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01

episodes = 10_000


# --------------------------------------------------
# Training
# --------------------------------------------------

for episode in range(episodes):

    if episode % 100 == 0:
        print(episode)

    observation, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        state = discretize(observation)

        # Exploration vs exploitation
        if np.random.random() < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(
                Q_table[state]
            )


        # Take action
        new_observation, reward, terminated, truncated, info = env.step(action)

        new_state = discretize(new_observation)


        # Q-learning target
        if terminated:

            target = reward

        else:

            target = (
                reward
                + gamma * np.max(Q_table[new_state])
            )


        # Q-learning update
        Q_table[state + (action,)] += (
            alpha
            * (
                target
                - Q_table[state + (action,)]
            )
        )


        observation = new_observation


    # Reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


env.close()


print(Q_table)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

eval_episodes = 10

for episode in range(eval_episodes):

    observation, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0

    while not (terminated or truncated):

        state = discretize(observation)

        # No exploration during evaluation
        action = np.argmax(Q_table[state])

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

    print(
        f"Evaluation episode {episode + 1}: "
        f"reward = {total_reward}"
    )


env.close()