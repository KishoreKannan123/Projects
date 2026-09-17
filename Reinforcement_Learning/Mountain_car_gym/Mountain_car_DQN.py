
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

#To do - Reduce no. of time steps it takes to reach goal during eval

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Environment
# ============================================================

environment = "MountainCar-v0"

env = gym.make(environment)

# Seed action space as well
env.action_space.seed(SEED)


# ============================================================
# Q Network
# ============================================================

class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.ReLU(),

            nn.Linear(100, env.action_space.n)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Networks
# ============================================================

Q_model = QNetwork().to(device)

target_model = QNetwork().to(device)

# Target network starts identical to Q network
target_model.load_state_dict(Q_model.state_dict())

target_model.eval()


# ============================================================
# Loss and Optimizer
# ============================================================

loss_fn = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(
    Q_model.parameters(),
    lr=0.001
)


# ============================================================
# Hyperparameters
# ============================================================

gamma = 0.99

# Epsilon-greedy exploration
epsilon_start = 1.0
epsilon_min = 0.05

# Number of environment steps over which
# epsilon decreases from 1.0 -> 0.05
epsilon_decay_steps = 50_000

epsilon = epsilon_start

# Training
episodes = 1000

# Replay buffer
batch_size = 64
replay_buffer_size = 50_000

# Start training after collecting this many transitions
learning_starts = 1_000

# Hard target-network update
target_update_frequency = 1_000


# ============================================================
# Replay Buffer
# ============================================================

replay_buffer = deque(
    maxlen=replay_buffer_size
)


# ============================================================
# Training Statistics
# ============================================================

episode_rewards = []

total_steps = 0

total_loss = 0.0
loss_count = 0


# ============================================================
# Training
# ============================================================

for episode in range(episodes):

    observation, info = env.reset(
        seed=SEED + episode
    )

    terminated = False
    truncated = False

    episode_reward = 0

    while not (terminated or truncated):

        # ----------------------------------------------------
        # Convert state to tensor
        # ----------------------------------------------------

        state = torch.tensor(
            observation,
            dtype=torch.float32,
            device=device
        )


        # ----------------------------------------------------
        # Epsilon-Greedy Action
        # ----------------------------------------------------

        if random.random() < epsilon:

            # Explore
            action = env.action_space.sample()

        else:

            # Exploit
            with torch.no_grad():

                q_values = Q_model(
                    state.unsqueeze(0)
                )

                action = torch.argmax(
                    q_values,
                    dim=1
                ).item()


        # ----------------------------------------------------
        # Environment Step
        # ----------------------------------------------------

        new_observation, reward, terminated, truncated, info = env.step(
            action
        )
        reward = new_observation[0]

        if terminated:
            reward += 10


        # ----------------------------------------------------
        # Store Transition
        # ----------------------------------------------------

        replay_buffer.append(
            (
                observation,
                action,
                reward,
                new_observation,
                terminated
            )
        )


        # Update state
        observation = new_observation

        episode_reward += reward

        total_steps += 1


        # ----------------------------------------------------
        # Epsilon Decay
        # ----------------------------------------------------

        epsilon = max(
            epsilon_min,

            epsilon_start
            - (
                (epsilon_start - epsilon_min)
                * total_steps
                / epsilon_decay_steps
            )
        )


        # ----------------------------------------------------
        # Start Learning
        # ----------------------------------------------------

        if (
            total_steps >= learning_starts
            and len(replay_buffer) >= batch_size
        ):

            # ------------------------------------------------
            # Sample Random Batch
            # ------------------------------------------------

            batch = random.sample(
                replay_buffer,
                batch_size
            )

            (
                states_b,
                actions_b,
                rewards_b,
                next_states_b,
                dones_b
            ) = zip(*batch)


            # ------------------------------------------------
            # Convert Batch to Tensors
            # ------------------------------------------------

            states_b = torch.tensor(
                np.array(states_b),
                dtype=torch.float32,
                device=device
            )

            actions_b = torch.tensor(
                actions_b,
                dtype=torch.long,
                device=device
            ).unsqueeze(1)

            rewards_b = torch.tensor(
                rewards_b,
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)

            next_states_b = torch.tensor(
                np.array(next_states_b),
                dtype=torch.float32,
                device=device
            )

            dones_b = torch.tensor(
                dones_b,
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)


            # ------------------------------------------------
            # Current Q(s,a)
            # ------------------------------------------------

            current_q = Q_model(
                states_b
            ).gather(
                1,
                actions_b
            )


            # ------------------------------------------------
            # Target Q Value
            # ------------------------------------------------

            with torch.no_grad():

                max_next_q = target_model(
                    next_states_b
                ).max(
                    dim=1,
                    keepdim=True
                ).values


                target = (
                    rewards_b
                    + gamma
                    * max_next_q
                    * (1 - dones_b)
                )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = loss_fn(
                current_q,
                target
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_loss += loss.item()

            loss_count += 1


            # ------------------------------------------------
            # Target Network Update
            # ------------------------------------------------

            if total_steps % target_update_frequency == 0:

                target_model.load_state_dict(
                    Q_model.state_dict()
                )


    # ========================================================
    # End of Episode
    # ========================================================

    episode_rewards.append(
        episode_reward
    )


    # ========================================================
    # Logging
    # ========================================================

    if (episode + 1) % 50 == 0:

        avg_reward = np.mean(
            episode_rewards[-50:]
        )

        avg_loss = (
            total_loss / loss_count
            if loss_count > 0
            else 0
        )

        print(
            f"Episode: {episode + 1:4d} | "
            f"Avg Reward: {avg_reward:7.1f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Avg Loss: {avg_loss:.5f} | "
            f"Buffer: {len(replay_buffer)}"
        )

        total_loss = 0.0
        loss_count = 0


    # ========================================================
    # MountainCar Early Stop
    # ========================================================

    if (
        len(episode_rewards) >= 100
        and np.mean(episode_rewards[-100:]) >= -60
    ):

        print(
            f"\nEnvironment solved early "
            f"at episode {episode + 1}!"
        )

        break


# ============================================================
# Close Training Environment
# ============================================================

env.close()


# ============================================================
# Evaluation
# ============================================================

print("\nStarting evaluation...\n")


eval_env = gym.make(
    environment,
    render_mode="human"
)

Q_model.eval()


eval_episodes = 10

evaluation_rewards = []


for episode in range(eval_episodes):

    observation, info = eval_env.reset(
        seed=SEED + 10_000 + episode
    )

    terminated = False
    truncated = False

    total_reward = 0


    while not (terminated or truncated):

        # ----------------------------------------------------
        # Convert state to tensor
        # ----------------------------------------------------

        state = torch.tensor(
            observation,
            dtype=torch.float32,
            device=device
        )


        # ----------------------------------------------------
        # Greedy Action
        # ----------------------------------------------------

        with torch.no_grad():

            q_values = Q_model(
                state.unsqueeze(0)
            )

            action = torch.argmax(
                q_values,
                dim=1
            ).item()


        # ----------------------------------------------------
        # Environment Step
        # ----------------------------------------------------

        observation, reward, terminated, truncated, info = (
            eval_env.step(action)
        )

        total_reward += reward


    # --------------------------------------------------------
    # Store Evaluation Result
    # --------------------------------------------------------

    evaluation_rewards.append(
        total_reward
    )

    print(
        f"Evaluation episode {episode + 1}: "
        f"reward = {total_reward}"
    )


# ============================================================
# Evaluation Summary
# ============================================================

print(
    "\nAverage evaluation reward:",
    np.mean(evaluation_rewards)   
)

print(
    "Maximum evaluation reward:",
    np.max(evaluation_rewards)
)


# ============================================================
# Close Evaluation Environment
# ============================================================

eval_env.close()