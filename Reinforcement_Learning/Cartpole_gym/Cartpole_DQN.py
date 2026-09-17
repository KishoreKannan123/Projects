import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


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

env = gym.make("CartPole-v1")


# ============================================================
# Q Network
# ============================================================

class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.ReLU(),

            nn.Linear(100, 2)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Networks
# ============================================================

Q_model = QNetwork().to(device)
target_model = QNetwork().to(device)

# Target starts identical to Q network
target_model.load_state_dict(Q_model.state_dict())
target_model.eval()


# ============================================================
# Loss and optimizer
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

epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01

episodes = 600 # Reduced from 10000 since CartPole solves rapidly

batch_size = 64
replay_buffer_size = 50_000

# Changed to update every 1000 STEPS rather than episodes for stability
target_update_frequency = 1000
learning_starts = 1000


# ============================================================
# Replay Buffer
# ============================================================

replay_buffer = deque(
    maxlen=replay_buffer_size
)


# ============================================================
# Training
# ============================================================

episode_rewards = []
total_steps = 0
total_loss = 0
loss_count = 0

for episode in range(episodes):

    observation, info = env.reset(
        seed=SEED + episode
    )

    terminated = False
    truncated = False
    episode_reward = 0

    while not (terminated or truncated):

        # Convert state to tensor
        state = torch.tensor(
            observation,
            dtype=torch.float32,
            device=device
        )

        # ----------------------------------------------------
        # Epsilon-greedy action (Switched to standard random)
        # ----------------------------------------------------
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                # QNetwork expects a batch dimension, use unsqueeze(0)
                q_values = Q_model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()

        # Environment step
        new_observation, reward, terminated, truncated, info = env.step(action)

        # Store transition
        replay_buffer.append(
            (
                observation,
                action,
                reward,
                new_observation,
                terminated
            )
        )

        observation = new_observation
        episode_reward += reward
        total_steps += 1

        # ----------------------------------------------------
        # Epsilon decay (Bypassed to step-level for stable decay)
        # ----------------------------------------------------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Start learning after buffer has enough samples
        if (
            total_steps >= learning_starts
            and len(replay_buffer) >= batch_size
        ):

            # Sample random batch
            batch = random.sample(replay_buffer, batch_size)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

            # Convert batch to tensors
            states_b = torch.tensor(np.array(states_b), dtype=torch.float32, device=device)
            actions_b = torch.tensor(actions_b, dtype=torch.long, device=device).unsqueeze(1)
            rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=device).unsqueeze(1)
            next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32, device=device)
            dones_b = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)

            # Current Q(s,a)
            current_q = Q_model(states_b).gather(1, actions_b)

            # Target Q value (Strict Shape Keeping)
            with torch.no_grad():
                max_next_q = target_model(next_states_b).max(dim=1, keepdim=True).values
                target = rewards_b + (gamma * max_next_q * (1 - dones_b))

            # Calculate loss (Huber Loss / SmoothL1Loss)
            loss = loss_fn(current_q, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_count += 1

            # Target network update (1000 step frequency constraint)
            if total_steps % target_update_frequency == 0:
                target_model.load_state_dict(Q_model.state_dict())

    # ========================================================
    # End of episode
    # ========================================================
    episode_rewards.append(episode_reward)

    # Logging
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        avg_loss = (total_loss / loss_count) if loss_count > 0 else 0

        print(
            f"Episode: {episode + 1:4d} | "
            f"Avg Reward: {avg_reward:6.1f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Avg Loss: {avg_loss:.5f} | "
            f"Buffer: {len(replay_buffer)}"
        )

        total_loss = 0
        loss_count = 0
        
    # Early solve stop flag
    if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475.0:
        print(f"\nEnvironment solved early at episode {episode + 1}!")
        break

env.close()


# ============================================================
# Evaluation
# ============================================================
print("\nStarting evaluation...\n")
eval_env = gym.make("CartPole-v1", render_mode="human")
Q_model.eval()

eval_episodes = 10
evaluation_rewards = []

for episode in range(eval_episodes):
    observation, info = eval_env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        state = torch.tensor(observation, dtype=torch.float32, device=device)

        with torch.no_grad():
            q_values = Q_model(state)
            action = torch.argmax(q_values).item()

        new_observation, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        observation = new_observation

    evaluation_rewards.append(total_reward)
    print(f"Evaluation episode {episode + 1}: reward = {total_reward}")

# ============================================================
# Evaluation summary
# ============================================================
print("\nAverage evaluation reward:", np.mean(evaluation_rewards))
print("Maximum evaluation reward:", np.max(evaluation_rewards))
eval_env.close()
