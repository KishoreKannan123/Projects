"""
Q-Learning on FrozenLake from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Q-Learning on FrozenLake: train a tabular agent and evaluate its greedy policy."""
import numpy as np
import gymnasium as gym

from solution import (
    init_q_table,
    max_q_value,
    greedy_action,
    sample_random_action,
    should_explore,
    epsilon_greedy_action,
    decay_epsilon,
    td_target,
    td_error,
    q_learning_update,
    interaction_step,
    run_training_episode,
    train_q_learning,
    extract_greedy_policy,
    run_greedy_episode,
    evaluate_success_rate,
)


if __name__ == "__main__":
    np.random.seed(0)

    # Build a non-slippery FrozenLake for faster, more reliable learning.
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.action_space.seed(0)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"FrozenLake: {num_states} states, {num_actions} actions")

    # Train the tabular Q-learning agent.
    q_table, reward_history = train_q_learning(
        env,
        num_episodes=2000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=0,
        max_steps=200,
    )
    print(f"Q-table shape: {q_table.shape}")
    print(f"Reward history length: {len(reward_history)}")
    early_avg = float(np.mean(reward_history[:100]))
    late_avg = float(np.mean(reward_history[-100:]))
    print(f"Mean reward first 100 episodes: {early_avg:.3f}")
    print(f"Mean reward last 100 episodes:  {late_avg:.3f}")

    # Extract greedy policy and inspect a couple of Q-values.
    policy = extract_greedy_policy(q_table)
    print(f"Greedy policy (first 8 states): {policy[:8].tolist()}")
    print(f"Greedy action at state 0: {greedy_action(q_table, 0)}")
    print(f"Max Q-value at state 0: {max_q_value(q_table, 0):.4f}")

    # Run one greedy episode and report success.
    reached_goal = run_greedy_episode(env, policy, seed=0, max_steps=200)
    print(f"Single greedy episode reached goal: {bool(reached_goal)}")

    # Evaluate success rate over many greedy episodes.
    success_rate = evaluate_success_rate(env, policy, num_episodes=100, seed=0, max_steps=200)
    print(f"Greedy success rate over 100 episodes: {success_rate:.2f}")

    env.close()
