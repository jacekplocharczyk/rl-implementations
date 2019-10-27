from typing import List, Tuple

import gym
import numpy as np
import torch
from torch import multiprocessing

from push_ups import agent_base


def train_agent(
    agent: agent_base.Agent,
    env_name: str,
    epochs: int,
    batch_size: int,
    stats: bool = True,
    stats_frequency: int = 1,
) -> agent_base.Agent:
    """
    Loop for collecting data from the environment using agent for n epochs and updating
    the agent after each epoch.
    """
    record_mean_reward = -1e100
    for i in range(epochs):
        agent.switch_to_cpu()
        actions, observations, rewards = run_simulations_on_all_cores(
            agent, env_name, batch_size
        )

        agent.switch_to_gpu()
        agent.update_policy(actions, observations, rewards)

        finished, record_mean_reward = check_criterium_and_print_stats(
            agent.env, rewards, i, stats, stats_frequency, record_mean_reward
        )

        if finished:
            break

    return agent


def run_simulations_on_all_cores(
    agent: agent_base.Agent, env_name: str, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run multiple simulations using the same agent on multiple cores to obtaine actions,
    observations, and rewards.
    """

    cores = get_cores()
    steps_per_core = batch_size // cores

    actions = []
    observations = []
    rewards = []

    with multiprocessing.Pool(cores) as p:
        collected_data = p.map(
            collect_actions_observations_rewards,
            [(agent, env_name, steps_per_core)] * cores,
        )

        for acts, obs, rews in collected_data:
            actions += acts
            observations += obs
            rewards += rews

    return actions, observations, rewards


def collect_actions_observations_rewards(
    data: Tuple[agent_base.Agent, str, int]
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    """
    Run multiple simulations on a single core.
    """
    np.random.RandomState().uniform()  # add randomness  # TODO check randomness
    agent, env_name, timesteps = data

    env = gym.make(env_name)
    i = 0

    actions = []
    observations = []
    rewards = []

    while i < timesteps:
        acts, obs, rews, i = run_episode(agent, env, i, timesteps)

        actions.append(acts)
        observations.append(obs)
        rewards.append(rews)

    return actions, observations, rewards


def run_episode(
    agent: agent_base.Agent, env: gym.Env, i: int, timesteps: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Perform one simulation (episode) to collect taken actions, observations and rewards
    for given agent. It collects data until reaches timesteps limit.
    Return actions tensor, observations tensor, rewards tensor and steps count.
    """
    obs = env.reset()
    # TODO do it using Action/Observation class
    actions = torch.Tensor(0, 1).float()
    observations = torch.tensor(obs).view(-1, len(obs)).float()
    rewards = torch.Tensor(0, 1).float()

    while True:
        i += 1
        action = agent(obs)

        obs, reward, done, _ = env.step(action)
        actions = torch.cat([actions, torch.tensor(action).view(1, 1).float()])
        rewards = torch.cat([rewards, torch.tensor(reward).view(1, 1).float()])

        if done:  # TODO or i >= timesteps for cases not terminating
            break

        observations = torch.cat(
            [observations, torch.tensor(obs).view(-1, len(obs)).float()]
        )

    return actions, observations, rewards, i


def get_cores() -> int:
    """
    Get numbers of available cores.
    """
    return multiprocessing.cpu_count()


def check_criterium_and_print_stats(
    env: gym.core.Env,
    rewards: List[torch.tensor],
    i: int,
    stats: bool,
    stats_frequency: int,
    record_mean_reward: float,
) -> Tuple[bool, float]:
    scores = np.array([float(ep_reward.sum()) for ep_reward in rewards])
    mean = scores.mean()

    if mean > record_mean_reward:
        record_mean_reward = mean

    if mean > env.spec.reward_threshold:
        print(f"Problem solved at {i:5} step with the mean score of {mean}")
        return True
    elif stats and i % stats_frequency == 0:
        print(
            f"Step {i:>5}\tMean score: {mean:>10.8}\tStd: {scores.std():>10.8}\tCurrent max: {record_mean_reward:>10.8}"
        )
    return False, record_mean_reward
