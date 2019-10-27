from typing import Tuple

import gym
import torch

from push_ups import agent_base


def run_episode(
    agent: agent_base.Agent, env: gym.Env
) -> Tuple[agent_base.Agent, torch.Tensor]:
    """
    Perform one simulation (episode) to collect data for the policy update.
    """
    obs = env.reset()
    rewards = torch.Tensor(0, 1).float()

    for t in range(1, 10000):  # Don't infinite loop while learning
        action = agent(obs)
        obs, reward, done, _ = env.step(action)
        rewards = torch.cat([rewards, torch.tensor(reward).view(1, 1)])

        if done:
            break

    return agent, rewards


def print_epiode_stats(
    i_episode: int,
    epiosde_reward: float,
    running_reward: float,
    env: gym.Env,
    rewards: torch.Tensor,
):
    if i_episode % 50 == 0:
        print(
            "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\t".format(
                i_episode, epiosde_reward, running_reward
            )
        )

    if running_reward > env.spec.reward_threshold:
        t = rewards.size()[0]
        print(
            "Solved - epiosde {}! Running reward is now {} and "
            "the last episode runs to {} time steps!".format(
                i_episode, running_reward, t
            )
        )
