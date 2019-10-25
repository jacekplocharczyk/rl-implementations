import concurrent.futures
import multiprocessing
from typing import List, Tuple

import gym
import numpy as np
import torch

from push_ups import agent_base


def train_agent(
    agent: agent_base.Agent,
    env_name: str,
    steps: int,
    bath_size: int,
    stats: bool = True,
) -> agent_base.Agent:

    for i in range(steps):
        agent.switch_to_cpu()

        actions, observations, rewards = run_simulations_on_all_cores(
            agent, env_name, batch_size
        )
        agent.switch_to_gpu()

        agent.update_policy(actions, observations, rewards)
        print_stats(rewards, i)

    return agent


def run_simulations_on_all_cores(
    agent: agent_base.Agent, env_name: str, batch_size: int
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    """
    Run multiple simulations using the same agent on multiple cores to obtaine actions,
    observations, and rewards.
    """

    cores = get_cores() * 8  # multiple processes per core works better
    steps_per_core = batch_size // cores

    actions = []
    observations = []
    rewards = []

    with concurrent.futures.ProcessPoolExecutor() as executor:

        results = [
            executor.submit(
                collect_actions_observations_rewards, agent, env_name, steps_per_core
            )
            for _ in range(cores)
        ]

        for f in concurrent.futures.as_completed(results):
            acts, obs, rews = f.result()

            actions += acts
            observations += obs
            rewards += rews
    # acts, obs, rews = collect_actions_observations_rewards(
    #     agent, env_name, steps_per_core
    # )
    # actions += acts
    # observations += obs
    # rewards += rews

    return actions, observations, rewards


def collect_actions_observations_rewards(
    agent: agent_base.Agent, env_name: str, timesteps: int
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    np.random.RandomState().uniform()  # add randomness

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
) -> Tuple[list, torch.Tensor, torch.Tensor, int]:
    """
    Perform one simulation (episode) to collect data for the policy update.
    You will collect data  until you reach timesteps limit.
    Return actions list, observations tensor, rewards tensor and steps count.
    """
    obs = env.reset()

    actions = []
    observations = torch.tensor(obs).view(-1, len(obs)).float()
    rewards = torch.Tensor(0, 1).float()

    while True:
        i += 1
        action = agent(obs)
        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards = torch.cat([rewards, torch.tensor(reward).view(1, 1).float()])

        if done or i >= timesteps:
            break

        observations = torch.cat(
            [observations, torch.tensor(obs).view(-1, len(obs)).float()]
        )

    return actions, observations, rewards, i


def get_cores() -> int:
    """ Get numbers of available cores """
    return multiprocessing.cpu_count()


def print_stats(rewards: List[torch.tensor], i: int) -> None:
    scores = np.array([float(ep_reward.sum()) for ep_reward in rewards])
    print(f"Step {i:5} Mean score: {scores.mean()} Std: {scores.std()}")


##############################################
from push_ups.utils.default_network import SimpleNet
from torch.distributions import Categorical
from push_ups.utils import default_network
import sys
from typing import Any, Tuple

import gym
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class PolicyAgent(agent_base.Agent):
    def __init__(self, env: gym.core.Env, gamma: float = 0.9, lr=0.01, *args, **kwargs):
        """
        :param: gamma: discount factor used to calculate return
        :param: lr: learning rate used in the torch optimizer
        """
        # TODO add logs if cuda is unavailable
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        super().__init__(env, gamma, lr, *args, **kwargs)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def switch_to_cpu(self):
        self.policy.to(self.cpu)

    def switch_to_gpu(self):
        self.policy.to(self.gpu)

    def take_action(self, observation: np.array, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Take action based on observation and return its computional form
        (e.g. action log probability)
        """
        del args, kwargs  # unused
        if not self.discrete_actions:
            raise NotImplementedError

        observation = torch.from_numpy(observation).float().unsqueeze(0)

        with torch.no_grad():
            probabilities = self.policy(observation)

        m = Categorical(probabilities)
        action = m.sample()

        return action.item()

    def get_action_log_probabilty(self, observation: torch.tensor, action: int) -> Any:  # TODO add return type
        if not self.discrete_actions:
            raise NotImplementedError

        observation = observation.unsqueeze(0).to(self.gpu)
        action = torch.tensor(action).to(self.gpu)
        probabilities = self.policy(observation)

        distribution = Categorical(probabilities)
        log_prob = distribution.log_prob(action)
        return log_prob

    def get_policy(self) -> nn.Module:
        inputs_no = self.get_observations()
        outputs_no = self.get_actions()
        discrete_outputs = self.discrete_actions
        return default_network.SimpleNet(inputs_no, outputs_no, discrete_outputs)

    def update_policy(self, actions, observations, rewards, *args, **kwargs):
        del args, kwargs  # unused
        dataset = Batch(actions, observations, rewards, self.gamma)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        # self.policy = self.policy.to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)  # TODO learning rate
        loss = torch.tensor([]).float().to(self.gpu)
        for act, ret, obs in dataset:
            log_prob = self.get_action_log_probabilty(obs, act)
           
            ep_loss = -log_prob * ret.to(self.gpu)
            loss = torch.cat([loss, ep_loss])
        
        loss = loss.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # cpu_device = torch.device("cpu")
        # self.policy = self.policy.to(cpu_device)

    def calculate_returns(self, rewards: torch.tensor) -> torch.tensor:
        returns = torch.flip(rewards, [0])
        for idx, item in enumerate(returns):
            if idx == 0:
                continue
            returns[idx] = item + self.gamma * returns[idx - 1]
        return torch.flip(returns, [0])


class Batch(torch.utils.data.Dataset):
    def __init__(
        self,
        actions: List[List[torch.tensor]],
        observations: List[torch.tensor],
        rewards: List[torch.tensor],
        gamma: float = 0.9,
    ):
        self.gamma = gamma
        self.actions = self.get_actions(actions)
        self.returns = self.get_returns(rewards)
        self.observations = self.get_observations(observations)
        self.indices = np.random.permutation(len(self.actions))

        assert len(self.actions) == self.returns.shape[0]
        assert self.returns.shape[0] == self.observations.shape[0]

    def __len__(self):
        return self.returns.shape[0]

    def __getitem__(self, idx):
        rand_index = self.indices[idx]
        act = self.actions[rand_index]
        ret = self.returns[rand_index]
        obs = self.observations[rand_index]

        return act, ret, obs

    @staticmethod
    def get_actions(actions: List[List[torch.tensor]]) -> List[torch.tensor]:
        """ Version for log probabilities of taking discrete action """
        a = []
        for episode in actions:
            for timestep_action in episode:
                a.append(timestep_action)

        return a

    def get_returns(self, rewards: List[torch.tensor]) -> torch.tensor:
        """
        Calculate returns based on the rewards and concat all tensors into one.
        """
        returns = [self.calculate_epiosde_returns(ep_rewards) for ep_rewards in rewards]
        returns_all = torch.tensor([]).view(0, 1).float()

        for ep_returns in returns:
            returns_all = torch.cat(
                [returns_all, ep_returns.view(-1, 1).float()]
            )

        # normalization
        eps = np.finfo(np.float32).eps.item()
        returns_all = (returns_all - returns_all.mean()) / (returns_all.std() + eps)

        return returns_all

    @staticmethod
    def get_observations(observations: List[torch.tensor]) -> torch.tensor:
        assert len(observations) > 0
        observation_dim = observations[0].shape[1]
        obs_tensor = torch.tensor([]).view(-1, observation_dim).float()
        for ep_observations in observations:
            obs_tensor = torch.cat([obs_tensor, ep_observations])

        return obs_tensor

    def calculate_epiosde_returns(self, rewards: torch.tensor) -> torch.tensor:
        returns = torch.flip(rewards, [0])
        for idx, item in enumerate(returns):
            if idx == 0:
                continue
            returns[idx] = item + self.gamma * returns[idx - 1]
        return torch.flip(returns, [0])


##############################################

if __name__ == "__main__":
    # import time

    # start = time.perf_counter()

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    agent = PolicyAgent(env)
    batch_size = 20000
    steps = 20

    # run_simulations_on_all_cores(agent, env_name, batch_size)

    # finish = time.perf_counter()

    # print(f"Finished in {round(finish-start, 2)} seconds")

    train_agent(agent, env_name, steps, batch_size)
