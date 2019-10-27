import concurrent.futures
from typing import List, Tuple

import gym
import numpy as np
import torch
from torch import multiprocessing

from push_ups import agent_base


def train_agent(
    agent: agent_base.Agent,
    env_name: str,
    steps: int,
    bath_size: int,
    stats: bool = True,
    stats_frequency: int = 1,
) -> agent_base.Agent:

    for i in range(steps):
        agent.switch_to_cpu()
        # profiler.check_to_cpu(agent.switch_to_cpu)

        actions, observations, rewards = run_simulations_on_all_cores(
            agent, env_name, batch_size
        )
        # actions, observations, rewards = profiler.check_collect_data(
        #     run_simulations_on_all_cores, agent, env_name, batch_size
        # )
        agent.switch_to_gpu()
        # profiler.check_to_gpu(agent.switch_to_gpu)

        agent.update_policy(actions, observations, rewards)
        # profiler.check_update(agent.update_policy, actions, observations, rewards)
        if stats and i % stats_frequency == 0:
            print_stats(rewards, i)

    return agent


def run_simulations_on_all_cores(
    agent: agent_base.Agent, env_name: str, batch_size: int
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    """
    Run multiple simulations using the same agent on multiple cores to obtaine actions,
    observations, and rewards.
    """

    cores = get_cores()
    steps_per_core = batch_size // cores
    # cores = 1

    actions = []
    observations = []
    rewards = []

    with multiprocessing.Pool(cores) as p:
        collected_data = p.map(
            collect_actions_observations_rewards,
            [(agent, env_name, steps_per_core)] * cores,
        )

        for acts, obs, rews in collected_data:
            # print(type(acts), len(acts))
            actions += acts
            observations += obs
            rewards += rews

    # acts, obs, rews = collect_actions_observations_rewards(
    #     (agent, env_name, steps_per_core)
    # )

    # acts, obs, rews = profiler.check_run_single_collect_data(
    #     collect_actions_observations_rewards, (agent, env_name, steps_per_core))

    # actions += acts
    # observations += obs
    # rewards += rews

    return actions, observations, rewards


def collect_actions_observations_rewards(
    data: Tuple
    # agent: agent_base.Agent, env_name: str, timesteps: int
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    np.random.RandomState().uniform()  # add randomness
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
    Perform one simulation (episode) to collect data for the policy update.
    You will collect data  until you reach timesteps limit.
    Return actions tensor, observations tensor, rewards tensor and steps count.
    """
    obs = env.reset()

    actions = torch.Tensor(0, 1).float()
    # actions = []
    observations = torch.tensor(obs).view(-1, len(obs)).float()
    rewards = torch.Tensor(0, 1).float()

    while True:
        i += 1
        action = agent(obs)

        obs, reward, done, _ = env.step(action)

        # actions.append(action)
        actions = torch.cat([actions, torch.tensor(action).view(1, 1).float()])
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
from typing import Any, Tuple, Union

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
        # self.gpu = torch.device("cpu")
        self.cpu = torch.device("cpu")

        super().__init__(env, gamma, lr, *args, **kwargs)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def switch_to_cpu(self):
        self.policy = self.policy.to(self.cpu)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)

    def switch_to_gpu(self):
        self.policy = self.policy.to(self.gpu)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)

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

    def get_action_log_probabilty(
        self, observations: torch.tensor, actions: torch.tensor
    ) -> Any:  # TODO add return type
        if not self.discrete_actions:
            raise NotImplementedError

        # action = torch.tensor(action)
        probabilities = self.policy(observations)

        distribution = Categorical(probabilities)
        log_prob = distribution.log_prob(actions.T).view(-1, 1)
        return log_prob

    def get_policy(self) -> nn.Module:
        inputs_no = self.get_observations()
        outputs_no = self.get_actions()
        discrete_outputs = self.discrete_actions
        return default_network.SimpleNet(inputs_no, outputs_no, discrete_outputs)

    
    def get_loss(self, dataset):
        act, ret, obs = dataset.get_data()
        log_prob = self.get_action_log_probabilty(obs, act)

        loss = -log_prob * ret
        loss = loss.sum()
        return loss

    def update_policy(self, actions, observations, rewards, *args, **kwargs):
        del args, kwargs  # unused

        dataset = Batch(actions, observations, rewards, self.gamma, self.gpu)
        # dataset = profiler.check_batch(
        #     Batch, actions, observations, rewards, self.gamma, self.gpu
        # )

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        # self.policy = self.policy.to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)  # TODO learning rate

        loss = self.get_loss(dataset)
        # loss = profiler.check_loss(get_loss, dataset)

        self.optimizer.zero_grad()
        loss.backward()
        # profiler.check_backward(loss.backward)

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
        actions: List[torch.tensor],
        observations: List[torch.tensor],
        rewards: List[torch.tensor],
        gamma: float = 0.9,
        gpu: Union[torch.device, Any] = None,
    ):
        self.gamma = gamma
        self.actions = self.get_actions(actions)
        self.returns = self.get_returns(rewards)
        self.observations = self.get_observations(observations)
        if gpu:
            self.actions = self.actions.to(gpu)
            self.returns = self.returns.to(gpu)
            self.observations = self.observations.to(gpu)

        self.randomize_rows()

        # profiler.get_device(gpu)

        assert self.actions.shape[0] == self.returns.shape[0]
        assert self.returns.shape[0] == self.observations.shape[0]

    def __len__(self):
        return self.returns.shape[0]

    def __getitem__(self, idx):  # TODO add return
        act = self.actions[idx]
        ret = self.returns[idx]
        obs = self.observations[idx]

        return act, ret, obs

    def get_data(self):  # TODO add return
        return self.actions, self.returns, self.observations

    def randomize_rows(self):
        indicies = np.random.permutation(self.actions.shape[0])
        self.actions = self.actions[indicies]
        self.returns = self.returns[indicies]
        self.observations = self.observations[indicies]

    @staticmethod
    def get_actions(actions: List[torch.tensor]) -> torch.tensor:
        """ Version for log probabilities of taking discrete action """
        actions_all = torch.tensor([]).view(0, 1).float()
        for episode_actions in actions:
            actions_all = torch.cat(
                [actions_all, episode_actions]
            )
            # for timestep_action in episode_actions:

            #     actions_all = torch.cat(
            #         [actions_all, torch.tensor(timestep_action).view(-1, 1).float()]
            #     )

        return actions_all

    def get_returns(self, rewards: List[torch.tensor]) -> torch.tensor:
        """
        Calculate returns based on the rewards and concat all tensors into one.
        """
        returns = [self.calculate_epiosde_returns(ep_rewards) for ep_rewards in rewards]
        returns_all = torch.tensor([]).view(0, 1).float()

        for ep_returns in returns:
            returns_all = torch.cat([returns_all, ep_returns.view(-1, 1).float()])

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
    from push_ups.utils.profiler import Profiler
    prof = Profiler()
    # import time

    # start = time.perf_counter()
    run_simulations_on_all_cores = prof.check("run_simulations_on", run_simulations_on_all_cores)
    Batch = prof.check("Batch", Batch)
    PolicyAgent.get_loss = prof.check("get_loss", PolicyAgent.get_loss)


    env_name = "CartPole-v1"
    env = gym.make(env_name)

    agent = PolicyAgent(env)
    batch_size = 10_000
    steps = 10


    

    # run_simulations_on_all_cores(agent, env_name, batch_size)

    # finish = time.perf_counter()

    # print(f"Finished in {round(finish-start, 2)} seconds")

    train_agent(agent, env_name, steps, batch_size, stats=False)
    # profiler.print()
    prof.print()