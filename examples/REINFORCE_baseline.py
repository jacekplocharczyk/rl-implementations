from torch.distributions import Categorical
from push_ups.utils import default_network
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn

from push_ups import framework
from push_ups import agent_base
from examples.batches import return_batch


class PolicyAgent(agent_base.Agent):
    def __init__(self, env: gym.core.Env, gamma: float = 0.9, lr=0.003, reward_normalization: bool = False, *args, **kwargs):
        """
        :param: gamma: discount factor used to calculate return
        :param: lr: learning rate used in the torch optimizer
        """
        # TODO add logs if cuda is unavailable
        # TODO perserve optimizer status when coping between cpu and gpu
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.lr = lr
        self.reward_normalization = reward_normalization

        super().__init__(env, gamma, lr, *args, **kwargs)
        self.policy = self.policy.to(self.gpu)
        self.baseline = self.get_baseline().to(self.gpu)
        self.baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(), lr=self.lr)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        torch.save(self.optimizer.state_dict(), 'saved_optimizer_state.pt')
        torch.save(self.baseline_optimizer.state_dict(), 'saved_baseline_state.pt')

    def switch_to_cpu(self):
        # self.optimizer_status = self.optimizer.state_dict()
        self.policy = self.policy.to(self.cpu)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.baseline = self.get_baseline().to(self.cpu)
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=self.lr)
        # self.optimizer.load_state_dict(self.optimizer_status)

    def switch_to_gpu(self):
        # self.optimizer_status = self.optimizer.state_dict()
        self.policy = self.policy.to(self.gpu)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.baseline = self.get_baseline().to(self.gpu)
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=self.lr)
        # self.optimizer.load_state_dict(self.optimizer_status)

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
    ) -> torch.tensor:
        if not self.discrete_actions:
            raise NotImplementedError

        probabilities = self.policy(observations)

        distribution = Categorical(probabilities)
        log_prob = distribution.log_prob(actions.T).view(-1, 1)
        return log_prob

    def get_policy(self) -> nn.Module:
        inputs_no = self.get_observations()
        outputs_no = self.get_actions()
        discrete_outputs = self.discrete_actions
        return default_network.SimpleNet(inputs_no, outputs_no, discrete_outputs)

    def get_baseline(self) -> nn.Module:
        inputs_no = self.get_observations()
        return default_network.SimpleNet(inputs_no, 1, False)

    def get_loss(self, dataset):
        act, ret, obs = dataset.get_data()
        log_prob = self.get_action_log_probabilty(obs, act)

        delta = ret - self.baseline(obs)

        # print(delta.mean().item())

        loss = -log_prob * delta
        loss = loss.sum()
        return loss

    def update_policy(self, actions, observations, rewards, *args, **kwargs):
        del args, kwargs  # unused

        dataset = return_batch.DiscountedReturn(
            actions,
            observations,
            rewards,
            gamma=self.gamma,
            device=self.gpu,
            normalization=self.reward_normalization,
        )

        loss = self.get_loss(dataset)
        self.optimizer.load_state_dict(torch.load('saved_optimizer_state.pt', map_location="cuda:0"))
        self.baseline_optimizer.load_state_dict(torch.load('saved_baseline_state.pt', map_location="cuda:0"))

        # self.optimizer.load_state_dict(self.optimizer_status)
        self.optimizer.zero_grad()
        self.baseline_optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        self.baseline_optimizer.step()

        torch.save(self.optimizer.state_dict(), 'saved_optimizer_state.pt')
        torch.save(self.baseline_optimizer.state_dict(), 'saved_baseline_state.pt')


##############################################


if __name__ == "__main__":
    from push_ups.utils.profiler import Profiler

    prof = Profiler()
    # import time

    # start = time.perf_counter()
    framework.run_simulations_on_all_cores = prof.check(
        "run_simulations_on", framework.run_simulations_on_all_cores
    )
    return_batch.DiscountedReturn = prof.check("Batch", return_batch.DiscountedReturn)
    PolicyAgent.get_loss = prof.check("get_loss", PolicyAgent.get_loss)

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    agent = PolicyAgent(env, lr=0.01)
    batch_size = 10_000
    steps = 10000

    # run_simulations_on_all_cores(agent, env_name, batch_size)

    # finish = time.perf_counter()

    # print(f"Finished in {round(finish-start, 2)} seconds")

    agent = framework.train_agent(
        agent, env_name, steps, batch_size, stats=True, stats_frequency=1
    )
    # profiler.print()
    prof.print()
