"""
Prototype of the RL Agent for the OpenAI Gym environment.
"""
from abc import ABC, abstractmethod
from typing import List

import gym
import numpy as np

from base import action, observation


class Agent(ABC):
    def __init__(self, env: gym.core.Env, gamma: float = 0.9, *args, **kwargs):
        self.env = env
        self.actions = action.Actions(env)
        self.observations = observation.Observations(env)
        self.policy = self.get_policy()
        self.gamma = gamma

    def __call__(self, *args, **kwargs):
        return self.take_action(*args, **kwargs)

    @abstractmethod
    def get_policy(self, *args, **kwargs):
        """
        Implement your own policy (e.g. nn from the Torch)
        """
        pass

    @abstractmethod
    def take_action(self, *args, **kwargs):
        """
        Take action based on the observation.
        """
        pass

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        """
        Imporve the policy according to your algorithm.
        """
        pass

    @property
    def discrete_actions(self) -> bool:
        """
        Return true if environment requires discrete actions.
        """
        return self.actions.discrete

    def get_actions(self) -> int:
        """
        Return the number of possible discrete options or continuous variables.
        """
        if self.discrete_actions:
            return self.actions.discrete_space[0]
        else:
            # continuous space array has (min, max) values for each variable
            return self.actions.continuous_space.size // 2

    def describe(self) -> str:
        """
        Print out info about the agent.
        """
        env = str(self.env)
        observation_space = str(self.observations)
        action_space = str(self.actions)
        policy = str(self.policy)
        gamma = f"Gamma({self.gamma})"

        return env, observation_space, action_space, policy, gamma


class RandomAgent(Agent):
    def take_action(self, *args, **kwargs) -> List:
        if self.discrete_actions:
            return self.sample_discrete()[0]
        else:
            return self.sample_continuous()

    def sample_discrete(self) -> List[int]:
        discrete_possibilities = self.actions.discrete_space
        d_actions = []

        if discrete_possibilities.size != 0:
            for options_no in discrete_possibilities:
                a = np.random.randint(options_no)
                # d_actions = np.append(d_actions, a)
                d_actions.append(a)

        return d_actions

    def sample_continuous(self) -> List[float]:
        continuous_possibilities = self.actions.continuous_space
        c_actions = []

        if continuous_possibilities.size != 0:
            for min_, max_ in continuous_possibilities:
                a = np.random.rand() * (max_ - min_) + min_
                c_actions.append(a)

        return c_actions

    def update_policy(self, *args, **kwargs):
        pass

    def get_policy(self, *args, **kwargs):
        pass
