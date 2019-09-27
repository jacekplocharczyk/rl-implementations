"""
Prototype of the RL Agent for the OpenAI Gym environment.
"""
from abc import ABC, abstractmethod
from typing import List

import gym
import numpy as np

from base import action, observation


class Agent(ABC):
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        self.actions = action.Actions(env)
        self.observations = observation.Observations(env)

    def __call__(self, *args, **kwargs):
        return self.take_action(*args, **kwargs)

    @abstractmethod
    def take_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        pass


class RandomAgent(Agent):
    def take_action(self, *args, **kwargs) -> List:
        d_actions = self.sample_discrete()
        c_actions = self.sample_continuous()

        self.actions.check_actions(d_actions, c_actions)

        return d_actions + c_actions

    def sample_discrete(self) -> List[int]:
        discrete_possibilities, continuous_possibilities = self.actions.possible
        d_actions = []

        if discrete_possibilities.size != 0:
            for options_no in discrete_possibilities:
                a = np.random.randint(options_no)
                # d_actions = np.append(d_actions, a)
                d_actions.append(a)

        return d_actions

    def sample_continuous(self) -> List[float]:
        discrete_possibilities, continuous_possibilities = self.actions.possible
        c_actions = []

        if continuous_possibilities.size != 0:
            for min_, max_ in continuous_possibilities:
                a = np.random.rand() * (max_ - min_) + min_
                # c_actions = np.append(c_actions, a)
                c_actions.append(a)

        return c_actions

    def update_policy(self, *args, **kwargs):
        pass
