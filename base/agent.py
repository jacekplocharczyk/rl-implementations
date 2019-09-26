"""
Prototype of the RL Agent for the OpenAI Gym environment.
"""
from abc import ABC, abstractmethod

import gym

from base import action


class Agent(ABC):
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        self.action = action.Action(env)

    @abstractmethod
    def take_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        pass

