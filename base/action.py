from abc import ABC
from typing import List, Tuple

import gym
import numpy as np

from base import spaces, space_aggregator


class Actions(space_aggregator.SpaceAggregator):
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        self.action_space = self.import_space_type(env.action_space)

    def __repr__(self):
        return f"Actions({repr(self.action_space)})"

    def take(self, taken_actions: List) -> Tuple:
        """
        Check if selected action is inside the action space and return them.
        Order of variables: [Discrete, Continuous].
        """
        discrete = self.discrete_space
        continuous = self.continuous_space

        if len(taken_actions) == len(discrete) + len(continuous):
            raise ValueError("Action count mismatch")

        taken_discrete_actions = taken_actions[:len(discrete)]
        taken_continuous_actions = taken_actions[len(discrete):]

        self.check_actions(taken_discrete_actions, taken_continuous_actions)
        
        return taken_actions

    def check_actions(self, taken_discrete_actions: np.array,
                      taken_continuous_actions: np.array):

        for i in range(len(taken_discrete_actions)):
            self.check_discrete_action(taken_discrete_actions[i], 
                                       self.discrete_space[i])

        for i in range(len(taken_continuous_actions)):
            self.check_continuous_action(taken_continuous_actions[i], 
                                         self.continuous_space[:, i])

    @staticmethod
    def check_discrete_action(taken_action: int, allowed_actions: int) -> None:
        error = f'Not allowed discrete action: {taken_action} for range({allowed_actions})'
        if taken_action not in range(allowed_actions):
            raise ValueError(error)

    @staticmethod
    def check_continuous_action(taken_action: int,
                                allowed_range: np.array) -> None:
        error = f'Not allowed continuous action: {taken_action} for range {allowed_range})'
        is_lesser = taken_action >= allowed_range[0]
        is_greater = taken_action <= allowed_range[1]
        if not (is_lesser and is_greater):
            raise ValueError(error)

    @property
    def discrete_space(self) -> np.array:
        return self.action_space.discrete_space()

    @property
    def continuous_space(self) -> np.array:
        return self.action_space.continuous_space()