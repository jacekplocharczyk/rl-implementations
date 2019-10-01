from typing import List, Tuple

import gym
import numpy as np

from base import space_aggregator


class Actions(space_aggregator.SpaceAggregator):
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        self.action_space = self.import_space_type(env.action_space)
        super().__init__()

    def __repr__(self):
        return f"Actions({repr(self.action_space)})"

    @property
    def discrete_space(self) -> np.array:
        return self.action_space.discrete_space()

    @property
    def continuous_space(self) -> np.array:
        return self.action_space.continuous_space()
