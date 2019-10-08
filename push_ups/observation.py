import gym
import numpy as np

from push_ups import space_aggregator


class Observations(space_aggregator.SpaceAggregator):
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        self.observation_space = self.import_space_type(env.observation_space)
        super().__init__()

    def __repr__(self):
        return f"Observation({repr(self.observation_space)})"

    @property
    def discrete_space(self) -> np.array:
        return self.observation_space.discrete_space()

    @property
    def continuous_space(self) -> np.array:
        return self.observation_space.continuous_space()
