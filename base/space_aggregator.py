from abc import ABC, abstractproperty
from typing import Tuple


import gym
import numpy as np

from base import spaces


ALL_SPACES = [spaces.DiscreteSpace, spaces.BoxSpace]


class SpaceAggregator(ABC):
    """
    Class with space checking mechanism.
    """

    SPACES = ALL_SPACES

    def __init__(self, *args, **kwargs):
        assert self.discrete_space.size == 0 or self.continuous_space.size == 0

    @classmethod
    def import_space_type(cls, space: gym.spaces.Space) -> spaces.Space:
        for s in cls.SPACES:
            if s.check_type(space):
                return s(space)

        raise NotImplementedError(f"Unknown space type: {repr(space)}")

    @property
    def discrete(self):
        assert self.discrete_space.size == 0 or self.continuous_space.size == 0
        if self.discrete_space.size != 0:
            return True
        else:
            False

    @abstractproperty
    def discrete_space(self) -> np.array:
        pass

    @abstractproperty
    def continuous_space(self) -> np.array:
        pass
