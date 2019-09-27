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

    @classmethod
    def import_space_type(cls, space: gym.spaces.Space) -> spaces.Space:
        for s in cls.SPACES:
            if s.check_type(space):
                return s(space)
        
        raise NotImplementedError(f'Unknown space type: {repr(space)}')

    @property
    def possible(self) -> Tuple[np.array, np.array]:
        """
        Return two object tuple with np.arrays. First object reffers to the 
        discrete states and the second to the continuous states.
        """
        discrete = self.discrete_space
        continuous = self.continuous_space

        return discrete, continuous

    @abstractproperty
    def discrete_space(self) -> np.array:
        pass

    @abstractproperty
    def continuous_space(self) -> np.array:
        pass