from abc import ABC

import gym

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
