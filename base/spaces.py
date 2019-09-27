from abc import ABC, abstractproperty

import gym
import numpy as np

from typing import Any


class Space(ABC):
    TYPE = None
    
    def __init__(self, space: gym.spaces.Space, *args, **kwargs):
        self.repr = repr(space)
        
    def __repr__(self):
        return self.repr

    @abstractproperty
    def discrete_space(self) -> np.array:
        """
        Return the number of available possibilites for each of the discrete
        variables.
        """
        pass

    @abstractproperty
    def continuous_space(self) -> np.array:
        """
        Return the lower and upper bounds for each of the  continious variables.
        """
        pass

    @classmethod
    def check_type(cls, obj: Any):
        return isinstance(obj, cls.TYPE)

    def __eq__(self, other: Any) -> bool:
        discrete_equal = np.array_equal(self.discrete_space, 
                                        other.discrete_space)
        continuous_equal = np.array_equal(self.continuous_space, 
                                          other.continuous_space)
        return discrete_equal and continuous_equal


class BoxSpace(Space):
    TYPE = gym.spaces.Box

    def __init__(self, space: gym.spaces.Space, *args, **kwargs):
        assert self.check_type(space), 'Space is not the type Box.'
        self.dtype = space.dtype
        self.shape = space.shape
        self.low = space.low
        self.high = space.high
        super().__init__(space, *args, **kwargs)

    def discrete_space(self) -> np.array:
        return np.array([])

    def continuous_space(self) -> np.array:
        if len(self.shape) == 1:
            low = np.reshape(self.low, (1, len(self.low)))
            high = np.reshape(self.high, (1, len(self.high)))
            bounds = np.concatenate([low, high])
        else:
            raise NotImplementedError('Multidimensional spaces not implemented')
        return bounds


class DiscreteSpace(Space):
    TYPE = gym.spaces.Discrete

    def __init__(self, space: gym.spaces.Space, *args, **kwargs):
        assert self.check_type(space), 'Space is not the type Discrete.'
        self.n = space.n
        super().__init__(space, *args, **kwargs)

    def discrete_space(self) -> np.array:
        return np.array([self.n])

    def continuous_space(self) -> np.array:
        return np.array([[]])
