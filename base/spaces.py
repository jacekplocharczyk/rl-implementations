from abc import ABC, abstractproperty

import gym
import numpy as np

from typing import Tuple


class Space(ABC):
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


class BoxSpace(Space):
    def __init__(self, space: gym.spaces.Space, *args, **kwargs):
        assert isinstance(space, gym.spaces.Box)
        self.dtype = space.dtype
        self.shape = space.shape
        self.low = space.low
        self.high = space.high
        self.repr = repr(space)
        
    def __repr__(self):
        return self.repr

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


class DiscreteSpace:
    def __init__(self, space: gym.spaces.Space, *args, **kwargs):
        assert isinstance(space, gym.spaces.Discrete)
        self.n = space.n
        self.repr = repr(space)

    def __repr__(self):
        return self.repr

    def discrete_space(self) -> np.array:
        return np.array([n])

    def continuous_space(self) -> np.array:
        return np.array([[]])
