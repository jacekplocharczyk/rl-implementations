
import gym
import pytest

from base import action


def test_action_init():
    env = gym.make('CartPole-v1')
    actions = action.Actions(env)
    assert actions.discrete == True
    assert actions.n == 2


def test_action_init_not_implemented():
    env = gym.make('MountainCarContinuous-v0')
    with pytest.raises(NotImplementedError):
        action.Actions(env)

