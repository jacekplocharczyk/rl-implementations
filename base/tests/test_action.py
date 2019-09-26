
import gym
import pytest

from base import action


def test_action_init():
    env = gym.make('CartPole-v1')
    action_ = action.Action(env)
    assert action_.discrete == True
    assert action_.actions_ == 2


def test_action_init_not_implemented():
    env = gym.make('MountainCarContinuous-v0')
    with pytest.raises(NotImplementedError):
        action_ = action.Action(env)

