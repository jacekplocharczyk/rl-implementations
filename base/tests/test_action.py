import gym
import pytest

from base import action


@pytest.fixture
def env_discrete():
    return gym.make("CartPole-v1")


@pytest.fixture
def env_continuous():
    return gym.make("MountainCarContinuous-v0")


def test_actions_discrete(env_discrete):
    actions = action.Actions(env_discrete)
    result = actions.discrete
    assert result


def test_actions_not_discrete(env_continuous):
    actions = action.Actions(env_continuous)
    result = actions.discrete
    assert not result
