import gym
import pytest
import numpy as np

from base import action


@pytest.fixture
def env_discrete():
    return gym.make("CartPole-v1")


@pytest.fixture
def env_continuous():
    return gym.make("MountainCarContinuous-v0")


def test_actions_take_discrete(env_discrete):
    actions = action.Actions(env_discrete)
    example = np.array([1])
    result = actions.take(example)
    np.testing.assert_array_equal(example, result)


def test_actions_take_discrete_wrong(env_discrete):
    actions = action.Actions(env_discrete)
    example = np.array([5])
    with pytest.raises(ValueError):
        actions.take(example)


def test_actions_take_continuous(env_continuous):
    actions = action.Actions(env_continuous)
    example = np.array([1])
    result = actions.take(example)
    np.testing.assert_array_equal(example, result)


def test_actions_take_continuous_wrong(env_continuous):
    actions = action.Actions(env_continuous)
    example = np.array([5])
    with pytest.raises(ValueError):
        actions.take(example)
