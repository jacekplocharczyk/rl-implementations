import gym
import numpy as np
import pytest

from base import spaces


@pytest.fixture
def env():
    return gym.make("CartPole-v1")


def test_equal_spaces(env):
    space_1 = spaces.BoxSpace(env.observation_space)
    space_2 = spaces.DiscreteSpace(env.action_space)

    assert space_1 == space_1
    assert space_1 != space_2


def test_BoxSpace_repr(env):
    space = spaces.BoxSpace(env.observation_space)
    assert repr(space) == "Box(4,)"


def test_DiscreteSpace_repr(env):
    space = spaces.DiscreteSpace(env.action_space)
    assert repr(space) == "Discrete(2)"


def test_BoxSpace_init(env):
    space = spaces.BoxSpace(env.observation_space)
    expected_result = np.array(
        [4.8000002e00, 3.4028235e38, 4.1887903e-01, 3.4028235e38], dtype=np.float32
    )

    assert space.shape == (4,)
    np.testing.assert_array_almost_equal(space.high, expected_result)


def test_BoxSpace_discrete_space(env):
    space = spaces.BoxSpace(env.observation_space)
    result = space.discrete_space()
    expected_result = np.array([])
    np.testing.assert_array_equal(result, expected_result)


def test_BoxSpace_continuous_space(env):
    space = spaces.BoxSpace(env.observation_space)
    result = space.continuous_space()
    expected_result = np.array(
        [
            [-4.8000002e00, -3.4028235e38, -4.1887903e-01, -3.4028235e38],
            [4.8000002e00, 3.4028235e38, 4.1887903e-01, 3.4028235e38],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_almost_equal(result, expected_result)


def test_DiscreteSpace_init(env):
    space = spaces.DiscreteSpace(env.action_space)
    assert space.n == 2


def test_DiscreteSpace_super_init(env):
    with pytest.raises(AssertionError):
        spaces.DiscreteSpace(env.observation_space)
