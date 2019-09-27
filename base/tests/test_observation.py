
import gym
import pytest
import numpy as np

from base import observation


@pytest.fixture
def env_discrete():
    return gym.make('FrozenLake-v0')


@pytest.fixture
def env_continuous():
    return gym.make('MountainCarContinuous-v0')


def test_observations_discrete_space(env_discrete):
    observations = observation.Observations(env_discrete)
    expected_result = np.array([16])
    result = observations.discrete_space
    np.testing.assert_array_equal(result, expected_result)

    expected_result = np.array([[]])
    result = observations.continuous_space
    np.testing.assert_array_equal(result, expected_result)

def test_observations_continuous_space(env_continuous):
    observations = observation.Observations(env_continuous)
    expected_result = np.array([])
    result = observations.discrete_space
    np.testing.assert_array_equal(result, expected_result)

    expected_result = np.array([[-1.2 , -0.07], [ 0.6 ,  0.07]])
    result = observations.continuous_space
    np.testing.assert_array_almost_equal(result, expected_result)

