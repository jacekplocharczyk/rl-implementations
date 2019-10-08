import gym
import pytest

from push_ups.space_aggregator import SpaceAggregator
from push_ups.spaces import DiscreteSpace


@pytest.fixture
def env():
    return gym.make("CartPole-v1")


def test_space_aggregator_import_space_type(env):
    result = SpaceAggregator.import_space_type(env.action_space)
    expected_result = DiscreteSpace(env.action_space)
    assert type(result) == type(expected_result)
