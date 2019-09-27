
import gym
import pytest

from base.space_aggregator import SpaceAggregator
from base.spaces import DiscreteSpace


class TestSpaceAggregator():
    env = gym.make('CartPole-v1')

    def test_import_space_type(self):
        result = SpaceAggregator.import_space_type(self.env.action_space)
        expected_result = DiscreteSpace(self.env.action_space)
        assert type(result) == type(expected_result)

