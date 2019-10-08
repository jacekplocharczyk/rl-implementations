import gym
import pytest

from push_ups import agent_base


@pytest.fixture
def env_discrete():
    return gym.make("CartPole-v1")


@pytest.fixture
def env_continuous():
    return gym.make("MountainCarContinuous-v0")


def test_agent_discrete_actions(env_discrete):
    agent = agent_base.RandomAgent(env_discrete)
    assert agent.discrete_actions


def test_agent_continuous_actions(env_continuous):
    agent = agent_base.RandomAgent(env_continuous)
    assert not agent.discrete_actions


def test_agent_get_actions_discrete(env_discrete):
    agent = agent_base.RandomAgent(env_discrete)
    assert agent.get_actions() == 2


def test_agent_get_actions_continuous(env_continuous):
    agent = agent_base.RandomAgent(env_continuous)
    assert agent.get_actions() == 1


def test_agent_get_observations_continuous(env_continuous):
    agent = agent_base.RandomAgent(env_continuous)
    assert agent.get_observations() == 2
