"""
Example of using custom profiler.
"""


import gym

from push_ups import framework
from push_ups.agent_base import RandomAgent
from push_ups.utils.profiler import profiler

prof = profiler.Profiler()


def fun_a(a, b):
    return a + b


fun_a = prof.check("fun_a", fun_a)
framework.torch.cat = prof.check("framework.torch.cat", framework.torch.cat)


if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = RandomAgent(env)

    for _ in range(10):
        fun_a(1, 2)
    for _ in range(300):
        framework.collect_actions_observations_rewards((agent, env_name, 100))

    prof.print()
