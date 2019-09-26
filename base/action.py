import gym


class Actions:
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            self.discrete_ = True
            self.n_ = env.action_space.n
            self.repr = repr(env.action_space)
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.repr

    @property
    def n(self):
        if self.discrete:
            return self.n_

    @property
    def discrete(self):
        return self.discrete_
