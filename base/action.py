import gym


class Action:
    def __init__(self, env: gym.core.Env, *args, **kwargs):
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            self.discrete = True
            self.actions_ = env.action_space.n
            self.repr = repr(env.action_space)
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.repr

    @property
    def actions(self):
        if self.discrete:
            return self.actions

