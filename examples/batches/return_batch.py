from typing import Union, List

import numpy as np
import torch

from push_ups import batch


class DiscountedReturn(batch.Batch):
    def __init__(
        self,
        actions: List[torch.tensor],
        observations: List[torch.tensor],
        rewards: List[torch.tensor],
        device: Union[torch.device, None] = None,
        randomize: bool = True,
        gamma: float = 0.9,
        normalization: bool = False,
        *args,
        **kwargs
    ):
        self.gamma = gamma
        self.normalization = normalization
        super().__init__(
            actions, observations, rewards, device, randomize, *args, **kwargs
        )

    @staticmethod
    def get_actions(actions: List[torch.tensor]) -> torch.tensor:
        actions_all = torch.tensor([]).view(0, 1).float()
        for episode_actions in actions:
            actions_all = torch.cat([actions_all, episode_actions])

        return actions_all

    def get_returns(self, rewards: List[torch.tensor]) -> torch.tensor:

        returns = [self.calculate_epiosde_returns(ep_rewards) for ep_rewards in rewards]
        returns_all = torch.tensor([]).view(0, 1).float()

        for ep_returns in returns:
            returns_all = torch.cat([returns_all, ep_returns.view(-1, 1).float()])

        if self.normalization:
            eps = np.finfo(np.float32).eps.item()
            returns_all = (returns_all - returns_all.mean()) / (returns_all.std() + eps)

        return returns_all

    @staticmethod
    def get_observations(observations: List[torch.tensor]) -> torch.tensor:
        assert len(observations) > 0
        observation_dim = observations[0].shape[1]
        obs_tensor = torch.tensor([]).view(-1, observation_dim).float()
        for ep_observations in observations:
            obs_tensor = torch.cat([obs_tensor, ep_observations])

        return obs_tensor

    def calculate_epiosde_returns(self, rewards: torch.tensor) -> torch.tensor:
        returns = torch.flip(rewards, [0])
        for idx, item in enumerate(returns):
            if idx == 0:
                continue
            returns[idx] = item + self.gamma * returns[idx - 1]
        return torch.flip(returns, [0])
