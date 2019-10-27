from abc import abstractstaticmethod
from typing import List, Tuple, Union

import numpy as np
import torch


class Batch(torch.utils.data.Dataset):
    def __init__(
        self,
        actions: List[torch.tensor],
        observations: List[torch.tensor],
        rewards: List[torch.tensor],
        device: Union[torch.device, None] = None,
        randomize: bool = True,
        *args,
        **kwargs
    ):
        del args, kwargs  # unused
        self.actions = self.get_actions(actions)
        self.returns = self.get_returns(rewards)
        self.observations = self.get_observations(observations)
        if device:
            self.actions = self.actions.to(device)
            self.returns = self.returns.to(device)
            self.observations = self.observations.to(device)

        if randomize:
            self.randomize_rows()

        assert self.actions.shape[0] == self.returns.shape[0]
        assert self.returns.shape[0] == self.observations.shape[0]

    def __len__(self) -> int:
        return self.returns.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        act = self.actions[idx]
        ret = self.returns[idx]
        obs = self.observations[idx]

        return act, ret, obs

    def get_data(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.actions, self.returns, self.observations

    def randomize_rows(self):
        indicies = np.random.permutation(self.actions.shape[0])
        self.actions = self.actions[indicies]
        self.returns = self.returns[indicies]
        self.observations = self.observations[indicies]

    @abstractstaticmethod
    def get_actions(actions: List[torch.tensor]) -> torch.tensor:
        pass

    @abstractstaticmethod
    def get_returns(self, rewards: List[torch.tensor]) -> torch.tensor:
        pass

    @abstractstaticmethod
    def get_observations(observations: List[torch.tensor]) -> torch.tensor:
        pass
