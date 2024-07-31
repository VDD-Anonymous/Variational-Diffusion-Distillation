from typing import Union

import numpy as np
import torch as ch

from common.normalizers import NormalizerBase


class DummyNormalizer(NormalizerBase):
    # Dummy normalizer that does nothing
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def normalize_obs(self, obs: Union[np.ndarray, ch.Tensor], train: bool) -> Union[np.ndarray, ch.Tensor]:
        return obs

    def denormalize_obs(self, obs: Union[np.ndarray, ch.Tensor]) -> Union[np.ndarray, ch.Tensor]:
        return obs

    def normalize_reward(self, reward: np.ndarray, done_mask: np.ndarray, train: bool) -> np.ndarray:
        return reward

    def denormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        raise reward

    def normalize_action(self, action: Union[np.ndarray, ch.Tensor], train: bool) -> Union[np.ndarray, ch.Tensor]:
        raise action

    def denormalize_action(self, action: Union[np.ndarray, ch.Tensor]) -> Union[np.ndarray, ch.Tensor]:
        return action