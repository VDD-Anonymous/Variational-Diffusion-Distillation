from abc import ABC, abstractmethod
from typing import List, Optional, Union
import pickle

import numpy as np
import torch as ch


class NormalizerBase(ABC):

    def __init__(self,
                 obs_dim: int = 1,
                 action_dim: int = 1,
                 norm_obs: bool = False,
                 norm_reward: bool = False,
                 norm_action: bool = False,
                 clip_obs: float = np.inf,
                 clip_reward: float = np.inf,
                 clip_action: float = np.inf,
                 gamma: float = 0.99,
                 epsilon: float = 1e-8,
                 norm_obs_keys: Optional[List[str]] = None,
                 ):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.norm_obs = norm_obs
        self.norm_action = norm_action
        self.norm_reward = norm_reward
        self.clip_obs = np.inf if clip_obs == 0. else clip_obs
        self.clip_reward = np.inf if clip_reward == 0. else clip_reward
        self.clip_action = np.inf if clip_action == 0. else clip_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm_obs_keys = norm_obs_keys

    @abstractmethod
    def normalize_obs(self, obs: Union[np.ndarray, ch.Tensor], train: bool) -> Union[np.ndarray, ch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def denormalize_obs(self, obs: Union[np.ndarray, ch.Tensor]) -> Union[np.ndarray, ch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def normalize_reward(self, reward: np.ndarray, done_mask: np.ndarray, train: bool) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def denormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def normalize_action(self, action: Union[np.ndarray, ch.Tensor], train: bool) -> Union[np.ndarray, ch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def denormalize_action(self, action: Union[np.ndarray, ch.Tensor]) -> Union[np.ndarray, ch.Tensor]:
        raise NotImplementedError

    def save(self, save_path: str) -> None:
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, load_path: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load_from_path(load_path: str):
        with open(load_path, 'rb') as f:
            return pickle.load(f)
