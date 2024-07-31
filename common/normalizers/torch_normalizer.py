from typing import List, Optional, Union

import torch
import torch as ch
import numpy as np

from common.normalizers.normalizer_base import NormalizerBase
from common.normalizers.normalizer_utils import TorchRunningMeanStd


class TorchNormalizer(NormalizerBase):

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 n_envs: int = 1,
                 norm_obs: bool = True,
                 norm_reward: bool = False,
                 norm_action: bool = False,
                 clip_obs: float = np.inf,
                 clip_reward: float = np.inf,
                 clip_action: float = np.inf,
                 gamma: float = 0.99,
                 epsilon: float = 1e-8,
                 norm_obs_keys: Optional[List[str]] = None,
                 device: str = 'cpu',
                 dtype: str = 'float32',
                 ):
        super().__init__(obs_dim, action_dim, norm_obs, norm_reward,
                         norm_action, clip_obs, clip_reward, clip_action,
                         gamma, epsilon, norm_obs_keys)
        self.dtype = torch.float32 if dtype == 'float32' else torch.float64
        self.obs_rms = TorchRunningMeanStd(shape=(self.obs_dim,), device=device, dtype=self.dtype)
        self.action_rms = TorchRunningMeanStd(shape=(self.action_dim,), device=device, dtype=self.dtype)
        self.ret_rms = TorchRunningMeanStd(shape=(), device=device, dtype=self.dtype)
        self.device = device
        self.n_envs = n_envs
        self.returns = torch.zeros(self.n_envs, device=self.device, dtype=self.dtype)
        self.epsilon = self.to_tensor(self.epsilon)
        self.clip_obs = self.to_tensor(self.clip_obs)
        self.clip_action = self.to_tensor(self.clip_action)
        self.clip_reward = self.to_tensor(self.clip_reward)

    def to_tensor(self, x):
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    def load_dataset(self, observations: Union[torch.Tensor, np.ndarray],
                           actions: Union[torch.Tensor, np.ndarray],
                           rewards: Union[torch.Tensor, np.ndarray] = None):

        if isinstance(observations, np.ndarray):
            observations = self.to_tensor(observations)
        if isinstance(actions, np.ndarray):
            actions = self.to_tensor(actions)
        if isinstance(rewards, np.ndarray):
            rewards = self.to_tensor(rewards)

        self.obs_rms.update(observations)
        self.action_rms.update(actions)

        if rewards is not None:
            self.ret_rms.update(rewards)

    @torch.no_grad()
    def _normalize_vector(self, vector: torch.Tensor, rms: TorchRunningMeanStd,
                          clip: torch.Tensor, normalize: bool, training: bool):
        if not normalize:
            return vector
        if training:
            rms.update(vector.reshape(-1, vector.shape[-1]))
        return torch.clip((vector - rms.mean)/torch.sqrt(rms.var+self.epsilon), -clip, clip)

    @torch.no_grad()
    def _denormalize_vector(self, vector: torch.Tensor, rms: TorchRunningMeanStd,
                            normalize: bool):
        if not normalize:
            return vector
        return (vector * torch.sqrt(rms.var + self.epsilon)) + rms.mean

    def normalize_obs(self, obs: ch.Tensor, train: bool = False) -> ch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = self.to_tensor(obs)
        return self._normalize_vector(obs, self.obs_rms, self.clip_obs, self.norm_obs,
                                      training=train)

    def denormalize_obs(self, obs: ch.Tensor) -> ch.Tensor:
        return self._denormalize_vector(obs, self.obs_rms, self.norm_obs)

    def normalize_action(self, action: ch.Tensor, train: bool = False) -> ch.Tensor:
        if isinstance(action, np.ndarray):
            action = self.to_tensor(action)
        return self._normalize_vector(action, self.action_rms, self.clip_action, self.norm_action,
                                      training=train)

    def denormalize_action(self, action: ch.Tensor) -> ch.Tensor:
        return self._denormalize_vector(action, self.action_rms, self.norm_action)

    def normalize_reward(self, reward: ch.Tensor, done_mask: ch.Tensor, train: bool) -> ch.Tensor:
        if not self.norm_reward:
            return reward
        if train:
            assert done_mask is not None, "done_mask must be provided during training"
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(self.returns.reshape(-1))
            reward = torch.clip(reward / torch.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
            self.returns[done_mask] = 0.
            return reward
        raise torch.clip(reward / torch.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)

    def denormalize_reward(self, reward: ch.Tensor) -> ch.Tensor:
        raise NotImplementedError

    def to_device(self, device: str) -> None:
        self.device = device
        self.obs_rms.to_device(self.device)
        self.action_rms.to_device(self.device)
        self.ret_rms.to_device(self.device)
        self.epsilon = self.epsilon.to(self.device)
        self.clip_obs = self.clip_obs.to(self.device)
        self.clip_action = self.clip_action.to(self.device)
        self.clip_reward = self.clip_reward.to(self.device)

