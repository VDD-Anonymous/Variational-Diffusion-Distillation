from typing import List, Optional

import numpy as np

from common.normalizers.normalizer_base import NormalizerBase
from common.normalizers.normalizer_utils import RunningMeanStd
from common.normalizers.torch_normalizer import TorchNormalizer

import copy


class NpNormalizer(NormalizerBase):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 n_envs: int,
                 norm_obs: bool = True,
                 norm_reward: bool = False,
                 norm_action: bool = False,
                 clip_obs: float = np.inf,
                 clip_reward: float = np.inf,
                 clip_action: float = np.inf,
                 gamma: float = 0.99,
                 epsilon: float = 1e-8,
                 norm_obs_keys: Optional[List[str]] = None,
                 ):

        super(NpNormalizer, self).__init__(obs_dim, action_dim, norm_obs, norm_reward,
                                           norm_action, clip_obs, clip_reward, clip_action,
                                           gamma, epsilon, norm_obs_keys)

        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.action_rms = RunningMeanStd(shape=(self.action_dim,))
        self.ret_rms = RunningMeanStd(shape=())
        self.n_envs = n_envs
        self.returns = np.zeros(self.n_envs)

    def _normalize_vector(self, vector: np.ndarray, rms: RunningMeanStd,
                          clip: float, normalize: bool, train: bool) -> np.ndarray:
        if not normalize:
            return vector
        if train:
            rms.update(vector)
        return np.clip((vector - rms.mean) / np.sqrt(rms.var + self.epsilon),
                       -clip, clip)

    def _denormalize_vector(self, vector: np.ndarray, rms: RunningMeanStd,
                            normalize: bool) -> np.ndarray:
        if not normalize:
            return vector
        return (vector * np.sqrt(rms.var + self.epsilon)) + rms.mean

    def normalize_obs(self, obs: np.ndarray, train: bool = False) -> np.ndarray:
        return self._normalize_vector(obs, self.obs_rms, self.clip_obs, self.norm_obs, train)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self._denormalize_vector(obs, self.obs_rms, self.norm_obs)

    def normalize_reward(self, reward: np.ndarray, done_mask: np.array = None, train: bool = False) -> np.ndarray:
        if not self.norm_reward:
            return reward
        if train:
            assert done_mask is not None, "done_mask must be provided during training"
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(self.returns)
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
            self.returns[done_mask] = 0.
            return reward
        return np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)

    def denormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        return self._denormalize_vector(reward, self.ret_rms, self.norm_reward)

    def normalize_action(self, action: np.ndarray, train: bool = False) -> np.ndarray:
        return self._normalize_vector(action, self.action_rms, self.clip_action, self.norm_action, train)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return self._denormalize_vector(action, self.action_rms, self.norm_action)

    @staticmethod
    def create_from_torch_normalizer(torch_normalizer: TorchNormalizer):
        np_normalizer = NpNormalizer(obs_dim=torch_normalizer.obs_dim,
                                     action_dim=torch_normalizer.action_dim,
                                     n_envs=torch_normalizer.n_envs,
                                     norm_obs=torch_normalizer.norm_obs,
                                     norm_reward=torch_normalizer.norm_reward,
                                     norm_action=torch_normalizer.norm_action,
                                     clip_obs=torch_normalizer.clip_obs,
                                     clip_reward=torch_normalizer.clip_reward.to('cpu').numpy(),
                                     clip_action=torch_normalizer.clip_action.to('cpu').numpy(),
                                     gamma=torch_normalizer.gamma,
                                     epsilon=torch_normalizer.epsilon.to('cpu').numpy(),
                                     norm_obs_keys=torch_normalizer.norm_obs_keys)

        def copy_rms_from_torch(np_rms, torch_rms):
            np_rms.mean = copy.deepcopy(torch_rms.mean.numpy())
            np_rms.var = copy.deepcopy(torch_rms.var.numpy())
            np_rms.count = copy.deepcopy(torch_rms.count.numpy())

        for np_rms, torch_rms in zip(
                [np_normalizer.obs_rms, np_normalizer.action_rms, np_normalizer.ret_rms],
                [torch_normalizer.obs_rms, torch_normalizer.action_rms, torch_normalizer.ret_rms]
        ):
            copy_rms_from_torch(np_rms, torch_rms)

        return np_normalizer

