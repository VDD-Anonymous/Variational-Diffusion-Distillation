from abc import ABC, abstractmethod

import torch as ch

import torch.nn as nn

from common.utils.network_utils import get_mlp, initialize_weights
from common.utils.torch_utils import inverse_softplus

from common.models.abstract_gaussian_policy import AbstractGaussianPolicy


class AbstractResidualPolicy(AbstractGaussianPolicy):

    def __init__(self, pretrained_policy: AbstractGaussianPolicy, residual_ratio: float = 0.5,
                 hidden_sizes=(64, 64), activation: str = "tanh", layer_norm: bool = False, contextual_std: bool = False,
                 trainable_std: bool = True, init_std: float = 1., share_weights=False, vf_model=None, minimal_std: float = 1e-5,
                 scale: float = 1e-4, gain: float = 0.01, **kwargs):

        self.action_dim = pretrained_policy.action_dim

        self.obs_dim = pretrained_policy.obs_dim

        self.residual_ratio = residual_ratio

        super().__init__(obs_dim=self.obs_dim, action_dim=self.action_dim,
                         hidden_sizes=hidden_sizes, activation=activation, layer_norm=layer_norm,
                         contextual_std=contextual_std, trainable_std=trainable_std, init_std=init_std,
                         share_weights=share_weights, vf_model=vf_model, minimal_std=minimal_std, scale=scale,
                         gain=gain, **kwargs)

        self.base_policy = pretrained_policy


