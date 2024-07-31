from typing import Tuple

import numpy as np
import torch as ch
import torch.nn as nn

from common.models.residual.abstract_residual_policy import AbstractResidualPolicy
from common.utils.network_utils import initialize_weights


class ResidualGaussianPolicyDiag(AbstractResidualPolicy):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and std vector, which parameterize a diagonal gaussian distribution.
    """

    def _get_std_parameter(self, action_dim, scale=0.01):
        std = ch.normal(0, scale, (action_dim,))
        return nn.Parameter(std)

    def _get_std_layer(self, prev_size, action_dim, init, gain=0.01, scale=1e-4):
        std = nn.Linear(prev_size, action_dim)
        initialize_weights(std, init, gain=gain, scale=scale)
        return std

    def forward(self, x: ch.Tensor, train=True):
        self.train(train)

        raw_obs = x

        for affine in self._affine_layers:
            x = affine(x)
        std = self._pre_std(x) if self.contextual_std else self._pre_std
        std = (self.diag_activation(std + self._pre_activation_shift) + self.minimal_std)
        # std = std.clamp(max=STD_MAX)
        std = std.diag_embed().expand(x.shape[:-1] + (-1, -1))

        with ch.no_grad():
            base_mean, base_std = self.base_policy(raw_obs, train=False)

        return self._mean(x) * self.residual_ratio + base_mean, std

    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        return self.rsample(p, n).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        means, std = p
        std = std.diagonal(dim1=-2, dim2=-1)
        eps = ch.randn((n,) + means.shape, dtype=std.dtype, device=std.device)
        samples = means + eps * std
        # squeeze when n == 1
        return samples.squeeze(0)

    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
        mean, std = p
        k = x.shape[-1]

        maha_part = self.maha(x, mean, std)
        const = np.log(2.0 * np.pi) * k
        logdet = self.log_determinant(std)

        nll = -0.5 * (maha_part + const + logdet)
        return nll

    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]):
        _, std = p
        logdet = self.log_determinant(std)
        k = std.shape[-1]
        return .5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def log_determinant(self, std: ch.Tensor):
        """
        Returns the log determinant of a diagonal matrix
        Args:
            std: a diagonal matrix
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        std = std.diagonal(dim1=-2, dim2=-1)
        return 2 * std.log().sum(-1)

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        std = std.diagonal(dim1=-2, dim2=-1)
        diff = mean - mean_other
        return (diff / std).pow(2).sum(-1)

    def precision(self, std: ch.Tensor):
        return (1 / self.covariance(std).diagonal(dim1=-2, dim2=-1)).diag_embed()

    def covariance(self, std: ch.Tensor):
        return std.pow(2)

    def set_std(self, std: ch.Tensor) -> None:
        assert not self.contextual_std
        self._pre_std.data = self.diag_activation_inv(std.diagonal() - self.minimal_std) - self._pre_activation_shift

    @property
    def is_diag(self):
        return True