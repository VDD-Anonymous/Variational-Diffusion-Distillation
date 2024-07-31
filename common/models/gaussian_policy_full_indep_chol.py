import numpy as np
import torch
import torch as ch
import torch.nn as nn
from typing import Tuple

from common.models.abstract_gaussian_policy import AbstractGaussianPolicy
from common.utils.network_utils import initialize_weights
from common.utils.torch_utils import diag_bijector, fill_triangular, fill_triangular_inverse
from common.utils.network_utils import get_mlp, initialize_weights


class GaussianPolicyFullIndepChol(AbstractGaussianPolicy):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and a cholesky matrix, which parameterize a full gaussian distribution.
    """

    def _get_std_parameter(self, action_dim: int):
        # std = inverse_softplus(ch.ones(action_dim) * init_std).diagflat()
        # flat_chol = fill_triangular_inverse(std)
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = ch.normal(0, 0.01, (chol_shape,))
        return nn.Parameter(flat_chol)

    def _get_std_layer(self, prev_size: int, action_dim: int, init: str, gain=0.01, scale=1e-4):
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = nn.Linear(prev_size, chol_shape)
        initialize_weights(flat_chol, init, gain=gain, scale=scale)
        return flat_chol

    def _mean_forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)
        for affine in self._affine_layers:
            x = affine(x)
        return self._mean(x)

    def _std_forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)
        for std_affine in self._std_layers:
            x = std_affine(x)
        return x

    def forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)
        mean = self._mean_forward(x)
        x = self._std_forward(x) if self.contextual_std else self._pre_std

        flat_chol = self._pre_std(x) if self.contextual_std else self._pre_std
        chol = fill_triangular(flat_chol).expand(x.shape[:-1] + (-1, -1))
        chol = diag_bijector(lambda z: self.diag_activation(z + self._pre_activation_shift) + self.minimal_std, chol)

        if torch.isnan(chol).any():
            print("nan in chol")
            print(chol)
            print(flat_chol)
            print(x)
            print(self._pre_std)
            print(self._pre_activation_shift)
            print(self.minimal_std)
            exit()

        return mean, chol

    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        return self.rsample(p, n).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        means, chol = p
        eps = ch.randn((n,) + means.shape).to(dtype=chol.dtype, device=chol.device)[..., None]
        samples = (chol @ eps).squeeze(-1) + means
        return samples.squeeze(0)

    # def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
    #     mean, std = p
    #     k = mean.shape[-1]
    #
    #     logdet = self.log_determinant(std)
    #     mean_diff = self.maha(x, mean, std)
    #     if torch.isinf(mean_diff).any() or torch.isinf(logdet).any():
    #         print("nan in log prob")
    #         print(mean_diff)
    #         print(logdet)
    #         print(x)
    #         print(mean)
    #         print(std)
    #         exit()
    #     nll = 0.5 * (k * np.log(2 * np.pi) + logdet + mean_diff)
    #     return -nll

    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
        mean, chol = p
        mvn = torch.distributions.MultivariateNormal(mean, scale_tril=chol, validate_args=False)
        log_prob = mvn.log_prob(x)
        if torch.isinf(log_prob).any():
            print("nan in log prob")
            print(log_prob)
            print(x)
            print(mean)
            print(chol)
            exit()
        return log_prob

    def ch_r_sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        mean, chol = p
        mvn = torch.distributions.MultivariateNormal(mean, scale_tril=chol, validate_args=False)
        return mvn.rsample((n,))

    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]):
        mean, chol = p
        mvn = torch.distributions.MultivariateNormal(mean, scale_tril=chol, validate_args=False)
        return mvn.entropy()

    def log_determinant(self, std: ch.Tensor):
        """
        Returns the log determinant of a cholesky matrix
        Args:
             std: a cholesky matrix
        Returns:
            The determinant of mat, aka product of the diagonal
        """
        return 2 * std.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        diff = (mean - mean_other)[..., None]
        return ch.triangular_solve(diff, std, upper=False)[0].pow(2).sum([-2, -1])

    def precision(self, std: ch.Tensor):
        return ch.cholesky_solve(ch.eye(std.shape[-1], dtype=std.dtype, device=std.device), std, upper=False)

    def covariance(self, std: ch.Tensor):
        std = std.view((-1,) + std.shape[-2:])
        return std @ std.permute(0, 2, 1)

    def set_std(self, std: ch.Tensor) -> None:
        std = diag_bijector(lambda z: self.diag_activation_inv(z - self.minimal_std) - self._pre_activation_shift, std)
        assert self._pre_std.shape == fill_triangular_inverse(std).shape
        self._pre_std.data = fill_triangular_inverse(std)
