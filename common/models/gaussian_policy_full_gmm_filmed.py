import numpy as np
import torch
import torch as ch
import torch.nn as nn
from typing import Tuple

# from demo_guided_rl.models.value.vf_net import VFNet

from common.models.gaussian_policy_full import GaussianPolicyFull
from common.utils.network_utils import initialize_weights
from common.utils.torch_utils import diag_bijector, fill_triangular, fill_triangular_inverse
from common.models.film.film_mlps import FiLMMLPNetwork


class FilmedGMMPolicyFull(GaussianPolicyFull):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and a cholesky matrix, which parameterize a full gaussian distribution.
    """
    def __init__(self, obs_dim, action_dim, init, hidden_sizes=(64, 64), std_hidden_sizes=(64, 64), n_components=1,
                 activation: str = "tanh", layer_norm: bool = False, contextual_std: bool = False, trainable_std: bool = True,
                 init_std: float = 1., share_weights=False, vf_model=None, minimal_std: float = 1e-5,
                 scale: float = 1e-4, gain: float = 0.01, share_layers: bool = True, use_film: bool = True,
                 fix_mean_bias: bool = False, bias_init_bound = 0.5, **kwargs):

        self.n_components = n_components
        self.fix_mean_bias = fix_mean_bias
        self.bias_init_bound = bias_init_bound

        super().__init__(obs_dim, action_dim, init, hidden_sizes, std_hidden_sizes, activation, layer_norm, contextual_std,
                         trainable_std, init_std, share_weights, vf_model, minimal_std, scale, gain, **kwargs)

        self.share_layers = share_layers
        self.use_film = use_film

        if use_film:

            self._affine_layers = FiLMMLPNetwork(input_dim=obs_dim//2,
                                                 condition_dim=obs_dim//2,
                                                 hidden_dim=hidden_sizes[0],
                                                 num_hidden_layers=len(hidden_sizes),
                                                 output_dim=hidden_sizes[-1],
                                                 dropout=0,
                                                 activation=activation,
                                                 use_spectral_norm=False,
                                                 device=kwargs.get("device", 'cuda'))

            if not self.share_layers:
                self._std_layers = FiLMMLPNetwork(input_dim=obs_dim//2,
                                                  condition_dim=obs_dim//2,
                                                  hidden_dim=hidden_sizes[0],
                                                  num_hidden_layers=len(std_hidden_sizes),
                                                  output_dim=hidden_sizes[-1],
                                                  dropout=0,
                                                  activation=activation,
                                                  use_spectral_norm=False,
                                                  device=kwargs.get("device", 'cuda'))
            else:
                self._std_layers = None

    def _get_std_layer(self, prev_size: int, action_dim: int, init: str, gain=0.01, scale=1e-4):
        chol_shape = action_dim * (action_dim + 1) // 2 * self.n_components
        flat_chol = nn.Linear(prev_size, chol_shape)
        initialize_weights(flat_chol, init, gain=gain, scale=scale)
        return flat_chol

    @staticmethod
    def distribute_components(n, bound=0.5):
        # Calculate grid size
        grid_side = int(torch.ceil(torch.sqrt(torch.tensor(n).float())))  # Number of points along one dimension

        # Generate grid points
        linspace = torch.linspace(-bound, bound, grid_side)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing='ij')

        # Flatten the grid and take the first n points
        points_x = grid_x.flatten()[:n]
        points_y = grid_y.flatten()[:n]

        stacked = torch.stack([points_x, points_y], dim=-1)

        return stacked.flatten()

    def _get_mean(self, action_dim, prev_size=None, init=None, gain=0.01, scale=1e-4):
        mean = nn.Linear(prev_size, action_dim * self.n_components)
        initialize_weights(mean, init, gain=gain, scale=scale)
        mean.weight.data.fill_(0)
        mean.bias.data = self.distribute_components(self.n_components, self.bias_init_bound)


        if self.fix_mean_bias:
            mean.bias.requires_grad = False

        return mean

    def forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)


        if self.use_film:
            input, condition = x[..., :self.obs_dim // 2], x[..., self.obs_dim // 2:]
            pre_mean = self._affine_layers(input, condition)

            if self.share_layers:
                flat_chol = self._pre_std(pre_mean)
            else:
                flat_chol = self._pre_std(self._std_layers(input, condition))
        else:
            pre_mean = x
            pre_std = x
            for affine in self._affine_layers:
                pre_mean = affine(pre_mean)

            if self.share_layers:
                flat_chol = self._pre_std(pre_mean)
            else:
                for affine in self._std_layers:
                    pre_std = affine(pre_std)
                flat_chol = self._pre_std(pre_std)

        mean = self._mean(pre_mean)

        # reshape mean and chol to GMM shape (batch, n_components, action_dim)

        mean = mean.view(x.shape[:-1] + (self.n_components, -1))

        flat_chol = flat_chol.view(x.shape[:-1] + (self.n_components, -1))

        chol = fill_triangular(flat_chol).expand(x.shape[:-1] + (-1, -1, -1))

        chol = diag_bijector(lambda z: self.diag_activation(z + self._pre_activation_shift) + self.minimal_std, chol)

        if torch.isnan(chol).any():
            print("nan in chol")
            exit()

        return mean, chol