import torch
import torch as ch

from common.models.film.film_mlps import FiLMMLPNetwork
from common.models.gaussian_policy_full import GaussianPolicyFull
from common.utils.torch_utils import diag_bijector, fill_triangular
# from demo_guided_rl.models.value.vf_net import VFNet


class FilmedGaussianPolicyFull(GaussianPolicyFull):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and a cholesky matrix, which parameterize a full gaussian distribution.
    """

    def __init__(self, obs_dim, action_dim, init, hidden_sizes=(64, 64), std_hidden_sizes=(64, 64),
                 activation: str = "tanh",
                 layer_norm: bool = False, contextual_std: bool = False, trainable_std: bool = True,
                 init_std: float = 1., share_weights=False, vf_model = None, minimal_std: float = 1e-5,
                 scale: float = 1e-4, gain: float = 0.01, share_layers: bool = True, **kwargs):

        super().__init__(obs_dim, action_dim, init, hidden_sizes, std_hidden_sizes, activation, layer_norm,
                         contextual_std,
                         trainable_std, init_std, share_weights, vf_model, minimal_std, scale, gain, **kwargs)


        self._affine_layers = FiLMMLPNetwork(input_dim=obs_dim // 2,
                                             condition_dim=obs_dim // 2,
                                             hidden_dim=hidden_sizes[0],
                                             num_hidden_layers=len(hidden_sizes),
                                             output_dim=hidden_sizes[-1],
                                             dropout=0,
                                             activation=activation,
                                             use_spectral_norm=False,
                                             device=kwargs.get("device", 'cuda'))

        self.share_layers = share_layers

        if not self.share_layers:
            self._std_layers = FiLMMLPNetwork(input_dim=obs_dim // 2,
                                              condition_dim=obs_dim // 2,
                                              hidden_dim=hidden_sizes[0],
                                              num_hidden_layers=len(std_hidden_sizes),
                                              output_dim=hidden_sizes[-1],
                                              dropout=0,
                                              activation=activation,
                                              use_spectral_norm=False,
                                              device=kwargs.get("device", 'cuda'))
        else:
            self._std_layers = None

    def forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)

        input, condition = x[..., :self.obs_dim // 2], x[..., self.obs_dim // 2:]

        x = self._affine_layers(input, condition)

        if self.share_layers:
            flat_chol = self._pre_std(x)
        else:
            flat_chol = self._pre_std(self._std_layers(input, condition))

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

        return self._mean(x), chol
