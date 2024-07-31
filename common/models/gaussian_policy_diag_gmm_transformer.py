import einops
import torch
import torch as ch
import torch.nn as nn

from einops import rearrange

# from demo_guided_rl.models.value.vf_net import VFNet

from common.models.gaussian_policy_full import GaussianPolicyFull
from common.utils.network_utils import initialize_weights, get_mlp
from common.utils.torch_utils import diag_bijector, fill_triangular
from common.models.film.film_mlps import FiLMMLPNetwork

from common.models.transformer.transformers import GPTNetwork


class TransformerGMMPolicyDiag(GaussianPolicyFull):
    """
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and a cholesky matrix, which parameterize a full gaussian distribution.
    """
    def __init__(self, obs_dim, action_dim, init='orthogonal', goal_dim=2, embed_dim=72, embed_pdrop = 0.0, atten_pdrop = 0.1,
                 resid_pdrop: float = 0.1, gpt_n_layers: int = 2, n_heads: int = 6, window_size: int = 5,
                 goal_conditional: bool = True, goal_seq_len: int = 1,
                 linear_output: bool = True, pre_out_hidden_dim: int = 100, encode_actions: bool = False,
                 hidden_sizes=(64, 64), std_hidden_sizes=(64, 64), n_components=1,
                 activation: str = "tanh", layer_norm: bool = False, contextual_std: bool = False, trainable_std: bool = True,
                 init_std: float = 1., share_weights=False, vf_model = None, minimal_std: float = 1e-5,
                 scale: float = 1e-4, gain: float = 0.01, share_layers: bool = True, use_film: bool = True,
                 fix_mean_bias: bool = False, bias_init_bound = 0.5, **kwargs):

        self.n_components = n_components
        self.fix_mean_bias = fix_mean_bias
        self.bias_init_bound = bias_init_bound

        super().__init__(obs_dim, action_dim, init, hidden_sizes, std_hidden_sizes, activation, layer_norm, contextual_std,
                         trainable_std, init_std, share_weights, vf_model, minimal_std, scale, gain, mlp_policy=False, **kwargs)

        self.share_layers = share_layers

        self.window_size = window_size

        self._gpt = GPTNetwork(obs_dim=obs_dim,
                               goal_dim=goal_dim,
                               output_dim=hidden_sizes[0],
                               embed_dim=embed_dim,
                               embed_pdrop=embed_pdrop,
                               atten_pdrop=atten_pdrop,
                               resid_pdrop=resid_pdrop,
                               n_layers=gpt_n_layers,
                               n_heads=n_heads,
                               window_size=window_size,
                               goal_conditional=goal_conditional,
                               goal_seq_len=goal_seq_len,
                               linear_output=linear_output,
                               pre_out_hidden_dim=pre_out_hidden_dim,
                               encode_actions=encode_actions,
                               device=kwargs.get("device", 'cuda'))

        self._affine_layers = get_mlp(hidden_sizes[0], hidden_sizes, init, activation, layer_norm,
                                                                    True)

        if contextual_std:
            self._std_layers = get_mlp(hidden_sizes[0], std_hidden_sizes, init, activation, layer_norm,
                                       True) if contextual_std else None
        else:
            self._std_layers = None

    def _get_std_layer(self, prev_size: int, action_dim: int, init: str, gain=0.01, scale=1e-4):
        # chol_shape = action_dim * (action_dim + 1) // 2 * self.n_components
        chol_shape = action_dim * self.n_components
        flat_chol = nn.Linear(prev_size, chol_shape)
        initialize_weights(flat_chol, init, gain=gain, scale=scale)
        return flat_chol

    def _get_std_parameter(self, action_dim: int, scale=0.01):
        # std = ch.normal(0, scale, (action_dim,))
        std = ch.zeros((action_dim,))
        return nn.Parameter(std)

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
    def uniform_distribute_components(self, n, bound=0.5):

        size = self.action_dim * n

        return torch.rand(size) * 2 * bound - bound

    def kmean_init(self, data):
        assert data.shape[-1] == self.action_dim, "data shape does not match action dim, cannot use kmean init"
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_components)
        kmeans.fit(data.to('cpu'))
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self._gpt.device).to(torch.float32)
        self._mean.bias.data = einops.rearrange(cluster_centers, 'n d -> (n d)')

    def _get_mean(self, action_dim, prev_size=None, init=None, gain=0.01, scale=1e-4):
        mean = nn.Linear(prev_size, action_dim * self.n_components)
        initialize_weights(mean, init, gain=gain, scale=scale)
        mean.weight.data.fill_(0)
        if action_dim == 2:
            mean.bias.data = self.distribute_components(self.n_components, self.bias_init_bound)
        else:
            mean.bias.data = self.uniform_distribute_components(self.n_components, self.bias_init_bound)

        if self.fix_mean_bias:
            mean.bias.requires_grad = False

        return mean

    def forward(self, x: ch.Tensor, goals: ch.Tensor = None, train: bool = True):
        """

        Args:
            x:
            train:

        Returns:
            mean: (batch, time_steps, n_component, action_dim)
            chol: (batch, time_steps, n_component, action_dim, action_dim)
        """
        self.train(train)

        x = self._gpt(states=x, goals=goals)

        pre_mean = x
        pre_std = x
        for affine in self._affine_layers:
            pre_mean = affine(pre_mean)

        if self.share_layers:
            flat_chol = self._pre_std(pre_mean)
            raise NotImplementedError
        else:
            if self._std_layers is not None:
                for affine in self._std_layers:
                    pre_std = affine(pre_std)
                diag_params = self._pre_std(pre_std)
                diag_params = rearrange(diag_params, 'b t (n d) -> b n t d', n=self.n_components, d=self.action_dim)
                chol = ch.diag_embed(diag_params)
            else:
                flat_chol = self._pre_std
                chol = ch.diag_embed(flat_chol)

        mean = self._mean(pre_mean)

        # reshape mean and chol to GMM shape (batch, n_components, t, action_dim)

        mean = rearrange(mean, 'b t (n d) -> b n t d', n=self.n_components, d=self.action_dim)

        if self._std_layers is None:
            chol = rearrange(chol, 'd1 d2 -> 1 1 1 d1 d2')
            chol = einops.repeat(chol, '1 1 1 d1 d2 -> b n t d1 d2', b=mean.shape[0], n=mean.shape[1], t=mean.shape[2])

        chol = diag_bijector(lambda z: self.diag_activation(z + self._pre_activation_shift) + self.minimal_std, chol)

        if torch.isnan(chol).any():
            print("nan in chol")
            exit()

        return mean, chol

if __name__ == "__main__":
    obs_dim = 10
    action_dim = 2
    goal_dim = 2
    n_components = 6
    window_size = 5
    model = TransformerGMMPolicyDiag(obs_dim=obs_dim, action_dim=action_dim, goal_dim=goal_dim, n_components=n_components,
                                     window_size=window_size, goal_conditional=False, share_layers=False, contextual_std=True)
    x = torch.randn(2, 5, obs_dim)
    mean, chol = model(x)
    print(mean.shape, chol.shape)