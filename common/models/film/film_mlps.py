import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from common.models.film.film_layers import FiLM


def return_activiation_fcn(activation_type: str):
    # build the activation layer
    if activation_type == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activation_type == "tanh":
        act = torch.nn.Sigmoid()
    elif activation_type == "ReLU":
        act = torch.nn.ReLU()
    elif activation_type == "PReLU":
        act = torch.nn.PReLU()
    elif activation_type == "softmax":
        act = torch.nn.Softmax(dim=-1)
    elif activation_type == "Mish":
        act = torch.nn.Mish()
    elif activation_type == nn.GELU():
        act = nn.GELU()
    else:
        act = torch.nn.PReLU()
    return act

class FiLMCondTwoLayerPreActivationResNetLinear(nn.Module):

    def __init__(
            self,
            cond_dim: int,
            hidden_dim: int = 100,
            activation: str = 'relu',
            dropout_rate: float = 0.25
    ) -> None:
        super().__init__()

        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.act = return_activiation_fcn(activation)
        self.FiLM = FiLM(cond_dim, hidden_dim)
        self.normalizer = torch.nn.LayerNorm(hidden_dim, eps=1e-06)

    def forward(self, x, cond=None):
        x_input = x
        if cond is not None:
            x = self.FiLM(cond, x)
        else:
            x = self.normalizer(x)
        x = self.l1(self.dropout(self.act(x)))
        x = self.l2(self.dropout(self.act(x)))
        return x + x_input


class FiLMMLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network which can be generated with different
    activation functions with and without spectral normalization of the weights
    """

    def __init__(
            self,
            input_dim: int,
            condition_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            use_spectral_norm: bool = False,
            device: str = 'cuda'
    ):
        super(FiLMMLPNetwork, self).__init__()
        self.network_type = "mlp"
        # define number of variables in an input sequence
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = output_dim
        self.dropout = dropout
        self.spectral_norm = use_spectral_norm
        # set up the network
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        self.layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for i in range(1, self.num_hidden_layers)
            ]
        )
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        # cond model
        self.FiLM_layers = nn.ModuleList([FiLM(self.condition_dim, self.hidden_dim)])
        self.FiLM_layers.extend(
            [FiLM(self.condition_dim, self.hidden_dim)
             for i in range(1, self.num_hidden_layers)]
        )

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self._device = device
        self.layers.to(self._device)

    def forward(self, x, condition):

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = layer(x)
            else:
                out = layer(out)
            if idx < len(self.layers) - 1:
                out = self.act(out)
                out = self.FiLM_layers[idx](condition, out)
        return out

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()


class FiLMResidualMLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "Mish",
            use_spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            device: str = 'cuda'
    ):
        super(FiLMResidualMLPNetwork, self).__init__()
        self.network_type = "mlp"
        self._device = device
        self.cond_dim = cond_dim
        self.num_hidden_layers = num_hidden_layers
        # set up the network
        self.hidden_dim = hidden_dim
        assert num_hidden_layers % 2 == 0
        if use_spectral_norm:
            self.layers = nn.ModuleList([spectral_norm(nn.Linear(input_dim, hidden_dim))])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.layers.extend(
            [
                FiLMCondTwoLayerPreActivationResNetLinear(
                    cond_dim=cond_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout,
                )
                for i in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.to(self._device)

    def forward(self, x, condition):

        for idx, layer in enumerate(self.layers):
            if idx == 0 or idx == len(self.layers) - 1:
                x = layer(x.to(torch.float32))
            else:
                x = layer(x, condition)
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()
