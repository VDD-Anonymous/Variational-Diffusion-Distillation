import torch.nn as nn
from einops import rearrange


class FiLM(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens):
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert scale.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'
        # scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))
        return hiddens * (scale + 1) + shift



class ResFiLM(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens, idx=0):
        scale_1, shift_1, scale_2, shift_2 = self.net(conditions).chunk(4, dim = -1)
        assert scale_1.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'
        # scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))
        if idx == 0:
            return hiddens * (scale_1 + 1) + shift_1
        else:
            return hiddens * (scale_2 + 1) + shift_2
