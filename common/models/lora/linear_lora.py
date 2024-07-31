import math
from torch import nn


class LinearLoRA(nn.Module):
    """
    A low-rank adapted linear layer.
    Args:
        in_dim: int = An integer representing the input dimension of the linear layer
        out_dim: int = An integer representing the output dimension of the linear layer
        r: int = An integer representing the rank of the low-rank approximated matrices
        lora_alpha: int = An integer representing the numerator of the scaling constant alpha / r
        lora_dropout: float = A float between 0 and 1 representing the dropout probability
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        # Check that the rank is at least 1
        assert r > 0, "Variable 'r' is not greater than zero. Choose a rank of 1 or greater."
        # recreate the linear layer and freeze it (the actual weight values will be copied in outside of this class)
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False
        # create the low-rank A matrix and initialize with same method as in Hugging Face PEFT library
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # create the low-rank B matrix and initialize to zero
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        nn.init.constant_(self.lora_B.weight, 0)
        # scaling constant
        self.scaling = self.lora_alpha / self.r
    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling
        return pretrained_out + lora_out