from typing import Tuple

from common.losses.value_loss import AbstractCriticLoss, TargetQLoss, DoubleQLoss, \
    RetraceQLoss
import torch as ch

from demo_guided_rl.models.value.critic import BaseCritic
from demo_guided_rl.models.value.critic_factory import get_critic


def get_value_loss_and_critic(value_loss_type: str, retrace_lambda: float = 0, device: ch.device = "cpu",
                              dtype=ch.float32, **kwargs_critic) -> Tuple[AbstractCriticLoss, BaseCritic]:
    """
    Value loss and critic network factory
    Args:
        value_loss_type: what type of loss to use, one of 'double' (double Q-learning),
        'duelling' (Duelling Double Q-learning), and 'retrace' (Retrace by Munos et al. 2016)
        device: torch device
        discount_factor: reward discounting factor for computing Q target values
        retrace_lambda: lambda weight for retrace algorithm by Munos, et al., 2016
        dtype: torch dtype
        **kwargs_critic: critic arguments

    Returns:
        Value loss and critic instance
    """

    if value_loss_type == "target":
        value_loss = TargetQLoss()
        critic_type = value_loss_type
    elif value_loss_type == "double":
        value_loss = DoubleQLoss()
        critic_type = value_loss_type
    elif value_loss_type == "retrace":
        value_loss = RetraceQLoss(retrace_lambda)
        critic_type = "target"
    elif value_loss_type == "vlearn" or value_loss_type == "vlearn_double":
        value_loss = ch.empty(0)
        critic_type = value_loss_type
    elif value_loss_type == "vtrace":
        value_loss = ch.empty(0)
        critic_type = value_loss_type
    elif value_loss_type == "value":
        # value_loss = ValueLoss(discount_factor, clip_critic)
        value_loss = ch.empty(0)
        critic_type = value_loss_type
    else:
        raise ValueError(f"Invalid value_loss type {value_loss_type}. Select one of 'double', 'duelling', 'retrace'.")

    critic = get_critic(critic_type=critic_type, device=device, dtype=dtype, **kwargs_critic)

    return value_loss.to(device, dtype), critic
