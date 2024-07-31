import torch as ch

from common.models.abstract_gaussian_policy import AbstractGaussianPolicy

from common.models.lora.linear_lora import LinearLoRA



def createLoRAPolicy(pretrained_policy: AbstractGaussianPolicy,
                     lora_r: int = 8,
                     lora_alpha: int = 16,
                     lora_dropout: float = 0.):
    lora_policy = pretrained_policy
    for i, layer in enumerate(pretrained_policy._affine_layers):
        if isinstance(layer, ch.nn.Linear):
            old_layer = layer
            lora_policy._affine_layers[i] = LinearLoRA(layer.in_features, layer.out_features, lora_r, lora_alpha,
                                                       lora_dropout)
            lora_policy._affine_layers[i].pretrained.weight.data = old_layer.weight.data
            lora_policy._affine_layers[i].pretrained.bias.data = old_layer.bias.data


    old_mean = pretrained_policy._mean
    lora_policy._mean = LinearLoRA(pretrained_policy._mean.in_features,
                                   pretrained_policy._mean.out_features,
                                   lora_r, lora_alpha, lora_dropout)
    lora_policy._mean.pretrained.weight.data = old_mean.weight.data
    lora_policy._mean.pretrained.bias.data = old_mean.bias.data

    return lora_policy