import os
from typing import Dict

import torch as ch

from common.utils.file_utils import load_metadata_from_yaml, get_files_with_prefix
from common.utils.torch_utils import str2torchdtype, str2npdtype
from common.models.policy_factory import get_policy_network
from common.normalizers.normalizer_base import NormalizerBase
from common.normalizers.torch_normalizer import TorchNormalizer
from common.normalizers.np_normalizer import NpNormalizer

from demo_guided_rl.rl_utils import get_obs_action_dim


def load_policy_from_path(path: str = None, metadata: Dict = None, filename: str = None,
                          obs_dim: int = None, act_dim: int = None,
                          device: ch.device = 'cpu', dtype: str = 'float32'):
    if 'policy' in metadata['params'].keys():
        policy_dict = metadata['params']['policy']
    elif 'policy_params' in metadata['params'].keys():
        policy_dict = metadata['params']['policy_params']
    else:
        raise ValueError("No policy parameters found in metadata")

    if 'dataset' in metadata['params'].keys():
        env_id = metadata['params']['dataset']['dataset_name']
        obs_dim, act_dim = get_obs_action_dim(env_id)
    elif 'enviroment' in metadata['params'].keys():
        env_id = metadata['params']['env_id']
        obs_dim, act_dim = get_obs_action_dim(env_id)
    else:
        obs_dim = obs_dim
        act_dim = act_dim

    if policy_dict.get('device') is not None:
        device = policy_dict['device']
        policy_dict.pop('device')
    if policy_dict.get('dtype') is not None:
        dtype = policy_dict['dtype']
        policy_dict.pop('dtype')

    policy_dict.update({'obs_dim': obs_dim, 'action_dim': act_dim})
    policy = get_policy_network(device=device,
                                dtype=str2torchdtype(dtype),
                                **policy_dict)
    policy.load_state_dict(ch.load(os.path.join(path, filename)))
    return policy


def load_pretrained_policy(load_path: str = None, checkpoint: int = -1, obs_dim: int = None, act_dim: int = None,
                           device: ch.device = 'cpu', dtype: str = 'float32'):
    meta_data = load_metadata_from_yaml(load_path, 'config.yaml')
    pretrained_policy_path = get_files_with_prefix(load_path, 'model_state_dict')[checkpoint]
    return load_policy_from_path(load_path, meta_data, pretrained_policy_path, obs_dim, act_dim, device, dtype)


def load_pretrained_normalizer(path: str = None, checkpoint: int = -1,
                              device: ch.device = 'cpu', dtype: str = 'float32'):
    normalizer_path = get_files_with_prefix(path, 'policy_normalizer')[checkpoint]
    normalizer = NormalizerBase.load_from_path(os.path.join(path, normalizer_path))
    if isinstance(normalizer, TorchNormalizer):
        normalizer.to_device(device)
        if normalizer.dtype != str2torchdtype(dtype):
            raise ValueError(f"normalizer dtype {normalizer.dtype} does not match with {dtype}")
    elif normalizer.is_instance_of(NpNormalizer):
        if normalizer.dtype != str2npdtype(dtype):
            raise ValueError(f"normalizer dtype {normalizer.dtype} does not match with {dtype}")
    return normalizer