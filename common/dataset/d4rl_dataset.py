import torch as ch
from torch.utils.data import TensorDataset, random_split
import pathlib

import gym
import d4rl


class SimpleDataset(TensorDataset):
    def __init__(self,
                 dataset_name: str = None,
                 dataset_path: str = None,
                 device: str = 'cpu',
                 dtype: ch.dtype = ch.float32,
                 ):
        env = gym.make(dataset_name)
        if dataset_path is None:
            self.dataset = env.get_dataset()
        else:
            dataset_path = pathlib.Path(dataset_path)
            self.dataset = env.get_dataset(dataset_path)
        tensors = [ch.from_numpy(self.dataset['observations']), ch.from_numpy(self.dataset['actions'])]
        tensors = [t.to(device).type(dtype) for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.observations = self.tensors[0]
        self.actions = self.tensors[1]

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)


def get_d4rl_train_val_dataset(val_ratio: float = 0.2,
                               seed: int = 42,
                               dataset_name: str = None,
                               dataset_path: str = None,
                               device: str = 'cuda',
                               dtype: str = 'float32',
                               **kwargs):
    dataset = SimpleDataset(dataset_name=dataset_name,
                            dataset_path=dataset_path,
                            device=device,
                            dtype=ch.float32 if dtype == 'float32' else ch.float64)
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=ch.Generator().manual_seed(seed))
    return train_dataset, val_dataset
