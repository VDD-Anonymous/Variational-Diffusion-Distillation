from typing import Tuple, Dict
import torch
import numpy as np


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Modified from stable-baselines3
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Load the state of the ``RunningMeanStd`` from a ``state_dict``.

        :param state_dict: A ``state_dict`` for a ``RunningMeanStd``.
        """
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict.get("count", 1)


class TorchRunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (),
                 dtype=torch.float32, device='cpu'):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        assert device in ['cpu', 'cuda'], "Invalid device type"
        self.device = device
        self.dtype = dtype
        self.mean = torch.zeros(size=shape, dtype=dtype, device=device)
        self.var = torch.ones(size=shape, dtype=dtype, device=device)
        self.count = torch.tensor(epsilon, dtype=dtype, device=device)

    def copy(self) -> "TorchRunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = TorchRunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.detach().clone()
        new_object.var = self.var.detach().clone()
        new_object.count = self.count.detach().clone()
        return new_object

    def combine(self, other: "TorchRunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Load the state of the ``RunningMeanStd`` from a ``state_dict``.

        :param state_dict: A ``state_dict`` for a ``RunningMeanStd``.
        """
        self.mean = torch.tensor(state_dict["mean"], device=self.device, dtype=self.dtype)
        self.var = torch.tensor(state_dict["var"], device=self.device, dtype=self.dtype)
        self.count = torch.tensor(state_dict.get("count", 1), device=self.device, dtype=self.dtype)

    def to_device(self, device: str = 'cpu'):
        self.device = device
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
