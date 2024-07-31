import torch as ch
import torch.utils.data as Data
import numpy as np

from vi.experiment_managers.base_manager import BaseManager
from vi.score_functions.gmm_score import GMMScoreFunction

import einops

from common.utils.plot_utils import plot_2d_gaussians
import matplotlib.pyplot as plt


class ToyTask2DManager(BaseManager):

    def __init__(self, num_datapoints=1000, score_fn_params={}, datasets_config={}, seed=0, device='cuda', **kwargs):
        super().__init__(seed, device, **kwargs)
        r = score_fn_params.get("r", 1)
        std = score_fn_params.get("std", 0.4)
        n_component = score_fn_params.get("num_components", 4)
        thetas = ch.linspace(0, 2 * np.pi, n_component + 1)[:-1]
        means = ch.stack([r * ch.cos(thetas), r * ch.sin(thetas)], dim=-1)
        chols = ch.eye(2).view(1, 2, 2).repeat(n_component, 1, 1) * std
        self.score_function = GMMScoreFunction(means=means, chols=chols, device=device)
        self.scaler = None
        self.r = r
        # self.std = std
        # self.num_datapoints = num_datapoints
        self.actions = self.score_function.sample(num_datapoints).unsqueeze(1).to(device)
        self.test_actions = self.score_function.sample(num_datapoints).unsqueeze(1).to(device)
        self.states = ch.ones_like(self.actions[:, 0]).unsqueeze(1).to(device)
        self.dataset = Data.TensorDataset(self.states, self.actions)
        self.test_dataset = Data.TensorDataset(self.states, self.test_actions)

        self.train_loader = Data.DataLoader(self.dataset, batch_size=datasets_config['batch_size'], shuffle=True)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=datasets_config['batch_size'], shuffle=True)

    def env_rollout(self, agent, n_episodes: int, **kwargs):

        agent.eval()
        fig, ax = self.score_function.visualize_grad_and_cmps(x_range=[-1.5*self.r, 1.5*self.r],
                                                            y_range=[-1.5*self.r, 1.5*self.r], n=12)
        means, chols, gating = agent(self.states[:1, ...])

        means = einops.rearrange(means, '1 n 1 d -> n d').cpu().detach().numpy()
        chols = einops.rearrange(chols, '1 n 1 d1 d2 -> n d1 d2').cpu().detach().numpy()
        plot_2d_gaussians(means, chols, ax, color='orange')  # plot
        ax.set_aspect('equal')

        idx = kwargs.get("idx", None)
        if idx is not None:
            plt.axis('off')
            plt.show()
        agent.train()

        return {'mse': 0}

    def get_scaler(self, **kwargs):
        return None
    
    def preprocess_data(self, batch_data):
        return batch_data[0], batch_data[1], None

    def get_score_function(self, **kwargs):
        return self.score_function

    def get_train_and_test_datasets(self, **kwargs):
        return self.train_loader, self.test_loader

if __name__ == '__main__':
    device = 'cuda'
    manager = ToyTask2DManager(r=1, n_component=1, std=0.4, device=device, datasets_config={'batch_size': 64})
    manager.score_function.visualize_grad_and_cmps(x_range=[-2, 2], y_range=[-2, 2], n=15)
    plt.show()