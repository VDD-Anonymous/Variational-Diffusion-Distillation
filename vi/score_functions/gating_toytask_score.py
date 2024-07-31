from vi.score_functions.score_base import ScoreFunction

import torch as ch
import torch.distributions as D
import torch.nn.functional as F

import matplotlib.pyplot as plt

class ToyTaskScoreFunction(ScoreFunction):
    def __init__(self, model=None, mean_1=-1, mean_2=1, std_1=1.0, std_2=1.0, s_change=0.5, device='cuda'):
        super().__init__(model)
        self.mean_1 = ch.tensor(mean_1, device=device)
        self.std_1 = ch.tensor(std_1, device=device)
        self.mean_2 = ch.tensor(mean_2, device=device)
        self.std_2 = ch.tensor(std_2, device=device)
        self.s_change = ch.tensor(s_change, device=device)
        self.device = device

    def log_probability(self, x, s):
        mask = s < self.s_change
        log_probs = ch.zeros_like(x)
        log_probs[mask] = -0.5 * ((x[mask] - self.mean_1) / self.std_1) ** 2
        log_probs[~mask] = -0.5 * ((x[~mask] - self.mean_2) / self.std_2) ** 2
        return log_probs

    def sample(self, s, n: int = 1):
        mask = s < self.s_change
        samples = ch.zeros(s.shape + (n,), device=self.device)
        samples[mask, ...] = self.mean_1 + self.std_1 * ch.randn(s.shape + (n,), device=self.device)[mask, ...]
        samples[~mask, ...] = self.mean_2 + self.std_2 * ch.randn(s.shape + (n,), device=self.device)[~mask, ...]
        return samples

    def visualize_samples(self, s, n=10, ax=None):
        s = s.to(self.device)
        samples = self.sample(s, n)
        samples = samples.cpu().detach().numpy()
        s = s.unsqueeze(1).repeat(1, n).cpu().detach().numpy()
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(s.flatten(), samples.flatten())
            plt.show()
        else:
            ax.scatter(s.flatten(), samples.flatten())

    def __call__(self, samples:ch.Tensor, states:ch.Tensor):
        with ch.enable_grad():
            samples = samples.clone().detach().requires_grad_(True)
            samples.retain_grad()
            log_probs = self.log_probability(samples, states)
            log_probs = log_probs.sum()
            log_probs.backward()
        return samples.grad


if __name__ == '__main__':
    device = 'cuda'
    score = ToyTaskScoreFunction(std_1=0.1, std_2=0.1, s_change=0.3, device=device)
    states = ch.arange(0, 1, 0.01, device=device)

    score.visualize_samples(states, 10)