from common.utils.save_utils import load_pretrained_policy, load_pretrained_normalizer
from vi.score_functions.score_base import ScoreFunction
import torch


class GaussianScoreFunction(ScoreFunction):
    def __init__(self, model, normalizer=None):
        super().__init__(model)
        self.normalizer = normalizer
        self.model.eval()

    def __call__(self, samples, states):
        states = states[..., :60]
        obs = self.normalizer.normalize_obs(states)
        obs = obs.unsqueeze(1).repeat(1, samples.shape[1], 1)
        pred_mean, pred_chol = self.model(obs)

        ### Analytical gradient of log p(x) w.r.t. x
        ### grad = - \Sigma^{-1} (x-mu) =- L^-T L^-1 (x - mu)
        L_inv = torch.inverse(pred_chol)
        L_inv_T = torch.transpose(L_inv, -1, -2)
        residual = samples - pred_mean
        grad = - L_inv_T @ L_inv @ residual.unsqueeze(-1)
        grad = grad.squeeze(-1).detach()
        return grad
