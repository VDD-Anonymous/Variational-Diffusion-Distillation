from vi.score_functions.score_base import ScoreFunction

from beso.agents.diffusion_agents.beso_agent import BesoAgent
# from agents.beso_agent import BesoAgent

import torch
import einops

class BesoScoreFunction(ScoreFunction):
    def __init__(self, model: BesoAgent, sigma_index=-1, obs_dim=10, goal_dim=10, weights_type='srpo',
                 sigma_min=0.1, sigma_max=1.0, anneal_end_iter=1e6,
                 noise_level_type='uniform', device='cuda', **kwargs):
        super().__init__(model)
        self.sigma_index = sigma_index
        self.normalize_score = False
        self.goal_dim = goal_dim
        self.obs_dim = obs_dim
        self.weights_type = weights_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.annealing_end_iter = anneal_end_iter
        self.noise_level_type = noise_level_type
        self.device = device

    def __call__(self, samples, states, goals=None, iter=None, vision_task=False):
        # assert ctxt.shape[-1] == self.obs_dim + self.goal_dim, f"Expected context to have shape [..., {self.obs_dim + self.goal_dim}], got {ctxt.shape}"
        # observations = ctxt[..., :self.obs_dim]
        # goal_observations = ctxt[..., self.obs_dim:]
        return self._get_score(samples, states, goals, iter, vision_task)

    @torch.no_grad()
    def _get_score(self, samples, state, goal, iter=None, vision_task=False):
        self.model.model.eval()

        noise_level = self._get_noise_level(samples, noise_level_type=self.noise_level_type, iter=iter).to(self.device)

        weights = self._get_weights(noise_level[..., None, None], weights_type=self.weights_type).to(self.device)

        ### einpack the samples
        # b = samples.shape[0]
        # c = samples.shape[1]
        # v = samples.shape[2]

        (b, c, v, t) = samples.shape[:4]

        if vision_task:
            # self.model.model.obs_encoder.eval()
            ### hack for vision-based tasks
            agent_view_image = einops.rearrange(state[0], 'b t ... -> (b t) ... ')
            in_hand_image = einops.rearrange(state[1], 'b t ... -> (b t) ... ')
            robot_ee_pos = einops.rearrange(state[2], 'b t ... -> (b t) ... ')
            state_dict = {"agentview_image": agent_view_image,
                          "in_hand_image": in_hand_image,
                          "robot_ee_pos": robot_ee_pos}
            try:
                state = self.model.model.obs_encoder(state_dict)
            except Exception as e:
                print("error: ", e)
                print("Error in encoding the state")

            pack_state = einops.rearrange(state, '(b t) ... -> b t ...', b=b, t=t)
            pack_state = einops.repeat(pack_state, 'b t ... -> b c v t ...', c=c, v=v)
            pack_state = einops.rearrange(pack_state, 'b c v t ... -> (b c v) t ...')
        else:
            pack_state = einops.rearrange(state, 'b c v t ... -> (b c v) t ...')
        pack_samples = einops.rearrange(samples, 'b c v t ... -> (b c v) t ...')
        pack_goal = einops.rearrange(goal, 'b c v t ... -> (b c v) t ...') if goal is not None else None
        pack_noise_level = einops.rearrange(noise_level, 'b c v -> (b c v)')

        s_in = torch.ones_like(pack_noise_level).to(self.device)

        if vision_task:
            denoised = self.model.model.model(state=pack_state, action=pack_samples, goal=pack_goal, sigma=pack_noise_level * s_in)
        else:
            denoised = self.model.model(state=pack_state, action=pack_samples, goal=pack_goal, sigma=pack_noise_level * s_in)

        ### unpack the denoised samples
        denoised = einops.rearrange(denoised, '(b c v) t d -> b c v t d', b=b, c=c, v=v)

        ### score D(x;sigma) - x / sigma^2
        score = (denoised - samples) / noise_level[..., None, None] ** 2

        if self.normalize_score:
            score = score/torch.norm(score, dim=-1, keepdim=True)
        return score * weights, noise_level


    def _get_noise_level(self, samples, noise_level_type='uniform', iter=None):
        if noise_level_type == 'uniform':
            return torch.rand(samples.shape[:3]) * (self.sigma_max - self.sigma_min) + self.sigma_min
        elif noise_level_type == 'last_sigma':
            return torch.ones(samples.shape[:3]) * self.sigma_min
        elif noise_level_type == 'anneal':
            iter = min(iter, self.annealing_end_iter)
            annealed_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - iter / self.annealing_end_iter)
            return torch.ones(samples.shape[:3]) * annealed_sigma
        else:
            raise ValueError(f"Unknown noise level type: {noise_level_type}")

    def _get_weights(self, noise_level, weights_type='srpo'):
        if weights_type == 'srpo':
            return noise_level ** 2
        elif weights_type == 'stable':
            return torch.ones_like(noise_level)
        else:
            raise ValueError(f"Unknown weights type: {weights_type}")