import collections

import torch as ch
from torch import nn
from typing import Union

from demo_guided_rl.models.value.critic import BaseCritic, TargetCritic, DoubleCritic
from demo_guided_rl.samplers.dataclass import TrajectoryOffPolicy, TrajectoryOffPolicyLogpacs, \
    TrajectoryOnPolicy


def compute_td_target(critic: TargetCritic, batch: TrajectoryOffPolicy, actions_tp1: ch.Tensor,
                      entropy: Union[float, ch.Tensor] = None):
    """
    Compute td target values for Q-Learning
    Args:
        critic: DuellingCritic instance to predict qf values as well as target q values
        batch: NamedTuple with
            obs: batch observations
            actions: batch actions
            rewards: batch rewards
            next_obs: batch next observations
            dones: batch terminals
        actions_tp1: Actions for s_t+1. Usually newly generated in each iteration by the target policy network.
        entropy: Additional entropy weight for soft-Q learning update.

    Returns:

    """
    # pcont =  (1 - (done * (1 - timelimit_done))) * discount
    _, _, rewards, next_obs, pcont = batch

    if next_obs.dim() != actions_tp1.dim():
        # first dimension is sampling dimension of actions
        next_obs = next_obs[None, ...].expand((actions_tp1.shape[0],) + next_obs.shape)
    target_q_values = critic.target((next_obs, actions_tp1))
    # Check if target_q_values have been computed based on sample estimate -> MC approximation of V(s)
    target_v_values = (target_q_values.mean(0) if target_q_values.dim() >= 2 else target_q_values) + entropy
    q_target = (rewards + pcont * target_v_values).detach()
    return q_target, target_q_values


class AbstractCriticLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_criterion = nn.MSELoss()

    @property
    def loss_schema(self):
        return {}

    @property
    def stats_schema(self):
        return {}

    def forward(self, critic: BaseCritic, batch: TrajectoryOffPolicy, actions_tp1: ch.Tensor,
                logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        return self.compute_loss(critic, batch, actions_tp1, logpacs, entropy)

    def compute_loss(self, critic: BaseCritic, batch: TrajectoryOffPolicy, actions_tp1: ch.Tensor,
                     logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        """
        Compute critic loss based on replay buffer samples
        Args:
            critic: DuellingCritic instance to predict qf values as well as target q values
            batch: NamedTuple with
                obs: batch observations
                actions: batch actions
                rewards: batch rewards
                next_obs: batch next observations
                dones: batch terminals
            actions_tp1: Actions for s_t+1. Usually newly generated in each iteration by the target policy network.
            logpacs: logpacs for the actions_tp1. Used for retrace or in Soft-Q-Learning for the entropy value
            entropy: Additional entropy weight for soft-Q learning update.

        Returns:
            qf_loss, loss_info, logging_info

        """
        pass

    @property
    def requires_logpacs(self):
        return False


class TargetQLoss(AbstractCriticLoss):
    """
    Implements the Q-Learning loss with target network
    """

    @property
    def loss_schema(self):
        return {
            'qf_loss': float,
        }

    @property
    def stats_schema(self):
        return {
            'current_q'  : float,
            "bootstrap_q": float,
            'target_q'   : float,
        }

    def compute_loss(self, critic: TargetCritic, batch: TrajectoryOffPolicy, actions_tp1: ch.Tensor,
                     logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        obs, actions, rewards, next_obs, dones = batch

        current_qf = critic((obs, actions))

        with ch.no_grad():
            q_target, target_q_values = compute_td_target(critic, batch, actions_tp1, entropy)

        qf_loss = self.loss_criterion(current_qf, q_target)

        info_vals = collections.OrderedDict(current_q=current_qf.detach(),
                                            bootstrap_q=q_target.detach(),
                                            target_q=target_q_values.detach()
                                            )
        loss_dict = collections.OrderedDict(qf_loss=qf_loss.detach())

        return qf_loss, loss_dict, info_vals


class DoubleQLoss(AbstractCriticLoss):
    """
    Implements the clipped double Q-Learning loss
    """

    @property
    def loss_schema(self):
        return {
            'qf_loss' : float,
            "qf1_loss": float,
            "qf2_loss": float,
        }

    @property
    def stats_schema(self):
        return {
            'current_q1' : float,
            "current_q2" : float,
            "bootstrap_q": float,
            'target_q'   : float,
        }

    def compute_loss(self, critic: DoubleCritic, batch: TrajectoryOffPolicy, actions_tp1: ch.Tensor,
                     logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        obs, actions, rewards, next_obs, dones = batch

        current_q1, current_q2 = critic.q1((obs, actions)), critic.q2((obs, actions))

        with ch.no_grad():
            q_target, target_q_values = compute_td_target(critic, batch, actions_tp1, entropy)

        qf1_loss = self.loss_criterion(current_q1, q_target)
        qf2_loss = self.loss_criterion(current_q2, q_target)
        qf_loss = qf1_loss + qf2_loss

        info_vals = collections.OrderedDict(current_q1=current_q1.detach(),
                                            current_q2=current_q2.detach(),
                                            bootstrap_q=q_target.detach(),
                                            target_q=target_q_values.detach()
                                            )

        loss_dict = collections.OrderedDict(qf_loss=qf_loss.detach(),
                                            qf1_loss=qf1_loss.detach(),
                                            qf2_loss=qf2_loss.detach())

        return qf_loss, loss_dict, info_vals


class RetraceQLoss(AbstractCriticLoss):
    """
    Implements the retrace Q-Learning loss
    """

    def __init__(self, retrace_lambda):
        super(RetraceQLoss, self).__init__()

        self.retrace_lambda = retrace_lambda

    @property
    def loss_schema(self):
        return {
            'qf_loss': float,
        }

    @property
    def stats_schema(self):
        return {
            'current_q'  : float,
            "bootstrap_q": float,
            'target_q'   : float,
        }

    def _calc_retrace_weights(self, target_policy_logpacs, behaviour_policy_logpacs):
        """
        Calculates the retrace weights (truncated importance weights) c according to:
        c_t = min(1, π_target(a_t|s_t) / b(a_t|s_t)) where:
        π_target: target policy probabilities
        b: behaviour policy probabilities
        Args:
            target_policy_logpacs: log π_target(a_t|s_t)
            behaviour_policy_logpacs: log b(a_t|s_t)
        Returns:
            retrace weights c
        """

        log_retrace_weights = (target_policy_logpacs - behaviour_policy_logpacs).clamp(max=0)
        retrace_weights = log_retrace_weights.exp()
        assert not ch.isnan(log_retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        return self.retrace_lambda * retrace_weights

    def compute_loss(self, critic: TargetCritic, batch: TrajectoryOffPolicyLogpacs, actions_tp1: ch.Tensor,
                     logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        obs, actions, rewards, next_obs, pcont, timeout, old_logpacs = batch

        old_actions_tp1 = actions[:, 1:]
        old_actions = actions[:, :-1]

        # observations might not match for timeout states
        current_qf = critic((obs, old_actions))
        target_qf_tp1 = critic.target((next_obs, old_actions_tp1))

        if next_obs.dim() != actions_tp1.dim():
            # first dimension is sampling dimension of actions
            next_obs = next_obs[None, ...].expand((actions_tp1.shape[0],) + next_obs.shape)
        target_q_values_tp1 = critic.target((next_obs, actions_tp1))
        # V(s) = E[Q(s,a)] ~ mean(Q(s,a))
        target_v_values_tp1 = (target_q_values_tp1.mean(
            0) if target_q_values_tp1.dim() >= 3 else target_q_values_tp1) + entropy

        with ch.no_grad():
            # We don't want gradients from computing q_ret, since:
            # ∇φ (Q - q_ret)^2 ∝ (Q - q_ret) * ∇φ Q
            c_ret = self._calc_retrace_weights(logpacs, old_logpacs)
            q_ret = ch.zeros_like(target_qf_tp1)  # (B,T)

            q_ret[:, -1] = target_qf_tp1[:, -1]
            for t in reversed(range(current_qf.size(1) - 1)):
                # q_ret(xt,at) = rt + γ ̄ρt+1[q_ret(xt+1,at+1) −Q(xt+1,at+1)] + γV (xt+1)
                # In case a timeout stopped the trajectory, we only use the V-function to bootstrap of off the
                # last observed state but ignore the correction term as this is from the next trajectory
                correction = c_ret[:, t] * (q_ret[:, t + 1] - target_qf_tp1[:, t])
                q_ret[:, t] = rewards[:, t] + pcont[:, t] * (target_v_values_tp1[:, t] + ~timeout[:, t] * correction)

        qf_loss = self.loss_criterion(current_qf, q_ret)

        info_vals = collections.OrderedDict(current_q=current_qf.detach(),
                                            bootstrap_q=q_ret.detach(),
                                            target_q=target_q_values_tp1.detach()
                                            )

        loss_dict = collections.OrderedDict(qf_loss=qf_loss.detach())

        return qf_loss, loss_dict, info_vals

    @property
    def requires_logpacs(self):
        return True


class VLoss(nn.Module):
    """
    Implements the default value function loss for on-policy methods
    """

    def __init__(self, clip_critic: float = 0.0):
        super().__init__()
        self.clip_critic = clip_critic

    @property
    def loss_schema(self):
        return {}

    @property
    def stats_schema(self):
        return {}

    def compute_loss(self, critic: BaseCritic, batch: TrajectoryOnPolicy, actions_tp1: ch.Tensor,
                     logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        values = critic(batch.obs)
        returns, old_values = batch.returns, batch.values

        vf_loss = (returns - values).pow(2)

        if self.clip_critic > 0:
            # In OpenAI's PPO implementation, we clip the value function around the previous value estimate
            # and use the worse of the clipped and unclipped versions to train the value function
            vs_clipped = old_values + (values - old_values).clamp(-self.clip_critic, self.clip_critic)
            vf_loss_clipped = (vs_clipped - returns).pow(2)
            vf_loss = ch.max(vf_loss, vf_loss_clipped)

        return vf_loss.mean()


class VtraceVLoss(AbstractCriticLoss):

    def __init__(self,
                 clip_rho_threshold: ch.Tensor = 1.,
                 clip_rho_pg_threshold: ch.Tensor = 1.,
                 clip_c_threshold: ch.Tensor = 1., ):
        super().__init__()

        self.clip_rho_threshold = clip_rho_threshold
        self.clip_rho_pg_threshold = clip_rho_pg_threshold
        self.clip_c_threshold = clip_c_threshold

    def compute_loss(self, critic: BaseCritic, batch: TrajectoryOffPolicyLogpacs, actions_tp1: ch.Tensor,
                     logpacs: Union[None, ch.Tensor] = None, entropy: Union[float, ch.Tensor] = 0):
        obs, actions, rewards, next_obs, terminals, _, old_logpacs = batch
        with ch.no_grad():
            p = self.policy(obs, train=False)
            logpacs = self.policy.log_probability(p, actions)
            ratio = (logpacs - old_logpacs).exp()
            rhos = ch.min(self.clip_rho_threshold, ratio)
            cs = ch.min(self.clip_c_threshold, ratio)

            values, next_values = self.critic(obs), self.critic(next_obs)
            td_target = rewards + terminals * next_values
            delta = rhos * (td_target - values)

            vs_minus_v_xs_lst = []
            vs_minus_v_xs_tensor = ch.zeros(len(obs) + 1)
            vs_minus_v_xs = 0.0
            vs_minus_v_xs_lst.append([vs_minus_v_xs])

            for i in range(len(obs) - 1, -1, -1):
                vs_minus_v_xs = terminals[i] * cs[i][0] * vs_minus_v_xs + delta[i][0]
                vs_minus_v_xs_lst.append([vs_minus_v_xs])
                vs_minus_v_xs_tensor[i] = terminals[i] * cs[i][0] * vs_minus_v_xs_tensor[i + 1] + delta[i][0]
            vs_minus_v_xs_lst.reverse()

            vs_minus_v_xs = ch.tensor(vs_minus_v_xs_lst, dtype=ch.float)
            vs = vs_minus_v_xs[:-1] + values
            next_vs = vs_minus_v_xs[1:] + next_values
            advantage = rewards + terminals * next_vs - values
            pg_advantages = ch.min(self.clip_rho_pg_threshold) * advantage

            critic_loss = self.loss_criterion(critic(obs), vs)

            return critic_loss, pg_advantages
