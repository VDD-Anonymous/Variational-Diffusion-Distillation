import torch
import torch as ch
import einops

import copy

from vi.algorithms.avid_joint_gmm import VDD
from vi.models.joint_gmm_policy import JointGaussianMixtureModel
from common.utils.network_utils import get_lr_schedule, get_optimizer


class AmortizedExpectationMaximization(VDD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_e_steps = kwargs.get("n_e_steps", 1)
        self.n_m_steps = kwargs.get("n_m_steps", 1)

    def iterative_train(self, n):
        train_metrics_dict = {}
        for i in range(self.n_e_steps):
            try:
                batch = next(self.iter_train_dataloader)
            except StopIteration:
                self.iter_train_dataloader = iter(self.train_dataloader)
                batch = next(self.iter_train_dataloader)
            batch = self.exp_manager.preprocess_data(batch)
            train_metrics_dict = self.e_step(batch)
        for i in range(self.n_m_steps):
            try:
                batch = next(self.iter_train_dataloader)
            except StopIteration:
                self.iter_train_dataloader = iter(self.train_dataloader)
                batch = next(self.iter_train_dataloader)
            batch = self.exp_manager.preprocess_data(batch)
            train_metrics_dict.update(self.m_step(batch))

        return train_metrics_dict

    def e_step(self, batch):
        return self.gating_update(batch[0], batch[1], batch[2])

    def m_step(self, batch):
        states, actions, goal_states = batch
        logging_dict = {}
        pred_means, pred_chols, pred_gatings = self.agent(states, goal_states)

        b, c, t, a = pred_means.shape

        actions = einops.rearrange(actions, 'b t a -> b 1 1 t a')
        actions = einops.repeat(actions, 'b 1 1 t a -> b c 1 t a', c=c)
        actions = einops.rearrange(actions, 'b c v t a -> (b t) c v a')

        pred_means = einops.rearrange(pred_means, 'b c t a -> (b t) c a')
        pred_chols = einops.rearrange(pred_chols, 'b c t a1 a2 -> (b t) c a1 a2')
        pred_gatings = einops.rearrange(pred_gatings, 'b c t -> (b t) c')

        # shape ((bxt), c, 1)
        responsibilities = self.agent.log_responsibilities(pred_means.clone().detach(),
                                                           pred_chols.clone().detach(),
                                                           pred_gatings.clone().detach(),
                                                           actions.clone().detach()).squeeze(-1)

        loglikelihood = self.agent.joint_cmps.log_probability((pred_means, pred_chols), actions.squeeze(-2))

        m_loss = - (loglikelihood * responsibilities.exp()).mean()

        self.cmps_optimizer.zero_grad(set_to_none=True)
        m_loss.backward()
        self.cmps_optimizer.step()
        if self.cmps_lr_scheduler is not None:
            self.cmps_lr_scheduler.step()

        logging_dict.update({"m_loss": m_loss.item()})

        if torch.isnan(loglikelihood).any():
            print("Nan in m_loss")
        if torch.isnan(responsibilities).any():
            print("Nan in responsibilities")
        if torch.isnan(m_loss).any():
            print("Nan in m_loss")

        return logging_dict

    @staticmethod
    def create_em_agent(policy_params, optimizer_params, training_params, train_dataset, test_dataset, **kwargs):
        policy = JointGaussianMixtureModel(**policy_params, device=training_params['device'], dtype=training_params['dtype'])

        if policy_params['vision_task']:
            score_function = kwargs.get('score_function', None)
            if policy_params['copy_vision_encoder']:
                policy.vision_encoder = copy.deepcopy(score_function.model.model.obs_encoder)
            if policy_params['train_vision_encoder']:
                cmps_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                               model_parameters=list(policy.joint_cmps.parameters()) + list(
                                                   policy.vision_encoder.parameters()),
                                               learning_rate=optimizer_params['cmps_lr'],
                                               weight_decay=optimizer_params['cmps_weight_decay'])
            else:
                cmps_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                               model_parameters=list(policy.joint_cmps.parameters()),
                                               learning_rate=optimizer_params['cmps_lr'],
                                               weight_decay=optimizer_params['cmps_weight_decay'])
        else:
            score_function = None

            cmps_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                           model_parameters=list(policy.joint_cmps.parameters()),
                                          learning_rate=optimizer_params['cmps_lr'],
                                          weight_decay=optimizer_params['cmps_weight_decay'])

        cmps_lr_scheduler = get_lr_schedule(optimizer_params['cmps_lr_schedule'],
                                           cmps_optimizer, training_params['max_train_iters']) \
            if optimizer_params['cmps_lr_schedule'] is not None else None

        if policy_params['learn_gating']:
            gating_net_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                                 model_parameters=policy.gating_network.parameters(),
                                                 learning_rate=optimizer_params['gating_lr'],
                                                 weight_decay=optimizer_params['gating_weight_decay'])
            gating_lr_scheduler = get_lr_schedule(optimizer_params['gating_lr_schedule'],
                                                    gating_net_optimizer, training_params['max_train_iters']) \
                    if optimizer_params['gating_lr_schedule'] is not None else None
        else:
            gating_net_optimizer = None
            gating_lr_scheduler = None

        return AmortizedExpectationMaximization(agent=policy, cmps_optimizer=cmps_optimizer, gating_optimizer=gating_net_optimizer,
                                          cmps_lr_scheduler=cmps_lr_scheduler,
                                          gating_lr_scheduler=gating_lr_scheduler,
                                          score_function=score_function,
                                          train_dataloader=train_dataset,
                                          test_dataloader=test_dataset,
                                          learn_gating=policy_params['learn_gating'],
                                          **training_params)
