import torch
import os

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from tqdm import tqdm
from common.utils.file_utils import process_cw2_train_rep_config_file

from vi.algorithms.imc import IMC

from vi.experiment_managers.manager_factory import create_experiment_manager

from common.utils.torch_utils import global_seeding


class EMExperiments(experiment.AbstractIterativeExperiment):

    def initialize(
        self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray
    ) -> None:
        algo_config = cw_config["params"]

        gpu_id = algo_config.get("gpu_id", None)

        cpu_cores = cw_config.get("cpu_cores", None)

        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if gpu_id == 0:
                cpu_cores = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
            elif gpu_id == 1:
                cpu_cores = {6, 7, 8, 9, 10, 11}
            elif gpu_id == 2:
                cpu_cores = {12, 13, 14, 15, 16}
            elif gpu_id == 3:
                cpu_cores = {17, 18, 19, 20, 21}


        self.device = algo_config['train_params']['device']

        experiment_config = algo_config['experiment_params']

        experiment_config['device'] = self.device

        experiment_config['cpu_cores'] = tuple(cpu_cores)

        self.experiment_manager = create_experiment_manager(experiment_config)

        self.train_dataset, self.test_dataset = self.experiment_manager.get_train_and_test_datasets()

        score_function = self.experiment_manager.get_score_function()

        cw_config['seed'] = rep

        global_seeding(cw_config['seed'])

        self.em_agent = IMC.create_imc_agent(policy_params=algo_config['policy_params'],
                                                                         optimizer_params=algo_config['optimizer_params'],
                                                                         training_params=algo_config['train_params'],
                                                                         train_dataset=self.train_dataset,
                                                                         test_dataset=self.test_dataset,
                                                                         score_function=score_function,)

        self.em_agent.get_scaler(self.experiment_manager.scaler)
        self.em_agent.get_exp_manager(self.experiment_manager)

        train_config = algo_config['train_params']

        self.num_final_rollouts = train_config.get('num_final_rollouts', 100)
        self.num_final_contexts = train_config.get('num_final_contexts', 60)
        print(f'num_final_rollouts: {self.num_final_rollouts}, num_final_contexts: {self.num_final_contexts}')

        self.test_interval = algo_config['train_params']['test_interval']
        self.env_rollout_interval = algo_config['train_params']['env_rollout_interval']
        self.num_env_rollouts = algo_config['train_params']['num_rollouts']
        self.num_env_contexts = train_config.get('num_contexts', 5)
        self.save_model_dir = os.path.join(cw_config['_rep_log_path'], 'model')
        self.model_select_metric = algo_config['experiment_params']['model_select_metric']
        self.max_reward = -1e10

        self.progress_bar = tqdm(total=algo_config["train_params"]["max_train_iters"], disable=False)


    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        """
        Arguments:
            cw_config {dict} -- clusterwork experiment configuration
            rep {int} -- repetition counter
            n {int} -- iteration counter
        """

        train_metric_dict = self.em_agent.iterative_train(n)

        if n % self.test_interval == 0:
            test_metric_dict = self.em_agent.iterative_evaluate()
        else:
            test_metric_dict = {}

        if n % self.env_rollout_interval == 0 and n > 0:
            rollout_dict = self.experiment_manager.env_rollout(self.em_agent.agent, self.num_env_rollouts, num_ctxts=self.num_env_contexts)
            if rollout_dict[self.model_select_metric] > self.max_reward:
                self.em_agent.save_best_model(path=self.save_model_dir)
                self.max_reward = rollout_dict[self.model_select_metric]
                print(f'save new best model at iteration {n}, with {self.model_select_metric}: {rollout_dict[self.model_select_metric]}')
        else:
            rollout_dict = {}

        self.progress_bar.update(1)

        if n == cw_config["iterations"] - 1:
            best_model_path = os.path.join(self.save_model_dir, 'best_model.pt')
            best_agent = torch.load(best_model_path)
            self.experiment_manager.goal_idx_offset = 0
            final_rollout_dict = self.experiment_manager.env_rollout(best_agent, self.num_final_rollouts, num_ctxts=self.num_final_contexts)
            final_rollout_dict = {"final_" + k: v for k, v in final_rollout_dict.items()}
            print(f'Final rollout with best model: {final_rollout_dict}')
        else:
            final_rollout_dict = {}


        return {**train_metric_dict, **test_metric_dict, **rollout_dict, **final_rollout_dict}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:

        if (n + 1) % (cw_config['iterations']//cw_config['num_checkpoints']) == 0 \
                or (n+1) == cw_config["params"]["train_params"]["max_train_iters"]:

            self.em_agent.save_model(iteration=n + 1, path=self.save_model_dir)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False) -> None:
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(EMExperiments)
    if cw.config.exp_configs[0]['enable_wandb']:
        cw.add_logger(WandBLogger())

    process_cw2_train_rep_config_file(cw.config, overwrite=True)

    cw.run()
