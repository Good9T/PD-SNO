import torch
from logging import getLogger
from EnvEval import EnvEval
from Model import Model
from my_utils import *
from copy import deepcopy
import numpy as np

class Evaluator:
    def __init__(self,
                 env_params,
                 model_params,
                 eval_params):

        # params
        self.env_params = env_params
        self.model_params = model_params
        self.eval_params = eval_params
        self.batch_size = self.eval_params['batch_size']
        self.sample = self.eval_params['sample']
        self.sample_batch = self.eval_params['sample_batch']
        self.sample_size = self.eval_params['sample_size']
        self.augmentation = self.eval_params['augmentation']
        self.data = None

        # result, log
        self.logger = getLogger(name='evaluator')
        self.result_folder = get_result_folder()

        # cuda
        use_cuda = self.eval_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.eval_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # main components
        self.env = EnvEval(**self.env_params)
        self.model = Model(**self.model_params)

        # restore
        self.model_load = self.eval_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**self.model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score, score_aug = self.eval_1_sample(self.sample_batch)

        # logs
        self.logger.info('Evaluate done, score without aug: {:.4f}, score with aug: {:.4f}'
                         .format(score, score_aug))

    def eval_1_sample(self, sample_batch):
        # aug
        if self.augmentation['aug_8']:
            aug_type = '8'
        elif self.augmentation['aug_9']:
            aug_type = '9'
        elif self.augmentation['aug_16']:
            aug_type = '16'
        else:
            aug_type = None

        # prepare
        self.model.eval()

        for i in range(sample_batch):
            with torch.no_grad():
                if i == 0:
                    self.data = self.env.load_dataset_problem(self.sample_size, self.sample, aug_type)
                    self.env.backup_problem()
                else:
                    self.env.load_backup_problem()
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)

            # aug_number = self.data['aug_number']

            # mt rollout
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, mt)
                state, reward, done = self.env.step(selected)

            # score
            reward_aug = reward.reshape(-1, self.env.mt_size)
            # shape: (aug * sample, mt)
            reward_max_mt, _ = reward_aug.max(dim=1)
            # shape: (aug * sample)
            reward_max_aug_mt, _ = reward_max_mt.max(dim=0)
            # shape : (1)
            score_no_aug = -reward_max_mt[0].float()

            score_aug_temp = -reward_max_aug_mt
            score_aug = min(score_aug, score_aug_temp) if i > 0 else score_aug_temp
        score_aug = score_aug * self.data['scale']
        score_no_aug = score_no_aug * self.data['scale']

        return score_no_aug.item(), score_aug.item()
