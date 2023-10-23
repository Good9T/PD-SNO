import os
import sys
import logging

##########################################################################################
# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for utils
sys.path.append('')

from my_utils import create_logger
from Trainer import Trainer

##########################################################################################
# Machine Environment Config
debug_mode = False
use_cuda = True
cuda_device_num = 0


##########################################################################################
# parameters

env_params = {
    'customer_size': 50,
    'mt_size': 50,
    'capacity': 20,
    'demand_min': 1,
    'demand_max': 10,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 16,
    'clip': 10,
    'ff_hidden_dim': 512,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [1701],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': use_cuda,
    'cuda_device_num': cuda_device_num,
    'epochs': 2000,
    'episodes': 10000,
    'batch_size': 16,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 50,
        'img_save_interval': 200,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'obj.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'loss.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './result/',  # directory path of model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to load.

    },
    'augmentation': {
        'aug_8': True,
        'aug_9': False,
    }
}

logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if debug_mode:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 1
    trainer_params['episodes'] = 1
    trainer_params['batch_size'] = 1
    global model_params


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(debug_mode))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(use_cuda, cuda_device_num))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
