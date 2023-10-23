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
from Evaluator import Evaluator
from Evaluator_1_problem import Evaluator as Evaluator1

##########################################################################################
# Machine Environment Config
debug_mode = False
use_cuda = True
cuda_device_num = 0


##########################################################################################
# parameters
logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'run_log'
    }
}

env_params = {
    'customer_size': 10,
    'mt_size': 10,
    'load_path': '',
    'synthetic_dataset': True,
    'sample': True,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 16,
    'clip': 10,
    'ff_hidden_dim': 512,
    'sample':  True if env_params['sample'] else False,  # argmax, softmax
}


eval_params = {
    'use_cuda': use_cuda,
    'cuda_device_num': cuda_device_num,
    'model_load': {
        'path': './result/train_21',  # directory path of model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to load.
    },
    'episodes': 2000,
    'batch_size': 200,
    'sample':  True if env_params['sample'] else False,
    'sample_batch': 5,
    'sample_size': 200,
    'augmentation': {
        'aug_8': False,
        'aug_9': True,
    }
}




##########################################################################################
# main

def main():
    if debug_mode:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    if env_params['synthetic_dataset']:
        evaluator = Evaluator(
            env_params=env_params,
            model_params=model_params,
            eval_params=eval_params
        )
    else:
        evaluator = Evaluator1(
            env_params=env_params,
            model_params=model_params,
            eval_params=eval_params
        )


    evaluator.run()


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
