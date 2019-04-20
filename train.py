#written by Jinbae Park
#2019-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json

import numpy as np
import tensorflow as tf

# tensorpack
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils import DictRestore
from tensorpack.train import TrainConfig
from tensorpack.train import SimpleTrainer
from tensorpack.train import SyncMultiGPUTrainer
from tensorpack.train import AsyncMultiGPUTrainer
from tensorpack.train import SyncMultiGPUTrainerReplicated
from tensorpack.train import SyncMultiGPUTrainerParameterServer
from tensorpack.train import launch_train_with_config

# custom
import Dataset
import Models


def get_train_config(config):
    size, ds_trn, ds_tst = getattr(Dataset, config['dataset'])().load_data()

    model = getattr(Models, config['model'])(config, size)

    callbacks, max_epoch = model.get_callbacks(ds_tst)

    train_config = TrainConfig(model=model,
                               dataflow=ds_trn,
                               callbacks=callbacks,
                               max_epoch=max_epoch,
                               session_init=DictRestore(dict(np.load(config['load']))) if config['load'] != 'None' else None
                               )
    return train_config


if __name__ == '__main__':
    # set config
    config = json.load(open('config.json'))

    for i in range(1, len(sys.argv)):
        keys, value = sys.argv[i].split('=')
        if ':' not in keys:
            config[keys] = value
        else:
            key1, key2 = keys.split(':')
            config[key1][key2] = value

    # set GPU machine
    if config['gpu'] is not 'None':
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

    # set log directory
    if config['logdir'] == 'None':
        logger.auto_set_dir()
    else:
        logger.set_logger_dir('train_log/' + config['logdir'], action='d')
    with open(logger.get_logger_dir() + '/config.json', 'w') as outfile:
        json.dump(config, outfile)

    # get train config
    train_config = get_train_config(config)

    # train the model
    num_gpu = max(get_num_gpu(), 1)
    if num_gpu > 1:
        launch_train_with_config(train_config, SyncMultiGPUTrainerReplicated(num_gpu, mode='nccl'))
    else:
        launch_train_with_config(train_config, SimpleTrainer())
