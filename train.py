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
import Utils

def get_train_config(config):
    # get dataset
    size, nb_classes, ds_trn, ds_tst = getattr(Dataset, config['dataset'])().load_data(num_gpu=config['num_gpu'])

    # get the model
    model = getattr(Models, config['model'])(config, size, nb_classes)

    # get callbacks such as ModelSaver
    callbacks, max_epoch = model.get_callbacks(ds_tst)

    # load pre-trained model
    if config['load']['name'] in [None, 'None', '']:
        session_init = None
    else:
        saved_model = dict(np.load(config['load']['name']))
        if eval(config['load']['find_max']):
            saved_model = Utils.find_max(saved_model, config['load'])
        elif eval(config['load']['find_99th']):
            saved_model = Utils.find_99th(saved_model, config['load'])
        if eval(config['load']['make_mask']):
            saved_model = Utils.make_mask(saved_model, config)
        if eval(config['load']['clustering']):
            saved_model, cluster_idx_ls = Utils.clustering(saved_model, config['quantizer'])
            model.add_clustering_update(cluster_idx_ls)
        session_init = DictRestore(saved_model)

    # set a train configuration
    train_config = TrainConfig(model=model,
                               dataflow=ds_trn,
                               callbacks=callbacks,
                               max_epoch=max_epoch,
                               session_init=session_init,
                               steps_per_epoch=1281167//256 if 'ImageNet' in config['model'] else None
                               )
    return train_config


if __name__ == '__main__':
    # set config
    if len(sys.argv) > 1 and sys.argv[1] == 'config':
        config_file = sys.argv[2]
        argv = sys.argv[3::] if len(sys.argv) > 3 else []
    else:
        config_file = 'config.json'
        argv = sys.argv[1::] if len(sys.argv) > 1 else []
    config = json.load(open(config_file))

    for i in range(len(argv)):
        keys, value = argv[i].split('=')
        keys = keys.split(':')
        temp = config
        for i in range(len(keys) - 1):
            temp = temp[keys[i]]
        temp[keys[-1]] = value

    # set GPU machine
    if config['gpu'] in [None, 'None', '']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        num_gpu = 0
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        num_gpu = max(get_num_gpu(), 1)
    config['num_gpu'] = num_gpu

    # set log directory
    if config['logdir'] in [None, 'None', '']:
        logger.auto_set_dir()
    else:
        logger.set_logger_dir('train_log/' + config['logdir'], action='d')
    # save configuration
    with open(logger.get_logger_dir() + '/config.json', 'w') as outfile:
        json.dump(config, outfile)

    # get train config
    train_config = get_train_config(config)

    # train the model
    if num_gpu > 1:
        launch_train_with_config(train_config, SyncMultiGPUTrainerReplicated(num_gpu, mode='nccl'))
    else:
        launch_train_with_config(train_config, SimpleTrainer())
