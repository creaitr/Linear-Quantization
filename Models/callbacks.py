import os
import six

import tensorflow as tf

# tensorpack
from tensorpack.utils import logger
from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name
from tensorpack.callbacks.base import Callback


class CkptModifier(Callback):
    def __init__(self, model_name):
        self.model_dir = logger.get_logger_dir()
        self.model_name = model_name

    def _after_train(self):
        with open(self.model_dir + '/checkpoint', 'w') as outfile:
            outfile.write('model_checkpoint_path: \"{}\"'.format(self.model_name))
            outfile.write('all_model_checkpoint_paths: \"{}\"'.format(self.model_name))


class NpzConverter(Callback):
    def __init__(self):
        return

    def _after_train(self):
        model_dir = logger.get_logger_dir()
        ckpt_file = model_dir + '/checkpoint'
        out_file = model_dir.split('/')[-1] + '.npz'
        meta_file = None
        for file in os.listdir(model_dir):
            if '.meta' in file:
                meta_file = model_dir + '/' + file; break

        # this script does not need GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        try:
            tf.train.import_meta_graph(meta_file, clear_devices=True)
        except KeyError:
            print("If your graph contains non-standard ops, you need to import the relevant library first.")
            raise

        # loading...
        dic = varmanip.load_chkpt_vars(ckpt_file)
        dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}

        # save variables that are GLOBAL, and either TRAINABLE or MODEL
        var_to_dump = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_to_dump.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        if len(set(var_to_dump)) != len(var_to_dump):
            print("TRAINABLE and MODEL variables have duplication!")
        var_to_dump = list(set(var_to_dump))
        globvarname = set([k.name for k in tf.global_variables()])
        var_to_dump = set([k.name for k in var_to_dump if k.name in globvarname])

        for name in var_to_dump:
            assert name in dic, "Variable {} not found in the model!".format(name)

        dic_to_dump = {k: v for k, v in six.iteritems(dic) if k in var_to_dump}
        varmanip.save_chkpt_vars(dic_to_dump, out_file)
