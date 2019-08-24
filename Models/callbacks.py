import os
import six
import json
from datetime import datetime

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


class StatsChecker(Callback):
    def __init__(self):
        self.model_dir = logger.get_logger_dir()

    def _after_train(self):
        with open(self.model_dir + '/stats.json') as file:
            stats = json.load(file)

        errors = [x['validation_error_top1'] for x in stats]
        idx = errors.index(min(errors))

        best = stats[idx]
        
        errors1 = [x['validation_error_top1'] for x in stats]
        errors5 = [x['validation_error_top5'] for x in stats]

        cnt = 5
        
        sma1 = np.zeros(len(errors1) - cnt)
        sma5 = np.zeros(len(errors1) - cnt)

        for i in range(len(sma1)):
            sma1[i] = np.sum(errors1[i:i+cnt]) / cnt
            sma5[i] = np.sum(errors5[i:i+cnt]) / cnt

        idx = np.where(sma1 == min(sma1))[0][0]
        
        best['sma_error_top1'] = '{:.4f}'.format(sma1[idx])
        best['sma_error_top5'] = '{:.4f}'.format(sma5[idx])
        best['sma_epoch_num'] = int(idx + cnt)
        #best['sma_last_error_top1'] = '{:.4f}'.format(sma1[-1])
        #best['sma_last_error_top5'] = '{:.4f}'.format(sma5[-1])

        with open(self.model_dir + '/best.json', 'w') as outfile:
            json.dump(best, outfile)


class InitSaver(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, max_to_keep=10,
                 keep_checkpoint_every_n_hours=0.5,
                 checkpoint_dir=None,
                 var_collections=None):
        """
        Args:
            max_to_keep (int): the same as in ``tf.train.Saver``.
            keep_checkpoint_every_n_hours (float): the same as in ``tf.train.Saver``.
                Note that "keep" does not mean "create", but means "don't delete".
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        self._max_to_keep = max_to_keep
        self._keep_every_n_hours = keep_checkpoint_every_n_hours

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        if checkpoint_dir is not None:
            if not tf.gfile.IsDirectory(checkpoint_dir):  # v2: tf.io.gfile.isdir
                tf.gfile.MakeDirs(checkpoint_dir)  # v2: tf.io.gfile.makedirs
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        assert self.checkpoint_dir is not None, \
            "ModelSaver() doesn't have a valid checkpoint directory."
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver(
            var_list=vars,
            max_to_keep=self._max_to_keep,
            keep_checkpoint_every_n_hours=self._keep_every_n_hours,
            write_version=tf.train.SaverDef.V2,
            save_relative_paths=True)
        # Scaffold will call saver.build from this collection
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)

    def _before_train(self):
        # graph is finalized, OK to write it now.
        time = datetime.now().strftime('%m%d-%H%M%S')
        self.saver.export_meta_graph(
            os.path.join(self.checkpoint_dir,
                         'graph-{}.meta'.format(time)),
            collection_list=self.graph.get_all_collection_keys())

        # save
        try:
            self.saver.save(
                tf.get_default_session(),
                self.path,
                global_step=tf.train.get_global_step(),
                write_meta_graph=False)
            logger.info("Model saved to %s." % tf.train.get_checkpoint_state(self.checkpoint_dir).model_checkpoint_path)
        except (OSError, IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver!")
        exit()


'''
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
'''
