#written by Jinbae Park
#2019-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K

# tensorpack
from tensorpack import *
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils import logger
from tensorpack.tfutils.common import get_global_step_var

# custom
from .regularization import regularizers
from .regularization.custom import custom_regularize_cost
from .optimization.optimizers import get_optimizer
from .activation.activation_funcs import get_activation_func
from .quantization.quantizers import quantize_weight, quantize_activation, quantize_gradient
from .callbacks import CkptModifier
from .callbacks import StatsChecker
from .callbacks import InitSaver


class Model(ModelDesc):
    def __init__(self, config={}, size=32, nb_classes=1000):
        self.config = config

        self.size = size
        self.nb_classes = nb_classes
        self.load_config = config['load']
        self.initializer_config = config['initializer']
        self.activation = get_activation_func(config['activation'])
        self.regularizer_config = config['regularizer']
        self.quantizer_config = config['quantizer']
        self.optimizer_config = config['optimizer']
        
    def inputs(self):
        return [tf.TensorSpec([None, self.size, self.size, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        # get quantization function
        # quantize weights
        qw = quantize_weight(int(self.quantizer_config['BITW']), self.quantizer_config['name'], self.quantizer_config['W_opts'], self.quantizer_config)
        # quantize activation
        if self.quantizer_config['BITA'] in ['32', 32]:
            qa = tf.identity
        else:
            qa = quantize_activation(int(self.quantizer_config['BITA']), self.quantizer_config['name'], self.quantizer_config)
        # quantize gradient
        qg = quantize_gradient(int(self.quantizer_config['BITG']))

        def new_get_variable(v):
            name = v.op.name
            # don't quantize first and last layer
            if not name.endswith('/W') or 'conv1' in name or 'fct' in name:
                return v
            else:
                logger.info("Quantizing weight {}".format(v.op.name))
                return qw(v)

        def activate(x):
            return qa(self.activation(x))

        def resblock(x, channel, stride):
            def get_stem_full(x):
                return (LinearWrap(x)
                        .Conv2D('stem_conv_a', channel, 3)
                        .BatchNorm('stem_bn')
                        .apply(activate)
                        .Conv2D('stem_conv_b', channel, 3)())
            
            channel_mismatch = channel != x.get_shape().as_list()[3]
            if stride != 1 or channel_mismatch:
                if stride != 1:
                    x = AvgPooling('avgpool', x, stride, stride)
                x = BatchNorm('bn', x)
                x = activate(x)
                shortcut = Conv2D('shortcut', x, channel, 1)
                stem = get_stem_full(x)
            else:
                shortcut = x
                x = BatchNorm('bn', x)
                x = activate(x)
                stem = get_stem_full(x)
            return shortcut + stem

        def group(x, name, channel, nr_block, stride):
            with tf.variable_scope(name + 'blk1', reuse=tf.AUTO_REUSE):
                x = resblock(x, channel, stride)
            for i in range(2, nr_block + 1):
                with tf.variable_scope(name + 'blk{}'.format(i), reuse=tf.AUTO_REUSE):
                    x = resblock(x, channel, 1)
            return x

        def resblock_idt(x, channel, stride, first):
            def get_r(x):
                if 'InferenceTower' in x.op.name:
                    idx = x.op.name.index('/')
                    n = x.op.name[idx+1::]
                elif 'tower' in x.op.name:
                    idx = x.op.name.index('/')
                    n = x.op.name[idx+1::]
                else:
                    n = x.op.name
                n0 = n.split('blk')[0]
                n1 = n0 + 'blk1/shortcut/maxW'
                n2 = n.split('/output')[0] + '/maxW'

                if int(self.quantizer_config['BITW']) != 32: # and eval(self.quantizer_config['W_opts']['fix_max']):
                    n1 += '_stop_grad'
                    n2 += '_stop_grad'

                maxs = tf.get_collection('maxs')
                for tensor in maxs:
                    tn = tensor.op.name
                    if n1 == tn:
                        m1 = tensor
                    elif n2 == tn:
                        m2 = tensor

                r = m2 / m1

                temp = self.quantizer_config['mulR']

                if temp == '2R':
                    r2 = (1 / r) * (2.0 ** tf.floor(tf.log(r) / tf.log(2.0)))
                elif temp == 'R':
                    r2 = 1 / r

                return r2
            
            def get_stem_full(x):
                return (LinearWrap(x)
                        .Conv2D('stem_conv_a', channel, 3, strides=(stride, stride))
                        .BatchNorm('stem_bn')
                        .apply(activate)
                        .Conv2D('stem_conv_b', channel, 3, strides=(1, 1))())

            #channel_mismatch = channel != x.get_shape().as_list()[3]
            #if stride != 1 or channel_mismatch:
            if first:
                #shortcut = tf.concat([x[::, 0::2, 0::2, ::], x[::, 1::2, 1::2, ::]], -1)
                x = BatchNorm('bn', x)
                x = activate(x)
                #if stride != 1:
                #    shortcut = Conv2D('shortcut', x, channel, 1, strides=(stride, stride))
                #else:
                #    shortcut = Conv2D('shortcut', x, channel, 1)
                shortcut = Conv2D('shortcut', x, channel, 1, strides=(stride, stride))
                stem = get_stem_full(x)
            else:
                shortcut = x
                x = BatchNorm('bn', x)
                x = activate(x)
                stem = get_stem_full(x)

            if self.quantizer_config['mulR'] in ['2R', 'R']:
                r = get_r(stem)
                stem = stem * r
                
            return shortcut + stem

        def group_v2(x, name, channel, nr_block, stride):
            with tf.variable_scope(name + 'blk1', reuse=tf.AUTO_REUSE):
                x = resblock_idt(x, channel, stride, True)
            for i in range(2, nr_block + 1):
                with tf.variable_scope(name + 'blk{}'.format(i), reuse=tf.AUTO_REUSE):
                    x = resblock_idt(x, channel, 1, False)
            return x

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity,
                         kernel_initializer=tf.variance_scaling_initializer(scale=float(self.initializer_config['scale']),
                                                                            mode=self.initializer_config['mode'])):
            logits = (LinearWrap(image)
                      .Conv2D('conv1', 64, 7, strides=2)   # size=112
                      .MaxPooling('pool1', pool_size=3, strides=2, padding="SAME")      # size=56
                      #.BatchNorm('bn1')
                      #.apply(activate)
                      .apply(group_v2, 'res1', 64, 3, 1)  # size=56
                      .apply(group_v2, 'res2', 128, 4, 2)  # size=28
                      .apply(group_v2, 'res3', 256, 6, 2)  # size=14
                      .apply(group_v2, 'res4', 512, 3, 2)  # size=7
                      .BatchNorm('last_bn')
                      .apply(activate)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('fct', self.nb_classes)())
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        # regularization
        if self.regularizer_config['name'] not in [None, 'None']:
            reg_func = getattr(regularizers, self.regularizer_config['name'])().get_func(self.regularizer_config, self.quantizer_config)
            reg_cost = tf.multiply(float(self.regularizer_config['lmbd']), regularize_cost('.*/W', reg_func), name='reg_cost')
            total_cost = tf.add_n([cost, reg_cost], name='total_cost')
        else:
            total_cost = cost

        # summary
        def add_summary(logits, cost):
            err_top1 = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='err_top1')
            add_moving_summary(tf.reduce_mean(err_top1, name='train_error_top1'))
            err_top5 = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 5)), tf.float32, name='err_top5')
            add_moving_summary(tf.reduce_mean(err_top5, name='train_error_top5'))

            add_moving_summary(cost)
            add_param_summary(('.*/W', ['histogram']))  # monitor W
        add_summary(logits, cost)
            
        return total_cost

    def add_centralizing_update(self):
        def func(x):
            param_name = x.op.name
            if '/W' in param_name and 'conv1' not in param_name and 'fct' not in param_name:
                name_scope, device_scope = x.op.name.split('/W')

                inBIT, exBIT = eval(self.quantizer_config['W_opts']['threshold_bit'])
                ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

                with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
                    if eval(self.quantizer_config['W_opts']['fix_max']):
                        #max_x_name = 'post_op_internals/' + name_scope + '/maxW'
                        #max_x_name = name_scope + '/maxW'
                        max_x = tf.stop_gradient(tf.get_variable('maxW', shape=(), initializer=tf.ones_initializer, dtype=tf.float32))
                        max_x *= float(self.quantizer_config['W_opts']['max_scale'])
                    else:
                        max_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)))

                    thresh = max_x * ratio * 0.999

                    mask_name = name_scope + '/maskW'
                    mask = tf.get_variable('maskW', shape=x.shape, initializer=tf.zeros_initializer, dtype=tf.float32)

                new_x = tf.where(tf.equal(1.0, mask), tf.clip_by_value(x, -thresh, thresh), x)
                return tf.assign(x, new_x, use_locking=False).op

        self.centralizing = func

    def add_stop_grad(self):
        def func(grad, val):
            val_name = val.op.name
            if '/W' in val_name and 'conv1' not in val_name and 'fct' not in val_name:
                name_scope, device_scope = val.op.name.split('/W')
                
                with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
                    mask_name = name_scope + '/maskW'
                    mask = tf.get_variable('maskW', shape=val.shape, initializer=tf.zeros_initializer, dtype=tf.float32)

                    #zero_grad = tf.zeros(shape=grad.shape)

                    new_grad = tf.where(tf.equal(1.0, mask), grad, grad * 0.1)
                    
                return new_grad

        self.stop_grad = func
        
    def add_clustering_update(self, n_ls):
        def func(grad, val):
            val_name = val.op.name
            if '/W' in val_name and 'conv1' not in val_name and 'fct' not in val_name:
                cluster_mask_name = val_name.split('/W')[0] + '/cluster_maskW'
                cluster_mask = tf.get_variable(cluster_mask_name, shape=grad.shape, initializer=tf.zeros_initializer, dtype=tf.float32)

                total_grad = tf.zeros(shape=grad.shape)
                sum_grads = []

                for n in n_ls:
                    sum_grads.append(
                        tf.reduce_sum(tf.where(tf.equal(np.float32(n), cluster_mask), grad, total_grad)) /
                        tf.reduce_sum(tf.where(tf.equal(np.float32(n), cluster_mask), tf.ones(grad.shape), tf.zeros(grad.shape))))

                for i in range(len(n_ls)):
                    total_grad = tf.where(tf.equal(np.float32(n_ls[i]), cluster_mask),
                                          tf.fill(grad.shape, sum_grads[i]), total_grad)
                return total_grad

        self.clustering = func

    def add_masking_update(self):
        gamma = 0.0001
        crate = 3.

        inBIT, exBIT = eval(self.quantizer_config['W_opts']['threshold_bit'])
        ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

        def func(val):
            val_name = val.op.name
            if '/W' in val_name and 'conv1' not in val_name and 'fct' not in val_name:
                name_scope, device_scope = x.op.name.split('/W')

                with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
                    if eval(self.quantizer_config['W_opts']['fix_max']) ==True:
                        max_x = tf.stop_gradient(
                            tf.get_variable('maxW', shape=(), initializer=tf.ones_initializer, dtype=tf.float32))
                        max_x *= float(self.quantizer_config['W_opts']['max_scale'])
                    else:
                        max_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
                    mask = tf.get_variable('maskW', shape=val.shape, initializer=tf.zeros_initializer, dtype=tf.float32)

                probThreshold = (1 + gamma * get_global_step_var()) ** -1

                # Determine which filters shall be updated this iteration
                random_number = K.random_uniform(shape=(1, 1, 1, int(mask.shape[-1])))
                random_number1 = K.cast(random_number < probThreshold, dtype='float32')
                random_number2 = K.cast(random_number < (probThreshold * 0.1), dtype='float32')

                thresh = max_x * ratio * 0.999

                # Incorporate hysteresis into the threshold
                alpha = thresh
                beta = 1.2 * thresh

                # Update the significant weight mask by applying the threshold to the unmasked weights
                abs_kernel = K.abs(x=val)
                new_mask = mask - K.cast(abs_kernel < alpha, dtype='float32') * random_number1
                new_mask = new_mask + K.cast(abs_kernel > beta, dtype='float32') * random_number2
                new_mask = K.clip(x=new_mask, min_value=0., max_value=1.)
                return tf.assign(mask, new_mask, use_locking=False).op

        self.masking = func

    def optimizer(self):
        opt = get_optimizer(self.optimizer_config)

        if self.quantizer_config['name'] == 'linear' and eval(self.quantizer_config['W_opts']['stop_grad']):
            self.add_stop_grad()
            opt = optimizer.apply_grad_processors(opt, [gradproc.MapGradient(self.stop_grad)])
        if self.quantizer_config['name'] == 'linear' and eval(self.quantizer_config['W_opts']['centralized']):
            self.add_centralizing_update()
            opt = optimizer.PostProcessOptimizer(opt, self.centralizing)
        if self.quantizer_config['name'] == 'cent':
            self.add_centralizing_update()
            opt = optimizer.PostProcessOptimizer(opt, self.centralizing)
        if self.quantizer_config['name'] == 'cluster' and eval(self.load_config['clustering']):
            opt = optimizer.apply_grad_processors(opt, [gradproc.MapGradient(self.clustering)])
        if self.quantizer_config['name'] == 'linear' and eval(self.quantizer_config['W_opts']['pruning']):
            self.add_masking_update()
            opt = optimizer.PostProcessOptimizer(opt, self.masking)
        return opt

    def get_callbacks(self, ds_tst):
        if self.config['num_gpu'] == 1:
            runner = InferenceRunner(ds_tst,
                                     [ScalarStats('cross_entropy_loss'),
                                      ClassificationError('err_top1', summary_name='validation_error_top1'),
                                      ClassificationError('err_top5', summary_name='validation_error_top5')])
        elif self.config['num_gpu'] > 1:
            runner = DataParallelInferenceRunner(ds_tst,
                                                 [ScalarStats('cross_entropy_loss'),
                                                  ClassificationError('err_top1', summary_name='validation_error_top1'),
                                                  ClassificationError('err_top5', summary_name='validation_error_top5')],
                                                 list(range(self.config['num_gpu'])))
        callbacks=[
            ModelSaver(max_to_keep=1),
            runner,
            MinSaver('validation_error_top1'),
            CkptModifier('min-validation_error_top1'),
            StatsChecker()
        ]




        # scheduling learning rate
        if self.optimizer_config['lr_schedule'] not in [None, 'None']:
            if type(self.optimizer_config['lr_schedule']) == str:
                callbacks += [ScheduledHyperParamSetter('learning_rate', eval(self.optimizer_config['lr_schedule']))]
            else:
                callbacks += [ScheduledHyperParamSetter('learning_rate', self.optimizer_config['lr_schedule'])]
        else:
            callbacks += [ScheduledHyperParamSetter('learning_rate',
                                      [(0, 0.1), (30, 0.01), (60, 0.001), (90, 0.0001), (100, 0.00001)])]

        if eval(self.config['save_init']):
            callbacks = [InitSaver()]

        max_epoch = int(self.optimizer_config['max_epoch'])
        return callbacks, max_epoch
