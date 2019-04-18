import tensorflow as tf


def get_activation_func(name):
    if name == 'relu':
        return tf.nn.relu
    elif name == 'relu1':
        def relu1(x):
            return tf.clip_by_value(x, 0.0, 1.0, name='relu1')
        return relu1
