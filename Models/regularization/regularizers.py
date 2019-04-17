import tensorflow as tf


def Ridge(x):
    return tf.multiply(0.5, tf.reduce_sum(tf.square(x)), name='ridge')


def Lasso(x):
    return tf.reduce_sum(tf.abs(x), name='lasso')


def partial_clip(x, start, end):
    return tf.clip_by_value(tf.abs(x), start, end)


def Weighted_Ridge1(x):
    end = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
    start = tf.multiply(end, (3 / 7))
    weighted = tf.multiply(start, partial_clip(x, start, end))
    weighted = tf.multiply(0.5, weighted)
    return tf.add(Ridge(x), weighted, name='weighted_ridge1')


def Weighted_Ridge2(x):
    end = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
    start = tf.multiply(end, (3 / 7))
    weighted = partial_clip(x, start, end)
    weighted = tf.multiply(-(3 / 8), tf.square(weighted - end))
    weighted = tf.multiply(start, weighted)
    weighted = tf.multiply(0.5, weighted)
    return tf.add(Ridge(x), weighted, name='weighted_ridge2')
