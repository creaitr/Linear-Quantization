import tensorflow as tf

def Ridge(x, name):
    return tf.multiply(0.5, tf.reduce_sum(tf.square(x)))

def Lasso(x, name):
    return tf.reduce_sum(tf.abs(x))

def Partial_Lasso(x, name):

    return x

def Custom1(x, name):

    return x

def Custom2(x, name):

    return x
