import tensorflow as tf


class Ridge():
    def get_func(self):
        def func(x):
            return tf.multiply(0.5, tf.reduce_sum(tf.square(x)), name='ridge')
        return func


class Lasso():
    def get_func(self):
        def func(x):
            return tf.reduce_sum(tf.abs(x), name='lasso')
        return func


class Weighted_Ridge1():
    def get_func(self, config):
        relative_lmbd = float(config['relative_lmbd'])
        fix_max = eval(config['fix_max'])
        if type(config['sub_ratio']) == str:
            inBIT, exBIT = eval(config['sub_ratio'])
        else:
            inBIT, exBIT = config['sub_ratio']
        inBIT, exBIT = inBIT ** 2, exBIT ** 2
        ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

        def clipped_Lasso(x, threshold, max_x):
            return tf.clip_by_value(tf.abs(x), threshold, max_x)

        def func(x):
            if fix_max:
                param_name = x.op.name.split('/W')[0] + '/maxW'
                maxW = tf.stop_gradient(tf.get_variable(param_name))
            else:
                maxW = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
            threshold = maxW * ratio

            weighted = tf.reduce_sum(clipped_Lasso(x, threshold, maxW) - threshold)
            weighted = tf.multiply((threshold * relative_lmbd), weighted)
            return tf.add(Ridge().get_func()(x), weighted, name='weighted_ridge1')
        return func


class Weighted_Ridge2():
    def get_func(self, config):
        relative_lmbd = float(config['relative_lmbd'])
        fix_max = eval(config['fix_max'])
        if type(config['sub_ratio']) == str:
            inBIT, exBIT = eval(config['sub_ratio'])
        else:
            inBIT, exBIT = config['sub_ratio']
        inBIT, exBIT = inBIT ** 2, exBIT ** 2
        ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

        def clipped_Lasso(x, threshold, max_x):
            return tf.clip_by_value(tf.abs(x), threshold, max_x)

        def reverse_Ridge(x, threshold, max_x):
            return (1 / (2 * (threshold - max_x))) * tf.square(x - max_x) + ((threshold + max_x) / 2)

        def func(x):
            if fix_max:
                param_name = x.op.name.split('/W')[0] + '/maxW'
                maxW = tf.stop_gradient(tf.get_variable(param_name))
            else:
                maxW = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
            threshold = maxW * ratio

            weighted = clipped_Lasso(x, threshold, maxW)
            weighted = tf.reduce_sum(reverse_Ridge(weighted, threshold, maxW) - threshold)
            weighted = tf.multiply((threshold * relative_lmbd), weighted)
            return tf.add(Ridge().get_func()(x), weighted, name='weighted_ridge2')
        return func
