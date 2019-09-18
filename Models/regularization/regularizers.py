import tensorflow as tf


class Ridge():
    def get_func(self, *args, **kwargs):
        def func(x):
            return tf.multiply(0.5, tf.reduce_sum(tf.square(x)), name='ridge')
        return func


class Lasso():
    def get_func(self, *args, **kwargs):
        def func(x):
            return tf.reduce_sum(tf.abs(x), name='lasso')
        return func


class Weighted_Ridge1():
    def get_func(self, config, q_config):
        if q_config['name'] == 'intQ':
            relative_lmbd = float(config['sub_reg']['relative_lmbd'])
            fix_max = eval(config['sub_reg']['fix_max'])

            esl = int(q_config['BITW'])
            n = (esl - 1) / 2

            def clipped_Lasso(x, threshold, max_x):
                return tf.clip_by_value(tf.abs(x), threshold, max_x * 10)

            def func(x=None, temp_max=None):
                if 'conv1' not in x.op.name and 'fct' not in x.op.name:
                    if temp_max != None:
                        maxW = tf.stop_gradient(temp_max)
                    elif fix_max:
                        param_name = x.op.name.split('/W')[0] + '/maxW'
                        maxW = tf.stop_gradient(tf.Variable(1.0, name=param_name))
                    else:
                        maxW = tf.stop_gradient(tf.reduce_max(tf.abs(x)))

                        
                    threshold = 0.05 * maxW + (0.9 * maxW / n)
                    thresh_temp = threshold * 0.999

                    weighted = tf.reduce_sum(clipped_Lasso(x, threshold, maxW) - threshold)
                    weighted = tf.multiply((relative_lmbd * maxW), weighted)
                    return tf.add(Ridge().get_func()(x), weighted, name='weighted_ridge1')
                else:
                    return Ridge().get_func()(x)
            return func
        
        relative_lmbd = float(config['sub_reg']['relative_lmbd'])
        fix_max = eval(config['sub_reg']['fix_max'])
        if type(config['sub_reg']['sub_ratio']) == str:
            inBIT, exBIT = eval(config['sub_reg']['sub_ratio'])
        else:
            inBIT, exBIT = config['sub_reg']['sub_ratio']
        ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

        def clipped_Lasso(x, threshold, max_x):
            return tf.clip_by_value(tf.abs(x), threshold, max_x * 10)

        def func(x=None, temp_max=None):
            if 'conv1' not in x.op.name and 'fct' not in x.op.name:
                if temp_max != None:
                    maxW = tf.stop_gradient(temp_max)
                elif fix_max:
                    param_name = x.op.name.split('/W')[0] + '/maxW'
                    maxW = tf.stop_gradient(tf.Variable(1.0, name=param_name))
                else:
                    maxW = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
                threshold = maxW * ratio

                weighted = tf.reduce_sum(clipped_Lasso(x, threshold, maxW) - threshold)
                weighted = tf.multiply((relative_lmbd * maxW), weighted)
                return tf.add(Ridge().get_func()(x), weighted, name='weighted_ridge1')
            else:
                return Ridge().get_func()(x)
        return func


class Weighted_Ridge2():
    def get_func(self, config):
        relative_lmbd = float(config['sub_reg']['relative_lmbd'])
        fix_max = eval(config['sub_reg']['fix_max'])
        if type(config['sub_reg']['sub_ratio']) == str:
            inBIT, exBIT = eval(config['sub_reg']['sub_ratio'])
        else:
            inBIT, exBIT = config['sub_reg']['sub_ratio']
        ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

        def clipped_Lasso(x, threshold, max_x):
            return tf.clip_by_value(tf.abs(x), threshold, max_x)

        def reverse_Ridge(x, threshold, max_x):
            return (1 / (2 * (threshold - max_x))) * tf.square(x - max_x) + ((threshold + max_x) / 2)

        def func(x):
            if 'conv1' not in x.op.name and 'fct' not in x.op.name:
                if fix_max:
                    param_name = x.op.name.split('/W')[0] + '/maxW'
                    maxW = tf.stop_gradient(tf.Variable(1.0, name=param_name))
                else:
                    maxW = tf.stop_gradient(tf.reduce_max(tf.abs(x)))
                threshold = maxW * ratio

                weighted = clipped_Lasso(x, threshold, maxW)
                weighted = tf.reduce_sum(reverse_Ridge(weighted, threshold, maxW) - threshold)
                weighted = tf.multiply((threshold * relative_lmbd), weighted)
                return tf.add(Ridge().get_func()(x), weighted, name='weighted_ridge2')
            else:
                return Ridge().get_func()(x)
        return func
