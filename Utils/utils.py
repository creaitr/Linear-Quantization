#utils.py
import numpy as np

def find_max(dic={}, config={}):
    keys = list(dic.keys())
    for key in keys:
        if '/W:' in key and 'conv1' not in key and 'fct' not in key:
            name_scope, device = key.split('/W')
            max_val_name = name_scope + '/maxW' + device

            if eval(config['add_reg_prefix']) == True:
                max_val_name = 'regularize_cost_internals/' + max_val_name

            if max_val_name not in keys:
                max_val = np.amax(np.absolute(dic[key]))
                dic[max_val_name] = max_val
    return dic


def find_99th(dic={}, config={}):
    keys = list(dic.keys())
    for key in keys:
        if '/W:' in key and 'conv1' not in key and 'fct' not in key:
            name_scope, device = key.split('/W')
            max_val_name = name_scope + '/maxW' + device

            x = np.absolute(dic[key])
            x = x.flatten()
            x.sort()
            n = x.shape[0]
            n_99th = int((n * 0.9995) - 1)
            max_val = x[n_99th]
            max_x = np.max(x); print(key); print("99th: {} / max: {} / {}%".format(max_val, max_x, max_val / max_x * 100))
            dic[max_val_name] = max_val
    return dic


def pruning(dic={}, config={}):
    inBIT, exBIT = eval(config['quantizer']['W_opts']['threshold_bit'])
    ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

    keys = list(dic.keys())
    for key in keys:
        if '/W:' in key and 'conv1' not in key and 'fct' not in key:
            name_scope, device = key.split('/W')

            max_val_name = name_scope + '/maxW' + device
            if max_val_name not in keys:
                temp = 'regularize_cost_internals/' + max_val_name
                max_val = dic[temp]
                del dic[temp]
                dic[max_val_name] = max_val
            max_val = dic[max_val_name]

            threshold = max_val * ratio
            up_thresh = threshold * 1.2
            dw_thresh = threshold * 0.999

            x = dic[key]

            mask = np.where(np.logical_and(x_bas >= dw_thresh, x_abs < up_thresh), np.float32(1.), np.float32(0.))
            mask *= threshold
            x_new = np.where(mask == np.float32(threshold), mask, x)

            mask = np.where(np.logical_and(x_bas <= -dw_thresh, x_abs > -up_thresh), np.float32(1.), np.float32(0.))
            mask *= -threshold
            x_new = np.where(mask == np.float32(-threshold), mask, x_new)
            
            dic[key] = x_new
    return dic


def make_mask(dic={}, config=[]):
    inBIT, exBIT = eval(config['quantizer']['W_opts']['threshold_bit'])
    ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

    keys = list(dic.keys())
    for key in keys:
        if '/W:' in key and 'conv1' not in key and 'fct' not in key:
            name_scope, device = key.split('/W')

            max_val_name = name_scope + '/maxW' + device
            if max_val_name not in keys:
                temp = 'regularize_cost_internals/' + max_val_name
                max_val = dic[temp]
                del dic[temp]
                dic[max_val_name] = max_val
            max_val = dic[max_val_name]

            threshold = max_val * ratio

            mask_name = name_scope + '/maskW' + device
            if mask_name not in keys:
                mask = np.where(np.absolute(dic[key]) < threshold, np.float32(1.), np.float32(0.))
                dic[mask_name] = mask
    return dic


def clustering(dic={}, config={}):
    bitW = int(config['BITW'])
    is_Lv = eval(config['W_opts']['is_Lv'])

    keys = list(dic.keys())
    for key in keys:
        if '/W:' in key and 'conv1' not in key and 'fct' not in key:
            name_scope, device = key.split('/W')

            max_val_name = name_scope + '/maxW' + device
            if max_val_name not in keys:
                temp = 'regularize_cost_internals/' + max_val_name
                max_val = dic[temp]
                del dic[temp]
                dic[max_val_name] = max_val
            max_val = dic[max_val_name]

            x = dic[key] / max_val

            if is_Lv:
                n = float((bitW - 1) / 2)
                x = np.rint(x * n)
            else:
                n = float(2 ** (bitW - 1) - 0.5)
                x = (np.floor(x * n) + 0.5)
            n_ls = np.arange(-n, n + 1)

            cluster_mask_name = name_scope + '/cluster_maskW' + device
            cluster_mask = np.copy(x)

            x = x / n * max_val

            dic[key] = x
            dic[cluster_mask_name] = cluster_mask
    return dic, n_ls
