import os
import json
import numpy as np
import matplotlib.pyplot as plt

def find_min_error_epoch(logdir='.'):
    with open(logdir + '/stats.json') as file:
        stats = json.load(file)

    errors = [x['validation_error_top1'] for x in stats]
    idx = errors.index(min(errors))

    with open(logdir + '/best.json', 'w') as outfile:
        json.dump(stats[idx], outfile)


def make_ckpt(logdir='.', model_name='', ckpt_name=''):
    with open(logdir + '/' + ckpt_name, 'wb') as outfile:
        end_char = chr(0x0a)
        outfile.write(bytes('model_checkpoint_path: \"{}\"{}'.format(model_name, end_char), 'utf-8'))
        outfile.write(bytes('all_model_checkpoint_paths: \"{}\"{}'.format(model_name, end_char), 'utf-8'))


def dump_to_npz():
    return

def add_Ws():
    return

def draw_dist():
    return

def dump_to_quantized_npz():
    d = dict(np.load(infile))

    np.savez_compressed(outfile, **d)
    return
    

if __name__ == '__main__':
    path = './train_log'
    #path = './temp'
    for logdir in os.listdir(path):
        logdir = path + '/' + logdir


        # find best result info        
        find_min_error_epoch(logdir)
        
        # 1. make checkpoint file
        ckpt_file = 'checkpoint'
        make_ckpt(logdir, 'min-validation_error_top1', ckpt_file)
        # 2. finc meta file
        for file in os.listdir(logdir):
            if '.meta' in file:
                meta_file = file
                break
        # 3. dump ckpt to npz
        dump_py = 'utils/dump-model-params.py'
        ckpt_file = logdir + '/' + ckpt_file
        meta_file = logdir + '/' + meta_file
        outfile = logdir + '/' + 'best.npz'
        r = os.system('python %s --meta %s %s %s' % (dump_py, meta_file, ckpt_file, outfile))
        
        
        # draw distribution
        outfile = logdir + '/' + 'best.npz'
        dist_path = logdir + '/dist'
        r = os.system('mkdir ' + dist_path.replace('/', '\\'))
        dic = dict(np.load(outfile))
        for key in dic.keys():
            if 'W:0' in key:
                x = dic[key].flatten()
                max_val = np.amax(np.absolute(x))
                mid_val = max_val * (3/7)

                x_abs = np.absolute(x)
                cnt1 = 0; cnt2 = 0
                for i in range(x.shape[0]):
                    if x_abs[i] >= mid_val:
                        cnt2 += 1
                    else:
                        cnt1 += 1
                prob1 = cnt1 / x.shape[0] * 100
                prob2 = cnt2 / x.shape[0] * 100
                txt = 'lv3: {:.2f}%/ lv7: {:.2f}%'.format(prob1, prob2)
                #print(key)
                #print(max_val)
                a = plt.hist(x, bins=101, density=1)
                #print(a)
                max_prob = np.amax(np.absolute(a[0]))
                plt.plot([-max_val, -max_val], [0,max_prob], color='red')
                plt.plot([max_val, max_val], [0,max_prob], color='red')
                plt.plot([-mid_val, -mid_val], [0,max_prob], color='green')
                plt.plot([mid_val, mid_val], [0,max_prob], color='green')
                plt.text(-max_val,max_prob, txt)
                plt.ylabel('Probability')
                plt.xlabel(key)
                file_path = dist_path + '/' + key.split(':')[0].replace('/', '.') + '.png'
                plt.savefig(file_path)
                plt.close()
