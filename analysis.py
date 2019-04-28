import os
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = './train_log'
    #path = './temp'
    for logdir in os.listdir(path):
        logdir = path + '/' + logdir

        with open(logdir + '/config.json') as f:
            config = json.load(f)

        # 1. finc meta file
        for file in os.listdir(logdir):
            if '.meta' in file:
                meta_file = file
                break
        # 2. dump ckpt to npz
        dump_py = 'Scripts/dump-model-params.py'
        ckpt_file = logdir + '/checkpoint'
        meta_file = logdir + '/' + meta_file
        outfile = logdir + '/' + 'best.npz'
        r = os.system('python %s --meta %s %s %s' % (dump_py, meta_file, ckpt_file, outfile))
        
        
        # draw distribution
        #outfile = logdir + '/' + 'best.npz'
        dist_path = logdir + '/dist'

        r = os.system('mkdir ' + dist_path.replace('/', '\\'))
        dic = dict(np.load(outfile))

        keys = list(dic.keys())
        for key in keys:
            if '/W:0' in key:
                x = dic[key].flatten()

                name_scope, device = key.split('/W')
                maxW_name = 'regularize_cost_internals/' + name_scope + '/maxW' + device

                if 'conv1' in key or 'fct' in key:
                    maxW = np.amax(np.absolute(x))
                elif maxW_name in keys:
                    maxW = dic[maxW_name]
                    maxW_temp = np.amax(np.absolute(x))
                else:
                    maxW = np.amax(np.absolute(x))

                inBIT, exBIT = config['regularizer']['sub_reg']['sub_ratio']
                ratio = (1 / (1 + ((2 ** int(exBIT) - 1) / (2 ** int(inBIT) - 1))))

                thresh = maxW * ratio

                x_abs = np.absolute(x)
                cnt1 = 0; cnt2 = 0
                for i in range(x.shape[0]):
                    if x_abs[i] >= thresh:
                        cnt2 += 1
                    else:
                        cnt1 += 1
                prob1 = cnt1 / x.shape[0] * 100
                prob2 = cnt2 / x.shape[0] * 100
                txt = 'lv3: {:.2f}%/ lv7: {:.2f}%'.format(prob1, prob2)

                a = plt.hist(x, bins=101, density=1)

                max_prob = np.amax(np.absolute(a[0]))

                plt.plot([-maxW, -maxW], [0,max_prob], color='red')
                plt.plot([maxW, maxW], [0,max_prob], color='red')
                plt.plot([-thresh, -thresh], [0,max_prob], color='green')
                plt.plot([thresh, thresh], [0,max_prob], color='green')
                if maxW_name in keys:
                    plt.plot([-maxW_temp, -maxW_temp], [0, max_prob/4], color='purple')
                    plt.plot([maxW_temp, maxW_temp], [0, max_prob/4], color='purple')

                plt.text(-maxW,max_prob, txt)
                plt.ylabel('Probability')
                plt.xlabel(key)
                file_path = dist_path + '/' + key.split(':')[0].replace('/', '.') + '.png'
                plt.savefig(file_path)
                plt.close()
