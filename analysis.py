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

        inBIT, exBIT = config['regularizer']['sub_reg']['sub_ratio']
        inBIT = int(inBIT); exBIT = int(exBIT)
        ratio = (1 / (1 + ((2 ** exBIT - 1) / (2 ** inBIT - 1))))

        total_weights = 0
        total_in_prob = 0
        total_out_prob = 0
        result_dic = {}

        keys = list(dic.keys())
        for key in keys:
            if '/W:0' in key:
                x = dic[key].flatten()

                name_scope, device = key.split('/W')
                maxW_name = name_scope + '/maxW' + device

                def check_cond(key):
                    if 'conv1' in key or 'fct' in key:
                        return False
                    return True

                if not check_cond(key):
                    maxW = np.amax(np.absolute(x))
                elif maxW_name in keys:
                    maxW = dic[maxW_name]
                    maxW *= config['quantizer']['W_opts']['max_scale']
                    maxW_temp = np.amax(np.absolute(x))
                else:
                    maxW_name = 'regularize_cost_internals/' + maxW_name
                    if maxW_name in keys:
                        maxW = dic[maxW_name]
                        maxW_temp = np.amax(np.absolute(x))
                    else:
                        maxW = np.amax(np.absolute(x))
                thresh = maxW * ratio
                
                if check_cond(key):
                    if eval(config['quantizer']['W_opts']['is_Lv']):
                        n = ((2 ** exBIT - 1) + (2 ** inBIT - 1)) / 2
                        x_temp = np.round((np.clip(x, -maxW, maxW) / maxW) * n)
                    else:
                        n = 2 ** (float(config['quantizer']['BITW']) - 1) - 0.5
                        x_temp = np.floor((np.clip(x, -maxW, maxW) / maxW) * n) + 0.5
                    for i in range(x.shape[0]):
                        if x_temp[i] in result_dic.keys():
                            result_dic[x_temp[i]] += 1
                        else:
                            result_dic[x_temp[i]] = 1

                if check_cond(key):
                    x_abs = np.absolute(x)
                    cnt1 = 0; cnt2 = 0
                    for i in range(x.shape[0]):
                        if x_abs[i] >= thresh:
                            cnt2 += 1
                        else:
                            cnt1 += 1
                    prob1 = cnt1 / x.shape[0] * 100
                    prob2 = cnt2 / x.shape[0] * 100
                    txt = 'lv3: {:.6f}%/ lv{}: {:.6f}%'.format(prob1, int(n*2 + 1), prob2)
                    # for total
                    total_weights += x.shape[0]
                    total_in_prob += cnt1
                    total_out_prob += cnt2

                a = plt.hist(x, bins=101, density=1)

                max_prob = np.amax(np.absolute(a[0]))

                plt.plot([-maxW, -maxW], [0,max_prob], color='red')
                plt.plot([maxW, maxW], [0,max_prob], color='red')
                plt.plot([-thresh, -thresh], [0,max_prob], color='green')
                plt.plot([thresh, thresh], [0,max_prob], color='green')
                if maxW_name in keys:
                    plt.plot([-maxW_temp, -maxW_temp], [0, max_prob], color='purple')
                    plt.plot([maxW_temp, maxW_temp], [0, max_prob], color='purple')

                if check_cond(key):
                    plt.text(-maxW, max_prob, txt)
                plt.ylabel('Probability')
                plt.xlabel(key)
                file_path = dist_path + '/' + key.split(':')[0].replace('/', '.') + '.png'
                plt.savefig(file_path)
                plt.close()

        maxQ = max(list(result_dic.keys()))
        thresh = maxQ * ratio
        part1_keys = [key for key in result_dic.keys() if np.absolute(key) < thresh]
        part1_vals = [result_dic[key] for key in result_dic.keys() if np.absolute(key) < thresh]
        part2_keys = [key for key in result_dic.keys() if np.absolute(key) > thresh]
        part2_vals = [result_dic[key] for key in result_dic.keys() if np.absolute(key) > thresh]
        
        #print(part1_keys); print(part1_vals)
        plt.bar(part1_keys, part1_vals, color='blue')
        plt.bar(part2_keys, part2_vals, color='red')

        for key in result_dic.keys():
            plt.text(key, result_dic[key], str(result_dic[key]), horizontalalignment='center')

        print('total in prob:', total_in_prob / total_weights * 100)
        print('total out prob:', total_out_prob / total_weights * 100)

        prob1 = total_in_prob / total_weights * 100
        prob2 = total_out_prob / total_weights * 100
        txt = 'lv3: {:.3f}%/ etc: {:.3f}%'.format(prob1, prob2)
        x_temp = min([float(x) for x in part2_keys])
        y_temp = max([float(x) for x in part1_vals])
        plt.text(x_temp, y_temp, txt)

        plt.ylabel('Probability')
        plt.xlabel('Quantized Weights')
        file_path = dist_path + '/quantized.png'
        plt.savefig(file_path)
        plt.close()

        with open(logdir + '/dist/result_dict.txt', 'w') as file:
            file.write(str(result_dic))
            file.write('\n\n')
            file.write('total in prob: {}\n'.format(total_in_prob / total_weights * 100))
            file.write('total out prob: {}'.format(total_out_prob / total_weights * 100))


