import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = './train_log'
    #path = './temp'
    for logdir in os.listdir(path):
        logdir = path + '/' + logdir

        with open(logdir + '/best.json') as f:
            best = json.load(f)

        with open(logdir + '/stats.json') as f:
            stats = json.load(f)

        errors1 = [x['validation_error_top1'] for x in stats]
        errors5 = [x['validation_error_top5'] for x in stats]

        cnt = 5
        
        sma1 = np.zeros(len(errors1) - cnt)
        sma5 = np.zeros(len(errors1) - cnt)

        for i in range(len(sma1)):
            sma1[i] = np.sum(errors1[i:i+cnt]) / cnt
            sma5[i] = np.sum(errors5[i:i+cnt]) / cnt

        idx = np.where(sma1 == min(sma1))[0][0]
        
        best['sma_error_top1'] = '{:.4f}'.format(sma1[idx])
        best['sma_error_top5'] = '{:.4f}'.format(sma5[idx])
        best['sma_epoch_num'] = int(idx + cnt)
        #best['sma_last_error_top1'] = '{:.4f}'.format(sma1[-1])
        #best['sma_last_error_top5'] = '{:.4f}'.format(sma5[-1])

        with open(logdir + '/best.json', 'w') as outfile:
            json.dump(best, outfile)
