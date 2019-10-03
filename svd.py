import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit()
    else:
        f = sys.argv[1]

    ls = dict(np.load(f))

    pw_keys = [x for x in ls.keys() if 'pwconv' in x]

    for key in pw_keys:
        shape = ls[key].shape

        if shape[2] >= 20 and shape[3] >= 20:
            mat = ls[key]

            s, u, vh = np.linalg.svd(mat, False)

            s = s[:,:,:,:20] * u[:,:,:20]
            vh = vh[:,:,:20,:]

            name1 = key.split('/W')[0] + '1/W:0'
            name2 = key.split('/W')[0] + '2/W:0'

            ls[name1] = s
            ls[name2] = vh

            del ls[key]

    np.savez('new.npz', **ls)
