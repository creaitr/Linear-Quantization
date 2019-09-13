import os
import sys

gpu = '0'

cmd = 'python train.py config config{0}.json gpu={1}'.format(gpu[0], gpu)

cmd = cmd + ' logdir=ResNet18_{0}'

for i in range(2):
    r = os.system(cmd.format(i))
