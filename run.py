import os
import sys
import json

gpu = '0'

if len(sys.argv) == 2:
    gpu = sys.argv[1]
  
with open('./config{0}.json'.format(gpu[0])) as f:
    best = json.load(f)

cmd = 'python train.py config config{0}.json gpu={1}'.format(gpu[0], gpu)

cmd = cmd + ' logdir={}'.format(best['logdir']) + '_{0}'

for i in range(2):
    r = os.system(cmd.format(i))

