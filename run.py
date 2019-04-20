import os

#str1 = 'python train.py logdir=Weighted_Ridge2_0.0002 regularizer:name=Weighted_Ridge2'
str1 = 'python train.py logdir=QnnBase8W initializer:scale=1.0 activation=relu1 regularizer:name=None quantizer:BITW=16 quantizer:BITA=32'
r = os.system(str1)
