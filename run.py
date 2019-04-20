import os

#str1 = 'python train.py logdir=Weighted_Ridge2_0.0002 regularizer:name=Weighted_Ridge2'
str1 = 'python train.py logdir=Qnn_test activation=relu1 regularizer:name=None quantizer:BITW=2 quantizer:BITA=32'
r = os.system(str1)
