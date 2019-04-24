import os

cmd = 'python train.py '

# 1. Ridge and relu
lmbd_list = [0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]
option = 'logdir=Ridge_{0} regularizer:name=Ridge regularizer:lmbd={0} activation=relu'
for lmbd in lmbd_list:
    r = os.system(cmd + option.format(lmbd))

option = 'logdir=Ridge_{0}_{1} regularizer:name=Ridge regularizer:lmbd={0} activation=relu'
for i in range(1,6):
    for lmbd in lmbd_list:
        r = os.system(cmd + option.format(lmbd, i))

# 2. Ridge and relu1
lmbd_list = [0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]
scale_list = [1.0, 2.0]
option = 'logdir=Ridge_{0}_relu1_{1} regularizer:name=Ridge regularizer:lmbd={0} activation=relu1 initializer:scale={1}'
for lmbd in lmbd_list:
    for scale in scale_list:
        r = os.system(cmd + option.format(lmbd, scale))

option = 'logdir=Ridge_{0}_relu1_{1}_{2} regularizer:name=Ridge regularizer:lmbd={0} activation=relu1 initializer:scale={1}'
for i in range(1,6):
    for lmbd in lmbd_list:
        for scale in scale_list:
            r = os.system(cmd + option.format(lmbd, scale, i))


# 3. quantized baseline
lmbd_list = [0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]
scale_list = [1.0, 2.0]
option = 'logdir=3W4A_Ridge_{0}_relu1_{1}_{2} regularizer:name=Ridge regularizer:lmbd={0} activation= relu1 intializer:scale={1} quantizer:name=linear BITW=3 BITA=4 quantizer:W_options:is_Lv=False'
for i in range(1,6):
    for lmbd in lmbd_list:
        for scale in scale_list:
            r = os.system(cmd + option.format(lmbd, scale, i))

bitWA_list = [[3, 2, True], [3, 3, True], [3, 4, True], [3, 8, True],
              [2, 2, False], [2, 3, False], [2, 4, False], [2, 8, False],
              [7, 3, True], [7, 4, True], [7, 8, True],
              [3, 3, False], [3, 4, False], [3, 8, False],
              [4, 4, False], [4, 8, False]]
load_file = ''
option = 'logdir={0}{1}W{2}A load={3} activation=relu1 quantizer:name=linear BITW={0} BITA={2} quantizer:W_options:is_Lv={4}'
for bitW, bitA, is_Lv in bitWA_list:
    r = os.system(cmd + option.format(bitW, 'L' if is_Lv else '', bitA, load_file, is_Lv))
