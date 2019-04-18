import os


lmd_list = [0.00001, 0.0001, 0.0002, 0.001, 0.01, 0.1]

str1 = 'python train.py logdir=Ridge_{0} regularizer\\lmd={0}'
for i in lmd_list:
	r = os.system(str1.format(i))

str2 = 'python train.py logdir=Weighted_Ridge1_{0} regularizer\\name=Weighted_Ridge1 regularizer\lmd={0}'
for i in lmd_list:
	r = os.system(str2.format(i))

str3 = 'python train.py logdir=Weighted_Ridge2_{0} regularizer\\name=Weighted_Ridge2 regularizer\lmd={0}'
for i in lmd_list:
	r = os.system(str3.format(i))

