#written by Jinbae Park
#2019-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorpack.dataflow import dataset
from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent
from tensorpack.dataflow import BatchData
from tensorpack.dataflow import PrefetchData


class Cifar10:
    def __init__(self):
        self.batch_size = 128

    def get_data(self, train_or_test):
        isTrain = train_or_test == 'train'
        ds = dataset.Cifar10(train_or_test)
        pp_mean = ds.get_per_pixel_mean()
        if isTrain:
            augmentors = [
                imgaug.CenterPaste((40, 40)),
                imgaug.RandomCrop((32, 32)),
                imgaug.Flip(horiz=True),
                # imgaug.Brightness(20),
                # imgaug.Contrast((0.6,1.4)),
                imgaug.MapImage(lambda x: x - pp_mean),
            ]
        else:
            augmentors = [
                imgaug.MapImage(lambda x: x - pp_mean)
            ]
        ds = AugmentImageComponent(ds, augmentors)
        ds = BatchData(ds, self.batch_size, remainder=not isTrain)
        if isTrain:
            ds = PrefetchData(ds, 3, 2)
        return ds

    def load_data(self):
        size = 32
        ds_trn = self.get_data('train')
        ds_tst = self.get_data('test')
        return size, ds_trn, ds_tst
