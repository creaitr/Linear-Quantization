#written by Jinbae Park
#2019-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from tensorpack.dataflow import dataset
from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent
from tensorpack.dataflow import BatchData
from tensorpack.dataflow import PrefetchData
from tensorpack.dataflow import MultiProcessRunnerZMQ, MultiThreadMapData
from tensorpack.input_source import QueueInput, StagingInput


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_LINEAR
    # linear seems to have more stable performance.
    # but cubic for compatibility with old models
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


class ImageNet:
    def __init__(self):
        self.batch_size = 256
        self.datadir = '/data3/ImageNet/ILSVRC2012'

    def get_data(self, name, num_gpu):
        gpu_batch = self.batch_size // num_gpu

        assert name in ['train', 'val', 'test']
        isTrain = name == 'train'

        augmentors = fbresnet_augmentor(isTrain)
        assert isinstance(augmentors, list)

        parallel = 8

        if isTrain:
            ds = dataset.ILSVRC12(self.datadir, name, shuffle=True)
            ds = AugmentImageComponent(ds, augmentors, copy=False)
            ds = MultiProcessRunnerZMQ(ds, parallel)
            ds = BatchData(ds, gpu_batch, remainder=False)
        else:
            ds = dataset.ILSVRC12Files(self.datadir, name, shuffle=False)
            aug = imgaug.AugmentorList(augmentors)

            def mapf(dp):
                fname, cls = dp
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
                im = aug.augment(im)
                return im, cls

            ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
            ds = BatchData(ds, gpu_batch, remainder=True)
            ds = MultiProcessRunnerZMQ(ds, 1)
        return QueueInput(ds)


    def load_data(self, num_gpu=0):
        size = 224
        nb_classes = 1000
        ds_trn = self.get_data('train', num_gpu)
        ds_val = self.get_data('val', num_gpu)
        return size, nb_classes, ds_trn, ds_val
