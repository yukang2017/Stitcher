# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list, to_image_list_synthesize
from maskrcnn_benchmark.config import cfg


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

#NOTE: for slice training
class BatchCollatorSynthesize(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible
        self.imgs_per_batch = int(cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN / 1000)

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        isRegularFlag = (len(batch)==self.imgs_per_batch)
        if isRegularFlag:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
        else:
            images, targets, img_ids = to_image_list_synthesize(transposed_batch, self.size_divisible)
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

