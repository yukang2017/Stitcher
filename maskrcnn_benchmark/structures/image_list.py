# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, PolygonList, PolygonInstance
from maskrcnn_benchmark.config import cfg

import numpy as np

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def to_image_list_synthesize_4(transposed_info, size_divisible=0):
    tensors = transposed_info[0]
    if isinstance(tensors, (tuple, list)):
        targets = transposed_info[1]
        img_ids = transposed_info[2]
        #synthesize data:
        assert len(tensors) % 4 == 0, \
            'len(tensor) % 4 != 0, could not be synthesized ! uneven'
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors)//4,) + max_size
        syn_batched_imgs = tensors[0].new(*batch_shape).zero_()

        syn_targets = []
        with torch.no_grad():
            for idx, pad_img in enumerate(syn_batched_imgs):
                # currently suppose first w then h
                new_h, new_w = max_size[1]//2, max_size[2]//2

                #NOTE: interpolate api require first h then w !
                mode = 'nearest'
                topLeftImg = torch.nn.functional.interpolate(tensors[idx*4].unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                topRightImg = torch.nn.functional.interpolate(tensors[idx*4+1].unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomLeftImg = torch.nn.functional.interpolate(tensors[idx*4+2].unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomRightImg = torch.nn.functional.interpolate(tensors[idx*4+3].unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                c = topLeftImg.shape[0]
                assert c == topRightImg.shape[0] and c == bottomLeftImg.shape[0] and c == bottomRightImg.shape[0]

                pad_img[:c, :new_h, :new_w].copy_(topLeftImg)
                pad_img[:c, :new_h, new_w:].copy_(topRightImg)
                pad_img[:c, new_h:, :new_w].copy_(bottomLeftImg)
                pad_img[:c, new_h:, new_w:].copy_(bottomRightImg)

                # resize each of four sub-imgs into (new_h, new_w) scale
                # resize api require first w then h !
                topLeftBL = targets[idx*4].resize((new_w, new_h))
                topRightBL = targets[idx*4+1].resize((new_w, new_h))
                bottomLeftBL = targets[idx*4+2].resize((new_w, new_h))
                bottomRightBL = targets[idx*4+3].resize((new_w, new_h))
                assert topLeftBL.mode == 'xyxy'
                offsets = [torch.Tensor([0.0,0.0,0.0,0.0]), torch.Tensor([new_w,0.0,new_w,0.0]), torch.Tensor([0.0,new_h,0.0,new_h]),torch.Tensor([new_w,new_h,new_w,new_h])]
                # append offsets to box coordinates except for topLeftBL
                syn_bbox = torch.cat(
                    (topLeftBL.bbox + offsets[0],
                     topRightBL.bbox + offsets[1],
                     bottomLeftBL.bbox + offsets[2],
                     bottomRightBL.bbox + offsets[3]), dim=0)
                #NOTE: BoxList initialization require first w then h
                tmp_BoxList = BoxList(syn_bbox, (new_w*2, new_h*2), mode='xyxy')

                tmp_BoxList.add_field('labels', torch.cat((topLeftBL.extra_fields['labels'], topRightBL.extra_fields['labels'], bottomLeftBL.extra_fields['labels'], bottomRightBL.extra_fields['labels']), dim=-1))

                #NOTE: adjust the targets mask
                topLeftPoly = [poly.polygons[0] for poly in topLeftBL.extra_fields['masks'].instances.polygons]
                topRightPoly =  [poly.polygons[0] for poly in topRightBL.extra_fields['masks'].instances.polygons]
                bottomLeftPoly = [poly.polygons[0] for poly in bottomLeftBL.extra_fields['masks'].instances.polygons]
                bottomRightPoly = [poly.polygons[0] for poly in bottomRightBL.extra_fields['masks'].instances.polygons]

                offsets = [[0.0,0.0], [new_w,0.0], [0.0,new_h], [new_w,new_h]]
                syn_mask = [[list(np.array(poly)+np.array(offsets[0]*int(len(poly)/2)))] for poly in topLeftPoly] + \
                    [[list(np.array(poly)+np.array(offsets[1]*int(len(poly)/2)))] for poly in topRightPoly] + \
                    [[list(np.array(poly)+np.array(offsets[2]*int(len(poly)/2)))] for poly in bottomLeftPoly] + \
                    [[list(np.array(poly)+np.array(offsets[3]*int(len(poly)/2)))] for poly in bottomRightPoly]
                syn_mask = SegmentationMask(syn_mask, (new_w*2, new_h*2), mode='poly')
                tmp_BoxList.add_field('masks', syn_mask)
                
                # append a four-to-one BoxList object
                syn_targets.append(tmp_BoxList)

        syn_targets = tuple(syn_targets)

        assert len(img_ids)%4==0
        #since images are synthesized, id is meaningless, substitute with -1
        syn_img_ids = tuple([-1]*(len(syn_targets)))
        syn_image_sizes = [list(max_size)[-2:] for i in range(batch_shape[0])]

        return ImageList(syn_batched_imgs, syn_image_sizes), syn_targets, syn_img_ids
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def to_image_list_synthesize_batchstitch(transposed_info, num_images=4, size_divisible=0):
    tensors = transposed_info[0]
    if isinstance(tensors, (tuple, list)):
        targets = transposed_info[1]
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            divider = num_images**0.5
            max_size[1] = int(math.ceil(max_size[1] //divider / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] //divider / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        syn_batched_imgs = tensors[0].new(*batch_shape).zero_()

        new_h, new_w = max_size[1], max_size[2]

        with torch.no_grad():
            #NOTE: interpolate api require first h then w !
            #Imgs = torch.nn.functional.interpolate(torch.cat(list(tensors)),size=(new_h, new_w),mode='nearest')
            if cfg.STITCHER.USE_PAD:
                max_h, max_w = max([tensor.shape[1] for tensor in tensors]), max([tensor.shape[2] for tensor in tensors])
                padded_tensors = [torch.nn.functional.pad(tensor.unsqueeze(0), (0, max_w-tensor.shape[2], 0, max_h-tensor.shape[1]), 'replicate') for tensor in tensors]
                for target in targets:
                    target.size = (max_w, max_h)
                tensors = [padded_tensor.reshape(padded_tensor.shape[1:]) for padded_tensor in padded_tensors]

            Imgs = torch.cat([torch.nn.functional.interpolate(tensor.unsqueeze(0),size=(new_h, new_w),mode='nearest') for tensor in tensors])

            c = tensors[0].shape[0]

            syn_batched_imgs[:,:c,:,:].copy_(Imgs)

            # resize each of four sub-imgs into (new_h, new_w) scale
            # resize api require first w then h !
            BLs = [target.resize((new_w, new_h)) for target in targets]

            #NOTE: BoxList initialization require first w then h
            tmp_BoxLists = [BoxList(BL.bbox, (new_w, new_h), mode='xyxy') for BL in BLs]

            for idx, tmp_BoxList in enumerate(tmp_BoxLists):
                tmp_BoxList.add_field('labels', BLs[idx].extra_fields['labels'])

            #NOTE: adjust the targets mask
            Polys = [[poly.polygons[0] for poly in BL.extra_fields['masks'].instances.polygons] for BL in BLs]

            syn_masks = [[[list(np.array(poly))] for poly in Poly] for Poly in Polys]
            syn_masks = [SegmentationMask(syn_mask, (new_w, new_h), mode='poly') for syn_mask in syn_masks]
            for idx, tmp_BoxList in enumerate(tmp_BoxLists):
                tmp_BoxList.add_field('masks', syn_masks[idx])

        syn_targets = tuple(tmp_BoxLists)

        #since images are synthesized, id is meaningless, substitute with -1
        syn_img_ids = tuple([-1]*(len(syn_targets)))
        syn_image_sizes = [list(max_size)[-2:] for i in range(batch_shape[0])]
        return ImageList(syn_batched_imgs, syn_image_sizes), syn_targets, syn_img_ids
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def to_image_list_synthesize(transposed_info, size_divisible=0):
    num_images = cfg.STITCHER.NUM_IMAGES_STITCH

    if cfg.STITCHER.BATCH_STITCH:
        return to_image_list_synthesize_batchstitch(transposed_info, num_images, size_divisible=size_divisible)
    else:
        return to_image_list_synthesize_4(transposed_info,size_divisible=size_divisible)
