
""" Agumentation code for SSD network

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

import os, glob
import PIL.Image as Image
from torchvision.transforms import ToTensor as tvToTensor
from torch.autograd import Variable
import torch.nn.functional as F


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


# Done
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        image = [ img.astype(np.float32) for img in image ]
        return image, boxes, labels


# Done
class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image[0].shape
        for box in boxes:
            box[:, 0] *= width
            box[:, 2] *= width
            box[:, 1] *= height
            box[:, 3] *= height

        return image, boxes, labels

# Done
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image[0].shape
        for box in boxes:
            box[:, 0] /= width
            box[:, 2] /= width
            box[:, 1] /= height
            box[:, 3] /= height

        return image, boxes, labels

# Done
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image[0].shape
        scale_for_shorter_edge = 600.0 / min(width, height)
        longer_edge_after_scaling = max(width, height) * scale_for_shorter_edge
        scale_for_longer_edge = (1000.0 / longer_edge_after_scaling) if longer_edge_after_scaling > 1000 else 1
        scale = scale_for_shorter_edge * scale_for_longer_edge

        # image = [cv2.resize(img, ( round(width * scale), round(height * scale) )) for img in image]
        image = [cv2.resize(img, ( self.size , self.size )) for img in image]

        return image, boxes, labels

   
class Augmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            ToPercentCoords(),
            Resize(self.size),
            ToAbsoluteCoords()
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)




