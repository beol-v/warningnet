#### Burhan Ahmad Mudassar
#### ROI Offset Pooling in Pytorch...Adapted from jwyang's code for ROI pooling

import sys

sys.path.insert(0, "../../")

from torch.nn.modules.module import Module
from lib.roi_pooling.functions.roi_pool import RoIPoolFunction
from lib.roioffset_pooling_pytorch.functions.RoIOffsetPooling_PyTorch import RoIOffsetPoolFunction
import torch.nn as nn
import torch
from torch.nn import init
from lib.deformConvPyTorch.modules.deform_conv import ConvOffset2d


class _RoIOffsetPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, use_offset=True, offset=None, in_channels=1024, num_K=1):
        super(_RoIOffsetPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        # fc_channel = self.pooled_width * self.pooled_height

        def zero_init(m):
            m.weight.data.zero_()
            m.bias.data.zero_()

        # self.fc = nn.Linear(fc_channel, 2 * fc_channel).cuda()
        # self.fc.apply(zero_init)

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if (m.bias is not None):
                    m.bias.data.zero_()
        self.K = num_K
        self.in_channels = in_channels

        self.offset_pred = nn.Conv2d(self.in_channels * self.K, 3 * 3 * 2 * self.K, kernel_size=3, padding=1)
        # Offset Conv has weight initialization step within the module itself
        self.offset_conv = ConvOffset2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, num_deformable_groups=1)

        # self.offset_conv.apply(weights_init)
        self.offset_pred.apply(weights_init)

        self.gamma = 0.1
        self.use_offset = use_offset
        self.offset = offset

    def forward(self, features, rois):
        # n, c, h, w = features.size()
        num_rois = rois.size(0)

        # features = features.view(1, n*c, h, w)

        offsets = self.offset_pred(features)
        offsets = offsets.view(offsets.size(0), self.K, 3 * 3, 2, offsets.size(-2), offsets.size(-1))

        offset_features_per_frame = list()

        for i in range(self.K):
            f_i = features[:, i*self.in_channels: (i+1)*self.in_channels, :, :]
            o_i = offsets[:, i, :, :, :, :].view(offsets.size(0), -1, offsets.size(-2), offsets.size(-1))
            offset_features_per_frame.append(self.offset_conv(f_i, o_i))

        offset_features_per_frame = torch.cat(offset_features_per_frame, 1) # (1, 3*1024, 19, 19)

        pooled = RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(offset_features_per_frame, rois) # (128, 3*1024, 7, 7)
        pooled = pooled.view(num_rois, self.in_channels * self.K, self.pooled_height, self.pooled_width)  # (128, 3*1024, 7, 7)

        return pooled
        # convert index to bilinear
