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


class _RoIOffsetPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, use_offset=True, offset=None):
        super(_RoIOffsetPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        fc_channel = self.pooled_width * self.pooled_height

        def zero_init(m):
            m.weight.data.zero_()
            m.bias.data.zero_()

        self.fc = nn.Linear(fc_channel, 2 * fc_channel).cuda()
        self.fc.apply(zero_init)

        self.gamma = 0.1
        self.use_offset = use_offset
        self.offset = offset

    def forward(self, features, rois):
        n, c, h, w = features.size()
        num_rois = rois.size(0)
        features = features.view(1, n*c, h, w)
        pooled = RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
        # print(pooled.shape)
        pooled = pooled.view(num_rois * c * n, -1)  # (128, 512, 7, 7)

        #### Original Pool Method ###
        # pooled = []
        # for proposal_bbox in rois:
        #     start_x = max(min(round(proposal_bbox[1].item() *self.spatial_scale), w - 1), 0)      # [0, feature_map_width)
        #     start_y = max(min(round(proposal_bbox[2].item() *self.spatial_scale), h - 1), 0)     # (0, feature_map_height]
        #     end_x = max(min(round(proposal_bbox[3].item() *self.spatial_scale) + 1, w), 1)        # [0, feature_map_width)
        #     end_y = max(min(round(proposal_bbox[4].item() *self.spatial_scale) + 1, h), 1)       # (0, feature_map_height]
        #     roi_feature_map = features[..., start_y:end_y, start_x:end_x]
        #     pooled.append(F.adaptive_max_pool2d(roi_feature_map, 7))
        # pooled = torch.cat(pooled, dim=0)   # pool has shape (128, 512, 7, 7)
        # pooled = pooled.view(rois.size(0) * c, -1)

        offset = self.fc(pooled) * self.gamma # (128, 512, 14, 14)


        features = features.view(1, n, c, h, w)
        _, n, c, h, w = features.size()

        feature_size = torch.tensor(features.size())
        roi_size = torch.tensor(rois.size())

        offset = offset.view(1, roi_size[0], n, c, self.pooled_height, self.pooled_width, 2) # (1, 128, 3, 512, 7, 7, 2)
        # Aggregate the channels belonging to the same time step
        offset = torch.mean(offset, dim=3) # (128, 7, 7, 2)

        pooled_offset = []
        for proposal_ind, roi in enumerate(rois.clone()):
            time_pooled = RoIOffsetPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features[0], roi, offset[0, proposal_ind,:,:,:,:], feature_size)
            pooled_offset.append(time_pooled)

        pooled_offsets = torch.cat(pooled_offset).view(int(roi_size[0]), n * c, self.pooled_height, self.pooled_width)

        return pooled_offsets
        # convert index to bilinear
