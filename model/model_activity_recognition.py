import os
import time
from typing import Union, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from backbone.interface import Interface as BackboneInterface
from lib.bbox import BBox
from lib.nms.nms import NMS
from lib.rpn.region_proposal_network import RegionProposalNetwork
# from lib.roioffset_pooling_pytorch.modules import RoIOffsetPooling_PyTorch
from lib.roioffset_pooling_v2.modules import RoIOffsetPooling_PyTorch
from lib.roi_pooling.modules import roi_pool
#from feature_visualizer import feature_visualizer
#from model_denoise_forward import Model as DenoiseModel

import matplotlib.pyplot as plt
import numpy as np


class Model(nn.Module):

    class ForwardInput:
        class Train(object):
            def __init__(self, image: Tensor, gt_classes: Tensor, gt_bboxes: Tensor):
                self.image = image
                self.gt_classes = gt_classes
                self.gt_bboxes = gt_bboxes

        class Eval(object):
            def __init__(self, image: Tensor):
                self.image = image

    class ForwardOutput:
        class Train(object):
            def __init__(self, anchor_objectness_loss: Tensor, anchor_transformer_loss: Tensor, proposal_class_loss: Tensor, proposal_transformer_loss: Tensor):
                self.anchor_objectness_loss = anchor_objectness_loss
                self.anchor_transformer_loss = anchor_transformer_loss
                self.proposal_class_loss = proposal_class_loss
                self.proposal_transformer_loss = proposal_transformer_loss

        class Eval(object):
            def __init__(self, detection_bboxes: Tensor, detection_scores: Tensor):
                self.detection_bboxes = detection_bboxes
                self.detection_scores = detection_scores

    def __init__(self, backbone: BackboneInterface, num_classes, args, num_K=1, offset_flag=False, global_feat_flag=False, rnn_flag=False):
        super().__init__()
        self.NUM_CLASSES = num_classes
        self.K = num_K
        self.offset_flag = offset_flag
        self.global_feat_flag = global_feat_flag
        self.rnn_flag = rnn_flag
        self.args = args

        self.mean = torch.tensor(np.array([0.406, 0.456, 0.485]), dtype=torch.float).cuda()
        self.stds = torch.tensor(np.array([0.225, 0.224, 0.229]), dtype=torch.float).cuda()

        self.features_base, self.features_top = backbone.features()
        self._bn_modules = [it for it in self.features_base.modules() if isinstance(it, nn.BatchNorm2d)] + \
                            [it for it in self.features_top.modules() if isinstance(it, nn.BatchNorm2d)]

        self.rpn = RegionProposalNetwork()
        self.detection = Model.Detection(num_classes=num_classes,
                                         top=self.features_top,
                                         num_K=self.K,
                                         offset_flag=self.offset_flag,
                                         global_feat_flag=self.global_feat_flag,
                                         rnn_flag=self.rnn_flag)

        self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
        self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)

        self.roipooloffsets = None
        #self.denoise = DenoiseModel(self.args).cuda()

    def forward(self, forward_input: Union[ForwardInput.Train, ForwardInput.Eval]) -> Union[ForwardOutput.Train, ForwardOutput.Eval]:

        for bn_module in self._bn_modules:
            bn_module.eval()
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        # image = forward_input.image.unsqueeze(dim=0)
        image = forward_input.image
        image_height, image_width = image.shape[2], image.shape[3]

        for i in range(image.shape[1]):
            image[:,i,:,:] = image[:,i,:,:] - self.mean[i]
        for i in range(image.shape[1]):
            image[:,i,:,:] = image[:,i,:,:] / self.stds[i]


        features = self.features_base(image)
        fff = self.features_base[0:3](image)
        #feature_visualizer(fff)

        f_rpn = features[-1, :, :, :].unsqueeze(dim=0) #(Feed last frame features to rpn)

        anchor_bboxes, anchor_objectnesses, anchor_transformers, proposal_bboxes = self.rpn.forward(f_rpn, image_width, image_height)

        if self.training:
            # forward_input = Model.ForwardInput.Train

            anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers = self.rpn.sample(anchor_bboxes, anchor_objectnesses, anchor_transformers, forward_input.gt_bboxes, image_width, image_height)
            anchor_objectness_loss, anchor_transformer_loss = self.rpn.loss(anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers)

            proposal_bboxes, gt_proposal_classes, gt_proposal_transformers = self.sample(proposal_bboxes, forward_input.gt_classes, forward_input.gt_bboxes)
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            proposal_class_loss, proposal_transformer_loss = self.loss(proposal_classes, proposal_transformers, gt_proposal_classes, gt_proposal_transformers)

            forward_output = Model.ForwardOutput.Train(anchor_objectness_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss)
        else:
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            detection_bboxes, detection_scores = self._generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height)
            forward_output = Model.ForwardOutput.Eval(detection_bboxes, detection_scores)
        return forward_output
        #return detection_bboxes, detection_scores

    def sample(self, proposal_bboxes: Tensor, gt_classes: Tensor, gt_bboxes: Tensor):
        proposal_bboxes = proposal_bboxes.cpu()
        gt_classes = gt_classes.cpu()
        gt_bboxes = gt_bboxes.cpu()

        # find labels for each `proposal_bboxes`
        labels = torch.ones(len(proposal_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(proposal_bboxes, gt_bboxes)
        proposal_max_ious, proposal_assignments = ious.max(dim=1)
        labels[proposal_max_ious < 0.5] = 0
        labels[proposal_max_ious >= 0.5] = gt_classes[proposal_assignments[proposal_max_ious >= 0.5]]

        # select 128 samples
        fg_indices = (labels > 0).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 64)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 - len(fg_indices)]]
        select_indices = torch.cat([fg_indices, bg_indices])
        select_indices = select_indices[torch.randperm(len(select_indices))]

        proposal_bboxes = proposal_bboxes[select_indices]
        gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes[proposal_assignments[select_indices]])
        gt_proposal_classes = labels[select_indices]

        gt_proposal_transformers = (gt_proposal_transformers - self._transformer_normalize_mean) / self._transformer_normalize_std

        gt_proposal_transformers = gt_proposal_transformers.cuda()
        gt_proposal_classes = gt_proposal_classes.cuda()

        return proposal_bboxes, gt_proposal_classes, gt_proposal_transformers

    def loss(self, proposal_classes: Tensor, proposal_transformers: Tensor, gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor):
        cross_entropy = F.cross_entropy(input=proposal_classes, target=gt_proposal_classes)

        proposal_transformers = proposal_transformers.view(-1, self.NUM_CLASSES, 4)
        proposal_transformers = proposal_transformers[torch.arange(end=len(proposal_transformers), dtype=torch.long).cuda(), gt_proposal_classes]

        fg_indices = gt_proposal_classes.nonzero().view(-1)

        # NOTE: The default of `reduction` is `elementwise_mean`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
        # Burhan code debug for PyTorch 0.4 run
        # smooth_l1_loss = F.smooth_l1_loss(input=proposal_transformers[fg_indices], target=gt_proposal_transformers[fg_indices], reduction='sum')
        smooth_l1_loss = F.smooth_l1_loss(input=proposal_transformers[fg_indices], target=gt_proposal_transformers[fg_indices], reduce=True, size_average=False)

        if len(gt_proposal_transformers) > 0:
            smooth_l1_loss /= len(gt_proposal_transformers)
        else:
            smooth_l1_loss = 0

        return cross_entropy, smooth_l1_loss

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self


    def _generate_detections(self, proposal_bboxes: Tensor, proposal_scores: Tensor, proposal_transformers: Tensor, image_height, image_width):
        proposal_transformers = proposal_transformers.view(-1, self.NUM_CLASSES, 4)
        mean = self._transformer_normalize_mean.repeat(1, self.NUM_CLASSES, 1).cuda()
        std = self._transformer_normalize_std.repeat(1, self.NUM_CLASSES, 1).cuda()

        proposal_transformers = proposal_transformers * std - mean
        proposal_bboxes = proposal_bboxes.view(-1, 1, 4).repeat(1, self.NUM_CLASSES, 1)
        detection_bboxes = BBox.apply_transformer(proposal_bboxes.view(-1, 4), proposal_transformers.view(-1, 4))
        detection_bboxes = detection_bboxes.view(-1, self.NUM_CLASSES, 4)

        detection_bboxes[:, :, 0] /= image_width
        detection_bboxes[:, :, 1] /= image_height
        detection_bboxes[:, :, 2] /= image_width
        detection_bboxes[:, :, 3] /= image_height

        return detection_bboxes, proposal_scores

    def NMS_FN(self, detection_bboxes, proposal_scores, image_height, image_width, nms_thresh=0.3):
        '''

        :param detection_bboxes: normalized N x K x 4 tensor <x1 y1 x2 y2>
        :param proposal_scores: logits before softmax N X K classes
        :return:
                nms threshold
        '''

        detection_bboxes[:, :, 0] *= image_width
        detection_bboxes[:, :, 1] *= image_height
        detection_bboxes[:, :, 2] *= image_width
        detection_bboxes[:, :, 3] *= image_height

        detection_bboxes[:, :, [0, 2]] = detection_bboxes[:, :, [0, 2]].clamp(min=0, max=image_width)
        detection_bboxes[:, :, [1, 3]] = detection_bboxes[:, :, [1, 3]].clamp(min=0, max=image_height)

        proposal_probs = F.softmax(proposal_scores, dim=1)

        detection_bboxes = detection_bboxes.cpu()
        proposal_probs = proposal_probs.cpu()

        generated_bboxes = []
        generated_labels = []
        generated_probs = []

        for c in range(1, self.NUM_CLASSES):
            detection_class_bboxes = detection_bboxes[:, c, :]
            proposal_class_probs = proposal_probs[:, c]

            _, sorted_indices = proposal_class_probs.sort(descending=True)
            detection_class_bboxes = detection_class_bboxes[sorted_indices]
            proposal_class_probs = proposal_class_probs[sorted_indices]

            keep_indices = NMS.suppress(detection_class_bboxes.cuda(), threshold=nms_thresh)
            detection_class_bboxes = detection_class_bboxes[keep_indices]
            proposal_class_probs = proposal_class_probs[keep_indices]

            generated_bboxes.append(detection_class_bboxes)
            generated_labels.append(torch.ones(len(keep_indices)) * c)
            generated_probs.append(proposal_class_probs)

        generated_bboxes = torch.cat(generated_bboxes, dim=0)
        generated_labels = torch.cat(generated_labels, dim=0)
        generated_probs = torch.cat(generated_probs, dim=0)
        return generated_bboxes, generated_labels, generated_probs

    ### TODO: Separate class Detection to Pool Class and Classifier Class
    class Detection(nn.Module):

        def __init__(self, num_classes, top, num_K, offset_flag = False, global_feat_flag = False, rnn_flag = False):
            super().__init__()

            self.pooled_width = 7
            self.pooled_height = 7
            self.K = num_K
            self.offset_flag = offset_flag
            self.global_feat_flag = global_feat_flag
            self.rnn_flag = rnn_flag

            # self.fcs = nn.Sequential(
            #     nn.Linear(self.K * 512 * pooled_height * pooled_width, 4096),
            #     nn.ReLU().cuda(1),
            #     nn.Linear(4096, 4096),
            #     nn.ReLU()
            # )

            self.fcs = top

            if self.offset_flag:
                self._roiPoolMethod_tubelet = RoIOffsetPooling_PyTorch._RoIOffsetPooling(pooled_height=self.pooled_height,
                                                                                         pooled_width=self.pooled_width,
                                                                                         spatial_scale=1 / 16.0,
                                                                                         use_offset=True,
                                                                                         num_K = self.K,
                                                                                         in_channels = 1024)
            else:
                self._roiPoolMethod_tubelet = roi_pool._RoIPooling(pooled_height=7, pooled_width=7, spatial_scale=1/16.0)

            if self.global_feat_flag:
                fc_channel = 2 * 2048
            else:
                fc_channel = 2048

            if self.rnn_flag:
                self.RNN = nn.GRU(input_size=fc_channel, hidden_size=fc_channel, num_layers=1, batch_first=True)

            self._class = nn.Linear(fc_channel, num_classes)
            self._transformer = nn.Linear(fc_channel, num_classes * 4)


        def forward(self, features: Tensor, proposal_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
            _, _, feature_map_height, feature_map_width = features.shape
            num_rois = proposal_bboxes.size(0)
            n, c, h, w = features.size()
            #print("features: " + str(features.size))
            #feature_visualizer(features)
            # Adding pooling modules from jwyand/faster-rcnn-pytorch
            batch_id = proposal_bboxes.new(proposal_bboxes.shape[0],1).zero_()
            rois = torch.cat((batch_id, proposal_bboxes),1).cuda()

            pool = self._roiPoolMethod_tubelet.forward(features.view(1, n*c, h, w), rois) # (n_rois, channels, pool_h, pool_w) # (128, 3072, 7, 7)
            pool = pool.view(num_rois*n, c, self.pooled_height, self.pooled_width)  # (128*3, 1024, 7, 7)

            # Local Features
            h_local = self.fcs(pool) # (128 * 3, 2048, 4, 4]
            h_local = h_local.mean(-1).mean(-1)
            h_local = h_local.view(num_rois, self.K, -1)
            #print("h_local: " + str(h_local.shape))

            # Concatenate global features with local features
            if self.global_feat_flag:
                h_global = self.fcs(features) # (3,1024,19,19) -> (3,2048,10,10)
                h_global = h_global.mean(-1).mean(-1)
                h_global = h_global.view(1, self.K, -1).repeat(num_rois, 1, 1) # (3,2048) -> (1,3,2048) -> (128,3,2048)
                h = torch.cat([h_local, h_global], dim=2) # (128, 3, 4096)
            else:
                h = h_local # (128, 3, 2048)
            #print("h_global: " + str(h_global.shape))
            #feature_visualizer(h_local)

            if self.rnn_flag:
                h = h
                o_RNN, h_RNN = self.RNN(h)
                h = o_RNN.mean(1) # Take mean of time dimension

            else:
                h = h.mean(1) # Take mean of time dimension
            #print("o_RNN: " + str(o_RNN.shape))
            #feature_visualizer(o_RNN)


            classes = self._class(h)
            transformers = self._transformer(h)

            return classes, transformers
