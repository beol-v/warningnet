from typing import Tuple, Type

from torch import nn
import torchvision

class Base(object):

    OPTIONS = ['resnet18', 'resnet50', 'resnet101']

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'resnet18':
            from backbone.resnet18 import ResNet18
            return ResNet18
        elif name == 'resnet50':
            from backbone.resnet50 import ResNet50
            return ResNet50
        elif name == 'resnet101':
            #from backbone.resnet101 import ResNet101
            return ResNet101
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError


class ResNet101(Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())
        features = children[:-3]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out
