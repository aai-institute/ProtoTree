import argparse

import torch.nn as nn
import torch.nn.functional as F

from features.densenet_features import (
    densenet121_features,
    densenet161_features,
    densenet169_features,
    densenet201_features,
)
from features.resnet_features import (
    ResNet_features,
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet50_features_inat,
    resnet101_features,
    resnet152_features,
)
from features.vgg_features import (
    vgg11_bn_features,
    vgg11_features,
    vgg13_bn_features,
    vgg13_features,
    vgg16_bn_features,
    vgg16_features,
    vgg19_bn_features,
    vgg19_features,
)
from prototree.prototree import ProtoTree
from util.log import Log

base_architecture_to_features = {
    "resnet18": resnet18_features,
    "resnet34": resnet34_features,
    "resnet50": resnet50_features,
    "resnet50_inat": resnet50_features_inat,
    "resnet101": resnet101_features,
    "resnet152": resnet152_features,
    "densenet121": densenet121_features,
    "densenet161": densenet161_features,
    "densenet169": densenet169_features,
    "densenet201": densenet201_features,
    "vgg11": vgg11_features,
    "vgg11_bn": vgg11_bn_features,
    "vgg13": vgg13_features,
    "vgg13_bn": vgg13_bn_features,
    "vgg16": vgg16_features,
    "vgg16_bn": vgg16_bn_features,
    "vgg19": vgg19_features,
    "vgg19_bn": vgg19_bn_features,
}


def get_prototree_base_networks(
    out_channels: int, net="resnet50_inat", pretrained=True
) -> tuple[nn.Module, nn.Module]:
    """
    Returns the backbone network with pretrained features and a 1x1 convolutional layer with the selected
    out_channels that should be added at top, i.e. passed as add_on_layers to ProtoTree

    :param out_channels:
    :param net:
    :param pretrained:
    :return: (backbone_convnet, add_one_layers)
    """
    feature_convnet = base_architecture_to_features[net](pretrained=pretrained)

    add_on_layer = nn.Sequential(
        nn.Conv2d(
            in_channels=num_out_channels(feature_convnet),
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        ),
        nn.Sigmoid(),
    )
    return feature_convnet, add_on_layer


def num_out_channels(convnet: nn.Module):
    convnet_name = str(convnet).upper()
    if convnet_name.startswith("VGG") or convnet_name.startswith("RES"):
        n_out_channels = [i for i in convnet.modules() if isinstance(i, nn.Conv2d)][
            -1
        ].out_channels
    elif convnet_name.startswith("DENSE"):
        n_out_channels = [
            i for i in convnet.modules() if isinstance(i, nn.BatchNorm2d)
        ][-1].num_features
    else:
        raise ValueError(f"base_architecture {convnet_name} NOT implemented")
    return n_out_channels
