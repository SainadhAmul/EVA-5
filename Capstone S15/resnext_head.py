
import torch
import torch.nn as nn


## MIDAS BACK BONE!!


def _make_encoder(features, use_pretrained, groups=1, expand=False, exportable=True):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    return pretrained



def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)
