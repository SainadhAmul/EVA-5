
import torch
import torch.nn as nn


## MIDAS BACK BONE!!


def _resnext_backbone(use_pretrained):
    pretrained = _make_pretrained_resnext101_wsl()
    return pretrained



def _make_resnext_backbone(resnext):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnext.conv1, resnext.bn1, resnext.relu, resnext.maxpool, resnext.layer1
    )

    pretrained.layer2 = resnext.layer2
    pretrained.layer3 = resnext.layer3
    pretrained.layer4 = resnext.layer4

    return pretrained


def _make_pretrained_resnext101_wsl():
    resnext = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    # print('***********', resnext)
    return _make_resnext_backbone(resnext)
