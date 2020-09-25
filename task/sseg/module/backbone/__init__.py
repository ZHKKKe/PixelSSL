"""
This file is adapted from https://github.com/jfzhang95/pytorch-deeplab-xception
"""

from . import resnet


def build_backbone(backbone, output_stride, BatchNorm, pretrained_backbone_url):
    if backbone in ['resnet50']:
        return resnet.ResNet50(output_stride, BatchNorm, pretrained_backbone_url)
    elif backbone in ['resnet101', 'resnet101-coco']:
        return resnet.ResNet101(output_stride, BatchNorm, pretrained_backbone_url)
    else:
        raise NotImplementedError
