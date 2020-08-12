""" This file is adapted from the repository: https://github.com/hfslyc/AdvSemiSeg 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pixelssl

from .backbone import build_backbone


class DeepLabV2(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, 
                 sync_bn=True, freeze_bn=False, pretrained_backbone_url=None):
        super(DeepLabV2, self).__init__()

        if sync_bn == True:
            BatchNorm = pixelssl.SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, pretrained_backbone_url)
        self.classifier = build_classifier([6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        bx, _ = self.backbone(input)
        x = self.classifier(bx)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, bx

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, pixelssl.SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], pixelssl.SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.classifier]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], pixelssl.SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


# ---------------------------------------------------------------------
# Definition of Classifier Module
# ---------------------------------------------------------------------

def build_classifier(dilation_series, padding_series, num_classes):
    return Classifier_Module(dilation_series, padding_series, num_classes)


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out
