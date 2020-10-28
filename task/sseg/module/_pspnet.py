""" This file is adapted form the repository: https://github.com/yassouali/CCT 
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pixelssl

from .backbone import build_backbone


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()

        self.norm_layer = norm_layer

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            self.norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, self.norm_layer):
                self._init_norm(m)

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
        
    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = self.norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                       align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class _PSPNet(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=8, num_classes=21,
                 sync_bn=True, freeze_bn=False, pretrained_backbone_url=None):
        super(_PSPNet, self).__init__()

        if sync_bn == True:
            self.norm_layer = pixelssl.SynchronizedBatchNorm2d
        else:
            self.norm_layer = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, self.norm_layer, pretrained_backbone_url)
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6], norm_layer=self.norm_layer)
        self.decoder = upsample(512, num_classes, upscale=8)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        bx = self.backbone(input)
        x = self.psp(bx)
        x = self.decoder(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, bx

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, self.norm_layer):
                m.eval()

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_psp_params(self):
        return self.psp.parameters()

    def get_decoder_params(self):
        return self.decoder.parameters()
