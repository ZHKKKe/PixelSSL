import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import pixelssl

from module import deeplab_v2, _pspnet


def add_parser_arguments(parser):
    pixelssl.model_template.add_parser_arguments(parser)

    # arguments for DeepLab
    parser.add_argument('--output-stride', type=int, default=16, help='sseg - output stride of the ResNet backbone')
    parser.add_argument('--backbone', type=str, default='resnet101', help='sseg - architecture of the backbone network')
    parser.add_argument('--freeze-bn', type=pixelssl.str2bool, default=False, 
                        help='sseg - if true, the statistics in BatchNorm will not be updated')


def deeplabv2():
    return DeepLabV2


def pspnet():
    return PSPNet


class DeepLab(pixelssl.model_template.TaskModel):
    def __init__(self, args, version, pretrained_backbone_url=None):
        super(DeepLab, self).__init__(args)

        model_func = None
        if version == 'v2':
            model_func = deeplab_v2.DeepLabV2
        else:
            pixelssl.log_err('For Semantic Segmentation - DeepLab, '
                             'we do not support version: {0}\n'.format(version))

        self.model = model_func(backbone=self.args.backbone,
            output_stride=self.args.output_stride, num_classes=self.args.num_classes,
            sync_bn=True, freeze_bn=self.args.freeze_bn, 
            pretrained_backbone_url=pretrained_backbone_url)

        self.param_groups = [
            {'params': self.model.get_1x_lr_params(), 'lr': self.args.lr},
            {'params': self.model.get_10x_lr_params(), 'lr': self.args.lr * 10}
        ]
    
    def forward(self, inp):
        resulter, debugger = {}, {}

        if not len(inp) == 1:
            pixelssl.log_err(
                'Semantic segmentation model DeepLab requires only one input\n'
                'However, {0} inputs are given\n'.format(len(inp)))

        inp = inp[0]
        pred, latent = self.model.forward(inp)

        resulter['pred'] = (pred, )
        resulter['activated_pred'] = (F.softmax(pred, dim=1), )
        resulter['ssls4l_rc_inp'] = pred
        return resulter, debugger


class DeepLabV2(DeepLab):
    def __init__(self, args):

        if args.backbone == 'resnet50':
            self.pretrained_backbone_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        elif args.backbone == 'resnet101':
            pretrained_backbone_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        elif args.backbone == 'resnet101-coco':
            pretrained_backbone_url = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        else:
            pixelssl.log_err('DeepLabV2 does not support the backbone: {0}\n'
                             'You can support it for PSPNet in the file \'task/sseg/model.py\'\n')

        super(DeepLabV2, self).__init__(args, 'v2', pretrained_backbone_url)


class PSPNet(pixelssl.model_template.TaskModel):
    def __init__(self, args):
        super(PSPNet, self).__init__(args)

        if self.args.backbone == 'resnet50':
            self.pretrained_backbone_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        elif self.args.backbone == 'resnet101':
            self.pretrained_backbone_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        elif args.backbone == 'resnet101-coco':
            self.pretrained_backbone_url = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        else:
            pixelssl.log_err('PSPNet does not support the backbone: {0}\n'
                             'You can support it for PSPNet in the file \'task/sseg/model.py\'\n')

        self.model = _pspnet._PSPNet(backbone=self.args.backbone,
            output_stride=self.args.output_stride, num_classes=self.args.num_classes,
            sync_bn=True, freeze_bn=self.args.freeze_bn,
            pretrained_backbone_url=self.pretrained_backbone_url)

        self.param_groups = [
            {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 'lr': self.args.lr},
            {'params': filter(lambda p:p.requires_grad, self.model.get_psp_params()), 'lr': self.args.lr * 10},
            {'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params()), 'lr': self.args.lr * 10},
        ]

    def forward(self, inp):
        resulter, debugger = {}, {}

        if not len(inp) == 1:
            pixelssl.log_err(
                'Semantic segmentation model PSPNet requires only one input\n'
                'However, {0} inputs are given\n'.format(len(inp)))
            
        inp = inp[0]
        pred, latent = self.model.forward(inp)

        resulter['pred'] = (pred, )
        resulter['activated_pred'] = (F.softmax(pred, dim=1), )
        resulter['ssls4l_rc_inp'] = pred

        return resulter, debugger
