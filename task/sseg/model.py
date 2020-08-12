import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import pixelssl

from module import deeplab_v2


def add_parser_arguments(parser):
    pixelssl.model_template.add_parser_arguments(parser)

    # arguments for DeepLab
    parser.add_argument('--output-stride', type=int, default=16, help='')
    parser.add_argument('--backbone', type=str, default='resnet', help='')
    parser.add_argument('--freeze-bn', type=pixelssl.str2bool, default=False, help='')


def deeplabv2():
    return DeepLabV2


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
        resulter['s4l_feature'] = (latent, )
        return resulter, debugger


class DeepLabV2(DeepLab):
    def __init__(self, args):
        # pretrained_backbone_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        pretrained_backbone_url = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

        super(DeepLabV2, self).__init__(args, 'v2', pretrained_backbone_url)
