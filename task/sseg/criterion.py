import time
import numpy as np

import torch
import torch.nn as nn

import pixelssl


def add_parser_arguments(parser):
    pixelssl.criterion_template.add_parser_arguments(parser)


def sseg_criterion():
    return CommonSSEGCriterion


class CommonSSEGCriterion(pixelssl.criterion_template.TaskCriterion):
    def __init__(self, args):
        super(CommonSSEGCriterion, self).__init__(args)
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.args.ignore_index, reduction='none')

    def forward(self, pred, gt, inp):
        # NOTE: input 'pred' is not activated!

        if len(pred) != 1 or len(gt) != 1 or len(inp) != 1:
            pixelssl.log_err('DeepLab criterion for semantic segmentation requires\t=>\t'
                             'len(pred) == 1 \t len(gt) == 1 \t len(inp) == 1\n')

        pred, gt, inp = pred[0], gt[0], inp[0]
        n, c, h, w = pred.shape

        if len(gt.shape) == 4:
            gt = gt.view(n, h, w)
        
        loss = self.cross_entropy(pred, gt.long())
        return torch.mean(loss, dim=(1, 2))
