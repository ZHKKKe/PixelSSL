""" This file is adapted from the repository: https://github.com/jfzhang95/pytorch-deeplab-xception
"""

import cv2
import math
import time
import scipy.ndimage
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pixelssl


def task_func():
    return SemanticSegmentationFunc


class SemanticSegmentationFunc(pixelssl.func_template.TaskFunc):
    def __init__(self, args):
        super(SemanticSegmentationFunc, self).__init__(args)
        self._colorize = VOCColorize()
        self._category_tensor = torch.FloatTensor(
            [[[[_]] for _ in range(0, self.args.num_classes)]]).cuda()

        self.fdgt_generator = nn.DataParallel(FDGTGenerator(self.args)).cuda()

    # ---------------------------------------------------------------------
    # Functions for All Tasks
    # ---------------------------------------------------------------------

    def metrics(self, pred, gt, inp, meters, id_str=''):
        assert len(pred) == len(gt) == 1
        
        pred, gt = pred[0].data.cpu().numpy(), gt[0].data.cpu().numpy()
        
        pred = np.argmax(pred, axis=1)
        pred = np.expand_dims(pred, axis=1)
        mask = (gt >= 0) & (gt < self.args.num_classes)
        
        label = self.args.num_classes * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.args.num_classes ** 2)
        confusion_matrix = count.reshape(self.args.num_classes, self.args.num_classes)
        meters.update('{0}_confusion_matrix'.format(id_str), confusion_matrix)

        acc_str = '{0}_{1}_acc'.format(id_str, self.METRIC_STR)
        acc_class_str = '{0}_{1}_acc-class'.format(id_str, self.METRIC_STR)
        mIoU_str = '{0}_{1}_mIoU'.format(id_str, self.METRIC_STR)
        fwIoU_str = '{0}_{1}_fwIoU'.format(id_str, self.METRIC_STR)

        if meters.has_key(acc_str):
            meters.reset(acc_str)
        if meters.has_key(acc_class_str):
            meters.reset(acc_class_str)
        if meters.has_key(mIoU_str):
            meters.reset(mIoU_str)
        if meters.has_key(fwIoU_str):
            meters.reset(fwIoU_str)

        cmat_sum = meters['{0}_confusion_matrix'.format(id_str)].sum

        acc = np.diag(cmat_sum).sum() / cmat_sum.sum()
        meters.update(acc_str, acc)
        
        acc_class = np.diag(cmat_sum) / cmat_sum.sum(axis=1)
        acc_class = np.nanmean(acc_class)
        meters.update(acc_class_str, acc_class)

        IoU = np.diag(cmat_sum) / (np.sum(cmat_sum, axis=1) + np.sum(cmat_sum, axis=0) - np.diag(cmat_sum)) 
        
        mIoU = np.nanmean(IoU)
        meters.update(mIoU_str, mIoU)
        
        freq = np.sum(cmat_sum, axis=1) / np.sum(cmat_sum)
        fwIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
        meters.update(fwIoU_str, fwIoU)

    def visualize(self, out_path, id_str='', inp=None, pred=None, gt=None):
        if out_path.split('/')[-2] == 'train':
            # TODO: when there are multiple training sets
            #       only the first data set is currently supported for visualization
            dataset = list(self.args.trainset.keys())[0]
        elif out_path.split('/')[-2] == 'val':
            dataset = list(self.args.valset)[0]
        else:
            dataset = None
            pixelssl.log_err(
                'The arguments \'visual_train_path\' and \'visual_val_path\' auto-set by the file: \n'
                '\'pixelssl/task_template/proxy.py\' are changed.\n'
                'The specific names of them are required in semantic segmentation.\n' 
                'Please check the \'visualize\' function in \'task/sseg/func.py\' for details\n')

        # NOTE: different datasets have different 'mean' and 'std'
        if dataset.startswith('pascal_voc'):
            mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
            std = np.array([[[0.229]], [[0.224]], [[0.225]]])
        else:
            mean = np.array([[[0]], [[0]], [[0]]])
            std = np.array([[[1]], [[1]], [[1]]])
        
        if inp is not None:
            assert len(inp) == 1
            inp = inp[0].data.cpu().numpy()
            inp = inp * std + mean
            inp = np.clip(inp, 0, 1)
            inp = np.transpose(inp, (1, 2, 0))
            Image.fromarray((inp * 255).astype('uint8')).save(out_path + '_{0}1-inp.png'.format(id_str))

        if pred is not None:
            assert len(pred) == 1
            pred = pred[0].data.cpu().numpy()
            pred = np.argmax(pred, axis=0)
            pred = self._colorize(pred)
            pred = np.transpose(pred, (1, 2, 0))
            Image.fromarray((pred * 255).astype('uint8')).save(out_path + '_{0}2-pred.png'.format(id_str))

        if gt is not None:
            assert len(gt) == 1
            gt = gt[0].data.cpu().numpy()
            gt = self._colorize(gt[0])
            gt = np.transpose(gt, (1, 2, 0))
            Image.fromarray((gt * 255).astype('uint8')).save(out_path + '_{0}3-gt.png'.format(id_str))

    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Functions for SSL_ADV
    # ---------------------------------------------------------------------

    def ssladv_fcd_in_channels(self):
        return self.args.num_classes

    def ssladv_preprocess_fcd_criterion(self, fcd_pred, task_gt, is_real):
        # fcd_pred is not activated

        biclass = 1 if is_real else 0
        
        if task_gt is None:
            ignore_mask = np.zeros(fcd_pred.shape).astype(np.bool)
        else:
            ignore_mask = (task_gt.data.cpu().numpy() == self.args.ignore_index)

        fcd_gt = np.ones(ignore_mask.shape) * biclass
        fcd_gt[ignore_mask] = 255
        fcd_gt = Variable(torch.FloatTensor(fcd_gt)).cuda()

        fcd_mask = (fcd_gt >= 0) * (fcd_gt != self.args.ignore_index)
        fcd_mask = fcd_mask.float()
        fcd_gt = fcd_gt * fcd_mask
        fcd_pred = fcd_pred * fcd_mask

        return fcd_pred, fcd_gt

    def ssladv_convert_task_gt_to_fcd_input(self, task_gt):
        task_gt = task_gt.data.cpu().numpy()
        shape = list(task_gt.shape)
        assert len(shape) == 4
        shape[1] = self.args.num_classes

        one_hot = np.zeros(shape, dtype=task_gt.dtype)
        for i in range(self.args.num_classes):
            one_hot[:, i:i+1, ...] = (task_gt == i)

        return torch.FloatTensor(one_hot)
        
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Functions for SSL_GCT
    # ---------------------------------------------------------------------

    def sslgct_fd_in_channels(self):
        return self.args.num_classes + 3

    def sslgct_prepare_task_gt_for_fdgt(self, task_gt):
        gt_np = task_gt.data.cpu().numpy()
        shape = list(gt_np.shape)
        assert len(shape) == 4
        shape[1] = self.args.num_classes

        one_hot = torch.zeros(shape).cuda()
        for i in range(self.args.num_classes):
            one_hot[:, i:i+1, ...].add_((task_gt == i).float())
            # ignore segment boundary
            one_hot[:, i:i+1, ...].mul_((task_gt != self.args.ignore_index).float())

        # return torch.FloatTensor(one_hot)
        return one_hot

    def visualize_pseudo_gt(self, pseudo_gt, out_path, id_str):
        pseudo_gt = pseudo_gt[0].data.cpu().numpy()
        pseudo_gt = np.argmax(pseudo_gt, axis=0)
        pseudo_gt = self._colorize(pseudo_gt)
        pseudo_gt = np.transpose(pseudo_gt, (1, 2, 0))
        Image.fromarray((pseudo_gt * 255).astype('uint8')).save(out_path + '_{0}-pseudo-gt.png'.format(id_str))

    # ---------------------------------------------------------------------
    
    # ---------------------------------------------------------------------
    # Functions for SSL_S4L
    # ---------------------------------------------------------------------

    def ssls4l_rc_in_channels(self):
        return self.args.num_classes

    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Functions for SSL_CCT
    # ---------------------------------------------------------------------
    
    def sslcct_activate_ad_preds(self, ad_preds):
        activated_ad_preds = []
        for ad_pred in ad_preds:
            activated_ad_preds.append(F.softmax(ad_pred, dim=1))
        return activated_ad_preds

    # ---------------------------------------------------------------------


class FDGTGenerator(nn.Module):
    def __init__(self, args):
        super(FDGTGenerator, self).__init__()
        self.args = args

        blur_ksize = int(self.args.im_size / 8)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = pixelssl.nn.module.GaussianBlurLayer(1, blur_ksize)

        reblur_ksize = int(self.args.im_size / 4)
        reblur_ksize = reblur_ksize + 1 if reblur_ksize % 2 == 0 else reblur_ksize
        self.reblur = pixelssl.nn.module.GaussianBlurLayer(1, reblur_ksize)

        self.dilate = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        )

    def forward(self, pred, gt):
        # one_hot = torch.zeros(pred.shape).cuda()
        one_hot = pred.data.clone().mul_(0.0)
        for i in range(0, self.args.num_classes):
            one_hot[:, i:i+1, ...].add_((gt == i).float())
            # ignore the segment boundary
            one_hot[:, i:i+1, ...].mul_((gt != self.args.ignore_index).float())

        diff = torch.abs_(one_hot.sub_(pred.detach()))
        diff = torch.sum(diff, dim=1, keepdim=True).div_(2)

        diff = self.blur(diff)
        diff = self.dilate(diff)
        diff = self.reblur(diff)

        # normlize to 1 for each samples
        dmax = diff.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        dmin = diff.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        diff.sub_(dmin).div_(dmax - dmin + 1e-9)

        flawmap_gt = diff        
        return flawmap_gt


# ---------------------------------------------------------------------
# Functions for Segmentic Segmentation
# Following code from the repository: 
#   https://github.com/hfslyc/AdvSemiSeg.git
# ---------------------------------------------------------------------

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# ---------------------------------------------------------------------
