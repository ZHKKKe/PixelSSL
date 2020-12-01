import os
import cv2
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.uniform import Uniform

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, tool, cmd
from pixelssl.nn import func
from pixelssl.nn.module import patch_replication_callback

from . import ssl_base


""" Implementation of the CCT algorithm for pixel-wise SSL

This method is proposed in the paper:
    'Semi-Supervised Semantic Segmentation with Cross-Consistency Training' 

This implementation tries to follow the code from: https://github.com/yassouali/CCT
Since the code of the auxiliary decoders are adapted from above repository, they may 
be only suitable for pixel-wise classification. 
For the semantic segmentation task, we only experimented with CCT under PSPNet.

This implementation supports:
    RAMP-UP TYPES: sigmoid ramp-up function
    LOSS TYPES: MSE loss
for calculating the consistency constraint.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)

    parser.add_argument('--cons-scale', type=float, default=-1, help='sslcct - consistency constraint coefficient')
    parser.add_argument('--cons-rampup-epochs', type=int, default=-1, help='sslcct - ramp-up epochs of conistency constraint')
    
    parser.add_argument('--ad-lr-scale', type=float, default=-1, help='sslcct - learning rate scale for auxiliary decoders')

    parser.add_argument('--vat-dec-num', type=int, default=0, help='sslcct - number of the \'I-VAT\' auxiliary decoders')
    parser.add_argument('--vat-dec-xi', type=float, default=1e-6, help='sslcct - the argument \'xi\' for \'I-VAT\' auxiliary decoders')
    parser.add_argument('--vat-dec-eps', type=float, default=2.0, help='sslcct - the argument \'eps\' for \'I-VAT\' auxiliary decoders')

    parser.add_argument('--drop-dec-num', type=int, default=0, help='sslcct - number of the \'DropOut\' auxiliary decoders')
    parser.add_argument('--drop-dec-rate', type=float, default=0.5, help='sslcct - the argument \'rate\' for \'DropOut\' auxiliary decoders')
    parser.add_argument('--drop-dec-spatial', type=cmd.str2bool, default=True, help='sslcct - the argument \'spatial\' for \'DropOut\' auxiliary decoders')

    parser.add_argument('--cut-dec-num', type=int, default=0, help='sslcct - number of the \'G-Cutout\' auxiliary decoders')
    parser.add_argument('--cut-dec-erase', type=float, default=0.4, help='sslcct - the argument \'erase\' for \'G-Cutout\' auxiliary decoders')

    parser.add_argument('--context-dec-num', type=int, default=0, help='sslcct - number of the \'Con-Msk\' auxiliary decoders')

    parser.add_argument('--object-dec-num', type=int, default=0, help='sslcct - number of the \'Obj-Msk\' auxiliary decoders')

    parser.add_argument('--fn-dec-num', type=int, default=0, help='sslcct - number of the \'F-Noise\' auxiliary decoders')
    parser.add_argument('--fn-dec-uniform', type=float, default=0.3, help='sslcct - the argument \'uniform\' for \'F-Noise\' auxiliary decoders')

    parser.add_argument('--fd-dec-num', type=int, default=0, help='sslcct - number of the \'F-Drop\' auxiliary decoders')

def ssl_cct(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_CCT should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_CCT, the key of element_dict should be \'model\',\n'
                'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLCCT(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLCCT(ssl_base._SSLBase):
    NAME = 'ssl_cct'
    # TODO: support the regression tasks
    SUPPORTED_TASK_TYPES = [CLASSIFICATION]

    def __init__(self, args):
        super(SSLCCT, self).__init__(args)

        self.main_model = None
        self.auxiliary_decoders = None

        self.model = None
        self.optimizer = None
        self.lrer = None
        self.criterion = None

        self.cons_criterion = None

        # check SSL arguments
        if self.args.unlabeled_batch_size > 0:
            if self.args.cons_scale < 0:
                logger.log_err('The argument - cons_scale - is not set (or invalid)\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - cons_scale >= 0 - for training\n')
            elif self.args.cons_rampup_epochs < 0:
                logger.log_err('The argument - cons_rampup_epochs - is not set (or invalid)\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - cons_rampup_epochs >= 0 - for training\n')
            if self.args.ad_lr_scale < 0:
                logger.log_err('The argument - ad_lr_scale - is not set (or invalid)\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - ad_lr_scale >= 0 - for training\n')
        else:
            self.args.ad_lr_scale = 0

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create criterions
        # TODO: support more types of the consistency criterion
        self.cons_criterion = nn.MSELoss()
        self.criterion = criterion_funcs[0](self.args)
        self.criterions = {'criterion': self.criterion, 'cons_criterion': self.cons_criterion}

        # create the main task model
        self.main_model = func.create_model(model_funcs[0], 'main_model', args=self.args).module
        
        # create the auxiliary decoders
        vat_decoders = [
            VATDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels(), 
                xi=self.args.vat_dec_xi, 
                eps=self.args.vat_dec_eps
            ) for _ in range(0, self.args.vat_dec_num)
        ]
        drop_decoders = [
            DropOutDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels(), 
                drop_rate=self.args.drop_dec_rate, 
                spatial_dropout=self.args.drop_dec_spatial
            ) for _ in range(0, self.args.drop_dec_num)
        ]
        cut_decoders = [
            CutOutDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels(), 
                erase=self.args.cut_dec_erase
            ) for _ in range(0, self.args.cut_dec_num)
        ]
        context_decoders = [
            ContextMaskingDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels()
            ) for _ in range(0, self.args.context_dec_num)
        ]
        object_decoders = [
            ObjectMaskingDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels()
            ) for _ in range(0, self.args.object_dec_num)
        ]
        feature_drop_decoders = [
            FeatureDropDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels()
            ) for _ in range(0, self.args.fd_dec_num)
        ]
        feature_noise_decoders = [
            FeatureNoiseDecoder(
                self.task_func.sslcct_ad_upsample_scale(), 
                self.task_func.sslcct_ad_in_channels(), 
                self.task_func.sslcct_ad_out_channels(), 
                uniform_range=self.args.fn_dec_uniform
            ) for _ in range(0, self.args.fn_dec_num)
        ]

        self.auxiliary_decoders = nn.ModuleList(
            [
                *vat_decoders, 
                *drop_decoders, 
                *cut_decoders, 
                *context_decoders, 
                *object_decoders, 
                *feature_drop_decoders, 
                *feature_noise_decoders,
            ]
        )

        # wrap 'self.main_model' and 'self.auxiliary decoders' into a single model
        # NOTE: all criterions are wrapped into the model to save the memory of the main GPU
        self.model = WrappedCCTModel(self.args, self.main_model, self.auxiliary_decoders, 
                                     self.criterion, self.cons_criterion, self.task_func.sslcct_activate_ad_preds)
        self.model = nn.DataParallel(self.model).cuda()
        # call 'patch_replication_callback' to use the `sync_batchnorm` layer
        patch_replication_callback(self.model)
        self.models = {'model': self.model}

        # create optimizers
        self.optimizer = optimizer_funcs[0](self.model.module.param_groups)
        self.optimizers = {'optimizer': self.optimizer}

        # create lrers
        self.lrer = lrer_funcs[0](self.optimizer)
        self.lrers = {'lrer': self.lrer}

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.model.train()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            inp, gt = self._batch_prehandle(inp, gt)
            if len(gt) > 1 and idx == 0:
                self._data_err()

            # TODO: support more ramp-up functions
            # calculate the ramp-up coefficient of the consistency constraint
            cur_step = len(data_loader) * epoch + idx
            total_steps = len(data_loader) * self.args.cons_rampup_epochs
            cons_rampup_scale = func.sigmoid_rampup(cur_step, total_steps)

            self.optimizer.zero_grad()

            # -----------------------------------------------------------
            # For Labeled Data
            # -----------------------------------------------------------
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_inp = func.split_tensor_tuple(inp, 0, lbs)

            # forward the wrapped CCT model
            resulter, debugger = self.model.forward(l_inp, l_gt, False)
            l_pred = tool.dict_value(resulter, 'pred')
            l_activated_pred = tool.dict_value(resulter, 'activated_pred')

            task_loss = tool.dict_value(resulter, 'task_loss', err=True)
            task_loss = task_loss.mean()
            self.meters.update('task_loss', task_loss.data)

            # -----------------------------------------------------------
            # For Unlabeled Data
            # -----------------------------------------------------------
            if self.args.unlabeled_batch_size > 0:
                ul_gt = func.split_tensor_tuple(gt, lbs, self.args.batch_size)
                ul_inp = func.split_tensor_tuple(inp, lbs, self.args.batch_size)

                # forward the wrapped CCT model
                resulter, debugger = self.model.forward(ul_inp, ul_gt, True)
                ul_pred = tool.dict_value(resulter, 'pred')
                ul_activated_pred = tool.dict_value(resulter, 'activated_pred')
                ul_ad_preds = tool.dict_value(resulter, 'ul_ad_preds')

                cons_loss = tool.dict_value(resulter, 'cons_loss', err=True)
                cons_loss = cons_loss.mean()
                cons_loss = cons_rampup_scale * self.args.cons_scale * cons_loss
                self.meters.update('cons_loss', cons_loss.data)

            else:
                cons_loss = 0
                self.meters.update('cons_loss', cons_loss)

            # backward and update the wrapped CCT model
            loss = task_loss + cons_loss
            loss.backward()
            self.optimizer.step()

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\t'
                                'cons-loss: {meters[cons_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.lrer.step()
        
        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.lrer.step()

    def _validate(self, data_loader, epoch):
        self.meters.reset()

        self.model.eval()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            inp, gt = self._batch_prehandle(inp, gt)
            if len(gt) > 1 and idx == 0:
                self._data_err()
            
            resulter, debugger = self.model.forward(inp, gt, False)
            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')

            task_loss = tool.dict_value(resulter, 'task_loss', err=True)
            task_loss = task_loss.mean()
            self.meters.update('task_loss', task_loss.data)

            self.task_func.metrics(activated_pred, gt, inp, self.meters, id_str='task')

            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\t'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, False, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

        # metrics
        metrics_info = {'task': ''}
        for key in sorted(list(self.meters.keys())):
            if self.task_func.METRIC_STR in key:
                for id_str in metrics_info.keys():
                    if key.startswith(id_str):
                        metrics_info[id_str] += '{0}: {1:.6}\t'.format(key, self.meters[key])
            
        logger.log_info('Validation metrics:\n task-metrics\t=>\t{0}\n'.format(metrics_info['task'].replace('_', '-')))

    def _save_checkpoint(self, epoch):
        state = {
            'algorithm': self.NAME,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lrer': self.lrer.state_dict(),
        }

        checkpoint = os.path.join(self.args.checkpoint_path, 'checkpoint_{0}.ckpt'.format(epoch))
        torch.save(state, checkpoint)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume)

        checkpoint_algorithm = tool.dict_value(checkpoint, 'algorithm', default='unknown')

        if checkpoint_algorithm != self.NAME:
            logger.log_err('Unmatched SSL algorithm format in checkpoint => required: {0} - given: {1}\n'
                           .format(self.NAME, checkpoint_algorithm))

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lrer.load_state_dict(checkpoint['lrer'])

        self.main_model = self.model.module.main_model
        self.auxiliary_decoders = self.model.module.auxiliary_decoders

        return checkpoint['epoch']

    # -------------------------------------------------------------------------------------------
    # Tool Functions for SSL_CCT
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_unlabeled, inp, pred, gt):
        visualize_path = self.args.visual_train_path if is_unlabeled else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        self.task_func.visualize(out_path, id_str='task', inp=inp, pred=pred, gt=gt)

    def _batch_prehandle(self, inp, gt):
        # add extra data augmentation process here if necessary
        
        inp_var = []
        for i in inp:
            inp_var.append(Variable(i).cuda())
        inp = tuple(inp_var)
            
        gt_var = []
        for g in gt:
            gt_var.append(Variable(g).cuda())
        gt = tuple(gt_var)

        return inp, gt

    def _data_err(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_CCT\n'
                        'Currently, this implementation of CCT algorithm supports only one (pred & gt) pairs\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_CCT that\n' 
                        'supports more than one (pred & gt) pairs\n')

    def _algorithm_warn(self):
        logger.log_warn('This SSL_CCT algorithm reproducts the SSL algorithm from paper:\n'
                        '  \'Semi-Supervised Semantic Segmentation with Cross-Consistency Training\'\n'
                        'The code of the auxiliary decoders are adapted from the official repository:\n'
                        '   https://github.com/yassouali/CCT \n'
                        'These auxiliary decoders may only suitable for pixel-wise classification\n'
                        'Hence, this implementation does not currently support pixel-wise regression tasks\n'
                        'Besides, the auxiliary decoders will use huge GPU memory\n'
                        'Please reduce the number of the auxiliary decoders if you run out of GPU memory\n')


class WrappedCCTModel(nn.Module):
    def __init__(self, args, main_model, auxiliary_decoders, task_criterion, cons_criterion, ad_activation_func):
        super(WrappedCCTModel, self).__init__()
        self.args = args
        self.main_model = main_model
        self.auxiliary_decoders = auxiliary_decoders
        self.task_criterion = task_criterion
        self.cons_criterion = cons_criterion
        self.ad_activation_func = ad_activation_func

        self.param_groups = self.main_model.param_groups + \
            [{'params': self.auxiliary_decoders.parameters(), 'lr': self.args.lr * self.args.ad_lr_scale}]

    def forward(self, inp, gt, is_unlabeled):
        resulter, debugger = {}, {}

        # forward the task model
        m_resulter, m_debugger = self.main_model.forward(inp)

        if not 'pred' in m_resulter.keys() or not 'activated_pred' in m_resulter.keys():
            logger.log_err('In SSL_CCT, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                           '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                           '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                           'We need both of them since some losses include the activation functions,\n'
                           'e.g., the CrossEntropyLoss has contained SoftMax\n')

        resulter['pred'] = tool.dict_value(m_resulter, 'pred')
        resulter['activated_pred'] = tool.dict_value(m_resulter, 'activated_pred')

        if not len(resulter['pred']) == len(resulter['activated_pred']) == 1:
            logger.log_err('This implementation of SSL_CCT only support the task model with only one prediction (output). \n'
                           'However, there are {0} predictions.\n'.format(len(resulter['pred'])))

        # calculate the task loss
        resulter['task_loss'] = None if is_unlabeled else torch.mean(self.task_criterion.forward(resulter['pred'], gt, inp))

        # for the unlabeled data
        if is_unlabeled and self.args.unlabeled_batch_size > 0:
            if not 'sslcct_ad_inp' in m_resulter.keys():
                logger.log_err('In SSL_CCT, the \'resulter\' dict returned by the task model should contain the key:\n'
                               '    \'sslcct_ad_inp\'\t=>\tinputs of the auxiliary decoders (a 4-dim tensor)\n'
                               'It is the feature map encoded by the task model\n'
                               'Please add the key \'sslcct_ad_inp\' in your task model\'s resulter\n'
                               'Note that for different task models, the shape of \'sslcct_ad_inp\' may be different\n')

            ul_ad_inp = tool.dict_value(m_resulter, 'sslcct_ad_inp')
            ul_main_pred = resulter['pred'][0].detach()

            # forward the auxiliary decoders
            ul_ad_preds = []
            for ad in self.auxiliary_decoders:
                ul_ad_preds.append(ad.forward(ul_ad_inp, pred_of_main_decoder=ul_main_pred))

            resulter['ul_ad_preds'] = ul_ad_preds

            # calculate the consistency loss
            ul_ad_gt = resulter['activated_pred'][0].detach()
            ul_ad_preds = [F.interpolate(ul_ad_pred, size=(ul_ad_gt.shape[2], ul_ad_gt.shape[3]), mode='bilinear') for ul_ad_pred in ul_ad_preds]
            ul_activated_ad_preds = self.ad_activation_func(ul_ad_preds)
            cons_loss = sum([self.cons_criterion.forward(ul_activated_ad_pred, ul_ad_gt) for ul_activated_ad_pred in ul_activated_ad_preds])
            cons_loss = torch.mean(cons_loss) / len(ul_activated_ad_preds)
            resulter['cons_loss'] = cons_loss
        else:
            resulter['ul_ad_preds'] = None
            resulter['cons_loss'] = None

        return resulter, debugger


# =======================================================
# Following code is adapted form the repository:
#   https://github.com/yassouali/CCT 
# =======================================================

# Archtectures of the Auxiliary Decoders

class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        self._icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x

    def _icnr(self, x, scale=2, init=nn.init.kaiming_normal_):
        """
        Checkerboard artifact free sub-pixel convolution
        https://arxiv.org/abs/1707.02937
        """
        ni,nf,h,w = x.shape
        ni2 = int(ni/(scale**2))
        k = init(torch.zeros([ni2,nf,h,w]).cuda()).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale**2)
        k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
        x.data.copy_(k)


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


class VATDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        r_adv = self.get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        x = self.upsample(x + r_adv)
        return x

    def get_r_adv(self, x, decoder, it=1, xi=1e-1, eps=10.0):
        """ Virtual Adversarial Training from:
                https://arxiv.org/abs/1704.03976
        """
        x_detached = x.detach()
        with torch.no_grad():
            pred = F.softmax(decoder(x_detached), dim=1)

        d = torch.rand(x.shape).sub(0.5).cuda()
        d = self._l2_normalize(d)

        for _ in range(it):
            d.requires_grad_()
            pred_hat = decoder(x_detached + xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = self._l2_normalize(d.grad)
            decoder.zero_grad()

        r_adv = d * eps
        return r_adv

    def _l2_normalize(self, d):
        # Normalizing per batch axis
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d


class DropOutDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x = self.upsample(self.dropout(x))
        return x


class CutOutDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale 
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        maskcut = self.guided_cutout(pred_of_main_decoder, upscale=self.upscale, 
                                     erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = self.upsample(x)
        return x

    def guided_cutout(self, output, upscale, resize, erase=0.4, use_dropout=False):
        if len(output.shape) == 3:
            masks = (output > 0).float()
        else:
            masks = (output.argmax(1) > 0).float()

        if use_dropout:
            p_drop = random.randint(3, 6)/10
            maskdroped = (F.dropout(masks, p_drop) > 0).float()
            maskdroped = maskdroped + (1 - masks)
            maskdroped.unsqueeze_(0)
            maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

        masks_np = []
        for mask in masks:
            mask_np = np.uint8(mask.cpu().numpy())
            mask_ones = np.ones_like(mask_np)
            try: # Version 3.x
                _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except: # Version 4.x
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
            for poly in polys:
                min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
                min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
                bb_w, bb_h = max_w-min_w, max_h-min_h
                rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
                rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
                h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
                w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
                mask_ones[h_start:h_end, w_start:w_end] = 0
            masks_np.append(mask_ones)
        masks_np = np.stack(masks_np)

        maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
        maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

        if use_dropout:
            return maskcut.cuda(), maskdroped.cuda()
        return maskcut.cuda()


class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x_masked_context = self.guided_masking(x, pred_of_main_decoder, resize=(x.size(2), x.size(3)),
                                               upscale=self.upscale, return_msk_context=True)
        x_masked_context = self.upsample(x_masked_context)
        return x_masked_context

    def guided_masking(self, x, output, upscale, resize, return_msk_context=True):
        if len(output.shape) == 3:
            masks_context = (output > 0).float().unsqueeze(1)
        else:
            masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
        
        masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

        x_masked_context = masks_context * x
        if return_msk_context:
            return x_masked_context

        masks_objects = (1 - masks_context)
        x_masked_objects = masks_objects * x
        return x_masked_objects


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x_masked_obj = self.guided_masking(x, pred_of_main_decoder, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
        x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj

    def guided_masking(self, x, output, upscale, resize, return_msk_context=True):
        if len(output.shape) == 3:
            masks_context = (output > 0).float().unsqueeze(1)
        else:
            masks_context = (output.argmax(1) > 0).float().unsqueeze(1)
        
        masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

        x_masked_context = masks_context * x
        if return_msk_context:
            return x_masked_context

        masks_objects = (1 - masks_context)
        x_masked_objects = masks_objects * x
        return x_masked_objects


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)

    def forward(self, x, pred_of_main_decoder=None):
        x = self.feature_dropout(x)
        x = self.upsample(x)
        return x

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, in_channels, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(in_channels, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def forward(self, x, pred_of_main_decoder=None):
        x = self.feature_based_noise(x)
        x = self.upsample(x)
        return x

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).cuda().unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise
