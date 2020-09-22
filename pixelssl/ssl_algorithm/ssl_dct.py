import os
import cv2
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, cmd, tool
from pixelssl.nn import func
from pixelssl.nn.lrer import PolynomialLR
from pixelssl.nn.module import patch_replication_callback, SynchronizedBatchNorm2d, GaussianBlurLayer

from . import ssl_base


# """ Implementation of Guided Collaborative Training (GCT) for pixel-wise semi-supervised learning

# This method is proposed in paper:
#     'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning'

# This is the official implementation of the above paper.
# """


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)

    parser.add_argument('--cons-scale', type=float, default=-1.0, help='')
    parser.add_argument('--cons-rampup-epochs', type=int, default=-1, help='')


def ssl_dct(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict):
        logger.log_err('The len(element_dict) of SSL_DCT should be the same\n')

    if len(model_dict) == 1:
        if list(model_dict.keys())[0] != 'model':
            logger.log_err('In SSL_DCT, the key of 1-value element_dict should be \'model\',\n'
                           'but \'{0}\' is given\n'.format(model_dict.keys()))

        model_funcs = [model_dict['model'], model_dict['model']]
        optimizer_funcs = [optimizer_dict['model'], optimizer_dict['model']]
        lrer_funcs = [lrer_dict['model'], lrer_dict['model']]
        criterion_funcs = [criterion_dict['model'], criterion_dict['model']]
    
    elif len(model_dict) == 2:
        if 'lmodel' not in list(model_dict.keys()) or 'rmodel' not in list(model_dict.keys()):
            logger.log_err('In SSL_DCT, the key of 2-value element_dict should be \'(lmodel, rmodel)\', '
                           'but \'{0}\' is given\n'.format(model_dict.keys()))

        model_funcs = [model_dict['lmodel'], model_dict['rmodel']]
        optimizer_funcs = [optimizer_dict['lmodel'], optimizer_dict['rmodel']]
        lrer_funcs = [lrer_dict['lmodel'], lrer_dict['rmodel']]
        criterion_funcs = [criterion_dict['lmodel'], criterion_dict['rmodel']]

    else:
        logger.log_err('The SSL_DCT algorithm supports element_dict with 1 or 2 elements, '
                       'but given {0} elements\n'.format(len(model_dict)))

    algorithm = SSLDCT(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLDCT(ssl_base._SSLBase):
    NAME = 'ssl_dct'
    SUPPORTED_TASK_TYPES = [REGRESSION, CLASSIFICATION]
    
    def __init__(self, args):
        super(SSLDCT, self).__init__(args)

        # define the task model and the flaw detector
        self.l_model, self.r_model = None, None
        self.l_optimizer, self.r_optimizer = None, None
        self.l_lrer, self.r_lrer = None, None
        self.l_criterion, self.r_criterion = None, None
        
        self.cons_criterion = None

        # check SSL arguments
        # TODO

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        # 'l_' denotes the first task model while 'r_' denotes the second task model
        self.l_model = func.create_model(model_funcs[0], 'l_model', args=self.args)
        self.r_model = func.create_model(model_funcs[1], 'r_model', args=self.args)
        # call 'patch_replication_callback' to enable the `sync_batchnorm` layer
        patch_replication_callback(self.l_model)
        patch_replication_callback(self.r_model)
        self.models = {'l_model': self.l_model, 'r_model': self.r_model}

        # create optimizers
        self.l_optimizer = optimizer_funcs[0](self.l_model.module.param_groups)
        self.r_optimizer = optimizer_funcs[1](self.r_model.module.param_groups)
        self.optimizers = {'l_optimizer': self.l_optimizer, 'r_optimizer': self.r_optimizer}   

        # create lrers
        self.l_lrer = lrer_funcs[0](self.l_optimizer)
        self.r_lrer = lrer_funcs[1](self.r_optimizer)
        self.lrers = {'l_lrer': self.l_lrer, 'r_lrer': self.r_lrer}

        # create criterions
        self.l_criterion = criterion_funcs[0](self.args)
        self.r_criterion = criterion_funcs[1](self.args)
        self.cons_criterion = torch.nn.MSELoss()
        self.criterions = {'l_criterion': self.l_criterion, 'r_criterion': self.r_criterion, 'cons_criterion': self.cons_criterion}

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.l_model.train()
        self.r_model.train()

        # both 'inp' and 'gt' are tuples
        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            (l_inp, l_gt), (r_inp, r_gt) = self._batch_prehandle(inp, gt)
            if len(l_gt) == len(r_gt) > 1 and idx == 0:
                self._inp_warn()

            # calculate the ramp-up coefficient of the dynamic consistency constraint
            cur_steps = len(data_loader) * epoch + idx
            total_steps = len(data_loader) * self.args.cons_rampup_epochs
            cons_rampup_scale = func.sigmoid_rampup(cur_steps, total_steps)

            # -----------------------------------------------------------------------------
            # step-0: pre-forwarding to save GPU memory
            #   - forward the task models and the flaw detector
            #   - generate pseudo ground truth for the unlabeled data if the dynamic
            #     consistency constraint is enabled
            # -----------------------------------------------------------------------------
            with torch.no_grad():
                l_resulter, l_debugger = self.l_model.forward(l_inp)
                l_activated_pred = tool.dict_value(l_resulter, 'activated_pred')
                r_resulter, r_debugger = self.r_model.forward(r_inp)
                r_activated_pred = tool.dict_value(r_resulter, 'activated_pred')
            
            # -----------------------------------------------------------------------------
            # step-1: train the task models
            # -----------------------------------------------------------------------------

            # train the 'l' task model
            l_loss = self._task_model_iter(epoch, idx, True, 'l', lbs, l_inp, l_gt, r_activated_pred[0].detach(), cons_rampup_scale)
            self.l_optimizer.zero_grad()
            l_loss.backward()
            self.l_optimizer.step()

            # train the 'r' task model
            r_loss = self._task_model_iter(epoch, idx, True, 'r', lbs, r_inp, r_gt, l_activated_pred[0].detach(), cons_rampup_scale)
            self.r_optimizer.zero_grad()
            r_loss.backward()
            self.r_optimizer.step()

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  l-{3}\t=>\t'
                                'l-task-loss: {meters[l_task_loss]:.6f}\t'
                                'l-cons-loss: {meters[l_cons_loss]:.6f}\n'
                                '  r-{3}\t=>\t'
                                'r-task-loss: {meters[r_task_loss]:.6f}\t'
                                'r-cons-loss: {meters[r_cons_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))
                
            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.l_lrer.step()
                self.r_lrer.step()

        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.l_lrer.step()
            self.r_lrer.step()

    def _validate(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.l_model.eval()
        self.r_model.eval()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            (l_inp, l_gt), (r_inp, r_gt) = self._batch_prehandle(inp, gt)
            if len(l_gt) == len(r_gt) > 1 and idx == 0:
                self._inp_warn()

            l_resulter, l_debugger = self.l_model.forward(l_inp)
            l_activated_pred = tool.dict_value(l_resulter, 'activated_pred')
            r_resulter, r_debugger = self.r_model.forward(r_inp)
            r_activated_pred = tool.dict_value(r_resulter, 'activated_pred')

            l_loss = self._task_model_iter(epoch, idx, False, 'l', lbs, l_inp, l_gt, r_activated_pred[0].detach(), 1)
            r_loss = self._task_model_iter(epoch, idx, False, 'r', lbs, r_inp, r_gt, l_activated_pred[0].detach(), 1)

            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                 '  l-{3}\t=>\t'
                                'l-task-loss: {meters[l_task_loss]:.6f}\t'
                                'l-cons-loss: {meters[l_cons_loss]:.6f}\n'
                                '  r-{3}\t=>\t'
                                'r-task-loss: {meters[r_task_loss]:.6f}\t'
                                'r-cons-loss: {meters[r_cons_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))
                        
        # metrics
        metrics_info = {'l': '', 'r': ''}
        for key in sorted(list(self.meters.keys())):
            if self.task_func.METRIC_STR in key:
                for id_str in metrics_info.keys():
                    if key.startswith(id_str):
                        metrics_info[id_str] += '{0}: {1:.6f}\t'.format(key, self.meters[key])
        
        logger.log_info('Validation metrics:\n  l-metrics\t=>\t{0}\n  r-metrics\t=>\t{1}\n'
                        .format(metrics_info['l'].replace('_', '-'), metrics_info['r'].replace('_', '-')))

    def _save_checkpoint(self, epoch):
        state = {
            'algorithm': self.NAME,
            'epoch': epoch + 1,
            'l_model': self.l_model.state_dict(),
            'r_model': self.r_model.state_dict(),
            'l_optimizer': self.l_optimizer.state_dict(),
            'r_optimizer': self.r_optimizer.state_dict(),
            'l_lrer': self.l_lrer.state_dict(),
            'r_lrer': self.r_lrer.state_dict(),
        }

        checkpoint = os.path.join(self.args.checkpoint_path, 'checkpoint_{0}.ckpt'.format(epoch))
        torch.save(state, checkpoint)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume)
        
        checkpoint_algorithm = tool.dict_value(checkpoint, 'algorithm', default='unknown')
        if checkpoint_algorithm != self.NAME:
            logger.log_err('Unmatched ssl algorithm format in checkpoint => required: {0} - given: {1}\n'
                           .format(self.NAME, checkpoint_algorithm))
        
        self.l_model.load_state_dict(checkpoint['l_model'])
        self.r_model.load_state_dict(checkpoint['r_model'])
        self.l_optimizer.load_state_dict(checkpoint['l_optimizer'])
        self.r_optimizer.load_state_dict(checkpoint['r_optimizer'])
        self.l_lrer.load_state_dict(checkpoint['l_lrer'])
        self.r_lrer.load_state_dict(checkpoint['r_lrer'])
        
        return checkpoint['epoch']

    def _task_model_iter(self, epoch, idx, is_train, mid, lbs, 
                         inp, gt, pseudo_gt, cons_rampup_scale):
        if mid == 'l': 
            model, criterion = self.l_model, self.l_criterion
        elif mid == 'r':
            model, criterion = self.r_model, self.r_criterion
        else:
            model, criterion = None, None

        # forward the task model
        resulter, debugger = model.forward(inp)
        if not 'pred' in resulter.keys() or not 'activated_pred' in resulter.keys():
            self._pred_err()
        
        pred = tool.dict_value(resulter, 'pred')
        activated_pred = tool.dict_value(resulter, 'activated_pred')

        # calculate the supervised task constraint on the labeled data
        labeled_pred = func.split_tensor_tuple(pred, 0, lbs)
        labeled_gt = func.split_tensor_tuple(gt, 0, lbs)
        labeled_inp = func.split_tensor_tuple(inp, 0, lbs)
        task_loss = torch.mean(criterion.forward(labeled_pred, labeled_gt, labeled_inp))
        self.meters.update('{0}_task_loss'.format(mid), task_loss.data)

        # calculate the consistency constraint between models
        cons_loss = self.cons_criterion.forward(activated_pred[0], pseudo_gt)
        cons_loss = cons_rampup_scale * self.args.cons_scale * torch.mean(cons_loss)
        self.meters.update('{0}_cons_loss'.format(mid), cons_loss.data)

        # for validation
        if not is_train:
            self.task_func.metrics(activated_pred, gt, inp, self.meters, id_str=mid)
        
        # visualization
        if self.args.visualize and idx % self.args.visual_freq == 0:
            self._visualize(epoch, idx, is_train, mid,
                            func.split_tensor_tuple(inp, 0, 1, reduce_dim=True), 
                            func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                            func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

        loss = task_loss + cons_loss
        return loss

    def _visualize(self, epoch, idx, is_train, id_str,
                   inp, pred, gt):
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        self.task_func.visualize(out_path, id_str=id_str, inp=inp, pred=pred, gt=gt)
        
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

        # augment 'inp' for different task models here if necessary
        l_inp, r_inp = inp, inp
        l_gt, r_gt = gt, gt

        return (l_inp, l_gt), (r_inp, r_gt)

    def _inp_warn(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_DCT\n'
                        'You try to train the task model with more than one (pred & gt) pairs\n'
                        'Please make sure that:\n'
                        '  (1) The prediction tuple has the same size as the ground truth tuple\n'
                        '  (2) The elements with the same index in the two tuples are corresponding\n'
                        '  (3) The first element of (pred & gt) will be used to train the flaw detector\n'
                        '      and calculate the SSL constraints\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_DCT with\n' 
                        'multiple flaw detectors (for multiple predictions)\n')

    def _pred_err(self):
        logger.log_err('In SSL_DCT, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                       '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                       '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                       'We need both of them since some losses include the activation functions,\n'
                       'e.g., the CrossEntropyLoss has contained SoftMax\n')
