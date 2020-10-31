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


""" Implementation of the GCT algorithm for pixel-wise SSL

This method is proposed in the paper:
    'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning'

This is the official implementation of the above paper.
"""


MODE_GCT = 'gct'    # enable both SSL constraints
MODE_FC = 'fc'      # only enable the flaw correction constraint ('L_fc' in paper)
MODE_DC = 'dc'      # only enable the dynamic consistency constraint ('L_dc' in paper)


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)

    parser.add_argument('--ssl-mode', type=str, default=MODE_GCT, choices=[MODE_GCT, MODE_DC, MODE_FC], 
                        help='sslgct - select semi-supervised constraints for training (gct = dc + fc)')

    parser.add_argument('--fc-ssl-scale', type=float, default=-1.0, help='sslgct - flaw correction constraint coefficient')
    parser.add_argument('--dc-ssl-scale', type=float, default=-1.0, help='sslgct - dynamic consistency constraint coefficient')
    parser.add_argument('--dc-threshold', type=float, default=-1.0, help='sslgct - threshold of dynamic consistency constraint')
    parser.add_argument('--dc-rampup-epochs', type=int, default=-1, help='sslgct - ramp-up epochs of dynamic consistency constraint')

    parser.add_argument('--fd-lr', type=float, default=1e-4, help='sslgct - the initial learning rate of the flaw detector')
    parser.add_argument('--fd-scale', type=float, default=1.0, help='sslgct - coefficient of the flaw detector constraint')

    parser.add_argument('--mu', type=float, default=-1.0, help='sslgct - channel average coefficient of the flaw detector\'s ground truth generator')
    parser.add_argument('--nu', type=int, default=-1, help='sslgct - operations repeat coefficient of the flaw detector\'s ground truth generator')


def ssl_gct(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict):
        logger.log_err('The len(element_dict) of SSL_GCT should be the same\n')

    if len(model_dict) == 1:
        if list(model_dict.keys())[0] != 'model':
            logger.log_err('In SSL_GCT, the key of 1-value element_dict should be \'model\',\n'
                           'but \'{0}\' is given\n'.format(model_dict.keys()))

        model_funcs = [model_dict['model'], model_dict['model']]
        optimizer_funcs = [optimizer_dict['model'], optimizer_dict['model']]
        lrer_funcs = [lrer_dict['model'], lrer_dict['model']]
        criterion_funcs = [criterion_dict['model'], criterion_dict['model']]
    
    elif len(model_dict) == 2:
        if 'lmodel' not in list(model_dict.keys()) or 'rmodel' not in list(model_dict.keys()):
            logger.log_err('In SSL_GCT, the key of 2-value element_dict should be \'(lmodel, rmodel)\', '
                           'but \'{0}\' is given\n'.format(model_dict.keys()))

        model_funcs = [model_dict['lmodel'], model_dict['rmodel']]
        optimizer_funcs = [optimizer_dict['lmodel'], optimizer_dict['rmodel']]
        lrer_funcs = [lrer_dict['lmodel'], lrer_dict['rmodel']]
        criterion_funcs = [criterion_dict['lmodel'], criterion_dict['rmodel']]

    else:
        logger.log_err('The SSL_GCT algorithm supports element_dict with 1 or 2 elements, '
                       'but given {0} elements\n'.format(len(model_dict)))

    algorithm = SSLGCT(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLGCT(ssl_base._SSLBase):
    NAME = 'ssl_gct'
    SUPPORTED_TASK_TYPES = [REGRESSION, CLASSIFICATION]
    
    def __init__(self, args):
        super(SSLGCT, self).__init__(args)

        # define the task model and the flaw detector
        self.l_model, self.r_model, self.fd_model = None, None, None
        self.l_optimizer, self.r_optimizer, self.fd_optimizer = None, None, None
        self.l_lrer, self.r_lrer, self.fd_lrer = None, None, None
        self.l_criterion, self.r_criterion, self.fd_criterion = None, None, None
        
        # define the extra modules required by GCT
        self.flawmap_handler = None
        self.dcgt_generator = None
        self.fdgt_generator = None
        self.zero_df_gt = torch.zeros([self.args.batch_size, 1, self.args.im_size, self.args.im_size]).cuda()

        # prepare the arguments for multiple GPUs
        self.args.fd_lr *= self.args.gpus

        # check SSL arguments
        if self.args.unlabeled_batch_size > 0:
            if self.args.ssl_mode in [MODE_GCT, MODE_FC]:
                if self.args.fc_ssl_scale < 0:
                    logger.log_err('The argument - fc_ssl_scale - is not set (or invalid)\n'
                                   'You enable the flaw correction constraint\n'
                                   'Please set - fc_ssl_scale >= 0 - for training\n')
            if self.args.ssl_mode in [MODE_GCT, MODE_DC]:
                if self.args.dc_rampup_epochs < 0:
                    logger.log_err('The argument - dc_rampup_epochs - is not set (or invalid)\n'
                                   'You enable the dynamic consistency constraint\n'
                                   'Please set - dc_rampup_epochs >= 0 - for training\n')
                elif self.args.dc_ssl_scale < 0:
                    logger.log_err('The argument - dc_ssl_scale - is not set (or invalid)\n'
                                   'You enable the dynamic consistency constraint\n'
                                   'Please set - dc_ssl_scale >= 0 - for training\n')
                elif self.args.dc_threshold < 0:
                    logger.log_err('The argument - dc_threshold - is not set (or invalid)\n'
                                   'You enable the dynamic consistency constraint\n'
                                   'Please set - dc_threshold >= 0 - for training\n')
                elif self.args.mu < 0:
                    logger.log_err('The argument - mu - is not set (or invalid)\n'
                                   'Please set - 0 < mu <= 1 - for training\n')
                elif self.args.nu < 0:
                    logger.log_err('The argument - nu - is not set (or invalid)\n'
                                   'Please set - nu > 0 - for training\n')

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        # 'l_' denotes the first task model while 'r_' denotes the second task model
        self.l_model = func.create_model(model_funcs[0], 'l_model', args=self.args)
        self.r_model = func.create_model(model_funcs[1], 'r_model', args=self.args)
        self.fd_model = func.create_model(FlawDetector, 'fd_model', in_channels=self.task_func.sslgct_fd_in_channels())
        # call 'patch_replication_callback' to enable the `sync_batchnorm` layer
        patch_replication_callback(self.l_model)
        patch_replication_callback(self.r_model)
        patch_replication_callback(self.fd_model)
        self.models = {'l_model': self.l_model, 'r_model': self.r_model, 'fd_model': self.fd_model}

        # create optimizers
        self.l_optimizer = optimizer_funcs[0](self.l_model.module.param_groups)
        self.r_optimizer = optimizer_funcs[1](self.r_model.module.param_groups)
        self.fd_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.fd_model.parameters()), 
                                       lr=self.args.fd_lr, betas=(0.9, 0.99))
        self.optimizers = {'l_optimizer': self.l_optimizer, 'r_optimizer': self.r_optimizer, 'fd_optimizer': self.fd_optimizer}   

        # create lrers
        self.l_lrer = lrer_funcs[0](self.l_optimizer)
        self.r_lrer = lrer_funcs[1](self.r_optimizer)
        self.fd_lrer = PolynomialLR(self.fd_optimizer, self.args.epochs, self.args.iters_per_epoch, power=0.9, last_epoch=-1)
        self.lrers = {'l_lrer': self.l_lrer, 'r_lrer': self.r_lrer, 'fd_lrer': self.fd_lrer}

        # create criterions
        self.l_criterion = criterion_funcs[0](self.args)
        self.r_criterion = criterion_funcs[1](self.args)
        self.fd_criterion = FlawDetectorCriterion()
        self.dc_criterion = torch.nn.MSELoss()
        self.criterions = {'l_criterion': self.l_criterion, 'r_criterion': self.r_criterion, 
                           'fd_criterion': self.fd_criterion, 'dc_criterion': self.dc_criterion}

        # build the extra modules required by GCT
        self.flawmap_handler = nn.DataParallel(FlawmapHandler(self.args)).cuda()
        self.dcgt_generator = nn.DataParallel(DCGTGenerator(self.args)).cuda()
        self.fdgt_generator = nn.DataParallel(FDGTGenerator(self.args)).cuda()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.l_model.train()
        self.r_model.train()
        self.fd_model.train()

        # both 'inp' and 'gt' are tuples
        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            (l_inp, l_gt), (r_inp, r_gt) = self._batch_prehandle(inp, gt)
            if len(l_gt) == len(r_gt) > 1 and idx == 0:
                self._inp_warn()

            # calculate the ramp-up coefficient of the dynamic consistency constraint
            cur_steps = len(data_loader) * epoch + idx
            total_steps = len(data_loader) * self.args.dc_rampup_epochs
            dc_rampup_scale = func.sigmoid_rampup(cur_steps, total_steps)

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

            # 'l_flawmap' and 'r_flawmap' will be used in step-2
            fd_resulter, fd_debugger = self.fd_model.forward(l_inp, l_activated_pred[0])
            l_flawmap = tool.dict_value(fd_resulter, 'flawmap')
            fd_resulter, fd_debugger = self.fd_model.forward(r_inp, r_activated_pred[0])
            r_flawmap = tool.dict_value(fd_resulter, 'flawmap')
            
            l_dc_gt, r_dc_gt = None, None
            l_fc_mask, r_fc_mask = None, None
            
            # generate the pseudo ground truth for the dynamic consistency constraint
            if self.args.ssl_mode in [MODE_GCT, MODE_DC]:
                with torch.no_grad():
                    l_handled_flawmap = self.flawmap_handler.forward(l_flawmap)
                    r_handled_flawmap = self.flawmap_handler.forward(r_flawmap)
                    l_dc_gt, r_dc_gt, l_fc_mask, r_fc_mask = self.dcgt_generator.forward(
                        l_activated_pred[0].detach(), r_activated_pred[0].detach(), l_handled_flawmap, r_handled_flawmap)

            # -----------------------------------------------------------------------------
            # step-1: train the task models
            # -----------------------------------------------------------------------------
            for param in self.fd_model.parameters():
                param.requires_grad = False

            # train the 'l' task model
            l_loss = self._task_model_iter(epoch, idx, True, 'l', lbs, l_inp, l_gt, l_dc_gt, l_fc_mask, dc_rampup_scale)
            self.l_optimizer.zero_grad()
            l_loss.backward()
            self.l_optimizer.step()

            # train the 'r' task model
            r_loss = self._task_model_iter(epoch, idx, True, 'r', lbs, r_inp, r_gt, r_dc_gt, r_fc_mask, dc_rampup_scale)
            self.r_optimizer.zero_grad()
            r_loss.backward()
            self.r_optimizer.step()

            # -----------------------------------------------------------------------------
            # step-2: train the flaw detector
            # -----------------------------------------------------------------------------
            for param in self.fd_model.parameters():
                param.requires_grad = True
            
            # generate the ground truth for the flaw detector (on labeled data only)
            with torch.no_grad():
                l_flawmap_gt = self.fdgt_generator.forward(
                    l_activated_pred[0][:lbs, ...].detach(), self.task_func.sslgct_prepare_task_gt_for_fdgt(l_gt[0][:lbs, ...]))
                r_flawmap_gt = self.fdgt_generator.forward(
                    r_activated_pred[0][:lbs, ...].detach(), self.task_func.sslgct_prepare_task_gt_for_fdgt(r_gt[0][:lbs, ...])) 
            
            l_fd_loss = self.fd_criterion.forward(l_flawmap[:lbs, ...], l_flawmap_gt)
            l_fd_loss = self.args.fd_scale * torch.mean(l_fd_loss)
            self.meters.update('l_fd_loss', l_fd_loss.data)
            
            r_fd_loss = self.fd_criterion.forward(r_flawmap[:lbs, ...], r_flawmap_gt)
            r_fd_loss = self.args.fd_scale * torch.mean(r_fd_loss)
            self.meters.update('r_fd_loss', r_fd_loss.data)

            fd_loss = (l_fd_loss + r_fd_loss) / 2

            self.fd_optimizer.zero_grad()
            fd_loss.backward()
            self.fd_optimizer.step()

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  l-{3}\t=>\t'
                                'l-task-loss: {meters[l_task_loss]:.6f}\t'
                                'l-dc-loss: {meters[l_dc_loss]:.6f}\t'
                                'l-fc-loss: {meters[l_fc_loss]:.6f}\n'
                                '  r-{3}\t=>\t'
                                'r-task-loss: {meters[r_task_loss]:.6f}\t'
                                'r-dc-loss: {meters[r_dc_loss]:.6f}\t'
                                'r-fc-loss: {meters[r_fc_loss]:.6f}\n'
                                '  fd\t=>\t'
                                'l-fd-loss: {meters[l_fd_loss]:.6f}\t'
                                'r-fd-loss: {meters[r_fd_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))
                
            # the flaw detector uses polynomiallr [ITER_LRERS]
            self.fd_lrer.step()
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
        self.fd_model.eval()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            (l_inp, l_gt), (r_inp, r_gt) = self._batch_prehandle(inp, gt)
            if len(l_gt) == len(r_gt) > 1 and idx == 0:
                self._inp_warn()

            l_dc_gt, r_dc_gt = None, None
            l_fc_mask, r_fc_mask = None, None
            if self.args.ssl_mode in [MODE_GCT, MODE_DC]:
                l_resulter, l_debugger = self.l_model.forward(l_inp)
                l_activated_pred = tool.dict_value(l_resulter, 'activated_pred')
                r_resulter, r_debugger = self.r_model.forward(r_inp)
                r_activated_pred = tool.dict_value(r_resulter, 'activated_pred')

                fd_resulter, fd_debugger = self.fd_model.forward(l_inp, l_activated_pred[0])
                l_flawmap = tool.dict_value(fd_resulter, 'flawmap')
                fd_resulter, fd_debugger = self.fd_model.forward(r_inp, r_activated_pred[0])
                r_flawmap = tool.dict_value(fd_resulter, 'flawmap')

                l_handled_flawmap = self.flawmap_handler.forward(l_flawmap)
                r_handled_flawmap = self.flawmap_handler.forward(r_flawmap)
                l_dc_gt, r_dc_gt, l_fc_mask, r_fc_mask = self.dcgt_generator.forward(
                    l_activated_pred[0].detach(), r_activated_pred[0].detach(), l_handled_flawmap, r_handled_flawmap)

            l_loss = self._task_model_iter(epoch, idx, False, 'l', lbs, l_inp, l_gt, l_dc_gt, l_fc_mask, 1)
            r_loss = self._task_model_iter(epoch, idx, False, 'r', lbs, r_inp, r_gt, r_dc_gt, r_fc_mask, 1)

            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  l-{3}\t=>\t'
                                'l-task-loss: {meters[l_task_loss]:.6f}\t'
                                'l-dc-loss: {meters[l_dc_loss]:.6f}\t'
                                'l-fc-loss: {meters[l_fc_loss]:.6f}\n'
                                '  r-{3}\t=>\t'
                                'r-task-loss: {meters[r_task_loss]:.6f}\t'
                                'r-dc-loss: {meters[r_dc_loss]:.6f}\t'
                                'r-fc-loss: {meters[r_fc_loss]:.6f}\n'
                                '  fd\t=>\t'
                                'l-fd-loss: {meters[l_fd_loss]:.6f}\t'
                                'r-fd-loss: {meters[r_fd_loss]:.6f}\n'
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
            'fd_model': self.fd_model.state_dict(),
            'l_optimizer': self.l_optimizer.state_dict(),
            'r_optimizer': self.r_optimizer.state_dict(),
            'fd_optimizer': self.fd_optimizer.state_dict(),
            'l_lrer': self.l_lrer.state_dict(),
            'r_lrer': self.r_lrer.state_dict(),
            'fd_lrer': self.fd_lrer.state_dict()
        }

        checkpoint = os.path.join(self.args.checkpoint_path, 'checkpoint_{0}.ckpt'.format(epoch))
        torch.save(state, checkpoint)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume)
        
        checkpoint_algorithm = tool.dict_value(checkpoint, 'algorithm', default='unknown')
        if checkpoint_algorithm != self.NAME:
            logger.log_err('Unmatched SSL algorithm format in checkpoint => required: {0} - given: {1}\n'
                           .format(self.NAME, checkpoint_algorithm))
        
        self.l_model.load_state_dict(checkpoint['l_model'])
        self.r_model.load_state_dict(checkpoint['r_model'])
        self.fd_model.load_state_dict(checkpoint['fd_model'])
        self.l_optimizer.load_state_dict(checkpoint['l_optimizer'])
        self.r_optimizer.load_state_dict(checkpoint['r_optimizer'])
        self.fd_optimizer.load_state_dict(checkpoint['fd_optimizer'])
        self.l_lrer.load_state_dict(checkpoint['l_lrer'])
        self.r_lrer.load_state_dict(checkpoint['r_lrer'])
        self.fd_lrer.load_state_dict(checkpoint['fd_lrer'])
        
        return checkpoint['epoch']

    def _task_model_iter(self, epoch, idx, is_train, mid, lbs, 
                         inp, gt, dc_gt, fc_mask, dc_rampup_scale):
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

        fd_resulter, fd_debugger = self.fd_model.forward(inp, activated_pred[0])
        flawmap = tool.dict_value(fd_resulter, 'flawmap')

        # calculate the supervised task constraint on the labeled data
        labeled_pred = func.split_tensor_tuple(pred, 0, lbs)
        labeled_gt = func.split_tensor_tuple(gt, 0, lbs)
        labeled_inp = func.split_tensor_tuple(inp, 0, lbs)
        task_loss = torch.mean(criterion.forward(labeled_pred, labeled_gt, labeled_inp))
        self.meters.update('{0}_task_loss'.format(mid), task_loss.data)

        # calculate the flaw correction constraint
        if self.args.ssl_mode in [MODE_GCT, MODE_FC]:
            if flawmap.shape == self.zero_df_gt.shape:
                fc_ssl_loss = self.fd_criterion.forward(flawmap, self.zero_df_gt, is_ssl=True, reduction=False)
            else:
                fc_ssl_loss = self.fd_criterion.forward(flawmap, torch.zeros(flawmap.shape).cuda(), is_ssl=True, reduction=False)
            
            if self.args.ssl_mode == MODE_GCT:
                fc_ssl_loss = fc_mask * fc_ssl_loss
            
            fc_ssl_loss = self.args.fc_ssl_scale * torch.mean(fc_ssl_loss)
            self.meters.update('{0}_fc_loss'.format(mid), fc_ssl_loss.data)
        else:
            fc_ssl_loss = 0
            self.meters.update('{0}_fc_loss'.format(mid), fc_ssl_loss)

        # calculate the dynamic consistency constraint
        if self.args.ssl_mode in [MODE_GCT, MODE_DC]:
            if dc_gt is None:
                logger.log_err('The dynamic consistency constraint is enabled, '
                               'but no pseudo ground truth is given.')
                        
            dc_ssl_loss = self.dc_criterion.forward(activated_pred[0], dc_gt)
            dc_ssl_loss = dc_rampup_scale * self.args.dc_ssl_scale * torch.mean(dc_ssl_loss)
            self.meters.update('{0}_dc_loss'.format(mid), dc_ssl_loss.data)
        else:
            dc_ssl_loss = 0
            self.meters.update('{0}_dc_loss'.format(mid), dc_ssl_loss)

        with torch.no_grad():
            flawmap_gt = self.fdgt_generator.forward(
                activated_pred[0], self.task_func.sslgct_prepare_task_gt_for_fdgt(gt[0]))

        # for validation
        if not is_train:
            fd_loss = self.args.fd_scale * self.fd_criterion.forward(flawmap, flawmap_gt)
            self.meters.update('{0}_fd_loss'.format(mid), torch.mean(fd_loss).data)

            self.task_func.metrics(activated_pred, gt, inp, self.meters, id_str=mid)
        
        # visualization
        if self.args.visualize and idx % self.args.visual_freq == 0:
            with torch.no_grad():
                handled_flawmap = self.flawmap_handler(flawmap)[0]

            self._visualize(epoch, idx, is_train, mid,
                            func.split_tensor_tuple(inp, 0, 1, reduce_dim=True), 
                            func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                            func.split_tensor_tuple(gt, 0, 1, reduce_dim=True),
                            handled_flawmap, flawmap_gt[0], dc_gt[0])

        loss = task_loss + fc_ssl_loss + dc_ssl_loss
        return loss

    # -------------------------------------------------------------------------------------------
    # Tool Functions for SSL_GCT
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_train, id_str,
                   inp, pred, gt, flawmap, flawmap_gt, dc_gt=None):
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        self.task_func.visualize(out_path, id_str=id_str, inp=inp, pred=pred, gt=gt)
        
        flawmap = flawmap[0].data.cpu().numpy()
        Image.fromarray((flawmap * 255).astype('uint8'), mode='L').save(out_path + '_{0}-fmap.png'.format(id_str))
        flawmap_gt = flawmap_gt[0].data.cpu().numpy()
        Image.fromarray((flawmap_gt * 255).astype('uint8'), mode='L').save(out_path + '_{0}-fmap-gt.png'.format(id_str))

        if dc_gt is not None:
            self.task_func.visualize(out_path, id_str=id_str + '_dc', inp=None, pred=[dc_gt], gt=None)

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
        logger.log_warn('More than one ground truth of the task model is given in SSL_GCT\n'
                        'You try to train the task model with more than one (pred & gt) pairs\n'
                        'Please make sure that:\n'
                        '  (1) The prediction tuple has the same size as the ground truth tuple\n'
                        '  (2) The elements with the same index in the two tuples are corresponding\n'
                        '  (3) The first element of (pred & gt) will be used to train the flaw detector\n'
                        '      and calculate the SSL constraints\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_GCT with\n' 
                        'multiple flaw detectors (for multiple predictions)\n')

    def _pred_err(self):
        logger.log_err('In SSL_GCT, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                       '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                       '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                       'We need both of them since some losses include the activation functions,\n'
                       'e.g., the CrossEntropyLoss has contained SoftMax\n')


class FlawDetector(nn.Module):
    """ The FC Discriminator proposed in paper:
        'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning'
    """

    ndf = 64    # basic number of channels
	
    def __init__(self, in_channels):
        super(FlawDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, self.ndf, kernel_size=4, stride=2, padding=1)
        self.ibn1 = IBNorm(self.ndf)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1)
        self.ibn2 = IBNorm(self.ndf * 2)
        self.conv2_1 = nn.Conv2d(self.ndf * 2, self.ndf * 2, kernel_size=4, stride=1, padding=1)
        self.ibn2_1 = IBNorm(self.ndf * 2)
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1)
        self.ibn3 = IBNorm(self.ndf * 4)
        self.conv3_1 = nn.Conv2d(self.ndf * 4, self.ndf * 4, kernel_size=4, stride=1, padding=1)
        self.ibn3_1 = IBNorm(self.ndf * 4)
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1)
        self.ibn4 = IBNorm(self.ndf * 8)
        self.conv4_1 = nn.Conv2d(self.ndf * 8, self.ndf * 8, kernel_size=4, stride=1, padding=1)
        self.ibn4_1 = IBNorm(self.ndf * 8)
        self.classifier = nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, task_inp, task_pred):
        resulter, debugger = {}, {}

        task_inp = torch.cat(task_inp, dim=1)
        x = torch.cat((task_inp, task_pred), dim=1)
        x = self.leaky_relu(self.ibn1(self.conv1(x)))
        x = self.leaky_relu(self.ibn2(self.conv2(x)))
        x = self.leaky_relu(self.ibn2_1(self.conv2_1(x)))
        x = self.leaky_relu(self.ibn3(self.conv3(x)))
        x = self.leaky_relu(self.ibn3_1(self.conv3_1(x)))
        x = self.leaky_relu(self.ibn4(self.conv4(x)))
        x = self.leaky_relu(self.ibn4_1(self.conv4_1(x)))
        x = self.classifier(x)
        x = F.interpolate(x, size=(task_pred.shape[2], task_pred.shape[3]), mode='bilinear', align_corners=True)

        # x is not activated here since it will be activated by the criterion function
        assert x.shape[2:] == task_pred.shape[2:]
        resulter['flawmap'] = x
        return resulter, debugger


class IBNorm(nn.Module):
    """ This layer combines BatchNorm and InstanceNorm.
    """

    def __init__(self, num_features, split=0.5):
        super(IBNorm, self).__init__()

        self.num_features = num_features
        self.num_BN = int(num_features * split + 0.5)
        self.bnorm = SynchronizedBatchNorm2d(num_features=self.num_BN, affine=True)
        self.inorm = nn.InstanceNorm2d(num_features=num_features - self.num_BN, affine=False)

    def forward(self, x):
        if self.num_BN == self.num_features:
            return self.bnorm(x.contiguous())
        else:
            xb = self.bnorm(x[:, 0:self.num_BN, :, :].contiguous())
            xi = self.inorm(x[:, self.num_BN:, :, :].contiguous())

            return torch.cat((xb, xi), 1)


class FlawDetectorCriterion(nn.Module):
    """ Criterion of the flaw detector.
    """

    def __init__(self):
        super(FlawDetectorCriterion, self).__init__()

    def forward(self, pred, gt, is_ssl=False, reduction=True):    
        loss = F.mse_loss(pred, gt, reduction='none')
        if reduction:
            loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


class FlawmapHandler(nn.Module):
    """ Post-processing of the predicted flawmap.

    This module processes the predicted flawmap to fix some special 
    cases that may cause errors in the subsequent steps of generating
    pseudo ground truth.
    """
    
    def __init__(self, args):
        super(FlawmapHandler, self).__init__()
        self.args = args
        self.clip_threshold = 0.1

        blur_ksize = int(self.args.im_size / 16)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

    def forward(self, flawmap):
        flawmap = flawmap.data

        # force all values to be larger than 0
        flawmap.mul_((flawmap >= 0).float())
        # smooth the flawmap
        flawmap = self.blur(flawmap)
        # if all values in the flawmap are less than 'clip_threshold'
        # set the entire flawmap to 0, i.e., no flaw pixel
        fmax = flawmap.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        fmin = flawmap.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_matrix = fmax.repeat(1, 1, flawmap.shape[2], flawmap.shape[3])
        flawmap.mul_((max_matrix > self.clip_threshold).float())
        # normalize the flawmap
        flawmap = flawmap.sub_(fmin).div_(fmax - fmin + 1e-9)

        return flawmap


class DCGTGenerator(nn.Module):
    """ Generate the ground truth of the dynamic consistency constraint.
    """

    def __init__(self, args):
        super(DCGTGenerator, self).__init__()
        self.args = args

    def forward(self, l_pred, r_pred, l_handled_flawmap, r_handled_flawmap):
        l_tmp = l_handled_flawmap.clone()
        r_tmp = r_handled_flawmap.clone()

        l_bad = l_tmp > self.args.dc_threshold
        r_bad = r_tmp > self.args.dc_threshold

        both_bad = (l_bad & r_bad).float()

        l_handled_flawmap.mul_((l_tmp <= self.args.dc_threshold).float())
        r_handled_flawmap.mul_((r_tmp <= self.args.dc_threshold).float())

        l_handled_flawmap.add_((l_tmp > self.args.dc_threshold).float())
        r_handled_flawmap.add_((r_tmp > self.args.dc_threshold).float())

        l_mask = (r_handled_flawmap >= l_handled_flawmap).float()
        r_mask = (l_handled_flawmap >= r_handled_flawmap).float()

        l_dc_gt = l_mask * l_pred + (1 - l_mask) * r_pred
        r_dc_gt = r_mask * r_pred + (1 - r_mask) * l_pred

        return l_dc_gt, r_dc_gt, both_bad, both_bad


class FDGTGenerator(nn.Module):
    """ Generate the ground truth of the flaw detector, 
        i.e., pipeline 'C' in the paper.
    """

    def __init__(self, args):
        super(FDGTGenerator, self).__init__()
        self.args = args

        blur_ksize = int(self.args.im_size / 8)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

        reblur_ksize = int(self.args.im_size / 4)
        reblur_ksize = reblur_ksize + 1 if reblur_ksize % 2 == 0 else reblur_ksize
        self.reblur = GaussianBlurLayer(1, reblur_ksize)

        self.dilate = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        )

    def forward(self, pred, gt):
        diff = torch.abs_(gt - pred.detach())
        diff = torch.sum(diff, dim=1, keepdim=True).mul_(self.args.mu)
        
        diff = self.blur(diff)
        for _ in range(0, self.args.nu):
            diff = self.reblur(self.dilate(diff))

        # normlize each sample to [0, 1]
        dmax = diff.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        dmin = diff.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        diff.sub_(dmin).div_(dmax - dmin + 1e-9)

        flawmap_gt = diff
        return flawmap_gt
