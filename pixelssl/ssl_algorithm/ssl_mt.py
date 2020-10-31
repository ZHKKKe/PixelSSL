import os
import time
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, cmd, tool
from pixelssl.nn import func
from pixelssl.nn.module import patch_replication_callback, GaussianNoiseLayer

from . import ssl_base


""" Implementation of pixel-wise Mean Teacher (MT)
    
This method is proposed in the paper: 
    'Mean Teachers are Better Role Models:
        Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results'

This implementation only supports Gaussian noise as input perturbation, and the two-heads
outputs trick is not available.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)
    parser.add_argument('--cons-for-labeled', type=cmd.str2bool, default=True, help='sslmt - calculate the consistency constraint on the labeled data if True')
    parser.add_argument('--cons-scale', type=float, default=-1, help='sslmt - consistency constraint coefficient')
    parser.add_argument('--cons-rampup-epochs', type=int, default=-1, help='sslmt - ramp-up epochs of conistency constraint')

    parser.add_argument('--ema-decay', type=float, default=0.999, help='sslmt - EMA coefficient of teacher model')
    parser.add_argument('--gaussian-noise-std', type=float, default=None, help='sslmt - std of input gaussian noise (set to None to disable it)')


def ssl_mt(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_MT should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_MT, the key of element_dict should be \'model\',\n'
                       'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLMT(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLMT(ssl_base._SSLBase):
    NAME = 'ssl_mt'
    SUPPORTED_TASK_TYPES = [REGRESSION, CLASSIFICATION]

    def __init__(self, args):
        super(SSLMT, self).__init__(args)

        # define the student model and the teacher model
        self.s_model, self.t_model = None, None
        self.s_optimizer = None
        self.s_lrer = None
        self.s_criterion = None

        self.cons_criterion = None
        
        self.gaussian_noiser = None
        self.zero_tensor = torch.zeros(1)

        # check SSL arguments
        if self.args.cons_for_labeled or self.args.unlabeled_batch_size > 0:
            if self.args.cons_scale < 0:
                logger.log_err('The argument - cons_scale - is not set (or invalid)\n'
                               'You set argument - cons_for_labeled - to True\n'
                               'or\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - cons_scale >= 0 - for training\n')
            elif self.args.cons_rampup_epochs < 0:
                logger.log_err('The argument - cons_rampup_epochs - is not set (or invalid)\n'
                               'You set argument - cons_for_labeled - to True\n'
                               'or\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - cons_rampup_epochs >= 0 - for training\n')

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        self.s_model = func.create_model(model_funcs[0], 's_model', args=self.args)
        self.t_model = func.create_model(model_funcs[0], 't_model', args=self.args)
        # call 'patch_replication_callback' to use the `sync_batchnorm` layer
        patch_replication_callback(self.s_model)
        patch_replication_callback(self.t_model)
        # detach the teacher model
        for param in self.t_model.parameters():
            param.detach_()
        self.models = {'s_model': self.s_model, 't_model': self.t_model}

        # create optimizers
        self.s_optimizer = optimizer_funcs[0](self.s_model.module.param_groups)
        self.optimizers = {'s_optimizer': self.s_optimizer}

        # create lrers
        self.s_lrer = lrer_funcs[0](self.s_optimizer)
        self.lrers = {'s_lrer': self.s_lrer}

        # create criterions
        # TODO: support more types of the consistency criterion
        self.cons_criterion = nn.MSELoss()
        self.s_criterion = criterion_funcs[0](self.args)
        self.criterions = {'s_criterion': self.s_criterion, 'cons_criterion': self.cons_criterion}

        # create the gaussian noiser
        self.gaussian_noiser = nn.DataParallel(GaussianNoiseLayer(self.args.gaussian_noise_std)).cuda()

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.s_model.train()
        self.t_model.train()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()
            
            # 's_inp', 't_inp' and 'gt' are tuples
            s_inp, t_inp, gt = self._batch_prehandle(inp, gt, True)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()

            # calculate the ramp-up coefficient of the consistency constraint
            cur_step = len(data_loader) * epoch + idx
            total_steps = len(data_loader) * self.args.cons_rampup_epochs
            cons_rampup_scale = func.sigmoid_rampup(cur_step, total_steps)

            self.s_optimizer.zero_grad()

            # forward the student model
            s_resulter, s_debugger = self.s_model.forward(s_inp)
            if not 'pred' in s_resulter.keys() or not 'activated_pred' in s_resulter.keys():
                self._pred_err()
            s_pred = tool.dict_value(s_resulter, 'pred')
            s_activated_pred = tool.dict_value(s_resulter, 'activated_pred')

            # calculate the supervised task constraint on the labeled data
            l_s_pred = func.split_tensor_tuple(s_pred, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_s_inp = func.split_tensor_tuple(s_inp, 0, lbs)

            # 'task_loss' is a tensor of 1-dim & n elements, where n == batch_size
            s_task_loss = self.s_criterion.forward(l_s_pred, l_gt, l_s_inp)
            s_task_loss = torch.mean(s_task_loss)
            self.meters.update('s_task_loss', s_task_loss.data)

            # forward the teacher model
            with torch.no_grad():
                t_resulter, t_debugger = self.t_model.forward(t_inp)
                if not 'pred' in t_resulter.keys():
                    self._pred_err()
                t_pred = tool.dict_value(t_resulter, 'pred')
                t_activated_pred = tool.dict_value(t_resulter, 'activated_pred')
            
                # calculate 't_task_loss' for recording
                l_t_pred = func.split_tensor_tuple(t_pred, 0, lbs)
                l_t_inp = func.split_tensor_tuple(t_inp, 0, lbs)
                t_task_loss = self.s_criterion.forward(l_t_pred, l_gt, l_t_inp)
                t_task_loss = torch.mean(t_task_loss)
                self.meters.update('t_task_loss', t_task_loss.data)

            # calculate the consistency constraint from the teacher model to the student model
            t_pseudo_gt = Variable(t_pred[0].detach().data, requires_grad=False)

            if self.args.cons_for_labeled:
                cons_loss = self.cons_criterion(s_pred[0], t_pseudo_gt)
            elif self.args.unlabeled_batch_size > 0:
                cons_loss = self.cons_criterion(s_pred[0][lbs:, ...], t_pseudo_gt[lbs:, ...])
            else:
                cons_loss = self.zero_tensor
            cons_loss = cons_rampup_scale * self.args.cons_scale * torch.mean(cons_loss)
            self.meters.update('cons_loss', cons_loss.data)

            # backward and update the student model
            loss = s_task_loss + cons_loss
            loss.backward()
            self.s_optimizer.step()

            # update the teacher model by EMA
            self._update_ema_variables(self.s_model, self.t_model, self.args.ema_decay, cur_step)

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  student-{3}\t=>\t'
                                's-task-loss: {meters[s_task_loss]:.6f}\t'
                                's-cons-loss: {meters[cons_loss]:.6f}\n'
                                '  teacher-{3}\t=>\t'
                                't-task-loss: {meters[t_task_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(s_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(s_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(t_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(t_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.s_lrer.step()

        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.s_lrer.step()

    def _validate(self, data_loader, epoch):
        self.meters.reset()

        self.s_model.eval()
        self.t_model.eval()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            s_inp, t_inp, gt = self._batch_prehandle(inp, gt, False)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()

            s_resulter, s_debugger = self.s_model.forward(s_inp)
            if not 'pred' in s_resulter.keys() or not 'activated_pred' in s_resulter.keys():
                self._pred_err()
            s_pred = tool.dict_value(s_resulter, 'pred')
            s_activated_pred = tool.dict_value(s_resulter, 'activated_pred')

            s_task_loss = self.s_criterion.forward(s_pred, gt, s_inp)
            s_task_loss = torch.mean(s_task_loss)
            self.meters.update('s_task_loss', s_task_loss.data)

            t_resulter, t_debugger = self.t_model.forward(t_inp)
            if not 'pred' in t_resulter.keys() or not 'activated_pred' in t_resulter.keys():
                self._pred_err()
            t_pred = tool.dict_value(t_resulter, 'pred')
            t_activated_pred = tool.dict_value(t_resulter, 'activated_pred')

            t_task_loss = self.s_criterion.forward(t_pred, gt, t_inp)
            t_task_loss = torch.mean(t_task_loss)
            self.meters.update('t_task_loss', t_task_loss.data)

            t_pseudo_gt = Variable(t_pred[0].detach().data, requires_grad=False)
            cons_loss = self.cons_criterion(s_pred[0], t_pseudo_gt)
            cons_loss = self.args.cons_scale * torch.mean(cons_loss)
            self.meters.update('cons_loss', cons_loss.data)

            self.task_func.metrics(s_activated_pred, gt, s_inp, self.meters, id_str='student')
            self.task_func.metrics(t_activated_pred, gt, t_inp, self.meters, id_str='teacher')

            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  student-{3}\t=>\t'
                                's-task-loss: {meters[s_task_loss]:.6f}\t'
                                's-cons-loss: {meters[cons_loss]:.6f}\n'
                                '  teacher-{3}\t=>\t'
                                't-task-loss: {meters[t_task_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, False, 
                                func.split_tensor_tuple(s_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(s_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(t_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(t_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))
    
        # metrics
        metrics_info = {'student': '', 'teacher': ''}
        for key in sorted(list(self.meters.keys())):
            if self.task_func.METRIC_STR in key:
                for id_str in metrics_info.keys():
                    if key.startswith(id_str):
                        metrics_info[id_str] += '{0}: {1:.6}\t'.format(key, self.meters[key])

        logger.log_info('Validation metrics:\n  student-metrics\t=>\t{0}\n  teacher-metrics\t=>\t{1}\n'
            .format(metrics_info['student'].replace('_', '-'), metrics_info['teacher'].replace('_', '-')))

    def _save_checkpoint(self, epoch):
        state = {
            'algorithm': self.NAME,
            'epoch': epoch + 1, 
            's_model': self.s_model.state_dict(),
            't_model': self.t_model.state_dict(),
            's_optimizer': self.s_optimizer.state_dict(),
            's_lrer': self.s_lrer.state_dict()
        }

        checkpoint = os.path.join(self.args.checkpoint_path, 'checkpoint_{0}.ckpt'.format(epoch))
        torch.save(state, checkpoint)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume)

        checkpoint_algorithm = tool.dict_value(checkpoint, 'algorithm', default='unknown')
        if checkpoint_algorithm != self.NAME:
            logger.log_err('Unmatched SSL algorithm format in checkpoint => required: {0} - given: {1}\n'
                           .format(self.NAME, checkpoint_algorithm))

        self.s_model.load_state_dict(checkpoint['s_model'])
        self.t_model.load_state_dict(checkpoint['t_model'])
        self.s_optimizer.load_state_dict(checkpoint['s_optimizer'])
        self.s_lrer.load_state_dict(checkpoint['s_lrer'])

        return checkpoint['epoch']

    # -------------------------------------------------------------------------------------------
    # Tool Functions for SSL_MT
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_train, 
                   s_inp, s_pred, t_inp, t_pred, gt):

        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        self.task_func.visualize(out_path, id_str='student', inp=s_inp, pred=s_pred, gt=gt)
        self.task_func.visualize(out_path, id_str='teacher', inp=t_inp, pred=t_pred, gt=gt)

    def _batch_prehandle(self, inp, gt, is_train):
        # add extra data augmentation process here if necessary

        # 'self.gaussian_noiser' will add the noise to the first input element
        s_inp_var, t_inp_var = [], []
        for idx, i in enumerate(inp):
            if is_train and idx == 0:
                s_inp_var.append(self.gaussian_noiser.forward(Variable(i).cuda())) 
                t_inp_var.append(self.gaussian_noiser.forward(Variable(i).cuda())) 
            else:
                s_inp_var.append(Variable(i).cuda()) 
                t_inp_var.append(Variable(i).cuda())
        s_inp = tuple(s_inp_var)
        t_inp = tuple(t_inp_var)
        
        gt_var = []
        for g in gt:
            gt_var.append(Variable(g).cuda())
        gt = tuple(gt_var)

        return s_inp, t_inp, gt

    def _update_ema_variables(self, s_model, t_model, ema_decay, cur_step):
        # update the teacher model by exponential moving average
        ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(t_model.parameters(), s_model.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def _algorithm_warn(self):
        logger.log_warn('This SSL_MT algorithm reproducts the SSL algorithm from paper:\n'
                        '  \'Mean Teachers are Better Role Models: Weight-Averaged Consistency Targets '
                        'Improve Semi-Supervised Deep Learning Results\'\n'
                        'The main differences between this implementation and the original paper are:\n'
                        '  (1) This is an implementation for pixel-wise vision tasks\n'
                        '  (2) The two-heads outputs trick is disable in this implementation\n'
                        '  (2) No extra perturbations between the inputs of the teacher and the student\n'
                        '      (The Gaussian noiser is provied, but it will degrade the performance)\n')

    def _inp_warn(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_MT\n'
                        'You try to train the task model with more than one (pred & gt) pairs\n'
                        'Please make sure that: \n'
                        '  (1) The prediction tuple has the same size as the ground truth tuple\n'
                        '  (2) The elements with the same index in the two tuples are corresponding\n'
                        '  (3) The first element of (pred & gt) will be used to calculate the consistency constraint\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_MT to\n' 
                        'calculate multiple consisteny constraints (for multiple predictions)\n')

    def _pred_err(self):
        logger.log_err('In SSL_MT, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                       '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                       '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                       'We need both of them since some losses include the activation functions,\n'
                       'e.g., the CrossEntropyLoss has contained SoftMax\n')
