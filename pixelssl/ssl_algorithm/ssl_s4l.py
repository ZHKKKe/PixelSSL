import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, tool
from pixelssl.nn import func
from pixelssl.nn.module import patch_replication_callback

from . import ssl_base


""" Implementation of pixel-wise self-supervised semi-supervised learning (S4L)
    
This method is proposed in paper: 
    'S4L: Self-Supervised Semi-Supervised Learning'

This implementation only supports the rotation-based self-supervised pretext task.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)
    parser.add_argument('--rotated-sup-scale', type=float, default=-1, help='task-supervised coefficient for rotated labeled data')
    parser.add_argument('--rotation-scale', type=float, default=-1, help='rotation-based self-supervised coefficient')


def ssl_s4l(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_S4L should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_S4L, the key of element_dict should be \'model\',\n'
                'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLS4L(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLS4L(ssl_base._SSLBase):
    NAME = 'ssl_s4l'
    SUPPORTED_TASK_TYPES = [REGRESSION, CLASSIFICATION]

    def __init__(self, args):
        super(SSLS4L, self).__init__(args)

        self.task_model = None
        self.rotation_classifier = None

        self.model = None
        self.optimizer = None
        self.lrer = None
        self.criterion = None

        # check SSL arguments
        if self.args.rotation_scale < 0:
            logger.log_err('The argument - rotation_scale - is not set (or invalid)\n'
                           'Please set - rotation_scale >= 0 - for training\n')
        if self.args.rotated_sup_scale < 0:
            logger.log_err('The argument - rotated_sup_scale - is not set (or invalid)\n'
                           'Please set - rotated_sup_scale >= 0 - for training\n')

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        self.task_model = func.create_model(model_funcs[0], 'task_model', args=self.args).module
        self.rotation_classifier = RotationClassifer(self.task_func.ssls4l_rc_in_channels())

        # wrap 'self.task_model' and 'self.rotation_classifier' into a single model
        self.model = WrappedS4LModel(self.args, self.task_model, self.rotation_classifier)
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

        # create criterions
        self.criterion = criterion_funcs[0](self.args)
        self.rotation_criterion = nn.CrossEntropyLoss()
        self.criterions = {'criterion': self.criterion, 'rotation_criterion': self.rotation_criterion}

        # the batch size is doubled in S4L since it creates an extra rotated sample for each sample
        self.args.batch_size *= 2
        self.args.labeled_batch_size *= 2
        self.args.unlabeled_batch_size *= 2

        logger.log_info('In SSL_S4L algorithm, batch size are doubled: \n'
                        '  Total labeled batch size: {1}\n'
                        '  Total unlabeled batch size: {2}\n'
                        .format(self.args.lr, self.args.labeled_batch_size, self.args.unlabeled_batch_size))

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        original_lbs = int(self.args.labeled_batch_size / 2)
        original_bs = int(self.args.batch_size / 2)

        self.model.train()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            # the rotated samples are generated in the 'self._batch_prehandle' function
            # both 'inp' and 'gt' are tuples
            # the last element in the tuple 'gt' is the ground truth of the rotation angle
            inp, gt = self._batch_prehandle(inp, gt, True)
            if len(gt) - 1 > 1 and idx == 0:
                self._inp_warn()

            self.optimizer.zero_grad()

            # forward the model
            resulter, debugger = self.model.forward(inp)
            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')
            pred_rotation = tool.dict_value(resulter, 'rotation')

            # calculate the supervised task constraint on the un-rotated labeled data
            l_pred = func.split_tensor_tuple(pred, 0, original_lbs)
            l_gt = func.split_tensor_tuple(gt, 0, original_lbs)
            l_inp = func.split_tensor_tuple(inp, 0, original_lbs)

            unrotated_task_loss = self.criterion.forward(l_pred, l_gt[:-1], l_inp)
            unrotated_task_loss = torch.mean(unrotated_task_loss)
            self.meters.update('unrotated_task_loss', unrotated_task_loss.data)
            
            # calculate the supervised task constraint on the rotated labeled data
            l_rotated_pred = func.split_tensor_tuple(pred, original_bs, original_bs + original_lbs)
            l_rotated_gt = func.split_tensor_tuple(gt, original_bs, original_bs + original_lbs)
            l_rotated_inp = func.split_tensor_tuple(inp, original_bs, original_bs + original_lbs)

            rotated_task_loss = self.criterion.forward(l_rotated_pred, l_rotated_gt[:-1], l_rotated_inp)
            rotated_task_loss = self.args.rotated_sup_scale * torch.mean(rotated_task_loss)
            self.meters.update('rotated_task_loss', rotated_task_loss.data)

            task_loss = unrotated_task_loss + rotated_task_loss
            
            # calculate the self-supervised rotation constraint
            rotation_loss = self.rotation_criterion.forward(pred_rotation, gt[-1])
            rotation_loss = self.args.rotation_scale * torch.mean(rotation_loss)
            self.meters.update('rotation_loss', rotation_loss.data)

            # backward and update the model
            loss = task_loss + rotation_loss
            loss.backward()
            self.optimizer.step()

            # calculate the accuracy of the rotation classifier
            _, angle_idx = pred_rotation.topk(1, 1, True, True)
            angle_idx = angle_idx.t()
            rotation_acc = angle_idx.eq(gt[-1].view(1, -1).expand_as(angle_idx))
            rotation_acc = rotation_acc.view(-1).float().sum(0, keepdim=True).mul_(100.0 / self.args.batch_size)
            self.meters.update('rotation_acc', rotation_acc.data[0])

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'unrotated-task-loss: {meters[unrotated_task_loss]:.6f}\t'
                                'rotated-task-loss: {meters[rotated_task_loss]:.6f}\n'
                                '  rotation-{3}\t=>\t'
                                'rotation-loss: {meters[rotation_loss]:.6f}\t'
                                'rotation-acc: {meters[rotation_acc]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))
                
            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt[:-1], 0, 1, reduce_dim=True))

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

            inp, gt = self._batch_prehandle(inp, gt, False)
            if len(gt) - 1 > 1 and idx == 0:
                self._inp_warn()

            resulter, debugger = self.model.forward(inp)
            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')
            pred_rotation = tool.dict_value(resulter, 'rotation')

            task_loss = self.criterion.forward(pred, gt[:-1], inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            rotation_loss = self.rotation_criterion.forward(pred_rotation, gt[-1])
            rotation_loss = self.args.rotation_scale * torch.mean(rotation_loss)
            self.meters.update('rotation_loss', rotation_loss.data)

            self.task_func.metrics(activated_pred, gt[:-1], inp, self.meters, id_str='task')

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\n'
                                '  rotation-{3}\t=>\t'
                                'rotation-loss: {meters[rotation_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt[:-1], 0, 1, reduce_dim=True))

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
            'epoch': epoch + 1,
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
            logger.log_err('Unmatched ssl algorithm format in checkpoint => required: {0} - given: {1}\n'
                           .format(self.NAME, checkpoint_algorithm))

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lrer.load_state_dict(checkpoint['lrer'])

        self.task_model = self.model.module.task_model
        self.rotation_classifier = self.model.module.rotation_classifier

        return checkpoint['epoch']

    # -------------------------------------------------------------------------------------------
    # Tool Functions for the SSL_S4L Framework
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_train, inp, pred, gt):
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))

        self.task_func.visualize(out_path, id_str='task', inp=inp, pred=pred, gt=gt)

    def _batch_prehandle(self, inp, gt, is_train):
        bs = inp[0].shape[0]
        rotation_angles = np.random.randint(low=1, high=4, size=inp[0].shape[0])

        inp_var = []
        for i in inp:
            i = i.cuda()

            if is_train:
                # create the extra rotated samples if 'is_train'
                assert i.shape[0] == bs
                rotated_i_shape = list(i.shape)
                rotated_i_shape[0] *= 2
                rotated_i = torch.zeros(rotated_i_shape).cuda()

                for sdx in range(0, bs):
                    rotated_i[sdx] = i[sdx]
                    rotated_i[bs + sdx] = self._rotate_tensor(i[sdx], angle_idx=rotation_angles[sdx])
                inp_var.append(Variable(rotated_i))

            else:
                inp_var.append(Variable(i))

        inp = tuple(inp_var)

        gt_var = []
        for g in gt:
            g = g.cuda()

            if is_train:
                # create the ground truth of the extra rotated samples if 'is_train'
                assert g.shape[0] == bs
                rotated_g_shape = list(g.shape)
                rotated_g_shape[0] *= 2
                rotated_g = torch.zeros(rotated_g_shape).cuda()

                for sdx in range(0, bs):
                    rotated_g[sdx] = g[sdx]
                    rotated_g[bs + sdx] = self._rotate_tensor(g[sdx], angle_idx=rotation_angles[sdx])
                gt_var.append(Variable(rotated_g))

            else:
                gt_var.append(Variable(g))
        
        # create the ground truth of the rotation classifier
        rotation_gt = torch.zeros(inp[0].shape[0]).cuda()
        for sdx in range(0, bs):
            rotation_gt[sdx] = 0
            if is_train:
                rotation_gt[bs + sdx] = float(rotation_angles[sdx])

        gt_var.append(Variable(rotation_gt.long()))
        gt = tuple(gt_var)

        return inp, gt

    def _rotate_tensor(self, tensor, angle_idx):
        if angle_idx == 1:
            tensor = tensor.transpose(1, 2).flip(2)
        elif angle_idx == 2:
            tensor = tensor.flip(2).flip(1)
        elif angle_idx == 3:
            tensor = tensor.transpose(1, 2).flip(1)
        
        return tensor

    def _inp_warn(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_S4L\n'
                        'You try to train the task model with more than one (pred & gt) pairs\n'
                        'Please make sure that:\n'
                        '  (1) The prediction tuple has the same size as the ground truth tuple\n'
                        '  (2) The elements with the same index in the two tuples are corresponding\n'
                        '  (3) All elements in the ground truth tuple should be 4-dim tensors since S4L\n'
                        '      will rotate them to match the rotated inputs\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_S4L that\n' 
                        'supports other formants (not 4-dim tensor) of the ground truth\n')

    def _algorithm_warn(self):
        logger.log_warn('This SSL_S4L algorithm reproducts the SSL algorithm from paper:\n'
                        '  \'S4L: Self-Supervised Semi-Supervised Learning\'\n'
                        'The main differences between this implementation and the original paper are:\n'
                        '  (1) This is an implementation for pixel-wise vision tasks\n'
                        '  (2) This implementation only supports the 4-angle (0, 90, 180, 270) rotation-based self-supervised pretext task\n')


class RotationClassifer(nn.Module):
    def __init__(self, in_channels):
        super(RotationClassifer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels * 2, 4)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, task_pred):
        x = self.leaky_relu(self.bn1(self.conv1(task_pred)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(task_pred.shape[0], -1)
        x = self.classifier(x)

        return x        


class WrappedS4LModel(nn.Module):
    def __init__(self, args, task_model, rotation_classifier):
        super(WrappedS4LModel, self).__init__()
        self.args = args
        self.task_model = task_model
        self.rotation_classifier = rotation_classifier

        self.param_groups = self.task_model.param_groups + \
            [{'params': self.rotation_classifier.parameters(), 'lr': self.args.lr}]
    
    def forward(self, inp):
        resulter, debugger = {}, {}

        t_resulter, t_debugger = self.task_model.forward(inp)

        if not 'pred' in t_resulter.keys() or not 'activated_pred' in t_resulter.keys():
            logger.log_err('In SSL_S4L, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                           '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                           '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                           'We need both of them since some losses include the activation functions,\n'
                           'e.g., the CrossEntropyLoss has contained SoftMax\n')

        if not 'ssls4l_rc_inp' in t_resulter.keys():
            logger.log_err('In SSL_S4L, the \'resulter\' dict returned by the task model should contain the key:\n'
                           '    \'ssls4l_rc_inp\'\t=>\tinputs of the rotation classifier (a 4-dim tensor)\n'
                           'It can be the feature map encoded by the task model or the output of the task model\n'
                           'Please add the key \'ssls4l_rc_inp\' in your task model\'s resulter\n')

        rc_inp = tool.dict_value(t_resulter, 'ssls4l_rc_inp')
        pred_rotation = self.rotation_classifier.forward(rc_inp)

        resulter['pred'] = tool.dict_value(t_resulter, 'pred')
        resulter['activated_pred'] = tool.dict_value(t_resulter, 'activated_pred')
        resulter['rotation'] = pred_rotation

        return resulter, debugger
