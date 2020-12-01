
import os
import time
import random
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
from pixelssl.nn.module import patch_replication_callback

from . import ssl_base


""" Implementation of the CutMix algorithm for pixel-wise SSL

This method is proposed in the paper: 
    'Semi-Supervised Semantic Segmentation Needs Strong, Varied Perturbations'

This implementation follows the code from: https://github.com/Britefury/cutmix-semisup-seg 
Only the CutMix algorithm proposed by them is implemented.
Since this algorithm has a hyper-parameter 'cons-threshold' that requires prediction probability, it 
is only compatible with piexl-wise classification.
Although this method is based on the Mean Teacher (SSL_MT), we implement it separately due to some 
special operations within it.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)
    
    parser.add_argument('--cons-type', type=str, default='mse', choices=['mse'], help='sslcutmix - consistency constraint type [mse, ]')
    parser.add_argument('--cons-scale', type=float, default=-1, help='sslcutmix - consistency constraint coefficient')
    parser.add_argument('--cons-rampup-epochs', type=int, default=-1, help='sslcutmix - ramp-up epochs of conistency constraint')
    parser.add_argument('--cons-threshold', type=float, default=-1, help='sslcutmix - the confidence threshold for the consistency constraint')

    parser.add_argument('--ema-decay', type=float, default=0.99, help='sslcutmix - EMA coefficient of teacher model')
    parser.add_argument('--mask-prop-range', type=cmd.str2floatlist, default='(0.5, 0.5)', help='sslcutmix - mixing ratio range')


def ssl_cutmix(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_CUTMIX should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_CUTMIX, the key of element_dict should be \'model\',\n'
                       'but \'{0}\' is given\n'.format(model_dict.keys()))
    
    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLCUTMIX(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLCUTMIX(ssl_base._SSLBase):
    NAME = 'ssl_cutmix'
    SUPPORTED_TASK_TYPES = [CLASSIFICATION]

    def __init__(self, args):
        super(SSLCUTMIX, self).__init__(args)

        self.s_model, self.t_model = None, None
        self.s_optimizer = None
        self.s_lrer = None
        self.s_criterion = None

        # define the auxiliary modules required by CUTMIX
        self.mask_generator = None

        # check SSL arguments
        if self.args.unlabeled_batch_size > 0:
            if not self.args.unlabeled_batch_size > 2 or not self.args.unlabeled_batch_size % 2 == 0:
                logger.log_err('This implementation of SSL_CUTMIX requires the unlabeled batch size: \n'
                            '    1. larger than 2 \n'
                            '    2. is divisible by 2 \n')
            if self.args.cons_scale < 0:
                logger.log_err('The argument - cons_scale - is not set (or invalid)\n'
                                'Please set - cons_scale >= 0 - for training\n')
            if self.args.cons_rampup_epochs < 0:
                logger.log_err('The argument - cons_rampup_epochs - is not set (or invalid)\n'
                            'Please set - cons_rampup_epochs >= 0 - for training\n')
            if self.args.cons_threshold < 0 or self.args.cons_threshold > 1:
                logger.log_err('The argument - cons_threshold - is not set (or invalid)\n'
                            'Please set - 0 <= cons_threshold < 1 - for training\n')

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
        self.s_criterion = criterion_funcs[0](self.args)
        # TODO: support more types of the consistency criterion
        if self.args.cons_type == 'mse':
            self.cons_criterion = nn.MSELoss()
        self.criterions = {'s_criterion': self.s_criterion, 'cons_criterion': self.cons_criterion}

        # build the auxiliary modules required by CUTMIX
        # NOTE: this setting follow the original paper of CUTMIX
        self.mask_generator = BoxMaskGenerator(prop_range=self.args.mask_prop_range, 
            boxes_num=1, random_aspect_ratio=True, area_prop=True, within_bounds=True, invert=True)

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size
        ubs = self.args.unlabeled_batch_size

        self.s_model.train()
        self.t_model.train()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            # 'inp' and 'gt' are tuples
            inp, gt, mix_u_inp, mix_u_mask = self._batch_prehandle(inp, gt, True)
            if len(inp) > 1 and idx == 0:
                self._inp_warn()
            if len(gt) > 1 and idx == 0:
                self._gt_warn()

            # calculate the ramp-up coefficient of the consistency constraint
            cur_step = len(data_loader) * epoch + idx
            total_steps = len(data_loader) * self.args.cons_rampup_epochs
            cons_rampup_scale = func.sigmoid_rampup(cur_step, total_steps)

            self.s_optimizer.zero_grad()

            # -------------------------------------------------
            # For Labeled Samples
            # -------------------------------------------------
            l_inp = func.split_tensor_tuple(inp, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)

            # forward the labeled samples by the student model
            l_s_resulter, l_s_debugger = self.s_model.forward(l_inp)
            if not 'pred' in l_s_resulter.keys() or not 'activated_pred' in l_s_resulter.keys():
                self._pred_err()
            l_s_pred = tool.dict_value(l_s_resulter, 'pred')
            l_s_activated_pred = tool.dict_value(l_s_resulter, 'activated_pred')

            # calculate the supervised task loss on the labeled samples
            task_loss = self.s_criterion.forward(l_s_pred, l_gt, l_inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            # -------------------------------------------------
            # For Unlabeled Samples
            # -------------------------------------------------
            if self.args.unlabeled_batch_size > 0:
                u_inp = func.split_tensor_tuple(inp, lbs, self.args.batch_size)

                # forward the original samples by the teacher model
                with torch.no_grad():
                    u_t_resulter, u_t_debugger = self.t_model.forward(u_inp)
                if not 'pred' in u_t_resulter.keys() or not 'activated_pred' in u_t_resulter.keys():
                    self._pred_err()
                u_t_activated_pred = tool.dict_value(u_t_resulter, 'activated_pred')

                # mix the activated pred from the teacher model as the pseudo gt
                u_t_activated_pred_1 = func.split_tensor_tuple(u_t_activated_pred, 0, int(ubs / 2)) 
                u_t_activated_pred_2 = func.split_tensor_tuple(u_t_activated_pred, int(ubs / 2), ubs) 

                mix_u_t_activated_pred = []
                mix_u_t_confidence = []
                for up_1, up_2 in zip(u_t_activated_pred_1, u_t_activated_pred_2):
                    mp = mix_u_mask * up_1 + (1 - mix_u_mask) * up_2
                    mix_u_t_activated_pred.append(mp.detach())

                    # NOTE: here we just follow the official code of CutMix to calculate the confidence
                    #       but it is odd that all the samples use the same confidence (mean confidence)
                    u_t_confidence = (mp.max(dim=1)[0] > self.args.cons_threshold).float().mean()
                    mix_u_t_confidence.append(u_t_confidence.detach())

                mix_u_t_activated_pred = tuple(mix_u_t_activated_pred)

                # forward the mixed samples by the student model
                u_s_resulter, u_s_debugger = self.s_model.forward(mix_u_inp)
                if not 'pred' in u_s_resulter.keys() or not 'activated_pred' in u_s_resulter.keys():
                    self._pred_err()
                mix_u_s_activated_pred = tool.dict_value(u_s_resulter, 'activated_pred')

                # calculate the consistency constraint
                cons_loss = 0
                for msap, mtap, confidence in zip(mix_u_s_activated_pred, mix_u_t_activated_pred, mix_u_t_confidence):
                    cons_loss += torch.mean(self.cons_criterion(msap, mtap)) * confidence
                cons_loss = cons_rampup_scale * self.args.cons_scale * torch.mean(cons_loss)
                self.meters.update('cons_loss', cons_loss.data)
            else:
                cons_loss = 0
                self.meters.update('cons_loss', cons_loss)

            # backward and update the student model
            loss = task_loss + cons_loss
            loss.backward()
            self.s_optimizer.step()

            # update the teacher model by EMA
            self._update_ema_variables(self.s_model, self.t_model, self.args.ema_decay, cur_step)

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  student-{3}\t=>\t'
                                's-task-loss: {meters[task_loss]:.6f}\t'
                                's-cons-loss: {meters[cons_loss]:.6f}\n'
                                .format(epoch + 1, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(l_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(l_s_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(l_gt, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(mix_u_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(mix_u_s_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(mix_u_t_activated_pred, 0, 1, reduce_dim=True),
                                mix_u_mask[0])

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

            inp, gt, _, _ = self._batch_prehandle(inp, gt, False)
            if len(inp) > 1 and idx == 0:
                self._inp_warn()
            if len(gt) > 1 and idx == 0:
                self._gt_warn()

            s_resulter, s_debugger = self.s_model.forward(inp)
            if not 'pred' in s_resulter.keys() or not 'activated_pred' in s_resulter.keys():
                self._pred_err()
            s_pred = tool.dict_value(s_resulter, 'pred')
            s_activated_pred = tool.dict_value(s_resulter, 'activated_pred')

            s_task_loss = self.s_criterion.forward(s_pred, gt, inp)
            s_task_loss = torch.mean(s_task_loss)
            self.meters.update('s_task_loss', s_task_loss.data)   

            t_resulter, t_debugger = self.t_model.forward(inp)
            if not 'pred' in t_resulter.keys() or not 'activated_pred' in t_resulter.keys():
                self._pred_err()
            t_pred = tool.dict_value(t_resulter, 'pred')
            t_activated_pred = tool.dict_value(t_resulter, 'activated_pred')

            t_task_loss = self.s_criterion.forward(t_pred, gt, inp)
            t_task_loss = torch.mean(t_task_loss)
            self.meters.update('t_task_loss', t_task_loss.data)

            t_pseudo_gt = []
            for tap in t_activated_pred:
                t_pseudo_gt.append(tap.detach())
            t_pseudo_gt = tuple(t_pseudo_gt)
            
            cons_loss = 0
            for sap, tpg in zip(s_activated_pred, t_pseudo_gt):
                cons_loss += torch.mean(self.cons_criterion(sap, tpg))
            cons_loss = self.args.cons_scale * torch.mean(cons_loss)
            self.meters.update('cons_loss', cons_loss.data)

            self.task_func.metrics(s_activated_pred, gt, inp, self.meters, id_str='student')
            self.task_func.metrics(t_activated_pred, gt, inp, self.meters, id_str='teacher')

            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  student-{3}\t=>\t'
                                's-task-loss: {meters[s_task_loss]:.6f}\t'
                                's-cons-loss: {meters[cons_loss]:.6f}\n'
                                '  teacher-{3}\t=>\t'
                                't-task-loss: {meters[t_task_loss]:.6f}\n'
                                .format(epoch + 1, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, False, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(s_activated_pred, 0, 1, reduce_dim=True),
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
            'epoch': epoch,
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
    # Tool Functions for SSL_CUTMIX
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_train, l_inp, l_pred, l_gt, 
                   m_u_inp=None, m_u_s_pred=None, m_u_t_pred=None, mix_u_mask=None):
        
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))
        
        self.task_func.visualize(out_path, id_str='s-labeled', inp=l_inp, pred=l_pred, gt=l_gt)
        
        if m_u_inp is not None and m_u_s_pred is not None:
            self.task_func.visualize(out_path, id_str='s-mixed', inp=m_u_inp, pred=m_u_s_pred)
        if m_u_inp is not None and m_u_t_pred is not None:
            self.task_func.visualize(out_path, id_str='t-mixed', inp=m_u_inp, pred=m_u_t_pred)
        
        if mix_u_mask is not None:
            mix_u_mask = mix_u_mask[0].data.cpu().numpy()
            Image.fromarray((mix_u_mask * 255).astype('uint8'), mode='L').save(out_path + '_m-mask.png')

    def _batch_prehandle(self, inp, gt, is_train):
        # add extra data augmentation process here if necessary
        
        inp_var = []
        for i in inp:
            inp_var.append(Variable(i).cuda())
        inp = tuple(inp_var)
            
        gt_var = []
        for g in gt:
            gt_var.append(Variable(g).cuda())
        gt = tuple(gt_var)

        mix_u_inp = None
        mix_u_mask = None

        # -------------------------------------------------
        # Operations for CUTMIX
        # -------------------------------------------------
        if is_train:
            lbs = self.args.labeled_batch_size
            ubs = self.args.unlabeled_batch_size

            # check the shape of input and gt
            # NOTE: this implementation of CUTMIX supports multiple input and gt
            #       but all input and gt should have the same image size
            
            sample_shape = (inp[0].shape[2], inp[0].shape[3])
            for i in inp:
                if not tuple(i.shape[2:]) == sample_shape:
                    logger.log_err('This SSL_CUTMIX algorithm requires all inputs have the same shape \n')
            for g in gt:
                if not tuple(g.shape[2:]) == sample_shape:
                    logger.log_err('This SSL_CUTMIX algorithm requires all ground truths have the same shape \n')

            # generate the mask for mixing the unlabeled samples
            mix_u_mask = self.mask_generator.produce(int(ubs / 2), sample_shape)
            mix_u_mask = torch.tensor(mix_u_mask).cuda()

            # mix the unlabeled samples
            u_inp_1 = func.split_tensor_tuple(inp, lbs, int(lbs + ubs / 2))
            u_inp_2 = func.split_tensor_tuple(inp, int(lbs + ubs / 2), self.args.batch_size)

            mix_u_inp = []
            for ui_1, ui_2 in zip(u_inp_1, u_inp_2):
                mi = mix_u_mask * ui_1 + (1 - mix_u_mask) * ui_2
                mix_u_inp.append(mi)
            mix_u_inp = tuple(mix_u_inp)

        return inp, gt, mix_u_inp, mix_u_mask

    def _update_ema_variables(self, s_model, t_model, ema_decay, cur_step):
        # update the teacher model by exponential moving average
        ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(t_model.parameters(), s_model.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def _algorithm_warn(self):
        logger.log_warn('This SSL_CUTMIX algorith reproduces the SSL algorithm from the paper: \n'
                        '  \'Semi-Supervised Semantic Segmentation Needs Strong, Varied Perturbations\'\n'
                        'This implementation supports pixel-wise classification only due to the hyper-parameter: \n'
                        '  \'cons-threshold\' \n'
                        'The \'CutOut\' mode proposed by their paper is not implemented in this code\n')

    def _inp_warn(self):
        logger.log_warn('More than one input of the task model is given in SSL_CUTMIX\n'
                        'You try to train the task model with more than one input\n'
                        'All inputs are preprocessed with a CutMix operation using the same mask\n')

    def _gt_warn(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_CUTMIX\n'
                        'You try to train the task model with more than one ground truth\n'
                        'All ground truths are preprocessed with a CutMix operation using the same mask\n')

    def _pred_err(self):
        logger.log_err('In SSL_CUTMIX, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                       '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                       '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                       'We need both of them since some losses include the activation functions,\n'
                       'e.g., the CrossEntropyLoss has contained SoftMax\n')


# =======================================================
# Following code is adapted form the repository:
#   https://github.com/Britefury/cutmix-semisup-seg 
# =======================================================

class BoxMaskGenerator:
    def __init__(self, prop_range, boxes_num=1, random_aspect_ratio=True, 
                 area_prop=True, within_bounds=True, invert=False):

        self.prop_range = prop_range
        self.boxes_num = boxes_num
        self.random_aspect_ratio = random_aspect_ratio
        self.area_prop = area_prop
        self.within_bounds = within_bounds
        self.invert = invert

    def produce(self, mask_num, mask_shape):
        """ Generate box masks.

        Box masks can be generated quickly on the CPU so do it there.

        Arguments:
            mask_num (int): number of masks to generate
            mask_shape (tuple): shape of masks as a `(height, width)` tuple

        Returns:
            numpy.array: masks as a `(N, 1, H, W)` array
        """
        
        if self.area_prop:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = np.random.uniform(
                self.prop_range[0], self.prop_range[1], size=(mask_num, self.boxes_num))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(
                    np.random.uniform(low=0.0, high=1.0, size=(mask_num, self.boxes_num)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)

            fac = np.sqrt(1.0 / self.boxes_num)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0

        else:
            if self.random_aspect_ratio:
                y_props = np.random.uniform(
                    self.prop_range[0], self.prop_range[1], size=(mask_num, self.boxes_num))
                x_props = np.random.uniform(
                    self.prop_range[0], self.prop_range[1], size=(mask_num, self.boxes_num))
            else:
                x_props = y_props = np.random.uniform(
                    self.prop_range[0], self.prop_range[1], size=(mask_num, self.boxes_num))
            fac = np.sqrt(1.0 / self.boxes_num)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((mask_num, 1) + mask_shape)
        else:
            masks = np.ones((mask_num, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]

        masks = masks.astype(np.float32)
        return masks
