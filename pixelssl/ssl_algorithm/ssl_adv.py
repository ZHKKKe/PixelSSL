import os
import time
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
from pixelssl.nn.module import patch_replication_callback
from . import ssl_base


""" Implementation of pixel-wise adversarial-based semi-supervised learning (AdvSSL)
    
This method is proposed in the paper: 
    'Adversarial Learning for Semi-supervised Semantic Segmentation'

This implementation does not support the constraint named \'L_semi\' in the original paper 
because it can only be used for pixel-wise classification.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)
    parser.add_argument('--adv-for-labeled', type=cmd.str2bool, default=False, help='ssladv - calculate the adversarial constraint on the labeled data if True')
    parser.add_argument('--labeled-adv-scale', type=float, default=-1, help='ssladv - adversarial constraint coefficient of labeled data')
    parser.add_argument('--unlabeled-adv-scale', type=float, default=-1, help='ssladv - adversarial constraint coefficient of unlabeled data')
    
    parser.add_argument('--discriminator-lr', type=float, default=1e-4, help='ssladv - the initial learning rate of the FC discriminator')
    parser.add_argument('--discriminator-power', type=float, default=0.9, help='ssladv - power value of the PolynomialLR strategy used by the FC discriminator')
    parser.add_argument('--unlabeled-for-discriminator', type=cmd.str2bool, default=False, help='ssladv - train FC discriminator with unlabeled data if True')
    parser.add_argument('--discriminator-scale', type=float, default=1.0, help='ssladv - coefficient of the FC discriminator constraint')


def ssl_adv(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_ADV should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_ADV, the key of element_dict should be \'model\',\n'
                       'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLADV(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


class SSLADV(ssl_base._SSLBase):
    NAME = 'ssl_adv'
    SUPPORTED_TASK_TYPES = [REGRESSION, CLASSIFICATION]

    def __init__(self, args):
        super(SSLADV, self).__init__(args)

        # define the task model and the FC discriminator
        self.model, self.d_model = None, None
        self.optimizer, self.d_optimizer = None, None
        self.lrer, self.d_lrer = None, None
        self.criterion, self.d_criterion = None, None

        # prepare the arguments for multiple GPUs
        self.args.discriminator_lr *= self.args.gpus

        # check SSL arguments
        if self.args.adv_for_labeled:
            if self.args.labeled_adv_scale < 0:
                logger.log_err('The argument - labeled_adv_scale - is not set (or invalid)\n'
                                'You set argument - adv_for_labeled - to True\n'
                                'Please set - labeled_adv_scale >= 0 - for calculating the'
                                'adversarial loss on the labeled data\n')
        if self.args.unlabeled_batch_size > 0:
            if self.args.unlabeled_adv_scale < 0:
                logger.log_err('The argument - unlabeled_adv_scale - is not set (or invalid)\n'
                                'You set argument - unlabeled_batch_size - larger than 0\n'
                                'Please set - unlabeled_adv_scale >= 0 - for calculating the' 
                                'adversarial loss on the unlabeled data\n')

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        self.model = func.create_model(model_funcs[0], 'model', args=self.args)
        self.d_model = func.create_model(FCDiscriminator, 'd_model', in_channels=self.task_func.ssladv_fcd_in_channels())
        # call 'patch_replication_callback' to enable the `sync_batchnorm` layer
        patch_replication_callback(self.model)
        patch_replication_callback(self.d_model)
        self.models = {'model': self.model, 'd_model': self.d_model}

        # create optimizers
        self.optimizer = optimizer_funcs[0](self.model.module.param_groups)
        self.d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.d_model.parameters()), 
                                      lr=self.args.discriminator_lr, betas=(0.9, 0.99))
        self.optimizers = {'optimizer': self.optimizer, 'd_optimizer': self.d_optimizer}

        # create lrers
        self.lrer = lrer_funcs[0](self.optimizer)
        self.d_lrer = PolynomialLR(self.d_optimizer, self.args.epochs, self.args.iters_per_epoch, 
                                   power=self.args.discriminator_power, last_epoch=-1)
        self.lrers = {'lrer': self.lrer, 'd_lrer': self.d_lrer}

        # create criterions
        self.criterion = criterion_funcs[0](self.args)
        self.d_criterion = FCDiscriminatorCriterion()
        self.criterions = {'criterion': self.criterion, 'd_criterion': self.d_criterion}

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.model.train()
        self.d_model.train()

        # both 'inp' and 'gt' are tuples
        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            inp, gt = self._batch_prehandle(inp, gt)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()

            # -----------------------------------------------------------------------------
            # step-1: train the task model
            # -----------------------------------------------------------------------------
            self.optimizer.zero_grad()

            # forward the task model
            resulter, debugger = self.model.forward(inp)
            if not 'pred' in resulter.keys() or not 'activated_pred' in resulter.keys():
                self._pred_err()

            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')

            # forward the FC discriminator
            # 'confidence_map' is a tensor
            d_resulter, d_debugger = self.d_model.forward(activated_pred[0])
            confidence_map = tool.dict_value(d_resulter, 'confidence')

            # calculate the supervised task constraint on the labeled data
            l_pred = func.split_tensor_tuple(pred, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_inp = func.split_tensor_tuple(inp, 0, lbs)

            # 'task_loss' is a tensor of 1-dim & n elements, where n == batch_size
            task_loss = self.criterion.forward(l_pred, l_gt, l_inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            # calculate the adversarial constraint
            # calculate the adversarial constraint for the labeled data
            if self.args.adv_for_labeled:
                l_confidence_map = confidence_map[:lbs, ...]

                # preprocess prediction and ground truch for the adversarial constraint
                l_adv_confidence_map, l_adv_confidence_gt = \
                    self.task_func.ssladv_preprocess_fcd_criterion(l_confidence_map, l_gt[0], True)
                l_adv_loss = self.d_criterion(l_adv_confidence_map, l_adv_confidence_gt)
                labeled_adv_loss = self.args.labeled_adv_scale * torch.mean(l_adv_loss)
                self.meters.update('labeled_adv_loss', labeled_adv_loss.data)
            else:
                labeled_adv_loss = 0
                self.meters.update('labeled_adv_loss', labeled_adv_loss)
            
            # calculate the adversarial constraint for the unlabeled data
            if self.args.unlabeled_batch_size > 0:
                u_confidence_map = confidence_map[lbs:self.args.batch_size, ...]

                # preprocess prediction and ground truch for the adversarial constraint
                u_adv_confidence_map, u_adv_confidence_gt = \
                    self.task_func.ssladv_preprocess_fcd_criterion(u_confidence_map, None, True)
                u_adv_loss = self.d_criterion(u_adv_confidence_map, u_adv_confidence_gt)
                unlabeled_adv_loss = self.args.unlabeled_adv_scale * torch.mean(u_adv_loss)
                self.meters.update('unlabeled_adv_loss', unlabeled_adv_loss.data)
            else:
                unlabeled_adv_loss = 0
                self.meters.update('unlabeled_adv_loss', unlabeled_adv_loss)

            adv_loss = labeled_adv_loss + unlabeled_adv_loss

            # backward and update the task model
            loss = task_loss + adv_loss
            loss.backward()
            self.optimizer.step()

            # -----------------------------------------------------------------------------
            # step-2: train the FC discriminator
            # -----------------------------------------------------------------------------
            self.d_optimizer.zero_grad()

            # forward the task prediction (fake)
            if self.args.unlabeled_for_discriminator:
                fake_pred = activated_pred[0].detach()
            else:
                fake_pred = activated_pred[0][:lbs, ...].detach()

            d_resulter, d_debugger = self.d_model.forward(fake_pred)
            fake_confidence_map = tool.dict_value(d_resulter, 'confidence')
            
            l_fake_confidence_map = fake_confidence_map[:lbs, ...]
            l_fake_confidence_map, l_fake_confidence_gt = \
                self.task_func.ssladv_preprocess_fcd_criterion(l_fake_confidence_map, l_gt[0], False) 
                
            if self.args.unlabeled_for_discriminator and self.args.unlabeled_batch_size != 0:
                u_fake_confidence_map = fake_confidence_map[lbs:self.args.batch_size, ...]
                u_fake_confidence_map, u_fake_confidence_gt = \
                    self.task_func.ssladv_preprocess_fcd_criterion(u_fake_confidence_map, None, False)

                fake_confidence_map = torch.cat((l_fake_confidence_map, u_fake_confidence_map), dim=0)
                fake_confidence_gt = torch.cat((l_fake_confidence_gt, u_fake_confidence_gt), dim=0)
            
            else:
                fake_confidence_map, fake_confidence_gt = l_fake_confidence_map, l_fake_confidence_gt

            fake_d_loss = self.d_criterion.forward(fake_confidence_map, fake_confidence_gt)
            fake_d_loss = self.args.discriminator_scale * torch.mean(fake_d_loss)
            self.meters.update('fake_d_loss', fake_d_loss.data)

            # forward the ground truth (real)
            # convert the format of ground truch 
            real_gt = self.task_func.ssladv_convert_task_gt_to_fcd_input(l_gt[0])
            d_resulter, d_debugger = self.d_model.forward(real_gt)
            real_confidence_map = tool.dict_value(d_resulter, 'confidence')

            real_confidence_map, real_confidence_gt = \
                self.task_func.ssladv_preprocess_fcd_criterion(real_confidence_map, l_gt[0], True)

            real_d_loss = self.d_criterion(real_confidence_map, real_confidence_gt)
            real_d_loss = self.args.discriminator_scale * torch.mean(real_d_loss)
            self.meters.update('real_d_loss', real_d_loss.data)            

            # backward and update the FC discriminator
            d_loss = (fake_d_loss + real_d_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\t'
                                'labeled-adv-loss: {meters[labeled_adv_loss]:.6f}\t'
                                'unlabeled-adv-loss: {meters[unlabeled_adv_loss]:.6f}\n'
                                '  fc-discriminator\t=>\t'
                                'fake-d-loss: {meters[fake_d_loss]:.6f}\t'
                                'real-d-loss: {meters[real_d_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))
                    
            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                u_inp_sample, u_pred_sample, u_cmap_sample = None, None, None
                if self.args.unlabeled_batch_size > 0:
                    u_inp_sample = func.split_tensor_tuple(inp, lbs, lbs+1, reduce_dim=True)
                    u_pred_sample = func.split_tensor_tuple(activated_pred, lbs, lbs+1, reduce_dim=True)
                    u_cmap_sample = torch.sigmoid(fake_confidence_map[lbs])

                self._visualize(epoch, idx, True, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True),
                                torch.sigmoid(confidence_map[0]), u_inp_sample, u_pred_sample, u_cmap_sample)

            # the FC discriminator uses polynomiallr [ITER_LRERS]
            self.d_lrer.step()
            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.lrer.step()
        
        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.lrer.step()

    def _validate(self, data_loader, epoch):
        self.meters.reset()
        
        self.model.eval()
        self.d_model.eval()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            inp, gt = self._batch_prehandle(inp, gt)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()
            
            resulter, debugger = self.model.forward(inp)
            if not 'pred' in resulter.keys() or not 'activated_pred' in resulter.keys():
                self._pred_err()
            
            pred = tool.dict_value(resulter, 'pred')
            activated_pred = tool.dict_value(resulter, 'activated_pred')

            task_loss = self.criterion.forward(pred, gt, inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            d_resulter, d_debugger = self.d_model.forward(activated_pred[0])
            unhandled_fake_confidence_map = tool.dict_value(d_resulter, 'confidence')
            fake_confidence_map, fake_confidence_gt = \
                self.task_func.ssladv_preprocess_fcd_criterion(unhandled_fake_confidence_map, gt[0], False)
            
            fake_d_loss = self.d_criterion.forward(fake_confidence_map, fake_confidence_gt)
            fake_d_loss = self.args.discriminator_scale * torch.mean(fake_d_loss)
            self.meters.update('fake_d_loss', fake_d_loss.data)

            real_gt = self.task_func.ssladv_convert_task_gt_to_fcd_input(gt[0])
            d_resulter, d_debugger = self.d_model.forward(real_gt)
            unhandled_real_confidence_map = tool.dict_value(d_resulter, 'confidence')
            real_confidence_map, real_confidence_gt = \
                self.task_func.ssladv_preprocess_fcd_criterion(unhandled_real_confidence_map, gt[0], True)
            
            real_d_loss = self.d_criterion.forward(real_confidence_map, real_confidence_gt)
            real_d_loss = self.args.discriminator_scale * torch.mean(real_d_loss)
            self.meters.update('real_d_loss', real_d_loss.data)

            self.task_func.metrics(activated_pred, gt, inp, self.meters, id_str='task')
            
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  task-{3}\t=>\t'
                                'task-loss: {meters[task_loss]:.6f}\t'
                                '  fc-discriminator\t=>\t'
                                'fake-d-loss: {meters[fake_d_loss]:.6f}\t'
                                'real-d-loss: {meters[real_d_loss]:.6f}\n'
                                .format(epoch, idx, len(data_loader), self.args.task, meters=self.meters))

            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, False, 
                                func.split_tensor_tuple(inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True),
                                torch.sigmoid(unhandled_fake_confidence_map[0]))

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
            'd_model': self.d_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'lrer': self.lrer.state_dict(),
            'd_lrer': self.d_lrer.state_dict()
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
        self.d_model.load_state_dict(checkpoint['d_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.lrer.load_state_dict(checkpoint['lrer'])
        self.d_lrer.load_state_dict(checkpoint['d_lrer'])

        return checkpoint['epoch']
        
    # -------------------------------------------------------------------------------------------
    # Tool Functions for SSL_ADV
    # -------------------------------------------------------------------------------------------

    def _visualize(self, epoch, idx, is_train, 
                   l_inp, l_pred, l_gt, l_cmap,
                   u_inp=None, u_pred=None, u_cmap=None):
        # 'cmap' is the output of the FC discriminator
        
        visualize_path = self.args.visual_train_path if is_train else self.args.visual_val_path
        out_path = os.path.join(visualize_path, '{0}_{1}'.format(epoch, idx))
        
        self.task_func.visualize(out_path, id_str='labeled', inp=l_inp, pred=l_pred, gt=l_gt)
        l_cmap = l_cmap[0].data.cpu().numpy()
        Image.fromarray((l_cmap * 255).astype('uint8'), mode='L').save(out_path + '_labeled-cmap.png')

        if u_inp is not None and u_pred and not None and u_cmap is not None:
            self.task_func.visualize(out_path, id_str='unlabeled', inp=u_inp, pred=u_pred, gt=None)
            u_cmap = u_cmap[0].data.cpu().numpy()
            Image.fromarray((u_cmap * 255).astype('uint8'), mode='L').save(out_path + '_unlabeled-cmap.png')

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

    def _algorithm_warn(self):
        logger.log_warn('This SSL_ADV algorithm reproducts the SSL algorithm from paper:\n'
                        '  \'Adversarial Learning for Semi-supervised Semantic Segmentation\'\n'
                        'The main differences between this implementation and the original paper are:\n'
                        '  (1) This implementation does not support the constraint named \'L_semi\' in the\n'
                        '      original paper since it can only be used for pixel-wise classification\n'
                        '\nThe semi-supervised constraint in this implementation refer to the constraint\n' 
                        'named \'L_adv\' in the original paper\n'
                        '\nSame as the original paper, the FC discriminator is trained by the Adam optimizer\n'
                        'with the PolynomialLR scheduler\n')

    def _inp_warn(self):
        logger.log_warn('More than one ground truth of the task model is given in SSL_ADV\n'
                        'You try to train the task model with more than one (pred & gt) pairs\n'
                        'Please make sure that:\n'
                        '  (1) The prediction tuple has the same size as the ground truth tuple\n'
                        '  (2) The elements with the same index in the two tuples are corresponding\n'
                        '  (3) The first element of (pred & gt) will be used to train the FC discriminator\n'
                        '      and calculate the SSL constraints\n'
                        'Please implement a new SSL algorithm if you want a variant of SSL_ADV with\n' 
                        'multiple FC discriminators (for multiple predictions)\n')

    def _pred_err(self):
        logger.log_err('In SSL_ADV, the \'resulter\' dict returned by the task model should contain the following keys:\n'
                       '   (1) \'pred\'\t=>\tunactivated task predictions\n'
                       '   (2) \'activated_pred\'\t=>\tactivated task predictions\n'
                       'We need both of them since some losses include the activation functions,\n'
                       'e.g., the CrossEntropyLoss has contained SoftMax\n')


""" The FC Discriminator proposed in paper:
    'Adversarial Learning for Semi-supervised Semantic Segmentation'
    arxiv: https://arxiv.org/abs/1802.07934
    github: https://github.com/hfslyc/AdvSemiSeg

The following code is adapted from the official implementation of the above paper.
The FC discriminator takes the prediction of the task model as input. 
It will output a confident map, which is activated in the confident pixels of prediction.
"""


class FCDiscriminator(nn.Module):
    ndf = 64    # base channal size
	
    def __init__(self, in_channels):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, self.ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, task_pred):
        resulter, debugger = {}, {}

        x = self.leaky_relu(self.conv1(task_pred))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        x = F.interpolate(x, size=(task_pred.shape[2], task_pred.shape[3]), mode='bilinear', align_corners=True)

        # 'x' is not activated here since it will be activated in 'FCDiscriminatorCriterion'
        assert x.shape[2:] == task_pred.shape[2:]
        resulter['confidence'] = x
        return resulter, debugger


class FCDiscriminatorCriterion(nn.Module):
    def __init__(self):
        super(FCDiscriminatorCriterion, self).__init__()

    def forward(self, pred, gt):
        # pred will be activated by F.binary_cross_entropy_with_logits (logger.log_softmax)
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')        
        return torch.mean(loss, dim=(1, 2, 3))
