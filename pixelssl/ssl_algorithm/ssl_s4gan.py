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


"""
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)

    parser.add_argument('--discriminator-lr', type=float, default=1e-4, help='ssls4gan - the initial learning rate of the FC discriminator')
    parser.add_argument('--discriminator-scale', type=float, default=1.0, help='ssls4gan - coefficient of the S4GAN discriminator constraint')
    parser.add_argument('--st-scale', type=float, default=-1.0, help='ssls4gan - coefficient of the Lst in the paper')
    parser.add_argument('--fm-scale', type=float, default=-1.0, help='ssls4gan - coefficient of the Lfm in the paper')

def ssl_s4gan(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_S4GAN should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_S4GAN, the key of element_dict should be \'model\',\n'
                       'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLS4GAN(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm


def SSLS4GAN(ssl_base._SSLBase):
    NAME = 'ssl_s4gan'
    SUPPORTED_TASK_TYPES = [CLASSIFICATION]

    def __init__(self, args):
        super(SSLS4GAN, self).__init__(args)

        # define the task model and the FC discriminator
        self.model, self.d_model = None, None
        self.optimizer, self.d_optimizer = None, None
        self.lrer, self.d_lrer = None, None
        self.criterion, self.d_criterion = None, None

        # prepare the arguments for multiple GPUs
        self.args.discriminator_lr *= self.args.gpus

        # check SSL arguments
        if self.args.unlabeled_batch_size > 0:
            if self.args.st_scale < 0:
                logger.log_err('The argument - st_scale - is not set (or invalid)\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - st_scale >= 0 - for calculating the' 
                               'semi-supervised loss on the unlabeled data\n')
            elif self.args.fm_scale < 0:
                logger.log_err('The argument - fm_scale - is not set (or invalid)\n'
                               'You set argument - unlabeled_batch_size - larger than 0\n'
                               'Please set - fm_scale >= 0 - for calculating the' 
                               'semi-supervised loss on the unlabeled data\n')

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self.task_func = task_func

        # create models
        self.model = func.create_model(model_funcs[0], 'model', args=self.args)
        self.d_model = func.create_model(S4GANDiscriminator, 'd_model', in_channels=self.task_func.ssls4gan_d_in_channels())
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
                                   power=0.9, last_epoch=-1)
        self.lrers = {'lrer': self.lrer, 'd_lrer': self.d_lrer}

        # create criterions
        self.criterion = criterion_funcs[0](self.args)
        self.d_criterion = S4GANDiscriminatorCriterion()
        self.criterions = {'criterion': self.criterion, 'd_criterion': self.d_criterion}

        self._algorithm_warn()

    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.model.train()
        self.d_model.train()

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

            # forward the S4GAN discriminator
            d_resulter, d_debugger = self.d_model.forward(activated_pred[0], inp)
            d_confidence = tool.dict_value(d_resulter, 'd_confidence')
            d_features = tool.dict_value(d_resulter, 'd_features')

            # calculate the supervised task constraint on the labeled data
            l_pred = func.split_tensor_tuple(pred, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_inp = func.split_tensor_tuple(inp, 0, lbs)
            
            task_loss = self.criterion.forward(l_pred, l_gt, l_inp)
            task_loss = torch.mean(task_loss)
            self.meters.update('task_loss', task_loss.data)

            # calculate the semi-supervised constraints
            # TODO
            fm_loss = 0
            st_loss = 0

            loss = task_loss + fm_loss + st_loss
            loss.backward()
            self.optimizer.step()

            # -----------------------------------------------------------------------------
            # step-2: train the S4GAN discriminator
            # -----------------------------------------------------------------------------
            self.d_optimizer.zero_grad()

            # forward the task prediction (fake)
            fake_pred = activated_pred[0].detach()
            d_resulter, d_debugger = self.d_model.forward(fake_pred, inp)
            fake_d_confidence = tool.dict_value(d_resulter, 'd_confidence')
            fake_d_confidence, fake_d_confidence_gt = \
                self.task_func.ssls4gan_preprocess_d_criterion(fake_d_confidence, gt[0], False)
            
            fake_d_loss = self.d_criterion.forward(fake_d_confidence, fake_d_confidence_gt)
            fake_d_loss = self.args.discriminator_scale * torch.mean(fake_d_loss)
            self.meters.update('fake_d_loss', fake_d_loss.data)

            # forward the ground truth (real)
            real_gt = self.task_func.ssls4gan_convert_task_gt_to_d_input(l_gt[0])
            d_resulter, d_debugger = self.d_model.forward(real_gt, l_inp)
            real_d_confidence = tool.dict_value(d_resulter, 'd_confidence')
            real_d_confidence, real_d_confidence_gt = \
                self.task_func.ssls4gan_preprocess_d_criterion(real_d_confidence, l_gt[0], True)

            real_d_loss = self.d_criterion(real_d_confidence, real_d_confidence_gt)
            real_d_loss = self.args.discriminator_scale * torch.mean(real_d_loss)
            self.meters.update('real_d_loss', real_d_loss.data)

            # backward and update the FC discriminator
            d_loss = (fake_d_loss + real_d_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()

            # logging
            # TODO



    def _validate(self, data_loader, epoch):
        pass

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
        logger.log_warn('')

    def _inp_warn(self):
        logger.log_warn('')

    def _pred_err(self):
        logger.log_err('')



""" The s4GAN Discriminator proposed in paper:
    'Semi-supevised Semantic Segmentation with High- and Low-level Consistency'
    arxiv: https://arxiv.org/pdf/1908.05724
    github: https://github.com/sud0301/semisup-semseg

The following code is adapted from the official implementation of the above paper.
The s4GAN discriminator takes the prediction of the task model as input. 
It will output a confident map and a laten feature vector.
"""


class S4GANDiscriminator(nn.Module):
    ndf = 64

    def __init__(self, in_channels):
        super(s4GAN_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d((20, 20))
        self.fc = nn.Linear(ndf * 8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, inp):
       resulter, debugger = {}, {}

        cat_inp = torch.cat(inp, dim=1)
        x = torch.cat((x, cat_inp), dim=1)

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))

        resulter['d_confidence'] = out
        resulter['d_features'] = conv4_maps
        
        return resulter, debugger


class S4GANDiscriminatorCriterion(nn.Module):
    def __init__(self):
        super(S4GANDiscriminator, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, gt):
        return torch.mean(self.criterion(pred, gt))    
