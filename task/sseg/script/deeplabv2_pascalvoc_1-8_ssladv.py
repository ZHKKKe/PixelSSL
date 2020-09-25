import os
import sys
import collections

try:
    import pixelssl
    pixelssl.log_info('Use installed pixelssl=={0}\n'.format(pixelssl.__version__))
except ImportError:
    sys.path.append('../..')
    import pixelssl
    pixelssl.log_warn('No installed pixelssl, the latest code of PixelSSL will be used.\n')

import proxy


config = collections.OrderedDict(
    [
        ('exp_id', os.path.basename(__file__).split(".")[0]),
        
        # arguments - SSL algorithm
        ('ssl_algorithm', pixelssl.SSL_ADV),

        ('adv_for_labeled', True),
        ('labeled_adv_scale', 0.01),
        ('unlabeled_adv_scale', 0.001),

        ('discriminator_lr', 0.0001),
        ('unlabeled_for_discriminator', True),

        # arguments - exp
        ('resume', 'pretrained/deeplabv2_pascalvoc_1-8_ssladv.ckpt'),
        ('validation', True),
        
        ('out_path', 'result'),
        
        ('visualize', False),
        ('debug', False),

        ('val_freq', 1),
        ('log_freq', 50),
        ('visual_freq', 50),
        ('checkpoint_freq', 10),

        # arguments - dataset / dataloader
        ('trainset', {'pascal_voc_aug': ['dataset/PascalVOC/VOCdevkit/VOC2012']}),
        ('valset', {'pascal_voc_aug': ['dataset/PascalVOC/VOCdevkit/VOC2012']}),
        ('num_workers', 2),
        ('im_size', 321),

        ('sublabeled_path', 'dataset/PascalVOC/sublabeled_prefix/1-8/0.txt'),
        ('ignore_unlabeled', False),

        # arguments - task specific components
        ('models', {'model': 'deeplabv2'}),
        ('optimizers', {'model': 'sgd'}),
        ('lrers', {'model': 'polynomiallr'}),
        ('criterions', {'model': 'sseg_criterion'}),

        # arguments - task specific optimizer / lr scheduler
        ('lr', 0.00025),
        ('momentum', 0.9),
        ('weight_decay', 0.0005),

        ('milestones', [10, 20]),
        ('gamma', 0.1),

        # arguments - task special model
        ('output_stride', 16),
        ('backbone', 'resnet101-coco'),

        # arguments - task special data
        ('val_rescaling', True),
        ('train_base_size', 400),

        # arguments - training details
        ('epochs', 20),
        ('batch_size', 4),
        ('unlabeled_batch_size', 2), 

    ]
)


if __name__ == '__main__':
    pixelssl.run_script(config, proxy, proxy.SemanticSegmentationProxy)
