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
        # arguments - SSL algorithm
        ('exp_id', os.path.basename(__file__).split(".")[0]),
        ('ssl_algorithm', pixelssl.SSL_DCT),

        ('cons_scale', 100.0),
        ('cons_rampup_epochs', 3),

        # arguments - exp
        # ('resume', 'pretrained/deeplabv2_pascalvoc_1-8_sslgct.ckpt'),
        # ('validation', True),
        
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
        ('criterions', {'model': 'deeplab_criterion'}),

        # arguments - task specific optimizer / lr scheduler
        ('lr', 0.00025),
        ('momentum', 0.9),
        ('weight_decay', 0.0005),

        # arguments - task special model
        ('output_stride', 16),
        ('backbone', 'resnet'),

        # arguments - training details
        ('epochs', 20),
        ('batch_size', 4),
        ('unlabeled_batch_size', 2), 

    ]
)


if __name__ == '__main__':
    pixelssl.run_script(config, proxy, proxy.SemanticSegmentationProxy)
