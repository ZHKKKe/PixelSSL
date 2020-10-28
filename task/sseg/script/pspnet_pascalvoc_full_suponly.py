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
        ('ssl_algorithm', pixelssl.SSL_NULL),

        # arguments - exp
        ('resume', 'pretrained/pspnet_pascalvoc_full_suponly.ckpt'),
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
        ('im_size', 513),

        # arguments - task specific components
        ('models', {'model': 'pspnet'}),
        ('optimizers', {'model': 'sgd'}),
        ('lrers', {'model': 'polynomiallr'}),
        ('criterions', {'model': 'sseg_criterion'}),
        
        # arguments - task specific optimizer / lr scheduler
        ('lr', 0.00025),
        ('momentum', 0.9),
        ('weight_decay', 0.0005),

        # arguments - task special model
        ('output_stride', 16),
        ('backbone', 'resnet101-coco'),

        # arguments - task special data
        ('val_rescaling', False),
        ('train_base_size', 513),

        # arguments - training details
        ('epochs', 80),
        ('batch_size', 4),
        ('unlabeled_batch_size', 0), 

    ]
)


if __name__ == '__main__':
    pixelssl.run_script(config, proxy, proxy.SemanticSegmentationProxy)
