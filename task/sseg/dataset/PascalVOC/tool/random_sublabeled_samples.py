import os
import numpy as np

# set ratio of the labeled samples
numerator = 1
denominator = 8
labeled_ratio = numerator / denominator

# read the samples list
samples_list = 'VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt'
if not os.path.exists(samples_list):
    print('The PascalVOC 2012 dataset is not prepared.\n'
          'Please run \'sh prepare.sh\' to prepare it.')

with open(samples_list, 'r') as f:
    samples = f.read().splitlines()
np.random.shuffle(samples)

# get the sublabeled list
labeled_num = int(len(samples) * labeled_ratio + 1)
labeled_list = samples[:labeled_num]

# create the output path and save the sublabeled list
out_path = 'sublabeled_prefix/{0}-{1}'.format(numerator, denominator)
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_file = os.path.join(out_path, '{0}.txt'.format(len(os.listdir(out_path))))
with open(out_file, 'w') as f:
    for sample in labeled_list:
        f.write(sample + '\n')
