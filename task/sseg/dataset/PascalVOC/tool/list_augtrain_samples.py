import os


# read all samples
all_samples_dir = 'VOCdevkit/VOC2012/SegmentationClassAug'
all_samples = os.listdir(all_samples_dir)
all_samples = [s.split('.')[0] for s in all_samples]

# read val samples
val_samples_file = 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
with open(val_samples_file, 'r') as f:
    val_samples = f.read().splitlines()

# create augmented train samples list
train_aug_samples = []
for s in all_samples:
    if s not in val_samples:
        train_aug_samples.append(s)

out_path = 'VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt'
with open(out_path, 'w') as f:
    for s in train_aug_samples:
        f.write(s + '\n')
