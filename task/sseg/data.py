""" This file is adapted from the repository: https://github.com/jfzhang95/pytorch-deeplab-xception
"""

import io
import os
import sys
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np

import torch
from torchvision import transforms

import pixelssl


def add_parser_arguments(parser):
    pixelssl.data_template.add_parser_arguments(parser)


def pascal_voc_aug():
    return PascalVocAugDataset


def pascal_voc_ori():
    return PascalVocOriDataset


class PascalVocDataset(pixelssl.data_template.TaskDataset):
    IMAGE = 'image'
    LABEL = 'label'
    PREFIX = 'prefix'

    def __init__(self, args, is_train, train_prefix_path, val_prefix_path):
        super(PascalVocDataset, self).__init__(args, is_train)
        self.im_size = self.args.im_size
        self.transform = None

        if self.is_train:
            self.fliplr = True
            self.prefix_path = os.path.join(self.root_dir, train_prefix_path)
        else:
            self.fliplr = False
            self.prefix_path = os.path.join(self.root_dir, val_prefix_path)

        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.label_dir = os.path.join(self.root_dir, 'SegmentationClassAug')

        with open(self.prefix_path, 'r') as f:
            lines = f.read().splitlines()
        
        for ii, line in enumerate(lines):
            image_path = os.path.join(self.image_dir, line + '.jpg')
            if not os.path.isfile(image_path):
                pixelssl.log_err('Cannot find image: {0} in Pascal VOC Dataset\n'.format(image_path))
            self.sample_list.append(line)

        self.idxs = [_ for _ in range(0, len(self.sample_list))]

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]

        image_path = os.path.join(self.image_dir, sample_name + '.jpg')
        label_path = os.path.join(self.label_dir, sample_name + '.png')

        has_label = os.path.exists(label_path)
        if not self.is_train and not has_label:
            pixelssl.log_err('The val sample of Pascal VOC dataset should has label\n'
                             'However, cannot find label: {0}\n'.format(label_path))

        image = self.im_loader.load(image_path).convert('RGB')
        label = self.im_loader.load(label_path) if has_label else None
        
        if self.is_train:
            image, label = self._train_prehandle(image, label)
        else:
            image, label = self._val_prehandle(image, label)

        label = label[None, :, :] if has_label else label

        return (image, ), (label, )

    def _train_prehandle(self, image, label):
        if label is None:
            sample = {self.IMAGE: image, self.LABEL: image} 
        else:
            sample = {self.IMAGE: image, self.LABEL: label}
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.args.im_size, crop_size=self.args.im_size),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        transformed_sample = composed_transforms(sample)

        if label is None:
            return transformed_sample[self.IMAGE], transformed_sample[self.IMAGE][0, ...] * 0.0 - 1.0
        else:
            return transformed_sample[self.IMAGE], transformed_sample[self.LABEL]

    def _val_prehandle(self, image, label):
        sample = {self.IMAGE: image, self.LABEL: label}
        composed_transforms = transforms.Compose([
            FixScaleCrop(crop_size=self.args.im_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        transformed_sample = composed_transforms(sample)

        return transformed_sample[self.IMAGE], transformed_sample[self.LABEL]


class PascalVocAugDataset(PascalVocDataset):
    def __init__(self, args, is_train):
        train_prefix_path = 'ImageSets/Segmentation/train_aug.txt'
        val_prefix_path = 'ImageSets/Segmentation/val.txt'

        super(PascalVocAugDataset, self).__init__(args, is_train, train_prefix_path, val_prefix_path) 


class PascalVocOriDataset(PascalVocDataset):
    def __init__(self, args, is_train):
        train_prefix_path = 'ImageSets/Segmentation/train.txt'
        val_prefix_path = 'ImageSets/Segmentation/val.txt'

        super(PascalVocOriDataset, self).__init__(args, is_train, train_prefix_path, val_prefix_path) 


# ---------------------------------------------------------------------
# Custom transforms for semantic segmentation dataset
# ---------------------------------------------------------------------

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img, 'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img, 'label': mask}
