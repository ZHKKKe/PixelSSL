import os
import itertools
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


""" This file implements dataset wrappers and batch samplers for SSL.
"""


class _SSLDatasetWrapper(Dataset):
    """ This is the superclass of SSL dataset wrapper.
    """

    def __init__(self):
        super(_SSLDatasetWrapper, self).__init__()

        self.labeled_idxs = []      # index of the labeled data
        self.unlabeled_idxs = []    # index of the unlabeled data


class SplitUnlabeledWrapper(_SSLDatasetWrapper):
    """ Split the fully labeled dataset into a labeled subset and an 
        unlabeled dataset based on a given sublabeled prefix list. 
    
    For a fully labeled dataset, a common operation is to remove the labels 
    of some samples and treat them as the unlabeled samples. 

    This dataset wrapper implements the dataset-split operation by using 
    the given sublabeled prefix list. Samples whose prefix in the list 
    are treated as the labeled samples, while others samples are treated as 
    the unlabeled samples.
    """

    def __init__(self, dataset, sublabeled_prefix, ignore_unlabeled=False):
        super(SplitUnlabeledWrapper, self).__init__()

        self.dataset = dataset
        self.sublabeled_prefix = sublabeled_prefix
        self.ignore_unlabeled = ignore_unlabeled

        self._split_labeled()

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _split_labeled(self):
        labeled_list, unlabeled_list = [], []
        for img in self.dataset.sample_list:
            is_labeled = False
            for pdx, prefix in enumerate(self.sublabeled_prefix):
                if img.startswith(prefix):
                    labeled_list.append(img)
                    is_labeled = True
                    break

            if not is_labeled:
                unlabeled_list.append(img)

        labeled_size, unlabeled_size = len(labeled_list), len(unlabeled_list)
        assert labeled_size + unlabeled_size == len(self.dataset.sample_list)          
        
        if self.ignore_unlabeled:
            self.dataset.sample_list = labeled_list
            self.dataset.idxs = [_ for _ in range(0, len(self.dataset.sample_list))]
            self.labeled_idxs = self.dataset.idxs
            self.unlabeled_idxs = []
        else:
            self.dataset.sample_list = labeled_list + unlabeled_list
            self.dataset.idxs = [_ for _ in range(0, len(self.dataset.sample_list))]
            self.labeled_idxs = [_ for _ in range(0, labeled_size)]
            self.unlabeled_idxs = [_ + labeled_size for _ in range(0, unlabeled_size)]


class JointDatasetsWrapper(_SSLDatasetWrapper):
    """ Combine several datasets (can be labeled or unlabeled) into one dataset.
    
    This dataset wrapper will combine multiple given dataset into one big dataset.
    The new dataset consists of a labeled subset and an unlabeled subset.
    """

    def __init__(self, labeled_datasets, unlabeled_datasets, ignore_unlabeled=False):
        super(JointDatasetsWrapper, self).__init__()

        self.labeled_datasets = labeled_datasets
        self.unlabeled_datasets = unlabeled_datasets
        self.ignore_unlabeled = ignore_unlabeled

        self.labeled_datasets_size = [len(d) for d in self.labeled_datasets]
        self.unlabeled_datasets_size = [len(d) for d in self.unlabeled_datasets]

        self.labeled_size = np.sum(np.asarray(self.labeled_datasets_size))        
        self.labeled_idxs = [_ for _ in range(0, self.labeled_size)]
        
        self.unlabeled_size = 0
        if not self.ignore_unlabeled:
            self.unlabeled_size = np.sum(np.asarray(self.unlabeled_datasets_size))
            self.unlabeled_idxs = [self.labeled_size + _ for _ in range(0, self.unlabeled_size)]

    def __len__(self):
        return int(self.labeled_size + self.unlabeled_size)

    def __getitem__(self, idx):
        assert 0 <= idx < self.__len__()

        if idx >= self.labeled_size:
            idx -= self.labeled_size
            datasets = self.unlabeled_datasets
            datasets_size = self.unlabeled_datasets_size
        else:
            datasets = self.labeled_datasets
            datasets_size = self.labeled_datasets_size

        accumulated_idxs = 0
        for ddx, dsize in enumerate(datasets_size):
            accumulated_idxs += dsize
            if idx < accumulated_idxs:
                return datasets[ddx].__getitem__(idx - (accumulated_idxs - dsize))


class TwoStreamBatchSampler(Sampler):
    """ This two stream batch sampler is used to read data from '_SSLDatasetWrapper'.

    It iterates two sets of indices simultaneously to read mini-batch for SSL.
    There are two sets of indices: 
        labeled_idxs, unlabeled_idxs
    An 'epoch' is defined by going through the longer indices once.
    In each 'epoch', the shorter indices are iterated through as many times as needed.
    """

    def __init__(self, labeled_idxs, unlabeled_idxs, labeled_batch_size, unlabeled_batch_size):
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size

        assert len(self.labeled_idxs) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_idxs) >= self.unlabeled_batch_size > 0

        self.unlabeled_batchs = len(self.unlabeled_idxs) // self.unlabeled_batch_size
        self.labeled_batchs = len(self.labeled_idxs) // self.labeled_batch_size

    def __iter__(self):
        if self.unlabeled_batchs >= self.labeled_batchs:
            unlabeled_iter = self.iterate_once(self.unlabeled_idxs)
            labeled_iter = self.iterate_eternally(self.labeled_idxs)
        else:
            unlabeled_iter = self.iterate_eternally(self.unlabeled_idxs)
            labeled_iter = self.iterate_once(self.labeled_idxs)

        return (labeled_batch + unlabeled_batch
                for (labeled_batch, unlabeled_batch) in zip(
                    self.grouper(labeled_iter, self.labeled_batch_size),
                    self.grouper(unlabeled_iter, self.unlabeled_batch_size)))

    def __len__(self):
        return max(self.unlabeled_batchs, self.labeled_batchs)

    def iterate_once(self, iterable):
        return np.random.permutation(iterable)

    def iterate_eternally(self, indices):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)

        return itertools.chain.from_iterable(infinite_shuffles())

    def grouper(self, iterable, n):
        # e.g., grouper('ABCDEFG', 3) --> ABC DEF"
        args = [iter(iterable)] * n
        return zip(*args)
