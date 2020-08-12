import os
import io
from PIL import Image

from torch.utils.data import Dataset


""" This file defines a template for task-specific dataset.
"""


def add_parser_arguments(parser):
    """ Add the arguments required by all types of dataset here.

    Please add task-specific dataset arguments into the function 
    of the same name in 'task/xxx/data.py'
    """

    pass


def task_dataset():
    """ Export function of task-specific dataset class.
    
    Export a task-specific dataset class so that it can be called by a string 
    in the script. We recommend that the name of this function corresponds to the 
    task-specific dataset class.
    
    Each task-specific dataset class should have its own export function.
    You can change the name of this function and use the new name in the script.
    """

    return TaskDataset


class TaskDataset(Dataset):
    """ Superclass of task-specific dataset. 

    You can treat it as a wrapper of 'torch.utils.data.Dataset'.
    Compare with 'torch.utils.data.Dataset', this wrapper constrains:
        (1) The input format of '__init__' is fixed
        (2) The output format of '__getitem__' is fixed
        (3) Some self variables are defined, e.g., 'self.sample_list', 'self.idxs'

    Inherit from this class to create a dataset class for your task.
    The following functions should be implemented:
        __getitem__, 

    This class process all data as the labeled data. The dataset wrappers for 
    semi-supervised learning are implemented in the file 'pixelssl/nn/data.py'.
    """

    def __init__(self, args=None, is_train=True):
        super(TaskDataset, self).__init__()

        self.args = args                # arguments dict for task-specific dataset
        self.is_train = is_train        # training mode if True (validation mode if False)
        self.root_dir = None            # root path of the dataset

        self.sample_list = []           # a list containing the name of all samples
        self.idxs = []                  # a list containing the index of all samples

        self.im_loader = ImageLoader()  # image loader based on PIL.Image

        if self.args is not None:
            assert len(self.args.trainset) <= 1
            assert len(self.args.valset) <= 1
        
        if is_train:
            self.root_dir = list(self.args.trainset.values())[0]
        else:
            self.root_dir = list(self.args.valset.values())[0]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """ Return the sample with [index == input 'idx'] in self.sample_list. 
        Each sample is organized into two tuples (inputs and labels) and returned.

        Arguments:
            idx (int): index of the sample in self.sample_list
        
        Returns:
            tuple, tuple: return a sample, the elements of the sample are orgnized 
                          into an input tuple of and a label tuple
                          for example -> 'return (input1, input2, ), (label1, )'
        """

        raise NotImplementedError


class ImageLoader:
    """ Image Loader based on PIL.Image.
    """

    def __init__(self):
        pass

    def load(self, name):
        image = Image.open(name)
        return image
