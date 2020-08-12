import torch

from pixelssl.utils import logger


""" This file provides a template (superclass) of the pixel-wise SSL algorithms.
"""

def add_parser_arguments(parser):
    """ Add the arguments related to the pixel-wise SSL algorithm.

    Arguments:
        parser (argparse.ArgumentParser): a instance of 'ArgumentParser'
    """

    pass


def ssl_base(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    """ Export function of the pixel-wise semi-supervised learning algorithm.

    Each algorithm requires an export function.
    The export function will pre-check/pre-process the format of task-specific dicts, 
    i.e., the element dicts. This function also creates an instance of the algorithm 
    and calls its 'build' function.

    Arguments:
        args (argparse.Namespace): arguments required by the algorithm
        xxx_dict (dict): element dicts used to build the algorithm
        task_func (pixelssl.task_template.func.TaskFunc): instance of 'TaskFunc'
            that contains the task-specific functions
    
    Returns:
        _SSLBase: an instance of the algorithm after calling 'algorithm.build()'
    """

    raise NotImplementedError



class _SSLBase:
    """ Superclass of pixel-wise semi-supervised learning algorithm.

    Subclassing from '_SSLBase' if you want to implement a new algorithm 
    for pixel-wise semi-supervised learning.

    Two constants should be set:
        NAME, SUPPORTED_TASK_TYPES

    Five functions should be implemented: 
        _build, _train, _validate, _save_checkpoint, _load_checkpoint

    The SSL algorithm is task-free, i.e., codes that follow 'pixelssl.task_template'
    are compatible with all subclasses of '_SSLBase'. Do not forget to register the 
    new algorithm in 'pixelssl/ssl_framwork/__init__.py'.
    """

    NAME = 'ssl_base'                           # unique name string of the algorithm
                                                #   'NAME' should be the same as the export function
    SUPPORTED_TASK_TYPES = []                   # list of supported types of pixel-wise tasks 
                                                #   pixelssl.REGRESSION / pixelssl.CLASSIFICATION

    def __init__(self, args):
        self.args = args                        # arguments required by SSL algorithm
        self.task_func = None                   # instance of 'TaskFunc' associated with a particular task
        self.meters = logger.AvgMeterSet()      # tool class for logging

        self.models = {}                        # dict of the models required by the task and algorithm
        self.optimizers = {}                    # dict of the optimizers required by the task and algorithm
        self.lrers = {}                         # dict of the learn rate required by the task and algorithm
        self.criterions = {}                    # dict of the criterions required by the task and algorithm

    # ---------------------------------------------------------------------
    # Interface for task proxy
    # ---------------------------------------------------------------------
    
    def build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        self._build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)

    def train(self, data_loader, epoch):
        self._train(data_loader, epoch)

    def validate(self, data_loader, epoch):
        self._validate(data_loader, epoch)
    
    def save_checkpoint(self, epoch):
        self._save_checkpoint(epoch)
    
    def load_checkpoint(self):
        return self._load_checkpoint()

    # ---------------------------------------------------------------------
    # All SSL algorithms should implement the following functions
    # ---------------------------------------------------------------------

    def _build(self, model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func):
        """ Build the SSL algorithm. 
        
        Each SSL algorithm contains both task-specific components and algorithm-special components.
        Each task-specific component has four parts -> (model, optimizer, lrer, criterion).

        This function takes the lists of task-specific components as the input and builds them.
        The algorithm-special components (e.g., the SSL constraints) will also be built in this function.
        Then, this function saves all required components into four dictionaries:
            self.models, self.optimizers, self.lrers, self.criterions

        Arguments:
            model_funcs (list): list of 'pixelssl.task_template.model.TaskModel'
            optimizer_funcs (list): list of optimizer function defined in 'pixelssl.nn.optimizer'
            lrer_funcs (list): list of learning rate adjust function defined in 'pixelssl.nn.lrer'
            criterion_funcs (list): list of 'pixelssl.task_template.criterion.TaskCriterion'
            task_func (pixelssl.task_template.func.TaskFunc): instance of 'pixelssl.task_template.func.TaskFunc'
                it contains the task-specific functions
        """

        raise NotImplementedError

    def _train(self, data_loader, epoch):
        """ Use the current SSL algorithm to train the task model (for one epoch).

        This function should be called after self.build().         
        One 'epoch' is defined by browsing the data_loader once.

        Arguments:
            data_loader (torch.utils.data.DataLoader): task-specific data loader for training 
            epoch (int): index of current epoch
        """

        raise NotImplementedError

    def _validate(self, data_loader, epoch):
        """ Validate the task model onece.

        This function should be called after self.build().

        Arguments:
            data_loader (torch.utils.data.DataLoader): task-specific data loader for validation
            epoch (int): index of current epoch
        """

        raise NotImplementedError
        
    def _save_checkpoint(self, epoch):
        """ Save the current state of the experiment to the checkpoint file.

        Arguments:
            epoch (int): index of current epoch
        """

        raise NotImplementedError
            
    def _load_checkpoint(self):
        """ Load the experiment status from the given checkpoint file (args.resume).

        Returns:
            int: index of current epoch
        """
        
        raise NotImplementedError
