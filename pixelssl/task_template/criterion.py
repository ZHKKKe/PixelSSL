import torch.nn as nn


""" This file defines a template of task-specific criterion.
"""


def add_parser_arguments(parser):
    """ Add the arguments required by all types of criterion here.
    
    Please add task-specific criterion arguments into the function 
    of the same name in 'task/xxx/criterion.py'
    """

    pass


def task_criterion():
    """ Export function of task-specific criterion class.

    Export a task-specific criterion class so that it can be called by a string 
    in the script. We recommend that the name of this function corresponds to the 
    task-specific criterion class.

    Each task-specific criterion classe should have its own export function.
    You can change the name of this function and use the new name in the script.
    """

    return TaskCriterion


class TaskCriterion(nn.Module):
    """ Superclass of task-specific criterion, i.e., the loss function.

    You can treat this class as a wrapper of 'torch.nn.Module'.
    Compare with 'torch.nn.module', this wrapper constrains:
        (1) The input format of '__init__' is fixed
        (2) The input and output format of 'forward' is fixed

    Inherit from this class to create a criterion class for your task.
    The following functions should be implemented:
        forward, 
        
    All criterions (losses) of the task model should be calculated in this 
    class since we need to ensure that the task code is compatible with the 
    SSL algorithms in PixelSSL.

    You can create multiple subclasses of 'TaskCriterion' for each task and 
    export them all. You can then switch them in the script.
    """

    def __init__(self, args=None):
        super(TaskCriterion, self).__init__()
        
        self.args = args        # arguments dict for task-specific criterion

    def forward(self, pred, gt, inp):
        """ Calculate the losses of the task model.

        All inputs - pred, gt, inp - are tuples. 
        In this function, you should:
            (1) check the length/format of - pred, gt, inp - to ensure 
                that they can be used to calculate losses
            (2) calculate all task-specific losses and return the values 
                on sample-level, i.e., a 1D tensor of shape == [batch_size]. 

        Arguments:
            pred (tuple): tuple contains all predictions of the task model
            gt (tuple): tuple contains all labels of the task model
            inp (tuple): tuple contains all inputs of the task model

        Returns:
            torch.Tensor: return a 1D tensor of shape == [batch_size], which 
                          allows the SSL algorithms to perform additional 
                          operations at the sample level.
        """

        raise NotImplementedError
