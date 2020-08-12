import torch.nn as nn


""" This file defines a template of task-specific model.
"""

def add_parser_arguments(parser):
    """ Add the arguments required by all types of model here.
    
    Please add task-specific model arguments into the function 
    of the same name in 'task/xxx/model.py'
    """

    pass


def task_model():
    """ Export function of task-specific model class.

    Export a task-specific model class so that it can be called by a string
    in the script. We recommend that the name of this function corresponds to 
    the task-specific model class.

    Each task-specific model class should have its own export function.
    You can change the name of this function and use the new name in the script.
    """

    return TaskModel


class TaskModel(nn.Module):
    """ Superclass of task-specific model, i.e., the task model.
    
    You can treat this class as a wrapper of 'torch.nn.Module'.
    Compare with 'torch.nn.module', this wrapper constrains:
        (1) The input format of '__init__' is fixed
        (2) The input and output format of 'forward' is fixed

    Inherit from this class to create a model class for your task.
    The following functions should be implemented:
        forward, 

    You can implement the model as suggested in the PyTorch official 
    document. 

    You can create multiple subclasses of 'TaskModel' for each task and 
    export them all. You can then switch them in the script.
    """

    def __init__(self, args=None):
        super(TaskModel, self).__init__()
        self.args = args        # arguments dict for task-specific model
        self.model = None       # instance of the task model ('torch.nn.Module')
        self.param_groups = []  # parameter groups list used for optimizer

    def forward(self, inp):
        """ Forward propagation function of the task model.

        The input and output formats of this function are fixed!
        'inp' is a tuple that includes all inputs for the task model,
        e.g., (inp1, inp2, ...).
        
        In this function, you should:
            (1) check the format/shape of 'inp' to ensure that it is suitable for 
                your task code.
            (2) create two dicts, named 'resulter' and 'debugger'. You need to save 
                all outputs into the 'resulter' dict to compatible with the SSL algorithms.
                Commonly, there are two outputs required by all SSL algorithms:
                    (a) resulter['pred'] - unactivated task predictions.
                    (b) resulter['activated_pred'] - activated task predictions.
                You can also save the temporary/debug variables in the 'debugger' dict and 
                access them in SSL algorithms.
        
        Some SSL algorithms require special variables, please save them in the 'result' dict 
        for return.
        
        Arguments:
            inp (tuple): tuple contains all inputs of the task model. 
                         Each element is an instance of 'torch.autograd.Variable.cuda'

        Returns:
            dict, dict: 'resulter' and 'debugger'. Both of them are dicts.
        """

        raise NotImplementedError
