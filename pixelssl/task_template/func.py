import torch

from pixelssl.utils import logger


""" This file defines a template for task-specific function.
"""

def task_func():
    """ Export function of task function class.

    Please do not change the name of this export function in 
    the file 'task/xxx/func.py'!
    The returned class is required by all SSL algorithms in PixelSSL.
    """
    
    return TaskFunc
    

class TaskFunc:
    """ Superclass of task-specific function.

    Inherit from this class to implement task-specific functions, 
    e.g., calculating metrics and visulizing images.
    If you want to use a certain SSL algorithm, you should implement 
    all the functions related to it, e.g., you should complete:
        sslgct_fd_in_channels, sslgct_prepare_task_gt_for_fdgt,
    for the SSL_GCT algorithm.
    """

    # special indetifier for the metric elements
    METRIC_STR = 'metric'
    
    def __init__(self, args=None):
        self.args = args

    # ---------------------------------------------------------------------
    # Functions for All Tasks
    # Following functions are required by all tasks.
    # ---------------------------------------------------------------------

    def metrics(self, pred, gt, inp, meters, id_str=''):
        """ Calculate metrics for the task model.

        This function calculates all performance metrics for the task model
        and saves them into 'meters' (with prefix '[id_str]-[str_METRIC_STR]_xxx').
        
        Arguments:
            pred (torch.Tensor): prediction of the task model
            gt (torch.Tensor): ground truth of the task model
            meters (pixelssl.utils.AvgMeterSet): recorder
            id_str (str): identifier for recording
        """

        logger.log_warn('No implementation of the \'metrics\' function for current task.\n'
                        'Please implement it in \'task/xxx/func.py\'.\n')

    def visualize(self, out_path, id_str='', inp=None, pred=None, gt=None):
        """ Visualize images during training/validation.

        Please refer to the implemented tasks to finish this function.

        Arguments:
            out_path (str): path to save the images
            id_str (str): identifier for recording
            inp (tuple): inputs of the task model
            pred (tuple): prediction of the task model
            gt (tuple): ground truth of the task model
        """

        logger.log_warn('No implementation of the \'visulize\' function for current task.\n'
                        'Please implement it in \'task/xxx/func.py\'.\n')
         
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Functions for SSL_ADV
    # ---------------------------------------------------------------------

    def ssladv_fcd_in_channels(self):
        """ Define the input channels of the Full Convolution Discriminator (FCD) used 
            in SSL_ADV.

        Since the prediction of different tasks have different channels, the input channels 
        of the FC discriminator will be different.
        Following the paper 'Adversarial Learning for Semi-supervised Semantic Segmentation', 
        we set:
          input channels of the FC discriminator = channels of the task predictionW

        Returns:
            int: number of the FC discriminator's input channels
        """

        raise NotImplementedError

    def ssladv_preprocess_fcd_criterion(self, fcd_pred, task_gt, is_real):
        """ Preprocess the task prediction (or the task ground truth) used to calculate 
            the adversarial loss of the FC discrminator used in SSL_ADV.

            In some tasks, a task-specific pipeline is required to generate the ground truth 
            for calculating the adversarial loss of the FC discriminator, e.g., we need to 
            ignore the segment boundary on the semantic segmentation task. This function performs 
            such a preprocessing pipeline and returns a tensor to calculate the adversarial loss 
            of the FC discriminator. 

            If this preprocessing pipeline is unnecessary for your task, please generate real/fake 
            ground truth based on the argument 'is_real' only.

        Arguments:
            pred (torch.Tensor): prediction of the FC discriminator
            gt (torch.Tensor or None): ground truth of the task model
                                       It may be used to generate the real/fake ground truth for the 
                                       FC discriminator. If 'fcd_pred' is an unlabeled sample, please 
                                       set 'gt' to None and generate a fake ground truth for the FC 
                                       discriminator
            is_real (bool): generate real ground truth if True; generate fake ground truth if False

        Returns:
            torch.Tensor, torch.Tensor: handled fcd_pred, ground truth of the FC discriminator
        """

        raise NotImplementedError

    def ssladv_convert_task_gt_to_fcd_input(self, task_gt):
        """ Convert the task ground truth to the input of the FC Discriminator used 
            in SSL_ADV.

        The FC discriminator requires the task ground truth as the 'real' input.
        This function preprocesses the task's ground truth for some special tasks. 
        For example, for semantic segmentation, we should convert the task ground 
        truth to a one-hot vector.

        Arguments:
            task_gt (torch.Tensor): ground truth of the task model
        
        Returns:
            torch.Tensor: handled task ground truth (as the input of the FC discriminator)
        """

        return task_gt

    # ---------------------------------------------------------------------
    
    # ---------------------------------------------------------------------
    # Functions for SSL_GCT
    # ---------------------------------------------------------------------

    def sslgct_fd_in_channels(self):
        """ Define the input channels of the flaw detector used in SSL_GCT.

        Since the input/prediction of different tasks have different channels, the 
        input channels of the flaw detector will be different.
        Following the paper 'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning',
        we set:
            input channels of the flaw detector = channels of the input + channels of the task prediction

        Returns:
            int: input channels of the flaw detector used in SSL_GCT
        """

        raise NotImplementedError

    def sslgct_prepare_task_gt_for_fdgt(self, task_gt):
        """ Prepare the ground truth of the task model for generating the ground truth 
            of the flaw detector used in SSL_GCT.
        
        To generate the ground truth of the flaw detector, the format of 'task_gt' should
        be the same as the task prediction. However, in some tasks, e.g., in semantic segmentation, 
        we should convert the task ground truth (task_gt) to a one-hot vector.

        Arguments:
            task_gt (torch.Tensor): ground truth of the task model

        Returns:
            torch.Tensor: handled task ground truth (has the same format as the task prediction)
        """

        return task_gt

    # ---------------------------------------------------------------------
    
    # ---------------------------------------------------------------------
    # Functions for SSL_S4L
    # ---------------------------------------------------------------------

    def ssls4l_rc_in_channels(self):
        """ Define the input channels of the rotation classifier used in SSL_S4L.

            Since different tasks use different model architectures, the input channel of the rotating 
            classifier should be changed. Usually, we use the feature map encoded by the task model or 
            the output of the task model as the input of the rotation classifier.

            Returns:
                int: input channels of the rotation classification used in SSL_S4L
        """

        raise NotImplementedError

    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Functions for SSL_CCT
    # ---------------------------------------------------------------------

    def sslcct_activate_ad_preds(self, ad_preds):
        """ Activate the predictions from the auxilary decoders used in SSL_CCT.

            Since different tasks use different activation functions, this function is task-specific.
            The output of this function will be used to calculate the consistency constraint.

            Arguments:
                ad_preds (list): list of the predictions from the auxilary decoders, and the type of 
                                 the element is torch.Tensor

            Returns:
                list: activated predictions from the auxilary decoders
        """

        raise NotImplementedError

    def sslcct_ad_in_channels(self):
        """ Define the input channels of the auxilary decoders used in SSL_CCT.

            Since different tasks use different model architectures, the input channel of the auxilary 
            decoders should be changed. Usually, we use the feature map encoded by the task model as 
            the input of the auxilary decoders.

            Returns:
                int: input channels of the auxilary decoders used in SSL_CCT
        """

        raise NotImplementedError

    def sslcct_ad_out_channels(self):
        """ Define the output channels of the auxilary decoders used in SSL_CCT.

            Since different tasks have different output format, the out channel of the auxilary 
            decoders should be changed. Usually, the outputs of the auxilary decoders have the 
            same format with the task prediction. 

            Returns:
                int: output channels of the auxilary decoders used in SSL_CCT
        """

        raise NotImplementedError

    def sslcct_ad_upsample_scale(self):
        """ Define the upsample of the auxilary decoders used in SSL_CCT.

            The auxiliary decoder in SSL_CCT use the low-resolution latent features of the task 
            model as input. As different task models have different downsampling ratios, we need 
            to specify different upsampling ratios for the auxiliary decoders to restore the 
            high-resolution output.

            Returns:
                int: upsampling scale of the auxilary decoders used in SSL_CCT
        """

        raise NotImplementedError

    # ---------------------------------------------------------------------
