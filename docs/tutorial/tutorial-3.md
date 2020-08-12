<div align="center">
  <img src="../img/pixelssl-logo.png" width="650"/>
</div>

---

## Tutorial 3 - Dataset Wrappers for Semi-Supervised Learning

In [Tutorial 2](tutorial-2.md), we have introduced how to encapsulate a vision task by the template provided in PixelSSL. The datasets inherited from the class `pixelssl/task_template/data.py/TaskFunc` are fully labeled. However, the SSL algorithms in PixelSSL support only one semi-supervised dataloader. Therefore, we provides some dataset wrappers in the file `pixelssl/nn/data.py` to preprocess the fully labeled datasets for semi-supervised learning. Currently, there are two common options:

1. Split a fully labeled dataset into a labeled subset and an unlabeled subset.  
Given a fully labeled dataset, we can remove the labels of some samples and treat them as unlabeled samples. To this end, we implement a `SplitUnlabeledWrapper`. It requires an additional file to indicate the prefix of the labeled samples. This operation is widely used in the research of semi-supervised learning.  
When using this dataset wrapper, the argument `sublabeled_path` should be set in the script.

2. Combine multiple datasets into a semi-supervised dataset.  
In practice, we may need to combine multiple datasets (including labeled and unlabeled) for semi-supervised learning. We provide a `JointDatasetsWrapper`, which (1) takes a list of labeled datasets and a list of unlabeled datasets as input, and (2) combines all given datasets into a large dataset. The new dataset consists of a labeled subset and an unlabeled subset.  
When using this dataset wrapper, the argument `unlabeledset` should be set in the script.
In this case, the argument `trainset` contains all labeled datasets while  `unlabeledset` contains all unlabeled datasets.

To implement a new dataset wrapper for semi-supervised learning, you should (assuming you are currently at the root path of the project): 
1. Create a new class inherited from the class `_SSLDatasetWrapper` in the file `pixlssl/nn/data.py`.

2. Implement the dataset wrapper refer to the implemented `SplitUnlabeledWrapper` and `JointDatasetWrapper`. The key is to divide the index list into two parts, labeled (`self.labeled_idxs`) and unlabeled (`self.unlabeled_idxs`). 

3. Implement the logic of calling the data wrapper in the function `pixelssl/task_template/proxy.py/_create_dataloader`. Typically, you can use the `TwoStreamBatchSampler` in the file `pixlssl/nn/data.py` to read the semi-supervised dataset.

Now you can use the new dataset wrapper in the script to create a semi-supervised dataset!