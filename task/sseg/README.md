<div align="center">
  <img src="../../docs/img/pixelssl-logo.png" width="650"/>
</div>

 ---

# Demo Task - Semantic Segmentation
This is the demo code of semantic segmentation used to validate the semi-supervised algorithms in PixelSSL.

## Introduction
Semantic segmentation takes an image as input and predicts a series of category masks, which link each pixel in the input image to a class. In this implementation, the supported task model is [DeepLab-v2](https://arxiv.org/abs/1606.00915) with the [ResNet-101](https://arxiv.org/abs/1512.03385) backbone. All experiments are conducted on the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) dataset, which comprises 20 foreground classes along with 1 background class. The extra annotation set from the [Segmentation Boundaries Dataset (SBD)](http://home.bharathh.info/pubs/codes/SBD/download.html) is combined to expand the dataset. 

The training rules proposed in paper [Guided Collaborative Training for Pixel-wise Semi-Supervised Learning](https://arxiv.org/abs/2008.05258) are applied to set the arguments. All experiments are trained with **4 Nvidia GPUs**. For semi-supervised learning, 1/16, 1/8, 1/4, 1/2 samples are randomly extracted as the labeled subset, and the rest of the training set is used as the unlabeled subset. Note that the same data splits are used in all experiments. The main experimental results (mIOU averaged over 3 runs) are as follows:
| SSL/Labels | 1/16 labels | 1/8 labels | 1/4 labels | 1/2 labels | full labels | 
| :---: | :---: | :---: | :---: | :---: | :---: |
| SupOnly | 64.55 | [68.38](https://drive.google.com/file/d/1QutIDEmxrz86tocLiW3cQPgPlfeVKoFO/view?usp=sharing) | 70.69 | 73.56| [75.32](https://drive.google.com/file/d/1sw1UspgnUsnJ5bOrTRVacX83B2P0Rng4/view?usp=sharing) |
| [MT](https://arxiv.org/abs/1703.01780) | 66.08 | [69.81](https://drive.google.com/file/d/1CkijR-hREoOCMVA5z_yc-MYkvLFecZcj/view?usp=sharing) | 71.28 | 73.23 | 75.28 | 
| [AdvSSL](https://arxiv.org/abs/1802.07934) | 65.67 | [69.89](https://drive.google.com/file/d/16zt6pBQEe8yBnbw12rEBoZR4npdYnoL0/view?usp=sharing) | 71.53 | 74.48 | 75.86 |
| [GCT](https://arxiv.org/abs/2008.05258) | 67.19 | [72.14](https://drive.google.com/file/d/1F21G7CtLOh5iyORnKgaetKent6oAAguZ/view?usp=sharing) | 73.62 | 74.82 | 75.73 |

**NOTE**: Please click the mIOU with the link to download the corresponding pretrained model.

The following sections will introduce how to prepare and run the code.  
We assume that you are currently at the root path of the task, i.e., `task/sseg`.

## Data Preparation
Please prepare the PascalVOC dataset augmented by SBD as follows:

1. Switch to the path of the PascalVOC dataset:
    ```
    cd dataset/PascalVOC
    ```

2. Download, unzip, and merge the PascalVOC dataset and SBD:
    ```
    sh prepare.sh
    ```
    The PascalVOC dataset augmented by SBD will be stored in the folder `dataset/PascalVOC/VOCdevkit`.  
    The augmented samples list used for training will be saved in the file `dataset/PascalVOC/VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt`.
  
3. Create the sublabeled prefix file for semi-supervised learning:  
    ```
    python tool/random_sublabeled_samples.py
    ```
    The sublabeled prefix file wil be saved in the folder `dataset/PascalVOC/sublabeled_prefix/[numerator]-[denominator]/[idx].txt`. Here, `[numerator]/[denominator]` is the ratio of the labeled data that can be changed in the file `random_sublabeled_samples.py`, and `[idx]` is the auto-set index.

4. Switch to the root path of the task:
    ```
    cd ../..
    ```

Now you can use the PascalVOC dataset to experiment. For semi-supervised learning, please set the experimental argument `sublabeled_path` to the `.txt` file created by step 3.

## Pretrained Models
Please click the mIOU results in the above table (in the [Introduction](#introduction) section) to download the pretrained models. All pretrained models can also be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1SOqm5s60WBnXIO-cNYS2XLiU2rz7O0lN?usp=sharing).

After downloading the pretrained models, please put them into the folder `pretrained`.

## Script Execution
In short, after [preparing the dataset](#data-preparation) and [downloading the pretrained models](#pretrained-models), you can run validate the pretrained model (take the GGT algorithm as an example) by:
```
pip install -r requirements.txt
python -m script.deeplabv2_pascalvoc_1-8_sslgct
```

Please refer to the [Getting Started](../../docs/getting_started.md) document for more details, e.g., how to retrain the model.  
**NOTE**: If you want to retrain the model, it is recommended to use **4 Nvidia GPUs** to obtain similar performance to the pretrained model.