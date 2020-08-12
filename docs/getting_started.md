<div align="center">
  <img src="img/pixelssl-logo.png" width="650"/>
</div>

---

## Getting Started

We provide some demo tasks under the folder `task`. Here, we introduce how to experiment based on PixelSSL.

The given task codes follow the task template provided by PixelSSL. Therefore, they are compatible with all integrated semi-supervised learning algorithms. We hope that these task codes can help users to validate their new semi-supervised algorithms and compare their algorithms with existing works.

In the following context, we refer to the task folder as `task/xxx`, where `xxx` is the unique name of the task. For example, the folder of semantic segmentation is named `task/sseg`.   
Currently supported tasks are [[semantic segmentation](../task/sseg), ].  
To run an experiment (assuming you are currently at the root path of the project): 

1. Switch to the root path of the task and install task-specific python dependencies:
    ```
    cd task/xxx
    pip install -r requirements.txt
    ```

2. Prepare the datasets according to the `Data Preparation` section in the file `task/xxx/README.md`.

3. If you want to validate the pretrained models provided by us, please download them according to the `Pretrained Models` section in the file `task/xxx/README.md` and put them in the folder `task/xxx/pretrained`. Then you can run:
    ```
    python -m script.[name_of_the_script]
    ```
    **NOTE**: You can specify the GPUs by adding `CUDA_VISIBLE_DEVICES=gpus-id` before the `python` command.  
    **NOTE**: The naming rule of the scripts and the pretrained models is `[model architecture]_[dataset]_[labels ratio]_[SSL algorithm]`.  

4. If you want to train the model, please comment the following two lines on the script as:
    ```
    # 'resume'  : 'pretrained/[name_of_the_pretrained_model]',
    # 'validation': True,
    ```
    Then you can run:
    ```
    python -m script.[name_of_the_script]
    ```

5. To check the help information for the arguments related to the current script, please run:
    ```
    python -m script.[name_of_the_script] --help
    ```

6. After starting the experiment, the folder `./task/xxx/result/name_of_the_script/yyyy-mm-dd_hh:mm:ss` will save all output files related to this run. The structure of the output folder is:
    ```
    task/xxx                            # task folder
    ├── result                          # output folder indicated in the script
    │   ├── name_of_the_script          # result folder of a specific script
    │   │   ├── yyyy-mm-dd_hh:mm:ss     # each run is named by its start time
    │   │   │   ├── train/val.log       # training or validating log
    │   │   │   ├── ckpt                # folder used to save checkpoints
    │   │   │   ├── visualization       # folder used to save images for visualization
    │   │   │   │   ├── debug           # folder used to save debugging images
    │   │   │   │   ├── train           # folder used to save training images
    │   │   │   │   ├── val             # folder used to save validating images
    ```
