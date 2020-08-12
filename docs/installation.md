<div align="center">
  <img src="img/pixelssl-logo.png" width="650"/>
</div>

---

## Installation

The basic requirements for installing PixelSSL are:
- Linux System
- Nvidia GPU with CUDA 8.0+
- Python 3+

**NOTE**: PixelSSL requires Nvidia GPU to run, i.e., the CPU only mode is currently not supported.


We recommand creating a new conda virtual environment for PixelSSL as follow:

1. Create a conda virtual environment named PixelSSL and activate it:
    ```
    conda create -n PixelSSL python=3.6
    source activate PixelSSL
    ```

2. Install PyTorch (>=1.0.0) and the corresponding torchvision following [the PyTorch official instructions](https://pytorch.org/).   
For example, if you use **CUDA 8.0**:
    ```
    conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch
    ```

3. Clone the repository of PixelSSL:
    ```
    git clone https://github.com/ZHKKKe/PixelSSL.git
    cd PixelSSL
    ```

4. Install other dependencies and PixelSSL:

    We provide two options for using PixelSSL as follow:

    (a) If you want to develop and validate a new semi-supervised learning algorithm (or try a new vision task) based on the latest code of PixelSSL, you need to install python dependencies:
    ```
    pip install -r pixelssl/requirements.txt
    ```

    (b) If you want to use the semi-supervised learning algorithms provided by PixelSSL in your own task project (should follow the task template in PixelSSL), you can compile and install PixelSSL into the current conda virtual environment:
    ```
    pip install .
    ```
    or
    ```
    python setup.py install
    ```
    Then, in any directory, you can import the package of PixelSSL in the current conda virtual environment:
    ```
    python
    >>> import pixelssl
    ```

5. After completing any of the above options, you can follow [Getting Started](getting_started.md) to run the integrated task code.
