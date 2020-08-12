<div align="center">
  <img src="../img/pixelssl-logo.png" width="650"/>
</div>

---

## Tutorial 1 - Implement A New Pixel-wise Semi-Supervised Algorithm

This is a guide for implementing a new pixel-wise semi-supervised learning (SSL) algorithm based on the interface provided by PixelSSL.

In PixelSSL, the codes of all SSL algorithms are located under the folder `pixelssl/ssl_algorithm`.
The `_SSLBase` class in the file `ssl_base.py` is the interface of the SSL algorithm.

To implement a new SSL algorithm, you should complete the following steps (assuming you are currently at the root path of the project):

1. Create a new python file named `ssl_xxx.py` under the folder `pixelssl/ssl_algorithm`, where `xxx` is the name of your SSL algorithm. 

2. Define a subclass of `_SSLBase`, named `SSLXXX`, and implement your SSL algorithm in it. Typically, you need to:  
(a) set two constants:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NAME, SUPPORTED_TASK_TYPES  
(b) finish five functions:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_build, _train, _validate, _save_checkpoint, _load_checkpoint  
Please refer to the comments of the `_SSLBase` class and other implemented SSL algorithms for more details.  
**NOTE**: If your SSL algorithm includes the task-specific functions that are vary between tasks, please:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) define these functions in the class `pixelssl/task_template/func.py/TaskFunc`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) override/implement these functions in the file `task/name/func.py` (`name` is the name of task)

3. Implement the export function of `SSLXXX`. The name of the export function should be `ssl_xxx`. Its input and output should be the same as the export function `pixelssl/ssl_algorithm/ssl_base.py/ssl_base`.

4. Implement the `add_parser_arguments` function in the file `ssl_xxx.py`. This function defines all algorithm-specific arguments.

5. In the file `pixelssl/ssl_algorithm/__init__.py`, register your new SSL algorithm as follow: 
    ``` 
    from .ssl_xxx import SSLXXX
    SSL_XXX = SSLXXX.NAME
    SSL_ALGORITHMS.append(SSL_XXX)
    del SSLXXX
    ```

6. If your SSL algorithm depends on specific python packages, please add their name to the file `pixelssl/requirements.txt` and install them.

Now you can use the unique key `pixelsl.SSL_XXX` and the algorithm-specific arguments to call your SSL algorithm in the script!
