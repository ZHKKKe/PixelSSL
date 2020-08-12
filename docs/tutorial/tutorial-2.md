<div align="center">
  <img src="../img/pixelssl-logo.png" width="650"/>
</div>

---

## Tutorial 2 - Implement A New Pixel-wise Task Based on the Task Template

Since different vision tasks require a unified format to be compatible with the semi-supervised algorithms in PixelSSL,  in the folder `pixelssl/task_template`, we provide a template for encapsulating the task code. The template consists of five files:
```
pixelssl/task_template          # task template folder
├── __init__.py
├── criterion.py                # template of the task criterions (losses)
├── data.py                     # template of the task datasets
├── func.py                     # template of the task-specific functions
├── model.py                    # template of the task models
├── proxy.py                    # template of the task proxy
```
In this template, we split the main components used in deep learning into the files `data.py`, `model.py`, and `criterion.py`. We put task-specific functions (required by different semi-supervised algorithms) in the file `func.py`. In addition, we define the deep learning pipeline in the file `proxy.py`.

To implement a new task code based on the latest PixelSSL project, you should complete the following steps (assuming you are currently at the root path of the project):
1. Create a new task folder `task/xxx`, where `xxx` is the unique name of the task. Then create the file `__init__.py`.

2. Create five files (`criterion.py`, `model.py`, `data.py`, `func.py`, `proxy.py`) under the folder `task/xxx`. These files should inherit from the corresponding files under the folder `pixelssl/task_template`.
Each file contains several classes and the corresponding export functions:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) Each class implements a task-specific component  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) Each export function exports the corresponding class for calling in the script  
Please refer to the comments of the files under the folder `pixelssl/task_template` and other implemented task codes for more details. 

3. Create the file `task/xxx/requirements.txt` and add task-specific python dependencies into it.

4. Create the folder `task/xxx/dataset` and put your task-specific datasets in it. 

5. Create the folder `task/xxx/script` and implement scripts inside it. Please refer to the scripts of other implemented task codes for more details. 

6. Create the folder `task/xxx/pretrained` to save the pretrained models if necessary.

7. Create the file `README.md` to describe how to prepare and run the task code.

Now you can run your task codes with the semi-supervised algorithms according to the [Getting Started](../getting_started.md) document!

If you want to call the provided semi-supervised algorithms in your own project, please install PixelSSL in your python environment and use our task template to encapsulate your code.
