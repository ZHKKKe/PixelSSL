<div align="center">
  <img src="../img/pixelssl-logo.png" width="650"/>
</div>

---

## Tutorial 4 - Support More Optimizers and LRSchedulers

Since PixelSSL is a script-based codebase, we need to export the PyTorch optimizers and LR schedulers so that the script can support them.

### Optimizers  
In the file `pixelssl/nn/optimizer.py`, we have exported the optimizers supported by PyTorch == 1.0.0. There are three steps to export a new optimizer (supported by PyTorch > 1.0.0):  
1. Refer to the `sgd` function in the file to implement the export function of the optimizer.

2. Note that the `pytorch_support` function should be called to check whether the current PyTorch version supports the optimizer.

3. Add the arguments related to the optimizer in the `add_parser_arguments` function.

Now you can call the LR optimizer in the script!

If you want to implement a new optimizer that PyTorch does not support, please refer to the official PyTorch documents for more details. After that, you can follow the above steps to export it. In the file `pixelssl/nn/optimizer.py`, we provide the `WDAdam (wdadam)` optimizer as an example.

### LR Schedulers
In the file `pixelssl/nn/lrer.py`, we have exported the LR schedulers supported by PyTorch == 1.0.0. There are four steps to export a new LR scheduler (supported by PyTorch > 1.0.0):
1. Refer to the `steplr` function in the file to implement the export function of the LR scheduler.

2. Note that the `pytorch_support` function should be called to check whether the current PyTorch version supports the LR scheduler.

3. Add the arguments related to the optimizer in the `add_parser_arguments` function.

4. Put the name of the export function into either `EPOCH_LRERS` or `ITER_LRERS`.  
In deep learning, the learning rate is updated after either each iteration or each epoch. `EPOCH_LRERS` contains the LR schedulers that are updated once per epoch, and `ITER_LRERS` contains the LR schedulers that are updated once per iteration.  
The LR schedulers supported by PyTorch usually belong to `EPOCH_LRERS`.

Now you can call the LR scheduler in the script!

If you want to implement a new LR scheduler that PyTorch does not support, please refer to the official PyTorch documents for more details. After that, you can follow the above steps to export it. In the file `pixelssl/nn/lrer.py`, we provide the `PolynomialLR` LR scheduler as an example.

