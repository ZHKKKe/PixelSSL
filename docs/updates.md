<div align="center">
  <img src="img/pixelssl-logo.png" width="650"/>
</div>

---

## Updates

- **[Dec 25 2020] PixelSSL v0.1.4 is Released!**  
  v0.1.4 supports the [CutMix](https://arxiv.org/abs/1906.01916) semi-supervised learning algorithm for pixel-wise classification.

<br>

- **[Nov 06 2020] PixelSSL v0.1.3 is Released!**  
  v0.1.3 supports the [CCT](https://arxiv.org/abs/2003.09005) semi-supervised learning algorithm for pixel-wise classification.

<br>

- **[Oct 28 2020] PixelSSL v0.1.2 is Released!**  
  v0.1.2 supports [PSPNet](https://arxiv.org/abs/1612.01105) and its SSL results for semantic segmentation task. 
  
<br>

- **[Sep 16 2020] PixelSSL v0.1.1 is Released!**  
  v0.1.1 supports the [S4L](https://arxiv.org/abs/1905.03670) semi-supervised learning algorihm and fixes some bugs in the demo code of semantic segmentation task.
  
  Bug fix:  
    - Fixed the bug of the ASPP module in the DeepLabV2 model.
    - Fixed the bug of sharing `confusion_matrix` when calculating the semantic segmentation metric in multi-model SSL algorithm, e.g., MT, GCT.  

<br>

- **[Aug 13 2020] PixelSSL v0.1.0 is Released!**  
  v0.1.0 supports supervised-only learning, three semi-supervised learning algorithms 
  ([MT](https://arxiv.org/abs/1703.01780), 
  [AdvSSL](https://arxiv.org/abs/1802.07934), 
  [GCT](https://arxiv.org/abs/2008.05258)) 
  and one example task (semantic segmentation).
