<div align="center">
  <img src="docs/img/pixelssl-logo.png" width="650"/>
</div>

---

PixelSSL is a PyTorch-based semi-supervised learning (SSL) codebase for pixel-wise (Pixel) vision tasks.

The purpose of this project is to promote the research and application of semi-supervised learning on pixel-wise vision tasks. PixelSSL provides two major features:
- Interface for implementing new semi-supervised algorithms
- Template for encapsulating diverse computer vision tasks

As a result, the SSL algorithms integrated in PixelSSL are compatible with all task codes inherited from the given template. 

In addition, PixelSSL provides the benchmarks for validating semi-supervised learning algorithms for some pixel-level tasks, which now include [semantic segmentation](task/sseg).


## News
- **[Dec 25 2020] PixelSSL v0.1.4 is Released!**  
  *Merry Christmas!* :christmas_tree:  
  v0.1.4 supports the [CutMix](https://arxiv.org/abs/1906.01916) semi-supervised learning algorithm for pixel-wise classification.

- **[Nov 06 2020] PixelSSL v0.1.3 is Released!**  
  v0.1.3 supports the [CCT](https://arxiv.org/abs/2003.09005) semi-supervised learning algorithm for pixel-wise classification.

- **[Oct 28 2020] PixelSSL v0.1.2 is Released!**  
  v0.1.2 supports [PSPNet](https://arxiv.org/abs/1612.01105) and its SSL results for semantic segmentation task (check [here](task/sseg)).
  
  [[More](docs/updates.md)]


## Supported Algorithms and Tasks
We are actively updating this project.  
The SSL algorithms and demo tasks supported by PixelSSL are summarized in the following table: 
| Algorithms / Tasks | [Segmentation](task/sseg) | Other Tasks | 
| :---: | :---: | :---: |
| SupOnly | v0.1.0 | Coming Soon |
| MT [[1]](https://arxiv.org/abs/1703.01780) | v0.1.0 | Coming Soon |
| AdvSSL [[2]](https://arxiv.org/abs/1802.07934) | v0.1.0 | Coming Soon |
| S4L [[3]](https://arxiv.org/abs/1905.03670) | v0.1.1 | Coming Soon | 
| CCT [[4]](https://arxiv.org/abs/2003.09005) | v0.1.3 | Coming Soon |
| GCT [[5]](https://arxiv.org/abs/2008.05258) | v0.1.0 | Coming Soon |
| CutMix [[6]](https://arxiv.org/abs/1906.01916) | v0.1.4 | Coming Soon |


[1] Mean Teachers are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Antti Tarvainen, and Harri Valpola. NeurIPS 2017.

[2] Adversarial Learning for Semi-Supervised Semantic Segmentation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Wei-Chih Hung, Yi-Hsuan Tsai, Yan-Ting Liou, Yen-Yu Lin, and Ming-Hsuan Yang. BMVC 2018.  

[3] S4L: Self-Supervised Semi-Supervised Learning  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. ICCV 2019.  

[4] Semi-Supervised Semantic Segmentation with Cross-Consistency Training  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yassine Ouali, Céline Hudelot, and Myriam Tami. CVPR 2020.

[5] Guided Collaborative Training for Pixel-wise Semi-Supervised Learning  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Zhanghan Ke, Di Qiu, Kaican Li, Qiong Yan, and Rynson W.H. Lau. ECCV 2020.

[6] Semi-Supervised Semantic Segmentation Needs Strong, Varied Perturbations  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Geoff French, Samuli Laine, Timo Aila, Michal Mackiewicz, and Graham Finlayson.  BMVC 2020.

## Installation
Please refer to the [Installation](docs/installation.md) document.  


## Getting Started
Please follow the [Getting Started](docs/getting_started.md) document to run the provided demo tasks.


## Tutorials
We provide the [API](docs/api.md) document and some tutorials for using PixelSSL.
- [Tutorial 1 - Implement A New Pixel-wise Semi-Supervised Algorithm](docs/tutorial/tutorial-1.md)
- [Tutorial 2 - Implement A New Pixel-wise Task Based on the Task Template](docs/tutorial/tutorial-2.md)
- [Tutorial 3 - Dataset Wrappers for Semi-Supervised Learning](docs/tutorial/tutorial-3.md)
- [Tutorial 4 - Support More Optimizers and LRSchedulers](docs/tutorial/tutorial-4.md)


## License
This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
We thank [City University of Hong Kong](https://www.cityu.edu.hk/) and [SenseTime](https://www.sensetime.com/) for their support to this project.


## Citation
This project is extended from our ECCV 2020 paper [Guided Collaborative Training for Pixel-wise Semi-Supervised Learning](https://arxiv.org/abs/2008.05258) (GCT). If this codebase or our method helps your research, please cite:

```bibtex
@InProceedings{ke2020gct,
  author = {Ke, Zhanghan and Qiu, Di and Li, Kaican and Yan, Qiong and Lau, Rynson W.H.},
  title = {Guided Collaborative Training for Pixel-wise Semi-Supervised Learning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  month = {August},
  year = {2020},
}
```

## Contact
This project is currently maintained by Zhanghan Ke ([@ZHKKKe](https://github.com/ZHKKKe)).  
If you have any questions, please feel free to contact `kezhanghan@outlook.com`.
