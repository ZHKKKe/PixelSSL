<div align="center">
  <img src="docs/img/pixelssl-logo.png" width="650"/>
</div>

---

PixelSSL is a PyTorch-based semi-supervised learning (SSL) codebase for pixel-wise (Pixel) vision tasks.

The purpose of this project is to promote the research and application of semi-supervised learning on pixel-wise vision tasks. PixelSSL provides two major features:
- Interface for implementing new semi-supervised algorithms
- Template for encapsulating diverse computer vision tasks

As a result, the SSL algorithms integrated in PixelSSL are compatible with all task codes inherited from the given template. 

This project includes the code of our ECCV 2020 paper [Guided Collaborative Training for Pixel-wise Semi-Supervised Learning](https://arxiv.org/abs/2008.05258) (GCT).

**News**:
- **[Aug 13 2019] PixelSSL v0.1.0 is Published!**  
  v0.1.0 supports supervised-only learning, three semi-supervised learning algorithms 
  ([MT](https://arxiv.org/abs/1703.01780), 
  [AdvSSL](https://arxiv.org/abs/1802.07934), 
  [GCT](https://arxiv.org/abs/2008.05258)) 
  and one example task (semantic segmentation).

  [[More](docs/updates.md)]


## Supported Algorithms and Tasks
We are actively updating this project.  
The algorithms and demo tasks supported by PixelSSL are summarized in the following table: 
| SSL/Tasks | [Segmentation](task/sseg) | Other Tasks | 
| :---: | :---: | :---: |
| SupOnly | v0.1.0 | Coming Soon |
| [MT](https://arxiv.org/abs/1703.01780) | v0.1.0 | Coming Soon |
| S4L | Coming Soon | Coming Soon | 
| [AdvSSL](https://arxiv.org/abs/1802.07934) | v0.1.0 | Coming Soon |
| [GCT](https://arxiv.org/abs/2008.05258) | v0.1.0 | Coming Soon |


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
