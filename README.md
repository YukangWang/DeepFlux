# DeepFlux for Skeletons in the Wild

## Introduction

The code and trained models of: 

DeepFlux for Skeletons in the Wild, CVPR 2019 [[Paper]](https://arxiv.org/abs/1811.12608)

## Citation

Please cite the related works in your publications if it helps your research:

```

@article{wang2018deepflux,
  title={DeepFlux for Skeletons in the Wild},
  author={Wang, Yukang and Xu, Yongchao and Tsogkas, Stavros and Bai, Xiang and Dickinson, Sven and Siddiqi, Kaleem},
  journal={arXiv preprint arXiv:1811.12608},
  year={2018}
}

```

## Prerequisite

* Caffe and VGG-16 pretrained model [[VGG_ILSVRC_16_layers.caffemodel]](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)

* Datasets: [[SK-LARGE]](https://drive.google.com/file/d/1eBIjpzU0kttcKEesRJ2y29_yMcc03sbh/view?usp=sharing), [[SYM-PASCAL]](https://drive.google.com/file/d/1PSfksp7X9fhF0xZ9jOaMb0f4eUs8ED9j/view?usp=sharing)

* OpenCV 3.4.3

* MATLAB

## Usage

#### 1. Install Caffe

```bash

cp Makefile.config.example Makefile.config
# adjust Makefile.config (for example, enable python layer)
make all -j16
# make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
make pycaffe

```
Please refer to [Caffe Installation](http://caffe.berkeleyvision.org/install_apt.html) to ensure other dependencies.

#### 2. Data and model preparation

```bash

# download datasets and pretrained model then
mkdir data && mv [your_dataset_folder] data/
mkdir models && mv [your_pretrained_model] models/
# data augmentation
cd data/[your_dataset_folder]
matlab -nodisplay -r "run augmentation.m; exit"

```  

#### 3. Training scripts
  

```bash

# an example on SK-LARGE dataset
cd examples/DeepFlux/
python train.py --gpu [your_gpu_id] --dataset sklarge --initmodel ../../models/VGG_ILSVRC_16_layers.caffemodel

```

#### 4. Evaluation scripts

```bash

# an example on SK-LARGE dataset
cd evaluation/
./eval.sh ../../data/SK-LARGE/images/test ../../data/SK-LARGE/groundTruth/test ../../models/sklarge_iter_40000.caffemodel

```

## Results and Trained Models

#### SK-LARGE

| Backbone | F-measure | Comment & Link |
|:-------------:|:-------------:|:-----:|
| VGG-16 | 0.732 | CVPR submission [[Google drive]](https://drive.google.com/file/d/1dYtxLqNgNRCTVnkL_2wzIvjWuRk363FM/view?usp=sharing) |
| VGG-16 | 0.735 | different_lr [[Available soon]]() |
| ResNet-101 | 0.752 | different_lr [[Available soon]]() |

#### SYM-PASCAL

| Backbone | F-measure | Comment & Link |
|:-------------:|:-------------:|:-----:|
| VGG-16 | 0.502 | CVPR submission [[Google drive]](https://drive.google.com/file/d/1jtWk_7Vt-Gb8IrW3eaR7EuW8_uitgKE1/view?usp=sharing) |
| VGG-16 | 0.558 | different_lr [[Available soon]]() |
| ResNet-101 | 0.584 | different_lr [[Available soon]]() |

>*different_lr means different learning rates for backbone and additional layers

>*lambda=0.4, k1=3, k2=4 for all models
