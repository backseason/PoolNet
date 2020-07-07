## A Simple Pooling-Based Design for Real-Time Salient Object Detection

### This is a PyTorch implementation of our CVPR 2019 [paper](https://arxiv.org/abs/1904.09569).

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)

## Update

1. We released our code for joint training with edge, which is also our best performance model.
2. You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).


## Usage

### 1. Clone the repository

```shell
git clone https://github.com/backseason/PoolNet.git
cd PoolNet/
```

### 2. Download the datasets

Download the following datasets and unzip them into `data` folder.

* [MSRA-B and HKU-IS](https://drive.google.com/open?id=14RA-qr7JxU6iljLv6PbWUCQG0AJsEgmd) dataset. The .lst file for training is `data/msrab_hkuis/msrab_hkuis_train_no_small.lst`.
* [DUTS](https://drive.google.com/open?id=1immMDAPC9Eb2KCtGi6AdfvXvQJnSkHHo) dataset. The .lst file for training is `data/DUTS/DUTS-TR/train_pair.lst`.
* [BSDS-PASCAL](https://drive.google.com/open?id=1qx8eyDNAewAAc6hlYHx3B9LXvEGSIqQp) dataset. The .lst file for training is `./data/HED-BSDS_PASCAL/bsds_pascal_train_pair_r_val_r_small.lst`.
* [Datasets for testing](https://drive.google.com/open?id=1eB-59cMrYnhmMrz7hLWQ7mIssRaD-f4o).

### 3. Download the pre-trained models for backbone

Download the following pre-trained models [GoogleDrive](https://drive.google.com/open?id=1Q2Fg2KZV8AzNdWNjNgcavffKJBChdBgy) | [BaiduYun](https://pan.baidu.com/s/1ehZheaqeU3pyvYQfRU9c6A) (pwd: **27p5**) into `dataset/pretrained` folder. 

### 4. Train

1. Set the `--train_root` and `--train_list` path in `train.sh` correctly.

2. We demo using ResNet-50 as network backbone and train with a initial lr of 5e-5 for 24 epoches, which is divided by 10 after 15 epochs.
```shell
./train.sh
```
3. We demo joint training with edge using ResNet-50 as network backbone and train with a initial lr of 5e-5 for 11 epoches, which is divided by 10 after 8 epochs. Each epoch runs for 30000 iters.
```shell
./joint_train.sh
```
4. After training the result model will be stored under `results/run-*` folder.

### 5. Test

For single dataset testing: `*` changes accordingly and `--sal_mode` indicates different datasets (details can be found in `main.py`)
```shell
python main.py --mode='test' --model='results/run-*/models/final.pth' --test_fold='results/run-*-sal-e' --sal_mode='e'
```
For all datasets testing used in our paper: `2` indicates the gpu to use
```shell
./forward.sh 2 main.py results/run-*
```
For joint training, to get salient object detection results use
```shell
./forward.sh 2 joint_main.py results/run-*
```
to get edge detection results use
```shell
./forward_edge.sh 2 joint_main.py results/run-*
```

All results saliency maps will be stored under `results/run-*-sal-*` folders in .png formats.


### 6. Pre-trained models, pre-computed results and evaluation results

We provide the pre-trained model, pre-computed saliency maps and evaluation results for:
1. PoolNet-ResNet50 w/o edge model [GoogleDrive](https://drive.google.com/open?id=12Zgth_CP_kZPdXwnBJOu4gcTyVgV2Nof) | [BaiduYun](https://pan.baidu.com/s/1m3BXHZt5PJO5lEdWF0MqBA ) (pwd: **2uln**).
2. PoolNet-ResNet50 w/ edge model (best performance) [GoogleDrive](https://drive.google.com/open?id=1sH5RKEt6SnG33Z4sI-hfLs2d21GmegwR) | [BaiduYun](https://pan.baidu.com/s/10AXBYc_YY3FYcEbCWX6f-A) (pwd: **ksii**).
3. PoolNet-VGG16 w/ edge model (pre-computed maps) [GoogleDrive](https://drive.google.com/open?id=1jbNyNUJFZPb_jhwkm_D70gsxXgbbv_S1) | [BaiduYun](https://pan.baidu.com/s/1gcl-BVwn1YZpaOV3XNxeBQ) (pwd: **3wgc**).

Note：

1. only support `bath_size=1`
2. Except for the backbone we do not use BN layer.

### 7. Contact
If you have any questions, feel free to contact me via: `j04.liu(at)gmail.com`.


### If you think this work is helpful, please cite
```latex
@inproceedings{Liu2019PoolSal,
  title={A Simple Pooling-Based Design for Real-Time Salient Object Detection},
  author={Jiang-Jiang Liu and Qibin Hou and Ming-Ming Cheng and Jiashi Feng and Jianmin Jiang},
  booktitle={IEEE CVPR},
  year={2019},
}
```

Thanks to [DSS](https://github.com/Andrew-Qibin/DSS) and [DSS-pytorch](https://github.com/AceCoooool/DSS-pytorch).
