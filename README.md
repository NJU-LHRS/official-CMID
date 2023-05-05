<h1 align="center"> CMID: A Unified Self-Supervised Learning Framework for Remote Sensing Image Understanding </h1> 

<h5 align="center"><em>Dilxat Muhtar, Xueliang Zhang, Pengfeng Xiao, Zhenshi Li, and Feng Gu</em></h5>

<p align="center">
  <a href="#News">News</a> |
  <a href="#Introduction">Introduction</a> |
  <a href="#Pre-trained Models">Pre-trained Models</a> |
  <a href="#Usage">Usage</a>|
  <a href="#Acknowledgement">Acknowledgement</a> |
  <a href="#Statement">Statement</a>
</p >
<p align="center">
<a href="https://arxiv.org/abs/2304.09670"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
<a href="https://ieeexplore.ieee.org/document/10105625"><img src="https://img.shields.io/badge/Paper-IEEE%20TGRS-red"></a>
</p>
## News

+ **\[May 5 2023\]:** The IEEE TGRS published version can be found at [IEEE Xplore](https://ieeexplore.ieee.org/document/10105625?source=authoralert).

+ **\[May 4 2023\]:**  Updated the acknowledgment. Many thanks to [A2MIM](https://github.com/Westlake-AI/A2MIM) and [OpenMixup](https://github.com/Westlake-AI/openmixup) for their awesome implementations of RGB mean input and the Focal Frequency loss!

+ **\[Apr 20 2023]**: The IEEE TGRS early access version can be found at [this website](https://ieeexplore.ieee.org/document/10105625).

+ **\[Apr 19 2023]**: This paper have beed released at [arxiv](https://arxiv.org/abs/2304.09670).

+ **\[Apr 15 2023]**: All of the codes have been released.

+ **[Apr 14 2023]**: This paper has been accepted by IEEE TGRS!

+ **[Jan 11 2023]**: All the pre-trained models and checkpoints of various downstream tasks are released. The code will be uploaded after the paper has been accepted.

## Introduction

This is the official repository for the paper “CMID: A Unified Self-Supervised Learning Framework for Remote Sensing Image Understanding”

**Abstract:** Self-supervised learning (SSL) has gained widespread attention in the remote sensing (RS) and earth observation (EO) communities owing to its ability to learn task-agnostic representations without human-annotated labels. Nevertheless, most existing RS SSL methods are limited to learning either global semantic separable or local spatial perceptible representations. We argue that this learning strategy is suboptimal in the realm of RS, since the required representations for different RS downstream tasks are often varied and complex. In this study, we proposed a unified SSL framework that is better suited for RS images representation learning. The proposed SSL framework, Contrastive Mask Image Distillation (CMID), is capable of learning representations with both global semantic separability and local spatial perceptibility by combining contrastive learning (CL) with masked image modeling (MIM) in a self-distillation way. Furthermore, our CMID learning framework is architecture-agnostic, which is compatible with both convolutional neural networks (CNN) and vision transformers (ViT), allowing CMID to be easily adapted to a variety of deep learning (DL) applications for RS understanding. Comprehensive experiments have been carried out on four downstream tasks (i.e. scene classification, semantic segmentation, object-detection, and change detection) and the results show that models pre-trained using CMID achieve better performance than other state-of-the-art SSL methods on multiple downstream tasks.

<figure>
<div align="center">
<img src=Figure/CMID.png width="90%">
</div>
</figure>


## Pre-trained Models

|    Method    | Backbone  | Pre-trained Dataset | Pre-trained Epochs |                      Pre-trained model                       |                        Backbone Only                         |
| :----------: | :-------: | :-----------------: | :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     CMID     | ResNet-50 |     MillionAID      |        200         | [NJU Box](https://box.nju.edu.cn/f/b29d28b279ab4724913b/?dl=1) | [NJU Box](https://box.nju.edu.cn/f/b2e7ad5d3aea4f569e82/?dl=1) |
|     CMID     |  Swin-B   |     MillionAID      |        200         | [NJU Box](https://box.nju.edu.cn/f/8453a93f652c4c4eb054/?dl=1) | [NJU Box](https://box.nju.edu.cn/f/82a3c7ca7bfc4887aef8/?dl=1) |
|     CMID     | ResNet-50 |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/578b015fc41d4119a2d5/?dl=1) | [NJU Box](https://box.nju.edu.cn/f/d12c2d027d2846119855/?dl=1) |
|     CMID     |  Swin-B   |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/dce6cdc3882a46c8b286/?dl=1) | [NJU Box](https://box.nju.edu.cn/f/92d3d42dce07467f9d8e/?dl=1) |
|     BYOL     | ResNet-50 |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/502b4a0d0f9b4b2d9a1c/?dl=1) |                              \                               |
| Barlow-Twins | ResNet-50 |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/6696b1d415fc4d8080cd/?dl=1) |                              \                               |
|   MoCo-v2    | ResNet-50 |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/f6984aab871949458079/?dl=1) |                              \                               |
|     MAE      |   ViT-B   |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/5d1c95dcacc245509e0b/?dl=1) |                              \                               |
|    SimMIM    |  Swin-B   |       Potsdam       |        400         | [NJU Box](https://box.nju.edu.cn/f/12fa42e0e0a84edbb473/?dl=1) |                              \                               |

### Scene Classification (UCM 8:2)

|      Method       |   Backbone   | Pre-trained Dataset | Pre-trained Epochs |  OA   |                           Weights                            |
| :---------------: | :----------: | :-----------------: | :----------------: | :---: | :----------------------------------------------------------: |
|       CMID        |  ResNet-50   |     MillionAID      |        200         | 99.22 | [NJU Box](https://box.nju.edu.cn/f/6c5cfbff667e45cea624/?dl=1) |
|       CMID        |    Swin-B    |     MillionAID      |        200         | 99.48 | [NJU Box](https://box.nju.edu.cn/f/bcf1f1292a2044bcb10d/?dl=1) |
|       BYOL        |  ResNet-50   |      ImageNet       |        200         | 99.22 | [NJU Box](https://box.nju.edu.cn/f/333c2125b4c645f69e8f/?dl=1) |
|   Barlow-Twins    |  ResNet-50   |      ImageNet       |        300         | 99.16 | [NJU Box](https://box.nju.edu.cn/f/715cb236b97f4af69402/?dl=1) |
|      MoCo-v2      |  ResNet-50   |      ImageNet       |        200         | 97.92 | [NJU Box](https://box.nju.edu.cn/f/20f42aa6fa894d90ac11/?dl=1) |
|       SwAV        |  ResNet-50   |      ImageNet       |        200         | 98.96 | [NJU Box](https://box.nju.edu.cn/f/46bf899af9d840199f7e/?dl=1) |
|       SeCo        |  ResNet-50   |       SeCo-1m       |        200         | 97.66 | [NJU Box](https://box.nju.edu.cn/f/b6aaf1ba169a442abada/?dl=1) |
| ResNet-50-SEN12MS |  ResNet-50   |       SEN12MS       |        200         | 96.88 | [NJU Box](https://box.nju.edu.cn/f/7bf4a152de6d4c129f65/?dl=1) |
|        MAE        |  ViT-B-RVSA  |     MillionAID      |        1600        | 98.56 | [NJU Box](https://box.nju.edu.cn/f/442c73fe6a9a4419b056/?dl=1) |
|        MAE        | ViTAE-B-RVSA |     MillionAID      |        1600        | 97.12 | [NJU Box](https://box.nju.edu.cn/f/cc64099a725948eb99fe/?dl=1) |

### Semantic Segmentation

|      Method       |   Backbone   | Pre-trained Dataset | Pre-trained Epochs | mIoU (Potsdam) |                      Weights (Potsdam)                       | mIoU (VH) |                         Weights (VH)                         |
| :---------------: | :----------: | :-----------------: | :----------------: | :------------: | :----------------------------------------------------------: | :-------: | :----------------------------------------------------------: |
|       CMID        |  ResNet-50   |     MillionAID      |        200         |     87.35      | [NJU Box](https://box.nju.edu.cn/f/753e96b398f84152bd40/?dl=1) |   79.44   | [NJU Box](https://box.nju.edu.cn/f/5e0a62032a7a4c66aae4/?dl=1) |
|       CMID        |    Swin-B    |     MillionAID      |        200         |     88.36      | [NJU Box](https://box.nju.edu.cn/f/cceb17ba88f44b7ea2b1/?dl=1) |   80.01   | [NJU Box](https://box.nju.edu.cn/f/f4cb2a1940594865890c/?dl=1) |
|       BYOL        |  ResNet-50   |      ImageNet       |        200         |     85.54      | [NJU Box](https://box.nju.edu.cn/f/95ac4ce4a9b541d38f02/?dl=1) |   72.52   | [NJU Box](https://box.nju.edu.cn/f/ae6a898cb4ad489faed8/?dl=1) |
|   Barlow-Twins    |  ResNet-50   |      ImageNet       |        300         |     83.16      | [NJU Box](https://box.nju.edu.cn/f/bfb1b6ce68d04ff79f85/?dl=1) |   71.86   | [NJU Box](https://box.nju.edu.cn/f/6c2b4b4178f040d09c2a/?dl=1) |
|      MoCo-v2      |  ResNet-50   |      ImageNet       |        200         |     87.02      | [NJU Box](https://box.nju.edu.cn/f/be8946e4e5f1405e8fda/?dl=1) |   79.16   | [NJU Box](https://box.nju.edu.cn/f/e3562a0c2fcf43a3b063/?dl=1) |
|       SwAV        |  ResNet-50   |      ImageNet       |        200         |     85.74      | [NJU Box](https://box.nju.edu.cn/f/93d58a4ff7934c92947c/?dl=1) |   73.76   | [NJU Box](https://box.nju.edu.cn/f/994fd0346292441fbdd8/?dl=1) |
|       SeCo        |  ResNet-50   |       SeCo-1m       |        200         |     85.82      | [NJU Box](https://box.nju.edu.cn/f/b4ce95a18a904084baed/?dl=1) |   78.59   | [NJU Box](https://box.nju.edu.cn/f/9b06e6841868495bbee2/?dl=1) |
| ResNet-50-SEN12MS |  ResNet-50   |       SEN12MS       |        200         |     83.17      | [NJU Box](https://box.nju.edu.cn/f/f483f9dd03fb496b9f01/?dl=1) |   73.99   | [NJU Box](https://box.nju.edu.cn/f/b071c6dec61d4393b69c/?dl=1) |
|        MAE        |  ViT-B-RVSA  |     MillionAID      |        1600        |     86.37      | [NJU Box](https://box.nju.edu.cn/f/9f41cda6a04b441fb116/?dl=1) |   77.29   | [NJU Box](https://box.nju.edu.cn/f/cf2c7092e38c4504b60d/?dl=1) |
|        MAE        | ViTAE-B-RVSA |     MillionAID      |        1600        |     86.61      | [NJU Box](https://box.nju.edu.cn/f/f99cd7703d11403fa7ac/?dl=1) |   78.17   | [NJU Box](https://box.nju.edu.cn/f/5d96a3ec3cf1497dad35/?dl=1) |

### Object Detection (DOTA V1.0 Dataset)

|    Method    |   Backbone   | Pre-trained Dataset | Pre-trained Epochs |  mAP  |                           Weights                            |
| :----------: | :----------: | :-----------------: | :----------------: | :---: | :----------------------------------------------------------: |
|     CMID     |  ResNet-50   |     MillionAID      |        200         | 76.63 | [NJU Box](https://box.nju.edu.cn/f/ad596092342f42f89da8/?dl=1) |
|     CMID     |    Swin-B    |     MillionAID      |        200         | 77.36 | [NJU Box](https://box.nju.edu.cn/f/56428634781348539763/?dl=1) |
|     BYOL     |  ResNet-50   |      ImageNet       |        200         | 73.62 | [NJU Box](https://box.nju.edu.cn/f/a0c217c9b0a44434986b/?dl=1) |
| Barlow-Twins |  ResNet-50   |      ImageNet       |        300         | 67.54 | [NJU Box](https://box.nju.edu.cn/f/bd780fb7ef414e12b419/?dl=1) |
|   MoCo-v2    |  ResNet-50   |      ImageNet       |        200         | 73.25 | [NJU Box](https://box.nju.edu.cn/f/0eee642365474eb1a73d/?dl=1) |
|     SwAV     |  ResNet-50   |      ImageNet       |        200         | 73.30 | [NJU Box](https://box.nju.edu.cn/f/c5875db670f145498231/?dl=1) |
|     MAE      |  ViT-B-RVSA  |     MillionAID      |        1600        | 78.08 | [NJU Box](https://box.nju.edu.cn/f/19aa5f84c7e945519d8d/?dl=1) |
|     MAE      | ViTAE-B-RVSA |     MillionAID      |        1600        | 76.96 | [NJU Box](https://box.nju.edu.cn/f/618643cf52ff43bbbedd/?dl=1) |

### Change Detection (CDD Dataset)

|      Method       | Backbone  | Pre-trained Dataset | Pre-trained Epochs |  mF1  |                           Weights                            |
| :---------------: | :-------: | :-----------------: | :----------------: | :---: | :----------------------------------------------------------: |
|       CMID        | ResNet-50 |     MillionAID      |        200         | 96.95 | [NJU Box](https://box.nju.edu.cn/f/42bc816b650c49208608/?dl=1) |
|       CMID        |  Swin-B   |     MillionAID      |        200         | 97.11 | [NJU Box](https://box.nju.edu.cn/f/6837337314984975a4c0/?dl=1) |
|       BYOL        | ResNet-50 |      ImageNet       |        200         | 96.30 | [NJU Box](https://box.nju.edu.cn/f/a8e9ee8e88114a60bdaf/?dl=1) |
|   Barlow-Twins    | ResNet-50 |      ImageNet       |        300         | 95.63 | [NJU Box](https://box.nju.edu.cn/f/4845c1e1273b47bc8a58/?dl=1) |
|      MoCo-v2      | ResNet-50 |      ImageNet       |        200         | 96.05 | [NJU Box](https://box.nju.edu.cn/f/1f6c7ab33a7440aba360/?dl=1) |
|       SwAV        | ResNet-50 |      ImageNet       |        200         | 95.89 | [NJU Box](https://box.nju.edu.cn/f/1fdbe3159ec8497786a4/?dl=1) |
|       SeCo        | ResNet-50 |       SeCo-1m       |        200         | 96.26 | [NJU Box](https://box.nju.edu.cn/f/ffd1db53aea94cc1b51a/?dl=1) |
| ResNet-50-SEN12MS | ResNet-50 |       SEN12MS       |        200         | 95.88 | [NJU Box](https://box.nju.edu.cn/f/62f1fd9636db4ee287eb/?dl=1) |

# Usage

+ Details about pre-training from scratch please refer to [pre-training instructions](./Pretrain/README.md).
+ After pre-training (or downloading the pre-trained models), please make sure extract the backbone using [extract.py](./extract.py).
+ Details about fine-tuning on the UCM scene classification task please refer to [classification instructions](./Pretrain/README.md).
+ Details about fine-tuning on semantic segmentation task please refer to [sementic segmentation instructions](./SemanticSegmentation/README.md).
+ Details about fine-tuning on the DOTA OBB detection task please refer to [OBB detection instructions](./Detection/README.md).
+ Details bout fine-tuning on the CDD change detection task please refer to [change detection instructions](./ChangeDetection/README.md).

## Acknowledgement

+ Many thanks to the following repos: [Remote-Sensing-RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)、[iBOT](https://github.com/bytedance/ibot)、[A2MIM](https://github.com/Westlake-AI/A2MIM)、[core-pytorch-utils](https://github.com/serend1p1ty/core-pytorch-utils)、[solo-learn](https://github.com/vturrisi/solo-learn)、[OpenMixup](https://github.com/Westlake-AI/openmixup)、[timm](https://github.com/huggingface/pytorch-image-models)、[Adan](https://github.com/sail-sg/Adan).

## Statement

+ This project is strictly forbidden for any commercial purpose. Any questions please contact [pumpKin-Co](https://github.com/pUmpKin-Co).
