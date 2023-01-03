## 
## Usage

### Installation

The code framework is mainly borrowed from the SNUNet, while we use BIT in the network part. Thus

Please refer to [BIT-Readme.md](https://github.com/justchenhao/BIT_CD/blob/master/README.md) for installing main packeges such as python, pytorch, etc.

Please refer to [SNUNet-Readme.md](https://github.com/RSCD-Lab/Siam-NestedUNet/blob/master/README.md) for other required packages.

### Data Preparation

- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)

```
│─train
│   ├─A
│   ├─B
│   └─OUT
│─val
│   ├─A
│   ├─B
│   └─OUT
└─test
    ├─A
    ├─B
    └─OUT
```
A: t1 image

B: t2 image

OUT: Binary label map

+ Please change the dataset path in `line 61 of train.py`

### Training

Training the BiT with CMID-ResNet-50 backbone on CDD dataset: 

+ First, Change the path of pre-trained backbone in `line 252 of models/resnet.py`. Then:

```shell
python train.py \
--backbone "resnet" --dataset "cdd" --mode "proposed"
```

Training the BiT with CMID-Swin-B backbone on CDD dataset: 

~~~shell
python train.py \ 
--backbone 'swin_base' --dataset "cdd" --mode "swin_base" --path [Pre-trained CMID-Swin-B Path]
~~~

### Inference

Evaluation using CMID pre-trained models on CDD dataset

```
python eval.py \
--backbone 'resnet' --dataset 'cdd' --mode 'proposed' \
--path [model path]

python eval.py \
--backbone 'swin_base' --dataset 'cdd' --mode 'proposed' \
--path [model path]
```

## References

The codes are mainly borrowed from this [repo](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Change%20Detection)

