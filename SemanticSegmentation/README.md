## Useage

+ The installation, data preparation and useage of mmsegmentation please refer to this [repo](https://github.com/open-mmlab/mmsegmentation).

+ For loading annotations of the Potsdam dataset and ignore the background:

    1. please replace the files of the same name under `mmseg/datasets/pipelines` with the file under this [folder](./mmseg/datasets/pipelines).

    2. please replace the configuration of Potsdam dataset in `mmseg/datasets/potsdam.py` with this [file](./mmseg/datasets/potsdam.py).

+ The config are under this [folder](./config).

+ The config fils for other ResNet-50 based SSL methods can be obtained by changing the checkpoint path in `config/CMID-ResNet50_config.py`

+ The additional needed files for ViT-B-RVSA and ViTAE-B-RVSA please refer to this [repo](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/tree/main/Semantic%20Segmentation).

+ The config file for the Vaigingen dataset is the same as the one under the [config](./config) folder, only the dataset name and path need to be changed.