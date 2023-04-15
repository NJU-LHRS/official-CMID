import torch.nn as nn
from .vit_rvsa import ViT_Win_RVSA
from .viate_rvsa import ViTAE_NC_Win_RVSA
from .clas_model import ClassifyModel, SwinClassifyModel
from .pretrain_model import MoCoBased, BYOLBased, MoCoBasedDDP


def build_pretrain_model(config) -> nn.Module:
    if config.base_model == "moco":
        if config.is_distribute:
            model = MoCoBasedDDP(config)
        else:
            model = MoCoBased(config)
    elif config.base_model == "byol":
        model = BYOLBased(config)
    else:
        raise NotImplementedError
    return model


def build_model(config, is_pretrain=True) -> nn.Module:
    if is_pretrain:
        model = build_pretrain_model(config)
    else:
        if config.backbone == "resnet50":
            model = ClassifyModel(config)
        elif config.backbone == "swin":
            model = SwinClassifyModel(config)
        elif config.backbone == "vitae":
            model = ViTAE_NC_Win_RVSA(num_classes=config.num_classes, drop_rate=0.1)  # official setting
        elif config.backbone == "vit":
            model = ViT_Win_RVSA(num_classes=config.num_classes, drop_rate=0.1)
        else:
            raise NotImplementedError

    return model