import timm
import torch
import ml_collections
import torch.nn as nn
from typing import Dict
from .swin_utils import resize_pos_embed
from mmseg.models.backbones import ResNet
from .swin_transformer import SwinTransformer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class ClassifyModel(nn.Module):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(ClassifyModel, self).__init__()
        self.embed_dim = config.hidden_dim
        self.num_classes = config.num_classes
        self.backbone = ResNet(depth=50, pretrained=None)
        self.cls_head = nn.Linear(self.embed_dim, self.num_classes)
        self.config = config

    @torch.no_grad()
    def register_criterion(self, mixup_fn):
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif self.config.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.config.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        self.criterion = criterion
        self.mixup_fn = mixup_fn
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def load_pretrain_model(self, state_dict: Dict):
        self.backbone.eval()
        self.backbone.load_state_dict(state_dict)

    @torch.no_grad()
    def extract(self, data: torch.Tensor, device: torch.device = torch.device("cpu")):
        return (self.backbone(data.to(device))[-1])

    def forward(self,
                x: torch.Tensor,
                device: torch.device = torch.device("cpu"),
                return_loss: bool = True,
                eval: bool = False):
        x, target = x
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if self.mixup_fn is not None and not eval:
            x, target = self.mixup_fn(x, target)

        feat = self.avg_pool(self.backbone(x)[-1]).squeeze()
        logit = self.cls_head(feat)

        if return_loss:
            return self.criterion(logit, target)
        else:
            return logit


class SwinClassifyModel(ClassifyModel):

    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 6, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [6, 12, 24, 48]}),
    }  # yapf: disable

    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(SwinClassifyModel, self).__init__(config)
        self.embed_dim = config.hidden_dim
        self.num_classes = config.num_classes
        self.arch_setting = self.arch_zoo["base"]
        self.backbone = SwinTransformer(embed_dims=self.arch_setting["embed_dims"],
                                        depths=self.arch_setting["depths"],
                                        num_heads=self.arch_setting["num_heads"],
                                        out_indices=(0, 1, 2, 3),
                                        drop_path_rate=0.5,
                                        pretrain_img_size=224)
        self.cls_head = nn.Linear(self.embed_dim, self.num_classes)
        self.config = config

    @torch.no_grad()
    def register_criterion(self, mixup_fn):
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif self.config.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.config.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        self.criterion = criterion
        self.mixup_fn = mixup_fn
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def load_pretrain_model(self, state_dict: Dict):
        self.backbone.eval()
        self.backbone.init_weights(pretrained=state_dict)

    @torch.no_grad()
    def get_selfattention(self, data: torch.Tensor, n: int = 1):
        x, hw_shape = self.backbone.patch_embed(data)
        if self.backbone.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.backbone.absolute_pos_embed, self.backbone.patch_resolution, hw_shape,
                self.backbone.interpolate_mode, self.backbone.num_extra_tokens)
        x = self.backbone.drop_after_pos(x)

        if n == 1:
            return self.get_last_selfattention(x, hw_shape)
        else:
            return self.get_all_selfattention(x, hw_shape)

    @torch.no_grad()
    def get_last_selfattention(self, x: torch.Tensor, hw_shape):
        for i, stage in enumerate(self.backbone.stages):
            if i < len(self.backbone.stages) - 1:
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
                x, hw_shape, out, out_hw_shape, attns = stage.forward_with_attention(x, hw_shape)
                return attns[0]

    @torch.no_grad()
    def get_all_selfattention(self, x: torch.Tensor, hw_shape):
        attn_out = []
        for stage in self.backbone.stages:
            x, hw_shape, out, out_hw_shape, attns = stage.forward_with_attention(x, hw_shape)
            attn_out += attns
        return attn_out
    
    @torch.no_grad()
    def extract(self, data: torch.Tensor, device: torch.device = torch.device("cpu")):
        return self.backbone(data.to(device), extract=True)

    def forward(self,
                x: torch.Tensor,
                device: torch.device = torch.device("cpu"),
                return_loss: bool = True,
                eval: bool = False):
        x, target = x
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if self.mixup_fn is not None and not eval:
            x, target = self.mixup_fn(x, target)

        feat = self.avg_pool(self.backbone(x)[-1]).squeeze()
        logit = self.cls_head(feat)

        if return_loss:
            return self.criterion(logit, target)
        else:
            return logit

