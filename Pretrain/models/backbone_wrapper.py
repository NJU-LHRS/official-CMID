import torch
import timm
import torch.nn as nn
from typing import Union, List
from .swin_transformer import SwinTransformer
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init, trunc_normal_


class MaskResNet(nn.Module):
    def __init__(self,
                 mask_layer: int,
                 out_stage: Union[int, List],
                 backbone: str = "resnet50",
                 depth: str = "50",
                 replace=False):
        super(MaskResNet, self).__init__()
        self.mask_stage = mask_layer
        self.out_stage = out_stage
        self.replace = replace
        if isinstance(self.out_stage, int):
            self.out_stage = [self.out_stage]
        self.model = timm.create_model(backbone, pretrained=False, features_only=False, num_classes=0, global_pool="")
        ARCH_DIMS = {
            **dict.fromkeys(
                ['18', '34'],
                [64, 128, 256, 512, ]),
            **dict.fromkeys(
                ['50', '101', '152', ],
                [64, 256, 512, 1024]),
        }
        self.mask_dim = ARCH_DIMS[depth][self.mask_stage]
        self.mask_token = nn.Parameter(torch.zeros(1, self.mask_dim, 1, 1))
        self.layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]

    def forward_mask(self, mim_feature: torch.Tensor, mask: torch.Tensor):
        B, _, H, W = mim_feature.size()
        mask_token = self.mask_token.expand(B, -1, H, W)
        mask = mask.view(B, 1, H, W).type_as(mask_token)
        if self.replace:
            x = mim_feature * (1. - mask) + mask_token * mask
        else:
            x = mim_feature + mask_token * mask
        return x

    def forward(self, mim_image: torch.Tensor, mask: torch.Tensor = None):
        x = self.model.conv1(mim_image)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        outs = []
        if -1 in self.out_stage:
            outs.append(x)

        for i, layer in enumerate(self.layers):
            # mask, add mask token
            if self.mask_stage == i and mask is not None:
                x = self.forward_mask(x, mask)

            x = layer(x)
            if i + 1 in self.out_stage:
                outs.append(x)
                if len(self.out_stage) == 1:
                    return outs
        return outs


class MaskSwin(nn.Module):
    def __init__(self,
                 mask_layer: int = 0,
                 replace: bool = False,
                 **kwargs):
        super(MaskSwin, self).__init__()
        self.mask_layer = mask_layer
        self.replace = replace
        assert self.mask_layer in [0, 1, 2, 3, 4]
        self.model = SwinTransformer(arch="base",
                                     drop_rate=0.,
                                     drop_path_rate=0.,
                                     **kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dims * (2 ** max(0, self.mask_layer - 1))))

    def init_weights(self, pretrained=None):
        self.model.init_weights(pretrained)

        if pretrained is not None:
            if self.model.use_abs_pos_embed:
                trunc_normal_(self.model.absolute_pos_embed, std=0.02)

            trunc_normal_(self.mask_token, std=0.02)

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0)

    def forward_mask(self, mim_feature: torch.Tensor, mask: torch.Tensor = None):
        assert mask is not None
        B, L, _ = mim_feature.shape
        mask_token = self.mask_token.expand(B, L, -1)
        mask = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        if self.replace:
            mim_feature = mim_feature * (1. - mask) + mask_token * mask
        else:
            mim_feature = mim_feature + mask_token * mask  # residual
        return mim_feature

    def forward(self, mim_image: torch.Tensor, mask: torch.Tensor = None):
        x, hw_shape = self.model.patch_embed(mim_image)

        if self.mask_layer == 0 and mask is not None:
            x = self.forward_mask(x, mask)

        if self.model.use_abs_pos_embed:
            x = x + self.model.absolute_pos_embed
        x = self.model.drop_after_pos(x)

        outs = []
        if -1 in self.model.out_indices:
            outs.append(
                x.view(x.size(0), *hw_shape, -1).permute(0, 3, 1, 2).contiguous())

        for i, stage in enumerate(self.model.stages):
            if self.mask_layer == i + 1:
                x = self.forward_mask(x, mask)

            x, hw_shape = stage(x, hw_shape)
            if i in self.model.out_indices:
                norm_layer = getattr(self.model, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs
