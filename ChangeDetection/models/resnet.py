import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os

import torchvision
torchvision.models.resnext50_32x4d()

from mmcv.cnn import (constant_init, kaiming_init)

#from ..backbones.custom_load import load_checkpoint
#from mmcv.utils.registry import BACKBONES

import warnings
from collections import OrderedDict
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer

from torch.nn.modules.batchnorm import _BatchNorm

#from mmcv.runner import load_checkpoint

__model_file = {
    18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_cfg=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        #self.bn1 = norm_layer(width)
        _, self.bn1 = build_norm_layer(norm_cfg, width, postfix=1)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        #self.bn2 = norm_layer(width)
        _, self.bn2 = build_norm_layer(norm_cfg, width, postfix=2)
        self.conv3 = conv1x1(width, planes * self.expansion)
        #self.bn3 = norm_layer(planes * self.expansion)
        _, self.bn3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args, block=Bottleneck, layers=[3,4,6,3], num_classes=51, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=None, init_cfg=None, norm_cfg=dict(type='BN', requires_grad=True)
                 ,frozen_stages=-1,norm_eval=False):

        # assert not (init_cfg and pretrained), \
        #     'init_cfg and pretrained cannot be setting at the same time'
        # if isinstance(pretrained, str):
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated, '
        #                   'please use "init_cfg" instead')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # elif pretrained is None:
        #     self.init_cfg = init_cfg
        # else:
        #     raise TypeError('pretrained must be a str or None')

        super(ResNet, self).__init__()

        self.args = args

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        _, self.bn1 = build_norm_layer(
                norm_cfg, self.inplanes, postfix=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_cfg=norm_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       norm_cfg=norm_cfg)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       norm_cfg=norm_cfg)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       norm_cfg=norm_cfg)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, norm_cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, norm_cfg=norm_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         # if self.deep_stem:
    #         #     self.stem.eval()
    #         #     for param in self.stem.parameters():
    #         #         param.requires_grad = False
    #         # else:
    #         self.bn1.eval()
    #         for m in [self.conv1, self.bn1]:
    #             for param in m.parameters():
    #                 param.requires_grad = False

    #     for i in range(1, self.frozen_stages + 1):
    #         m = getattr(self, f'layer{i}')
    #         m.eval()
    #         for param in m.parameters():
    #             param.requires_grad = False


    def init_weights(self):

        if self.args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
        elif self.args.mode == 'sen12ms':
            pretrained = r'F:\pumpkinCode\mmseg\pretrain_checkpoint\sen12ms-res50.pth'
        elif self.args.mode == 'mocov2':
            pretrained = r'F:\pumpkinCode\mmseg\pretrain_checkpoint\org_mocov2_res50_200ep.pth'
        elif self.args.mode == 'seco':
            pretrained = r'F:\pumpkinCode\mmseg\pretrain_checkpoint\seco1m_res50_200ep.pth'
        elif self.args.mode == "proposed":
            pretrained = r"F:\pumpkinCode\mmseg\pretrain_checkpoint\proposed_millionaid_200ep.pth"
        elif self.args.mode == "byol":
            pretrained = r"F:\pumpkinCode\mmseg\pretrain_checkpoint\org_byol_res50_200ep.pth"
        elif self.args.mode == "swav":
            pretrained = r"F:\pumpkinCode\mmseg\pretrain_checkpoint\org_swav_res50_200ep.pth"
        elif self.args.mode == "barlowtwins":
            pretrained = r"F:\pumpkinCode\mmseg\pretrain_checkpoint\org_barlowtwins_res50_300ep.pth"
        else:
            raise NotImplementedError


        if isinstance(pretrained, str):
            
            ckpt = torch.load(pretrained, map_location='cpu')
            
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}


#            print('##############')

#            print(self.state_dict().keys())

#            print('$$$$$$$$$$$$$$')

#            print(state_dict.keys())

            msg = self.load_state_dict(state_dict, False)

            print('load {}-{} weight: {}'.format(self.args.backbone, self.args.mode, pretrained))

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return [x1,x2,x3,x4]

    # def train(self, mode=True):
    #     """Convert the model into training mode while keep normalization layer
    #     freezed"""
    #     super(Our_ResNet, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = Our_ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


# def resnet18(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)


# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)


# def resnet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
#                    **kwargs)


# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)


# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)