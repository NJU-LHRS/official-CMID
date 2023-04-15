import os
import torch
import functools
import numpy as np
import torchvision
import torch.nn as nn
from collections import abc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from inspect import getfullargspec
from torch.cuda.amp import autocast
from mmcv.utils import TORCH_VERSION, digit_version
from .loss_utils import RegressionLoss, FocalFrequencyLoss


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
    torch.cuda.amp is used as the backend, otherwise, original mmcv
    implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=False):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


class PlotTensor:
    """Plot torch tensor as matplotlib figure.

    Args:
        apply_inv (bool): Whether to apply inverse normalization.
    """

    def __init__(self, apply_inv=True) -> None:
        trans_cifar = [
            torchvision.transforms.Normalize(
                mean=[0., 0., 0.], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.201]),
            torchvision.transforms.Normalize(
                mean=[-0.4914, -0.4822, -0.4465], std=[1., 1., 1.])]
        trans_in = [
            torchvision.transforms.Normalize(
                mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]
        if apply_inv:
            self.invTrans_cifar = torchvision.transforms.Compose(trans_cifar)
            self.invTrans_in = torchvision.transforms.Compose(trans_in)

    def plot(self,
             img, nrow=4, title_name=None, save_name=None,
             dpi=None, apply_inv=True, overwrite=False):
        assert save_name is not None
        assert img.size(0) % nrow == 0
        ncol = img.size(0) // nrow
        if ncol > nrow:
            ncol = nrow
            nrow = img.size(0) // ncol
        img_grid = torchvision.utils.make_grid(img, nrow=nrow, pad_value=0)

        cmap = None
        if img.size(1) == 1:
            cmap = plt.cm.gray
        if apply_inv:
            if img.size(2) <= 64:
                img_grid = self.invTrans_cifar(img_grid)
            else:
                img_grid = self.invTrans_in(img_grid)
        img_grid = torch.clip(img_grid * 255, 0, 255).int()
        img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure(figsize=(nrow * 2, ncol * 2))
        plt.imshow(img_grid, cmap=cmap)
        if title_name is not None:
            plt.title(title_name)
        if not os.path.exists(save_name) or overwrite:
            plt.savefig(save_name, dpi=dpi)
        plt.close()


class MIMHead(nn.Module):
    def __init__(self,
                 in_channels: int = 2048,
                 in_chans: int = 3,
                 kernel_size: int = 1,
                 encoder_stride: int = 32,
                 ):
        super(MIMHead, self).__init__()

        self.decoder_pred = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride ** 2 * in_chans,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, feature: torch.Tensor):
        return self.decoder_pred(feature[-1])


class MIMLossHead(nn.Module):
    """Head for A2MIM training.

    Args:
        loss (dict): Config of regression loss.
        encoder_in_channels (int): Number of input channels for encoder.
        unmask_weight (float): Loss weight for unmasked patches.
        fft_weight (float): Loss weight for the fft prediction loss. Default to 0.
        fft_focal (bool): Whether to adopt the focal fft loss. Default to False.
        fft_unmask_replace (str): Mode to replace (detach) unmask patches for the fft
            loss, in {None, 'target', 'prediction', 'mean', 'mixed',}.
        fft_unmask_weight (float): Loss weight to caculate the fft loss on unmask
            tokens. Default to 0.
    """

    def __init__(self,
                 loss=dict(
                     loss_weight=1.0, mode="l1_loss"),
                 encoder_in_channels=3,
                 unmask_weight=0,
                 fft_weight=0,
                 fft_focal=False,
                 fft_unmask_replace=None,
                 fft_unmask_weight=0,
                 ):
        super(MIMLossHead, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.unmask_weight = unmask_weight
        self.fft_weight = fft_weight
        self.fft_focal = fft_focal
        self.fft_unmask_weight = fft_unmask_weight
        self.fft_unmask_replace = fft_unmask_replace
        assert fft_unmask_replace in [None, 'target', 'prediction', 'mean', 'mixed', ]
        assert 0 <= unmask_weight <= 1 and 0 <= fft_unmask_weight <= 1
        assert loss is None or isinstance(loss, dict)

        self.criterion = RegressionLoss(**loss)
        # fft loss
        if fft_focal:
            fft_loss = dict(
                loss_weight=1.0, alpha=1.0,
                ave_spectrum=True, log_matrix=True, batch_matrix=True)
            self.fft_loss = FocalFrequencyLoss(**fft_loss)

    def forward(self, x, x_rec, mask):
        # upsampling mask
        scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)  # 下采样的stride
        if scale_h > 1:
            mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                int(scale_w), 2).unsqueeze(1).contiguous()
        else:
            mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                 scale_factor=(scale_h, scale_w), mode="nearest")

        # spatial loss
        if self.unmask_weight > 0.:
            # reweight unmasked patches
            mask_s = mask.clone()
            mask_s = mask_s + (1. - mask_s) * self.unmask_weight
        else:
            mask_s = mask
        loss_rec = self.criterion(x_rec, target=x, reduction_override='none')
        loss_rec = (loss_rec * mask_s).sum() / (mask_s.sum() + 1e-5) / self.encoder_in_channels

        # fourier domain loss
        if self.fft_weight > 0:
            # replace unmask patches (with detach)
            x_replace = None
            if self.fft_unmask_replace is not None:
                if self.fft_unmask_replace == 'target':
                    x_replace = x.clone()
                elif self.fft_unmask_replace == 'prediction':
                    x_replace = x_rec.clone().detach()
                elif self.fft_unmask_replace == 'mean':
                    x_replace = x.mean(dim=[2, 3], keepdim=True).expand(x.size())
                elif self.fft_unmask_replace == 'mixed':
                    x_replace = 0.5 * x_rec.clone().detach() + 0.5 * x.clone()
            if self.fft_unmask_weight < 1:
                mask_f = mask.clone()
                mask_f = mask_f + (1. - mask_f) * self.fft_unmask_weight
                x_rec = (x_rec * mask_f) + (x_replace * (1. - mask_f))  # replace unmask tokens

            # apply fft loss
            if self.fft_focal:
                loss_fft = self.fft_loss(x_rec, x)
                loss_rec += self.fft_weight * loss_fft

        return loss_rec