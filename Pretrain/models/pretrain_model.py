import timm
import math
import torch
import collections
import torch.nn as nn
import ml_collections
from itertools import repeat
from einops import rearrange
import torch.nn.functional as F
import torch.distributed as dist
from .swin_transformer import SwinTransformer
from Exploring import get_world_size, get_rank
from typing import List, Dict, Union, Tuple,Any
from .backbone_wrapper import MaskResNet, MaskSwin
from utils import accuracy_at_k, segmentation_accuracy
from .loss_utils import linear_loss, mocov2plus_loss_func, LocalLoss
from .mim_utils import MIMHead, MIMLossHead, force_fp32, PlotTensor
from .contrastive_utils import MLP, concat_all_gather, LocalHead, position_match


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module) -> None:
    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
                self.final_tau
                - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )


class PretrainModel(nn.Module):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(PretrainModel, self).__init__()

        self.ft_classify = config.ft_classify
        self.multi_label = config.dataset.lower() == "bigearthnet"
        self.kernel_size = config.kernel_size
        self.mim_head_in_channels = config.in_channels
        self.mim_in_chans = config.in_chans
        self.out_stride = config.out_stride
        self.segmentation = config.segmentation
        self.residual = config.residual
        self.save = False
        self.save_mask = config.mask_color == "zero"
        self.save_name = "reconstruct"
        if config.backbone == "swin":
            self.online_encoder = MaskSwin(config.mask_stage, replace=config.residual)
            self.branch_encoder = SwinTransformer(arch="base", drop_rate=0., drop_path_rate=0.)
            self.mim2org = MIMHead(in_channels=1024)
        else:
            assert config.backbone in timm.list_models("*"), f"Backbone {config.backbone} is not supported"
            self.online_encoder = MaskResNet(config.mask_stage,
                                             config.out_stage,
                                             config.backbone,
                                             depth="50",
                                             replace=config.residual)

            self.branch_encoder = timm.create_model(config.backbone, features_only=False, pretrained=False, num_classes=0,
                                                    global_pool="")
            self.mim2org = MIMHead()

        loss_dict = dict(
            loss_weight=config.loss_weight,
            reduction=config.reduction,
            activate=config.activate,
            alpha=config.alpha,
            focal_gamma=config.focal_gamma,
            residual=config.residual
        )

        fft_loss_dict = dict(
            fft_weight=config.fft_weight,
            fft_focal=config.fft_focal,
            fft_unmask_weight=config.fft_unmask_weight,
            fft_unmask_replace=config.fft_unmask_replace,
        )

        self.mim_loss = MIMLossHead(loss=loss_dict,
                                    encoder_in_channels=config.encoder_in_channels,
                                    unmask_weight=config.unmask_weight,
                                    **fft_loss_dict)

        if self.ft_classify:
            self.classifier = nn.Linear(config.in_channels, config.num_classes)
            self.cls_avg_pool = nn.AdaptiveAvgPool2d(1)
            if self.multi_label:
                self.cls_metrics = None
        elif self.segmentation:
            self.bn = nn.BatchNorm2d(config.in_channels)
            self.conv_seg = nn.Conv2d(config.in_channels, config.num_classes, kernel_size=1)
            self.loss_seg = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
            self.seg_ignore_index = config.ignore_index

        self.ema = MomentumUpdater(config.base_momentum, config.final_momentum)
        self.ploter = PlotTensor(apply_inv=True)

    @force_fp32(apply_to=('img', 'img_mim', 'img_rec', 'mask',))
    def plot_reconstruction(self, img, img_mim, img_rec, mask=None):
        """ visualize reconstruction results """
        nrow = 4
        img_mim = img_mim[:nrow]
        img_rec = img_rec[:nrow]
        img = img[:nrow]
        plot_args = dict(dpi=None, apply_inv=True)

        if mask is not None:
            mask = 1. - mask[:4].unsqueeze(1).type_as(img_mim)
            mask = F.interpolate(mask, scale_factor=img_mim.size(2) / mask.size(2), mode="nearest")
            img_mim = img_mim * mask

        img = torch.cat((img, img_mim, img_rec), dim=0)

        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=nrow, title_name="MIM", save_name=self.save_name, **plot_args)

    @force_fp32(apply_to=('img', 'img_mim', 'img_rec', 'mask',))
    def plot_residual_reconstruction(self, img, img_mim, img_rec, mask):
        """ visualize residual reconstruction results """
        nrow = 4
        img_mim = img_mim[:nrow]
        img_rec = img_rec[:nrow]
        img = img[:nrow]
        plot_args = dict(dpi=None, apply_inv=True)

        mask = mask[:4].unsqueeze(1).type_as(img_rec)
        mask = F.interpolate(mask, scale_factor=img_rec.size(2) / mask.size(2), mode="nearest")
        img_rec = img_rec * mask + img * (1 - mask)

        img = torch.cat((img, img_mim, img_rec), dim=0)

        assert self.save_name.find(".png") != -1
        self.ploter.plot(
            img, nrow=nrow, title_name="MIM", save_name=self.save_name, **plot_args)

    @property
    def momentum_pairs(self):
        return [
            (self.online_encoder.model, self.branch_encoder),
        ]

    def init_momentum_pairs(self) -> None:
        for momentum_pairs in self.momentum_pairs:
            initialize_momentum_params(momentum_pairs[0], momentum_pairs[1])

    def eval_forward(self, img: torch.Tensor, device=torch.device("cpu")):
        img = img.to(device)
        backbone_embedding = self.online_encoder(img)
        return backbone_embedding

    @torch.no_grad()
    def momentum_forward(self, cl_image: torch.Tensor):
        view_embedding = self.branch_encoder(cl_image)
        if not isinstance(view_embedding, list):
            view_embedding = [view_embedding]
        return view_embedding

    def mim_forward(self, imgBatch: Union[Dict, List], device=torch.device("cpu")) -> Dict:
        if isinstance(imgBatch, List):
            imgDict, targets = imgBatch
        else:
            imgDict = imgBatch

        org_image, mim_image, cl_image, mim_mask, locations, mim_view_image, view_mask = imgDict["img"].to(device, non_blocking=True), \
                                                                                        imgDict["mim_img"].to(device, non_blocking=True), \
                                                                                        imgDict["view"].to(device, non_blocking=True), \
                                                                                        imgDict["mask"].to(device, non_blocking=True), \
                                                                                        imgDict["locations"], \
                                                                                        imgDict["view_mim_img"].to(device, non_blocking=True), \
                                                                                        imgDict["view_mask"].to(device, non_blocking=True)
            # forward backbone
        locations = [loc.to(device, non_blocking=True) for loc in locations]
        mask_embedding_org = self.online_encoder(mim_image, mim_mask)
        mask_embedding_view = self.online_encoder(mim_view_image, view_mask)

        img_rec_org = self.mim2org(mask_embedding_org)
        loss_rec = self.mim_loss(org_image, img_rec_org, mim_mask)

        img_rec_view = self.mim2org(mask_embedding_view)
        loss_rec += self.mim_loss(cl_image, img_rec_view, view_mask)

        loss_rec /= 2

        with torch.no_grad():
            view_embedding_view = self.momentum_forward(cl_image)
            view_embedding_org = self.momentum_forward(org_image)

        if self.save:
            if not self.residual:
                self.plot_residual_reconstruction(org_image, mim_image, img_rec_org, mim_mask)
            elif self.save_mask:
                self.plot_reconstruction(org_image, mim_image, img_rec_org, mim_mask)
            else:
                self.plot_reconstruction(org_image, mim_image, img_rec_org)

        out = dict(loss_rec=loss_rec,
                   mask_embedding=[mask_embedding_org, mask_embedding_view],
                   view_embedding=[view_embedding_org, view_embedding_view],
                   locations=locations)

        # classifer loss
        if self.ft_classify:
            targets = targets.to(device)
            logits = self.classifier(self.cls_avg_pool(mask_embedding_org[-1]).squeeze().detach())
            if self.multi_label:
                classify_loss = F.multilabel_soft_margin_loss(logits, targets)
                acc = self.cls_metrics(torch.sigmoid(logits), targets) * 100
                out.update(acc=acc)
            else:
                top_k_max = min(1, logits.size(1))
                classify_loss = F.cross_entropy(logits, targets, ignore_index=-1)
                acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
                out.update(acc1=acc1, acc5=acc5)
            out.update(classify_loss=classify_loss)
        elif self.segmentation:
            targets = targets.to(device)
            feat = self.bn(mask_embedding_org[-1].detach())
            logits = self.conv_seg(feat)
            if logits.shape[2:] != targets.shape[1:]:
                logits = F.interpolate(logits,
                                       size=targets.shape[1:],
                                       mode="bilinear")
            seg_loss = self.loss_seg(logits, targets)
            acc = segmentation_accuracy(logits, targets, ignore_index=self.seg_ignore_index)
            out.update(acc=acc, seg_loss=seg_loss)

        return out


class MoCoBased(PretrainModel):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(MoCoBased, self).__init__(config)
        self.temperature = config.temperature
        self.queue_size = config.queue_size
        self.num_matches = config.num_matches
        self.global_branch = config.global_branch
        self.model_warmup = False

        self.online_projector = MLP(
            in_dim=config.in_channels,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            norm_layer=nn.Identity)

        self.branch_projector = MLP(
            in_dim=config.in_channels,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            norm_layer=nn.Identity)

        self.class_projector = LocalHead(
            in_dim=config.in_channels,
            use_bn=config.use_bn,
            out_dim=config.class_feat_size,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
        )

        self.branch_class_projector = LocalHead(
            in_dim=config.in_channels,
            use_bn=config.use_bn,
            out_dim=config.class_feat_size,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
        )

        self.class_loss = LocalLoss(
            class_feat_size=config.class_feat_size,
            center_momentum=config.center_momentum,
            warmup_branch_temp=config.warmup_branch_temp,
            warmup_branch_temp_epochs=config.warmup_branch_temp_epochs,
            online_temp=config.online_temp,
            num_epochs=config.epochs - config.model_warmup_epoch,
            branch_temp=config.branch_temp
        )

        self.register_buffer("queue", torch.randn(config.out_dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.init_momentum_pairs()

    @torch.no_grad()
    def extract_local_feature(self, data):
        dense_feature = self.online_encoder(data)[-1]
        B, _, H, W = dense_feature.shape
        dense_feature = rearrange(dense_feature, "B C H W -> B C (H W)")
        dense_feature = self.class_projector.mlp(dense_feature)
        return dense_feature.view(B, -1, H, W)

    @property
    def momentum_pairs(self):
        extra_paris = [
            (self.online_projector, self.branch_projector),
            (self.class_projector, self.branch_class_projector)
        ]

        return super().momentum_pairs + extra_paris

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(1, 0)
        self.queue[:, ptr: ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, imgBatch: List, device=torch.device("cpu")) -> Dict:
        backbone_dict = super(MoCoBased, self).mim_forward(imgBatch, device)
        loss_rec, mask_embedding, view_embedding, locations = backbone_dict["loss_rec"], \
                                                              backbone_dict["mask_embedding"], \
                                                              backbone_dict["view_embedding"], \
                                                              backbone_dict["locations"]

        if self.ft_classify:
            if self.multi_label:
                classify_loss, acc = backbone_dict["classify_loss"], \
                                     backbone_dict["acc"]
            else:
                classify_loss, acc1, acc5 = backbone_dict["classify_loss"], \
                                            backbone_dict["acc1"], \
                                            backbone_dict["acc5"]
        elif self.segmentation:
            seg_loss, acc = backbone_dict["seg_loss"], \
                            backbone_dict["acc"]

        mask_embedding_org, mask_embedding_view = mask_embedding
        view_embedding_org, view_embedding_view = view_embedding
        # get feature
        mask_avg_embedding_org = [self.avg_pool(feature).squeeze() for feature in mask_embedding_org]
        mask_avg_embedding_view = [self.avg_pool(feature).squeeze() for feature in mask_embedding_view]

        with torch.no_grad():
            view_avg_embedding_org = [self.avg_pool(feature).squeeze() for feature in view_embedding_org]
            view_avg_embedding_view = [self.avg_pool(feature).squeeze() for feature in view_embedding_view]

        out = dict(loss_rec=loss_rec)
        total_loss = loss_rec.clone()
        if self.global_branch or self.model_warmup:
            proj_embedding_org = self.online_projector(mask_avg_embedding_org)
            proj_embedding_org = [F.normalize(feature, dim=-1) for feature in proj_embedding_org]

            proj_embedding_view = self.online_projector(mask_avg_embedding_view)
            proj_embedding_view = [F.normalize(feature, dim=-1) for feature in proj_embedding_view]

            with torch.no_grad():
                view_proj_embedding_org = self.branch_projector(view_avg_embedding_org)
                view_proj_embedding_org = [F.normalize(feature, dim=-1) for feature in view_proj_embedding_org]

                view_proj_embedding_view = self.branch_projector(view_avg_embedding_view)
                view_proj_embedding_view = [F.normalize(feature, dim=-1) for feature in view_proj_embedding_view]

            queue = self.queue.clone().detach()

            global_loss = mocov2plus_loss_func(proj_embedding_org[-1], view_proj_embedding_view[-1].detach(), queue,
                                               self.temperature)
            global_loss += mocov2plus_loss_func(proj_embedding_view[-1], view_proj_embedding_org[-1].detach(), queue,
                                               self.temperature)

            global_loss /= 2
            self._dequeue_and_enqueue(gather(view_proj_embedding_view[-1]))
            self._dequeue_and_enqueue(gather(view_proj_embedding_org[-1]))
            total_loss += global_loss
            out.update(dict(global_loss=global_loss))

        if not self.model_warmup:
            mask_embedding_org = mask_embedding_org[-1]
            mask_embedding_view = mask_embedding_view[-1]

            view_embedding_org = view_embedding_org[-1]
            view_embedding_view = view_embedding_view[-1]

            location_1, location_2 = locations
            mask_embedding_org = rearrange(mask_embedding_org, " B C H W -> B (H W) C")
            mask_embedding_view = rearrange(mask_embedding_view, " B C H W -> B (H W) C")

            view_embedding_org = rearrange(view_embedding_org, " B C H W -> B (H W) C")
            view_embedding_view = rearrange(view_embedding_view, " B C H W -> B (H W) C")

            location_1 = location_1.flatten(1, 2)
            location_2 = location_2.flatten(1, 2)

            maps_1_filtered, maps_1_nn = position_match(
                location_1,
                location_2,
                mask_embedding_org,
                view_embedding_view,
                num_matches=self.num_matches
            )

            maps_2_filtered, maps_2_nn = position_match(
                location_2,
                location_1,
                mask_embedding_view,
                view_embedding_org,
                num_matches=self.num_matches
            )

            mask_proj_1 = self.class_projector(maps_1_filtered.permute(0, 2, 1))
            mask_proj_2 = self.class_projector(maps_2_filtered.permute(0, 2, 1))
            with torch.no_grad():
                view_proj_1 = self.branch_class_projector(maps_1_nn.permute(0, 2, 1))
                view_proj_2 = self.branch_class_projector(maps_2_nn.permute(0, 2, 1))

            pixel_loss = self.class_loss(mask_proj_1, view_proj_1.detach())
            pixel_loss += self.class_loss(mask_proj_2, view_proj_2.detach())
            pixel_loss /= 2

            total_loss += pixel_loss
            out.update(pixel_loss=pixel_loss)

        if self.ft_classify:
            if self.multi_label:
                out.update(classify_loss=classify_loss,
                           acc=acc)
            else:
                out.update(dict(classify_loss=classify_loss,
                                acc1=acc1,
                                acc5=acc5))
            total_loss += classify_loss
        elif self.segmentation:
            out.update(seg_loss=seg_loss,
                       acc=acc)
            total_loss += seg_loss

        out.update(dict(total_loss=total_loss))
        return out


class MoCoBasedDDP(MoCoBased):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(MoCoBasedDDP, self).__init__(config)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        if get_world_size() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, imgBatch: List, device=torch.device("cpu")) -> Dict:
        backbone_dict = super(MoCoBasedDDP, self).mim_forward(imgBatch, device)
        loss_rec, mask_embedding, view_embedding, locations = backbone_dict["loss_rec"], \
                                                              backbone_dict["mask_embedding"], \
                                                              backbone_dict["view_embedding"], \
                                                              backbone_dict["locations"]

        if self.ft_classify:
            if self.multi_label:
                classify_loss, acc = backbone_dict["classify_loss"], \
                                     backbone_dict["acc"]
            else:
                classify_loss, acc1, acc5 = backbone_dict["classify_loss"], \
                                            backbone_dict["acc1"], \
                                            backbone_dict["acc5"]
        elif self.segmentation:
            seg_loss, acc = backbone_dict["seg_loss"], \
                            backbone_dict["acc"]

        mask_embedding_org, mask_embedding_view = mask_embedding
        view_embedding_org, view_embedding_view = view_embedding
        # get feature
        mask_avg_embedding_org = [self.avg_pool(feature).squeeze() for feature in mask_embedding_org]
        mask_avg_embedding_view = [self.avg_pool(feature).squeeze() for feature in mask_embedding_view]

        with torch.no_grad():
            view_avg_embedding_org = [self.avg_pool(feature).squeeze() for feature in view_embedding_org]
            view_avg_embedding_view = [self.avg_pool(feature).squeeze() for feature in view_embedding_view]

        out = dict(loss_rec=loss_rec)
        total_loss = loss_rec.clone()
        if self.global_branch or self.model_warmup:
            proj_embedding_org = self.online_projector(mask_avg_embedding_org)
            proj_embedding_org = [F.normalize(feature, dim=-1) for feature in proj_embedding_org]

            proj_embedding_view = self.online_projector(mask_avg_embedding_view)
            proj_embedding_view = [F.normalize(feature, dim=-1) for feature in proj_embedding_view]

            with torch.no_grad():
                view_proj_embedding_org = self.branch_projector(view_avg_embedding_org)
                view_proj_embedding_org = [F.normalize(feature, dim=-1) for feature in view_proj_embedding_org]

                view_proj_embedding_view = self.branch_projector(view_avg_embedding_view)
                view_proj_embedding_view = [F.normalize(feature, dim=-1) for feature in view_proj_embedding_view]


            queue = self.queue.clone().detach()
            global_loss = mocov2plus_loss_func(proj_embedding_org[-1], view_proj_embedding_view[-1].detach(), queue,
                                               self.temperature)
            global_loss += mocov2plus_loss_func(proj_embedding_view[-1], view_proj_embedding_org[-1].detach(), queue,
                                               self.temperature)

            global_loss /= 2
            self._dequeue_and_enqueue(gather(view_proj_embedding_view[-1]))
            self._dequeue_and_enqueue(gather(view_proj_embedding_org[-1]))
            total_loss += global_loss
            out.update(dict(global_loss=global_loss))

        if not self.model_warmup:
            mask_embedding_org = mask_embedding_org[-1]
            mask_embedding_view = mask_embedding_view[-1]

            view_embedding_org = view_embedding_org[-1]
            view_embedding_view = view_embedding_view[-1]

            location_1, location_2 = locations
            mask_embedding_org = rearrange(mask_embedding_org, " B C H W -> B (H W) C")
            mask_embedding_view = rearrange(mask_embedding_view, " B C H W -> B (H W) C")

            view_embedding_org = rearrange(view_embedding_org, " B C H W -> B (H W) C")
            view_embedding_view = rearrange(view_embedding_view, " B C H W -> B (H W) C")

            location_1 = location_1.flatten(1, 2)
            location_2 = location_2.flatten(1, 2)

            maps_1_filtered, maps_1_nn = position_match(
                location_1,
                location_2,
                mask_embedding_org,
                view_embedding_view,
                num_matches=self.num_matches
            )

            maps_2_filtered, maps_2_nn = position_match(
                location_2,
                location_1,
                mask_embedding_view,
                view_embedding_org,
                num_matches=self.num_matches
            )

            mask_proj_1 = self.class_projector(maps_1_filtered.permute(0, 2, 1))
            mask_proj_2 = self.class_projector(maps_2_filtered.permute(0, 2, 1))
            with torch.no_grad():
                view_proj_1 = self.branch_class_projector(maps_1_nn.permute(0, 2, 1))
                view_proj_2 = self.branch_class_projector(maps_2_nn.permute(0, 2, 1))

            pixel_loss = self.class_loss(mask_proj_1, view_proj_1.detach())
            pixel_loss += self.class_loss(mask_proj_2, view_proj_2.detach())
            pixel_loss /= 2

            total_loss += pixel_loss
            out.update(pixel_loss=pixel_loss)

        if self.ft_classify:
            if self.multi_label:
                out.update(classify_loss=classify_loss,
                           acc=acc)
            else:
                out.update(dict(classify_loss=classify_loss,
                                acc1=acc1,
                                acc5=acc5))
            total_loss += classify_loss
        elif self.segmentation:
            out.update(seg_loss=seg_loss,
                       acc=acc)
            total_loss += seg_loss

        out.update(dict(total_loss=total_loss))
        return out


class BYOLBased(PretrainModel):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(BYOLBased, self).__init__(config)
        self.global_branch = config.global_branch
        self.num_matches = config.num_matches
        self.model_warmup = False

        self.online_projector = MLP(in_dim=2048,
                                    hidden_dim=config.hidden_dim,
                                    out_dim=config.out_dim)

        self.branch_projector = MLP(in_dim=2048,
                                    hidden_dim=config.hidden_dim,
                                    out_dim=config.out_dim)

        self.predictor = MLP(
            in_dim=config.out_dim,
            hidden_dim=config.predictor_hidden_dim,
            out_dim=config.out_dim)

        self.class_projector = LocalHead(
            in_dim=config.in_channels,
            use_bn=config.use_bn,
            out_dim=config.class_feat_size,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
        )

        self.branch_class_projector = LocalHead(
            in_dim=config.in_channels,
            use_bn=config.use_bn,
            out_dim=config.class_feat_size,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
        )

        self.class_loss = LocalLoss(
            class_feat_size=config.class_feat_size,
            center_momentum=config.center_momentum,
            warmup_branch_temp=config.warmup_branch_temp,
            warmup_branch_temp_epochs=config.warmup_branch_temp_epochs,
            online_temp=config.online_temp,
            num_epochs=config.epochs - config.model_warmup_epoch,
            branch_temp=config.branch_temp
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.init_momentum_pairs()

    @property
    def momentum_pairs(self):
        extra_paris = [
            (self.online_projector, self.branch_projector),
            (self.class_projector, self.branch_class_projector)
        ]

        return super().momentum_pairs + extra_paris

    def forward(self, imgBatch: List, device=torch.device("cpu")) -> Dict:
        backbone_dict = super(BYOLBased, self).mim_forward(imgBatch, device)
        loss_rec, mask_embedding, view_embedding, locations = backbone_dict["loss_rec"], \
                                                              backbone_dict["mask_embedding"], \
                                                              backbone_dict["view_embedding"], \
                                                              backbone_dict["locations"]

        if self.ft_classify:
            if self.multi_label:
                classify_loss, acc = backbone_dict["classify_loss"], \
                                     backbone_dict["acc"]
            else:
                classify_loss, acc1, acc5 = backbone_dict["classify_loss"], \
                                            backbone_dict["acc1"], \
                                            backbone_dict["acc5"]
        elif self.segmentation:
            seg_loss, acc = backbone_dict["seg_loss"], \
                            backbone_dict["acc"]

        mask_embedding_org, mask_embedding_view = mask_embedding
        view_embedding_org, view_embedding_view = view_embedding
        # get feature
        mask_avg_embedding_org = [self.avg_pool(feature).squeeze() for feature in mask_embedding_org]
        mask_avg_embedding_view = [self.avg_pool(feature).squeeze() for feature in mask_embedding_view]

        with torch.no_grad():
            view_avg_embedding_org = [self.avg_pool(feature).squeeze() for feature in view_embedding_org]
            view_avg_embedding_view = [self.avg_pool(feature).squeeze() for feature in view_embedding_view]

        out = dict(loss_rec=loss_rec)
        total_loss = loss_rec.clone()

        if self.global_branch or self.model_warmup:
            proj_embedding_org = self.online_projector(mask_avg_embedding_org)
            proj_embedding_org = self.predictor(proj_embedding_org)

            proj_embedding_view = self.online_projector(mask_avg_embedding_view)
            proj_embedding_view = self.predictor(proj_embedding_view)

            with torch.no_grad():
                view_proj_embedding_org = self.branch_projector(view_avg_embedding_org)

                view_proj_embedding_view = self.branch_projector(view_avg_embedding_view)

            global_loss = linear_loss(proj_embedding_org[-1], view_proj_embedding_view[-1])
            global_loss += linear_loss(proj_embedding_view[-1], view_proj_embedding_org[-1])

            global_loss /= 2
            total_loss += global_loss
            out.update(dict(global_loss=global_loss))

        if not self.model_warmup:
            mask_embedding_org = mask_embedding_org[-1]
            mask_embedding_view = mask_embedding_view[-1]

            view_embedding_org = view_embedding_org[-1]
            view_embedding_view = view_embedding_view[-1]

            location_1, location_2 = locations
            mask_embedding_org = rearrange(mask_embedding_org, " B C H W -> B (H W) C")
            mask_embedding_view = rearrange(mask_embedding_view, " B C H W -> B (H W) C")

            view_embedding_org = rearrange(view_embedding_org, " B C H W -> B (H W) C")
            view_embedding_view = rearrange(view_embedding_view, " B C H W -> B (H W) C")

            location_1 = location_1.flatten(1, 2)
            location_2 = location_2.flatten(1, 2)

            maps_1_filtered, maps_1_nn = position_match(
                location_1,
                location_2,
                mask_embedding_org,
                view_embedding_view,
                num_matches=self.num_matches
            )

            maps_2_filtered, maps_2_nn = position_match(
                location_2,
                location_1,
                mask_embedding_view,
                view_embedding_org,
                num_matches=self.num_matches
            )

            mask_proj_1 = self.class_projector(maps_1_filtered.permute(0, 2, 1))
            mask_proj_2 = self.class_projector(maps_2_filtered.permute(0, 2, 1))
            with torch.no_grad():
                view_proj_1 = self.branch_class_projector(maps_1_nn.permute(0, 2, 1))
                view_proj_2 = self.branch_class_projector(maps_2_nn.permute(0, 2, 1))

            pixel_loss = self.class_loss(mask_proj_1, view_proj_1.detach())
            pixel_loss += self.class_loss(mask_proj_2, view_proj_2.detach())
            pixel_loss /= 2

            total_loss += pixel_loss
            out.update(pixel_loss=pixel_loss)

        if self.ft_classify:
            if self.multi_label:
                out.update(classify_loss=classify_loss,
                           acc=acc)
            else:
                out.update(dict(classify_loss=classify_loss,
                                acc1=acc1,
                                acc5=acc5))
            total_loss += classify_loss
        elif self.segmentation:
            out.update(seg_loss=seg_loss,
                       acc=acc)
            total_loss += seg_loss

        out.update(dict(total_loss=total_loss))
        return out


if __name__ == '__main__':
    pass
