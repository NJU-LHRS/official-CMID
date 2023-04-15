import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def mocov2plus_loss_func(
        query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor, temperature=0.1
) -> torch.Tensor:
    """Computes MoCo's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the keys from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): temperature of the softmax in the contrastive
            loss. Defaults to 0.1.

    Returns:
        torch.Tensor: MoCo loss.
    """

    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    neg = torch.einsum("nc,ck->nk", [query, queue])
    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature
    targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


def linear_loss(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    if simplified:
        return (2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1)).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    return 2 - 2 * (p * z.detach()).sum(dim=1).mean()


def pixel_loss(pred, feat, view_index_mask):
    B, _, H, W = pred.shape
    pred = einops.rearrange(pred, "b c h w -> b (h w) c")
    feat = einops.rearrange(feat, "b c h w -> b (h w) c")
    view_index_mask = einops.rearrange(view_index_mask, "b h w -> b (h w)")
    img_index_mask = torch.arange(49, dtype=view_index_mask.dtype, device=view_index_mask.device).reshape(7,
                                                                                                          7).unsqueeze(
        0).repeat_interleave(B, 0)

    def make_same_obj(mask1, mask2):
        same_obj = torch.eq(mask1.contiguous().view(B, H * W, 1),
                            mask2.contiguous().view(B, 1, H * W))
        return same_obj

    same_obj = make_same_obj(img_index_mask, view_index_mask)

    pred = F.normalize(pred, dim=-1)
    feat = F.normalize(feat, dim=-1)
    logits = torch.bmm(pred, feat.detach().permute(0, 2, 1))

    pos_simi = torch.masked_select(logits, same_obj)
    loss = (2 - 2 * pos_simi).mean()

    return loss


def focal_l1_loss(pred, target,
                  alpha=0.2, gamma=1.0, activate='sigmoid', residual=False,
                  weight=None, reduction='mean', **kwargs):
    r"""Calculate Focal L1 loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): A balanced form for Focal Loss. Defaults to 0.2.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 1.0.
        activate (str): activate methods in Focal loss in {'sigmoid', 'tanh'}.
            Defaults to 'sigmoid'.
        residual (bool): Whether to use the original l1_loss, i.e., l1 + focal_l1.
            Defaults to False.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _loss = F.l1_loss(pred, target, reduction='none')
    if activate == 'tanh':
        loss = _loss * (torch.tanh(alpha * _loss)) ** gamma
    else:
        loss = _loss * (2. * torch.sigmoid(alpha * _loss) - 1.) ** gamma
    if residual:
        loss += _loss

    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class LocalLoss(nn.Module):
    def __init__(self,
                 class_feat_size: int,
                 warmup_branch_temp: float,
                 branch_temp: float,
                 warmup_branch_temp_epochs: float,
                 num_epochs: int,
                 online_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super(LocalLoss, self).__init__()
        self.epoch = 0
        self.online_temp = online_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, class_feat_size, 1))
        self.branch_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_branch_temp, branch_temp, warmup_branch_temp_epochs),
                np.ones(num_epochs - warmup_branch_temp_epochs) * branch_temp,
            )
        )

    def forward(self, online_output: torch.Tensor, branch_output: torch.Tensor) -> torch.Tensor:
        online_output = online_output / self.online_temp
        temp = self.branch_temp_schedule[self.epoch]
        branch_out = F.softmax((branch_output - self.center) / temp, dim=1)
        branch_out = branch_out.detach()

        loss = torch.sum(-branch_out * F.log_softmax(online_output, dim=1), dim=1)
        loss = loss.mean()

        self.update_center(branch_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        B, _, N = teacher_output.shape
        batch_center = torch.sum(teacher_output, dim=(0, 2), keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        batch_center = batch_center / (B * N)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class RegressionLoss(nn.Module):
    r"""Simple Regression Loss.

    Args:
        mode (bool): Type of regression loss. Notice that when using
            FP16 training, {'mse_loss', 'smooth_l1_loss'} should use
            'mmcv' mode. Defaults to "mse_loss".
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(RegressionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_kwargs = dict()

        assert reduction in [None, 'none', 'mean', 'sum']
        self.loss_kwargs = dict(
            alpha=kwargs.get('alpha', 0.2),
            gamma=kwargs.get('focal_gamma', 1.0),
            activate=kwargs.get('activate', 'sigmoid'),
            residual=kwargs.get('residual', False),
        )

        self.criterion = focal_l1_loss

    def forward(self,
                pred,
                target,
                weight=None,
                **kwargs):
        """caculate loss

        Args:
            pred (tensor): Predicted logits of (N, \*).
            target (tensor): Groundtruth label of (N, \*).
            weight (tensor): Sample-wise reweight of (N, \*) or element-wise reweight
                of (1, \*). Defaults to None.
            reduction_override (str): Reduction methods.
        """
        kwargs.update(self.loss_kwargs)

        loss_reg = self.loss_weight * self.criterion(
            pred, target, weight=weight, reduction=self.reduction, **kwargs)

        return loss_reg


class FocalFrequencyLoss(nn.Module):
    r"""Implements of focal frequency loss

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for
            flexibility. Default: 1.0
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm.
            Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using
            batch-based statistics. Default: False
    """

    def __init__(self,
                 loss_weight=1.0,
                 alpha=1.0,
                 ave_spectrum=False,
                 log_matrix=False,
                 batch_matrix=False):

        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def loss_formulation(self, f_pred, f_targ, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()  # predefined
        else:
            # if the matrix is calculated online: continuous, dynamic,
            #   based on current Euclidean distance
            matrix_tmp = (f_pred - f_targ) ** 2  # loss越大，loss权重越大
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = \
                    matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (f_pred - f_targ) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance

        return loss.mean()

    def forward(self, pred, target, matrix=None, **kwargs):
        r"""Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        f_pred = torch.fft.fft2(pred, dim=(2, 3), norm='ortho')
        f_targ = torch.fft.fft2(target, dim=(2, 3), norm='ortho')
        f_pred = torch.stack([f_pred.real, f_pred.imag], -1)
        f_targ = torch.stack([f_targ.real, f_targ.imag], -1)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            f_pred = torch.mean(f_pred, 0, keepdim=True)
            f_targ = torch.mean(f_targ, 0, keepdim=True)

        loss = self.loss_formulation(f_pred, f_targ, matrix) * self.loss_weight

        return loss
