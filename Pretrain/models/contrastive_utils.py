import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Union, List, Any
from timm.models.vision_transformer import trunc_normal_


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.where(sorted_indices_indices < num_matches, True, False)
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def position_match(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int = 256,
                 hidden_dim: int = 512,
                 out_dim: int = 256,
                 norm_layer=nn.BatchNorm1d,
                 act_layer=nn.ReLU,
                 drop: float = 0.):
        super(MLP, self).__init__()

        self.layer = nn.ModuleList()
        self.layer.append(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm_layer(hidden_dim) if norm_layer is not nn.Identity else norm_layer(),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(drop)
        ))

    def forward(self, x: Union[torch.Tensor, List]):
        out = []
        for feature, layer in zip(x, self.layer):
            out.append(layer(feature))
        return out


class LocalHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 use_bn: bool = True,
                 norm_last_layer: bool = True,
                 num_layers: int = 3,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256):
        super(LocalHead, self).__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Conv1d(in_dim, bottleneck_dim, 1)
        else:
            layers: List[Any] = [nn.Conv1d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv1d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Conv1d(bottleneck_dim, out_dim, 1, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the backbone, the projector and the last layer (prototypes).

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """

        x = self.mlp(x)
        x = F.normalize(x, dim=1)
        x = self.last_layer(x)
        return x