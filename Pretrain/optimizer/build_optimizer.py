import torch.nn as nn
import ml_collections
from timm.optim.optim_factory import create_optimizer_v2


def build_optimizer(model: nn.Module, config: ml_collections.ConfigDict):
    return create_optimizer_v2(model,
                               opt=config.optimizer,
                               lr=config.lr,
                               weight_decay=config.wd,
                               filter_bias_and_bn=True)