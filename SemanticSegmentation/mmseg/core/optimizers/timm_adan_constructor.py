import json
import warnings

from mmcv.runner import DefaultOptimizerConstructor
from ..builder import OPTIMIZER_BUILDERS
from timm.optim import create_optimizer_v2
from typing import Any, List, Dict


@OPTIMIZER_BUILDERS.register_module()
class TimmConstructor(DefaultOptimizerConstructor):
    def __init__(self, optimizer_cfg, paramwise_cfg=None) -> None:
        super(TimmConstructor, self).__init__(optimizer_cfg, paramwise_cfg)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return create_optimizer_v2(optimizer_cfg["params"], 
                                       opt=optimizer_cfg["type"],
                                       filter_bias_and_bn=True,
                                       weight_decay=optimizer_cfg["weight_decay"],
                                       lr=optimizer_cfg["lr"])
        params: List[Dict] = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return create_optimizer_v2(optimizer_cfg["params"], 
                                    opt=optimizer_cfg["type"],
                                    filter_bias_and_bn=optimizer_cfg["filter_bias_and_bn"],
                                    weight_decay=optimizer_cfg["weight_decay"],
                                    lr=optimizer_cfg["lr"])