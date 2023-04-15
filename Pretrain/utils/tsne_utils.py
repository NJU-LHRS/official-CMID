import mmcv
import torch
import numpy as np
import torch.nn as nn
from mmcv.runner import BaseModule


class MultiPooling(BaseModule):
    """Pooling layers for features from multiple depth.

    Args:
        pool_type (str): Pooling type for the feature map. Options are
            'adaptive' and 'specified'. Defaults to 'adaptive'.
        in_indices (Sequence[int]): Output from which backbone stages.
            Defaults to (0, ).
        backbone (str): The selected backbone. Defaults to 'resnet50'.
    """

    POOL_PARAMS = {
        'resnet50': [
            dict(kernel_size=10, stride=10, padding=4),
            dict(kernel_size=16, stride=8, padding=0),
            dict(kernel_size=13, stride=5, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=6, stride=1, padding=0)
        ],
        "vit": [
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
        ],
        "vitae": [
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
        ]
    }
    POOL_SIZES = {'resnet50': [12, 6, 4, 3, 1]}
    POOL_DIMS = {'resnet50': [9216, 9216, 8192, 9216, 8192]}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='resnet50'):
        super(MultiPooling, self).__init__()
        assert pool_type in ['adaptive', 'specified']
        if pool_type == 'adaptive':
            self.pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(self.POOL_SIZES[backbone][i])
                for i in in_indices
            ])
        else:
            self.pools = nn.ModuleList([
                nn.AvgPool2d(**self.POOL_PARAMS[backbone][i])
                for i in in_indices
            ])

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        return [p(xx) for p, xx in zip(self.pools, x)]


class MultiExtractProcess(object):
    """Multi-stage intermediate feature extraction process for `extract.py` and
    `tsne_visualization.py` in tools.

    This process extracts feature maps from different stages of backbone, and
    average pools each feature map to around 9000 dimensions.

    Args:
        pool_type (str): Pooling type in :class:`MultiPooling`. Options are
            "adaptive" and "specified". Defaults to "specified".
        backbone (str): Backbone type, now only support "resnet50".
            Defaults to "resnet50".
        layer_indices (Sequence[int]): Output from which stages.
            0 for stem, 1, 2, 3, 4 for res layers. Defaults to (0, 1, 2, 3, 4).
    """

    def __init__(self,
                 pool_type='specified',
                 backbone='resnet50',
                 layer_indices=(0, 1, 2, 3, 4)):
        self.multi_pooling = MultiPooling(
            pool_type, in_indices=layer_indices, backbone=backbone)
        self.layer_indices = layer_indices
        for i in self.layer_indices:
            assert i in [0, 1, 2, 3, 4]

    def _forward_func(self, model, **x):
        backbone_feats = model.extract(**x)
        pooling_feats = self.multi_pooling(backbone_feats)
        flat_feats = [xx.view(xx.size(0), -1) for xx in pooling_feats]
        feat_dict = {
            f'feat{self.layer_indices[i] + 1}': feat.cpu()
            for i, feat in enumerate(flat_feats)
        }
        return feat_dict

    def extract(self, model, data_loader, device: torch.device = torch.device("cpu")):
        model.eval()

        def func(**x):
            return self._forward_func(model, **x)

        results = []
        prog_bar = mmcv.ProgressBar(len(data_loader))
        for i, data in enumerate(data_loader):
            data, target = data
            input_data = dict(data=data, device=device)
            with torch.no_grad():
                result = func(**input_data)  # feat_dict
            results.append(result)  # list of feat_dict
            prog_bar.update()

        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate(
                [batch[k].numpy() for batch in results], axis=0)
            assert results_all[k].shape[0] == len(data_loader.dataset)

        return results_all