# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import logging
import numpy as np
import ml_collections
from Dataset import UCM
from models import build_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.transforms as T
from utils import MultiExtractProcess
from Exploring import init_distributed
from Exploring.utils import ConfigArgumentParser, setup_logger
logger = logging.getLogger("train")


def parse_args():
    parser = ConfigArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--data-root')
    parser.add_argument('--output', help='the dir to save logs and models')
    parser.add_argument(
        '--layer-ind',
        type=str,
        default='0,1,2,3,4',
        help='layer indices, separated by comma, e.g., "0,1,2,3,4"')
    parser.add_argument(
        '--pool-type',
        choices=['specified', 'adaptive'],
        default='specified',
        help='Pooling type in :class:`MultiPooling`')
    parser.add_argument(
        '--max-num-class',
        type=int,
        default=20,
        help='the maximum number of classes to apply t-SNE algorithms, now the'
        'function supports maximum 20 classes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # t-SNE settings
    parser.add_argument(
        '--n-components', type=int, default=2, help='the dimension of results')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms.')
    parser.add_argument(
        '--early-exaggeration',
        type=float,
        default=12.0,
        help='Controls how tight natural clusters in the original space are in'
        'the embedded space and how much space will be between them.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=200.0,
        help='The learning rate for t-SNE is usually in the range'
        '[10.0, 1000.0]. If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='Maximum number of iterations for the optimization. Should be at'
        'least 250.')
    parser.add_argument(
        '--n-iter-without-progress',
        type=int,
        default=300,
        help='Maximum number of iterations without progress before we abort'
        'the optimization.')
    parser.add_argument(
        '--init', type=str, default='random', help='The init method')


    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main():
    config = parse_args()

    torch.backends.cudnn.benchmark = True

    layer_ind = [int(idx) for idx in config.layer_ind.split(',')]
    config.rank, config.local_rank, config.world_size = init_distributed()
    config.is_distribute = config.world_size > 1
    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "tsne_features"), exist_ok=True)

    # set random seeds
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # build the dataset
    trans = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407])
    ])
    dataset = UCM(config.data_root, split="train", transform=trans)

    # compress dataset, select that the label is less then max_num_class
    tmp_imgs = []
    tmp_idx = []
    for i in range(len(dataset)):
        if dataset.cat_id[i] < config.max_num_class:
            tmp_idx.append(dataset.cat_id[i])
            tmp_imgs.append(dataset.imgs[i])
    dataset.cat_id = tmp_idx
    dataset.imgs = tmp_imgs
    logger.info(f'Apply t-SNE to visualize {len(dataset)} samples.')

    # build loader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=4,
                                              batch_size=64,
                                              shuffle=False)

    # build the model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(config, is_pretrain=False)
    with torch.no_grad():
        state_dict = torch.load(config.checkpoint)
        model.load_pretrain_model(state_dict)
        del state_dict
    model.to(device)

    # build extraction processor and run
    extractor = MultiExtractProcess(
        pool_type=config.pool_type, backbone='resnet50' if config.backbone in ["swin", "resnet50"] else config.backbone, layer_indices=layer_ind)
    features = extractor.extract(model, data_loader, device=device)
    labels = dataset.cat_id
    logger.info(f"Save features to {config.output}/tsne_features/")
    for key, val in features.items():
        output_file = os.path.join(config.output, "tsne_features")
        output_file = os.path.join(output_file, f"{config.name}_{key}.npy")
        np.save(output_file, val)

    tsne_model = TSNE(
        n_components=config.n_components,
        perplexity=config.perplexity,
        early_exaggeration=config.early_exaggeration,
        learning_rate=config.learning_rate,
        n_iter=config.n_iter,
        n_iter_without_progress=config.n_iter_without_progress,
        init=config.init)

    # run and get results
    save_dir = os.path.join(config.ouput, "save_picture")
    os.makedirs(save_dir, exist_ok=True)
    logger.info('Running t-SNE......')
    for key, val in features.items():
        result = tsne_model.fit_transform(val)
        res_min, res_max = result.min(0), result.max(0)
        res_norm = (result - res_min) / (res_max - res_min)
        plt.figure(figsize=(10, 10))
        plt.scatter(
            res_norm[:, 0],
            res_norm[:, 1],
            alpha=1.0,
            s=15,
            c=labels,
            cmap='tab20')
        plt.savefig(os.path.join(save_dir, f'{key}.png'))
    logger.info(f'Saved results to {save_dir}')


if __name__ == '__main__':
    main()
