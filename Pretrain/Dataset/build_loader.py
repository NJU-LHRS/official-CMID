import logging
from .UCM import UCM
from .potsdam import Potsdam
from .h5_dataset import H5Dataset
from .utils import InfiniteDataLoader
from torchvision.datasets import ImageFolder
# from .BigEarthNet import BigEarthNet, LMDBDataset
from torch.utils.data import DataLoader, DistributedSampler
from .build_transform import build_transform, build_cls_transform


logger = logging.getLogger("train")


def build_loader_pretrain(config):
    transform = build_transform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    if config.dataset == "potsdam":
        dataset = Potsdam(data_root=config.data_path, transform=transform, return_label=config.segmentation)
    elif config.dataset == "millionaid":
        if config.data_path.endswith("h5"):
            dataset = H5Dataset(config.data_path, transform)
        else:
            dataset = ImageFolder(config.data_path, transform)
    else:
        dataset = ImageFolder(config.data_path, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    if config.is_distribute:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    if config.infloader:
        dataloader = InfiniteDataLoader(dataset, config.batch_size, num_workers=config.workers, pin_memory=True,
                    drop_last=True, shuffle=True)
    else:
        dataloader = DataLoader(dataset, config.batch_size, sampler=sampler, num_workers=config.workers,
                                pin_memory=True, drop_last=True, shuffle=(sampler is None))

    return dataloader


def build_loader_finetune(config):
    transform = build_cls_transform(config, is_train=True)

    logger.info(f'Fine-tune data transform:\n{transform}')

    if config.dataset == "UCM":
        dataset = UCM(config.data_path, split="train", transform=transform)
    else:
        dataset = ImageFolder(config.data_path, transform)

    logger.info(f'Build dataset: train images = {len(dataset)}')

    if config.is_distribute:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    if config.infloader:
        dataloader = InfiniteDataLoader(dataset, config.batch_size, num_workers=config.workers, pin_memory=True,
                    drop_last=True, shuffle=True)
    else:
        dataloader = DataLoader(dataset, config.batch_size, sampler=sampler, num_workers=config.workers,
                                pin_memory=True, drop_last=True, shuffle=(sampler is None))

    return dataloader


def build_loader_evaluate(config):
    transform = build_cls_transform(config, is_train=False)

    logger.info(f'Evaluate data transform:\n{transform}')

    if config.dataset == "UCM":
        dataset = UCM(config.data_path, split="test", transform=transform)
    else:
        dataset = ImageFolder(config.data_path, transform)

    if config.is_distribute:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
        
    if config.infloader:
        dataloader = InfiniteDataLoader(dataset, config.batch_size, num_workers=config.workers, pin_memory=True,
                    drop_last=True, shuffle=True)
    else:
        dataloader = DataLoader(dataset, config.batch_size, sampler=sampler, num_workers=config.workers,
                                pin_memory=True, drop_last=True, shuffle=(sampler is None))

    return dataloader


def build_loader(config, mode: str = "pretrain"):
    assert mode in ["pretrain", "evaluate",
                    "finetune"], "Please choose mode for dataloder from [pretrain, finetune, evaluate]"
    if mode == "pretrain":
        return build_loader_pretrain(config)
    elif mode == "finetune":
        return build_loader_finetune(config)
    elif mode == "evaluate":
        return build_loader_evaluate(config)
