import os
import json
import torch
import wandb
import logging
import numpy as np
from models import build_model
import torch.distributed as dist
import ml_collections.config_dict
import torch.backends.cudnn as cudnn
from optimizer import build_optimizer
from Exploring.trainer import Trainer
from Dataset.build_loader import build_loader
from utils import auto_resume_helper, str2bool
from Exploring.utils import ConfigArgumentParser, setup_logger
from Exploring import init_distributed, is_main_process, get_world_size
from Exploring.hook import EMAHook, CounterHook, KnnEvaluate, PlotSaver, MoCoWarmup, DINOLossWarmUp

logger = logging.getLogger("train")


def parse_option():
    parser = ConfigArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # basic
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--eval-data-path', type=str, help='path to evaluate dataset')
    parser.add_argument('--workers', type=int, default=8, help="workers of dataloader")
    parser.add_argument('--auto_resume', action="store_true", help='resume from checkpoint')
    parser.add_argument("--resume_path", type=str, default=None, help="resume checkpoint path")
    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--enable_amp', type=str2bool, default=False,
                        help='mixed precision')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    parser.add_argument("--infloader", type=str2bool, default=False, help='Use Infinite loader if ture, else default datalodaer')

    # wandb
    parser.add_argument("--wandb", type=str2bool, default=False, help="wandb logger")
    parser.add_argument("--entity", type=str, default="pumpkinn", help="wandb entity")
    parser.add_argument("--project", type=str, default="MaskIndexNet", help="wandb project")

    # HardWare
    parser.add_argument("--accelerator", default="cpu", type=str, choices=["cpu", "gpu", "mps"], help="accelerator")

    # distributed training
    parser.add_argument("--local_rank", default=1, type=int, help="rank for distribute traning")
    parser.add_argument("--knn_eval", type=bool, default=False, help="If true, using knn evaluation")

    config = parser.parse_args(wandb=True)
    config = ml_collections.config_dict.ConfigDict(config)

    return config


def main(config):
    data_loader_train = build_loader(config, mode="pretrain")
    if config.knn_eval:
        data_loader_eval = build_loader(config, mode="evaluate")
    logger.info(f"Creating model: Backbone: {config.backbone}")

    model = build_model(config, is_pretrain=True)
    logger.info(str(model))
    optimizer = build_optimizer(model, config)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      lr_scheduler=config,
                      data_loader=data_loader_train,
                      max_epochs=config.epochs,
                      work_dir=config.output,
                      log_period=5,
                      accelerator=config.accelerator,
                      enable_amp=config.enable_amp,
                      wandb=config.wandb,
                      eval_data_loader=data_loader_eval if config.knn_eval else None,
                      gpus=0,
                      max_num_checkpoints=10,
                      is_distributed=config.is_distribute)

    hooks = [EMAHook(), MoCoWarmup(warmup_epoch=config.model_warmup_epoch), DINOLossWarmUp(warmup_epoch=config.model_warmup_epoch)]
    if is_main_process():
        hooks.extend([PlotSaver(save_interval=1)])
    trainer.register_hook(hooks)

    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            if config.resume_path is not None:
                logger.warning(f"auto-resume changing resume file from {config.resume_path} to {resume_file}")
            config.resume_path = resume_file
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.output}/checkpoint, ignoring auto resume")

    trainer.train(load_checkpoint=config.resume_path)


if __name__ == '__main__':
    config = parse_option()

    config.rank, config.local_rank, config.world_size = init_distributed()
    config.is_distribute = config.world_size > 1
    print(config)

    setup_logger("train", output=config.output, rank=config.rank)
    os.makedirs(config.output, exist_ok=True)
    os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=True)

    if config.is_distribute:
        seed = config.seed + dist.get_rank()
    else:
        seed = config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    config.lr = config.lr * get_world_size()
    # gradient accumulation need to scale the learning rate
    config.lr = config.lr * config.accumulation_steps
    config.min_lr = config.min_lr * config.accumulation_steps

    if config.rank == 0:
        path = os.path.join(config.output, "config.json")
        with open(path, "w") as f:
            configDict = dict(config.to_dict())
            json.dump(configDict, f, indent=4)
        logger.info(f"Full config saved to {path}")
        logger.info(config)

    if config.wandb and config.rank == 0:
        wandb.init(config=config.to_dict(),
                   entity=config.entity,
                   project=config.project)
        config = wandb.config

    main(config)
