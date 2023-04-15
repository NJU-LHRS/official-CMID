import os
import json
import wandb
import torch
import logging
import numpy as np
import ml_collections
from models import build_model
import torch.distributed as dist
from timm.data.mixup import Mixup
import torch.backends.cudnn as cudnn
from optimizer import build_optimizer
from Exploring.trainer import Trainer
from Dataset.build_loader import build_loader
from utils import auto_resume_helper, str2bool
from Exploring.hook import EvalHook, CounterHook
from Exploring.utils import ConfigArgumentParser, setup_logger
from Exploring import init_distributed, is_main_process, get_world_size
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
    parser.add_argument('--workers', type=int, default=8, help="workers of dataloader")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--auto_resume', action="store_true", help='resume from checkpoint')
    parser.add_argument("--resume_path", type=str, default=None, help="resume checkpoint path")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument("--seed", type=int, default=322, help="random seed")
    parser.add_argument("--gpus", type=int, default=0, help="gpus ID")
    parser.add_argument("--infloader", type=str2bool, default=False, help='Use Infinite loader if ture, else default datalodaer')
    parser.add_argument('--enable_amp', type=str2bool, default=False,
                        help='mixed precision')

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
    data_loader_train = build_loader(config, mode="finetune")
    data_loader_eval = build_loader(config, mode="evaluate")

    mixup_fn = None
    mixup_active = config.mixup > 0 or config.cutmix > 0. or config.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, cutmix_minmax=config.cutmix_minmax,
            prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
            label_smoothing=config.smoothing, num_classes=config.num_classes)


    logger.info(f"Creating model: Backbone: {config.backbone}")
    model = build_model(config, is_pretrain=False)
    model.register_criterion(mixup_fn)
    logger.info(str(model))
    optimizer = build_optimizer(model, config)

    with torch.no_grad():
        state_dict = torch.load(config.checkpoint_path)
        model.load_pretrain_model(state_dict)
        del state_dict

    if config.auto_resume:
        resume_file = auto_resume_helper(config.output)
        if resume_file:
            if config.resume_path is not None:
                logger.warning(f"auto-resume changing resume file from {config.resume_path} to {resume_file}")
            config.resume_path = resume_file
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.output}/checkpoint, ignoring auto resume")

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      lr_scheduler=config,
                      data_loader=data_loader_train,
                      max_epochs=config.epochs,
                      work_dir=config.output,
                      log_period=50,
                      accelerator=config.accelerator,
                      enable_amp=config.enable_amp,
                      wandb=config.wandb,
                      eval_data_loader=data_loader_eval,
                      gpus=0,
                      max_num_checkpoints=10,
                      is_distributed=config.is_distribute)

    hooks = [EvalHook(period=1), CounterHook()]
    trainer.register_hook(hooks)

    trainer.train(load_checkpoint=config.resume_path)


if __name__ == '__main__':
    config = parse_option()

    config.rank, config.local_rank, config.world_size = init_distributed()
    config.is_distribute = config.world_size > 1
    print(config)

    # assert config.checkpoint_path is not None, "Please choose the pretrain path"

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