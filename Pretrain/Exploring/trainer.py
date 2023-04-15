import os
import time
import torch
import weakref
import logging
import numpy as np
import torch.nn as nn
import os.path as osp
import ml_collections
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .utils import symlink, collect_env, is_main_process, get_world_size, gather, get_rank, MetricStroge
from .hook import HookBase, LoggerHook, CheckpointerHook, OptimizerHook, Fp16OptimizerHook, \
    GradientCumulativeOptimizerHook, GradientCumulativeFp16OptimizerHook, CosineAnnealingLrUpdaterHook, \
    FixedLrUpdaterHook, DistributedHook

logger = logging.getLogger("train")


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 lr_scheduler: ml_collections.config_dict.ConfigDict,
                 data_loader: DataLoader,
                 max_epochs: int,
                 work_dir: str = "work_dir",
                 max_num_checkpoints: int = None,
                 checkpoint_period: int = 1,
                 log_period: int = 50,
                 clip_grad_norm: Optional[float] = None,
                 enable_amp: bool = False,
                 wandb: bool = False,
                 accelerator: str = "cpu",
                 gpus: Optional[List] = None,
                 cumulative_iters: int = 1,
                 eval_data_loader: DataLoader = None,
                 is_distributed: bool = False,):
        """
        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer)
            lr_scheduler (optim.lr_scheduler._LRScheduler)
            data_loader (torch.utils.data.DataLoader): Training data loader.
            max_epochs (int): Total training epochs.
            work_dir (str): The working directory to save checkpoints and logs.
                Defaults to "work_dir".
            max_num_checkpoints (int): The maximum number of checkpoints to save.
                If None, save all checkpoints. Defaults to None.
            checkpoint_period (int): The period (epoch-based) to save checkpoint. Defaults to 1.
            log_period (int): The period (iter-based) to log. Defaults to 50.
            clip_grad_norm (float): Max norm of the gradients. If <= 0, will not clip gradients.
                Defaults to 0.
            enable_amp (bool): Enable the Automatic Mixed Precision (AMP) training.
                Defaults to False.
            warmup_method (str): Type of warmup used. It can be None (no warmup),
                "constant", "linear" or "exp". Defaults to None.
            warmup_iters (int): The number of iterations that warmup lasts. Defaults to 1000.
            warmup_factor (float): LR used at the beginning of warmup equals to
                ``warmup_factor * initial_lr``. Defaults to 0.001.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.eval_loader = eval_data_loader if eval_data_loader is not None else data_loader
        self.work_dir = work_dir
        self.metric_storage = MetricStroge()

        self.inner_iter: int = 0  # [0, epoch_len - 1]
        self.epoch: int = 0  # [0, max_epochs - 1]
        self.start_epoch = 0  # [0, max_epochs - 1]
        self.max_epochs = max_epochs
        self.wandb = wandb

        if accelerator == "cpu":
            self.device = torch.device(accelerator)
            self.autocast_type = "cpu"
        elif accelerator == "gpu":
            assert gpus is not None, "if using gpu, please choose the gpu index"
            if is_distributed:
                self.device = torch.device(get_rank())
            else:
                self.device = torch.device(gpus)
            self.autocast_type = "cuda"
        elif accelerator == "mps":
            self.device = torch.device("mps")
            self.autocast_type = "cpu"
        else:
            raise NotImplementedError

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._max_num_checkpoints = max_num_checkpoints
        self._checkpoint_period = checkpoint_period
        self._log_period = log_period
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp
        self._cumulative_iters = cumulative_iters
        self._is_distributed = is_distributed

        if is_main_process():
            self.register_hook(self._build_default_hook())
            logger.info(f"Registered default hooks for main process: {self.registered_hook_names}")

        logger.info("Environment info:\n" + collect_env())

    @property
    def registered_hook_names(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def epoch_len(self) -> int:
        return len(self.data_loader)

    @property
    def max_iters(self) -> int:
        return self.max_epochs * self.epoch_len

    @property
    def cur_iter(self) -> int:
        return self.epoch * self.epoch_len + self.inner_iter

    @property
    def start_iter(self) -> int:
        return self.start_epoch * self.epoch_len

    @property
    def ckpt_dir(self) -> str:
        return osp.join(self.work_dir, "checkpoints")

    @property
    def tb_log_dir(self) -> str:
        return osp.join(self.work_dir, "tf_logs")

    @property
    def log_file(self) -> str:
        return osp.join(self.work_dir, "log.txt")

    @property
    def model_or_module(self) -> nn.Module:
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            return self.model.module
        return self.model

    @property
    def hook_info(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    def log(self, *args, **kwargs):
        self.metric_storage.update(*args, **kwargs)

    def _prepare_for_training(self) -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        split_line = "-" * 50
        logger.info(
            f"\n{split_line}\n"
            f"Work directory: {self.work_dir}\n"
            f"Checkpoint directory: {self.ckpt_dir}\n"
            f"Tensorboard directory: {self.tb_log_dir}\n"
            f"Log file: {self.log_file}\n"
            f"{split_line}"
        )

        if self._cumulative_iters > 1 and self._enable_amp:
            optimizer_hook = GradientCumulativeFp16OptimizerHook(grad_clip=self._clip_grad_norm,
                                                                    cumulative_iters=self._cumulative_iters)
        elif self._enable_amp:
            optimizer_hook = Fp16OptimizerHook(grad_clip=self._clip_grad_norm)
        elif self._cumulative_iters > 1:
            optimizer_hook = GradientCumulativeOptimizerHook(grad_clip=self._clip_grad_norm,
                                                                cumulative_iters=self._cumulative_iters)
        else:
            optimizer_hook = OptimizerHook(grad_clip=self._clip_grad_norm)

        if self.lr_scheduler.name == "cosine":
            lr_scheduler = CosineAnnealingLrUpdaterHook(by_epoch=False,
                                                        warmup=self.lr_scheduler.warmup_method,
                                                        warmup_ratio=self.lr_scheduler.warmup_factor,
                                                        warmup_by_epoch=True,
                                                        min_lr=self.lr_scheduler.min_lr,
                                                        warmup_iters=self.lr_scheduler.warmup_epochs)
        elif self.lr_scheduler.name == "const":
            lr_scheduler = FixedLrUpdaterHook()

        self.register_hook([optimizer_hook, lr_scheduler, DistributedHook()])
        logger.info(f"Registered default hooks for all processes: {self.hook_info}")

    def _set_to_device(self) -> None:
        self.model.to(self.device)
        if self._is_distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[get_rank()])
        logger.info("Using %s for training " % self.device)

    def _build_default_hook(self) -> List[HookBase]:
        return [
            CheckpointerHook(self._checkpoint_period, self._max_num_checkpoints,),
            LoggerHook(self._log_period, tb_log_dir=self.tb_log_dir, use_wandb=self.wandb),
        ]

    def register_hook(self, hooks: List[Optional[HookBase]]) -> None:
        """Register hooks to the trainer.

        The hooks are executed in the order they are registered.

        Args:
            hooks (list[HookBase]): List of hooks.
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
            if self._hooks and isinstance(self._hooks[-1], LoggerHook):
                self._hooks.insert(len(self._hooks) - 1, h)
            else:
                self._hooks.append(h)

    def save_checkpoint(self, file_name: str) -> None:
        """Save "epoch", "model", "optimizer", "lr_scheduler", "metric_storage",
        "hooks" (optional), "grad_scaler" (optional).

        Args:
            filename (str): The name of the file to save.
        """
        data = {
            "epoch": self.epoch,
            "num_gpus": get_world_size(),
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metric_storage": self.metric_storage,
        }
        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states

        file_path = osp.join(self.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        dst_file = osp.join(self.ckpt_dir, "latest.pth")
        symlink(file_name, dst_file)

    def load_checkpoint(self, path: str = "", checkpoint: Dict[str, Any] = None):
        """Load the given checkpoint.

        Args:
            checkpoint (dict): The checkpoint to load.
            path (str): Path to the checkpoint. If empty, will not load anything.
                `checkpoint` and `path` can only be specified one.
        """
        assert (checkpoint is not None) ^ (path != "")
        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")

        num_gpus = get_world_size()
        ckpt_num_gpus = checkpoint["num_gpus"]
        assert num_gpus == ckpt_num_gpus, (
            f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
            f"but currently only have {num_gpus} GPUs.")

        # 1. load epoch
        self.start_epoch = checkpoint["epoch"] + 1

        # 2. load metric_storage
        self.metric_storage = checkpoint["metric_storage"]

        # 3. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 4. load model
        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning("Encounter missing keys when loading model weights:\n"
                           f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning("Encounter unexpected keys when loading model weights:\n"
                           f"{incompatible.unexpected_keys}")

        # 5. load hooks
        hook_states = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}")

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _log_iter_metrics(self,
                          loss_dict: Dict[str, torch.Tensor],
                          data_time: float,
                          iter_time: float,
                          lr: float) -> None:
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict.update(data_time=data_time, iter_time=iter_time)
        # gather metrics among all workers for logging
        all_metrics_dict = gather(metrics_dict)

        if is_main_process():
            self.log(self.cur_iter, lr=lr, smooth=False)

            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            self.log(self.cur_iter, data_time=data_time)

            iter_time = np.max([x.pop("iter_time") for x in all_metrics_dict])
            self.log(self.cur_iter, iter_time=iter_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }

            if "total_loss" in metrics_dict.keys():
                loss_value = metrics_dict.pop("total_loss")
            else:
                loss_value = sum(metrics_dict.values())
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at epoch={self.epoch}! loss_dict = {loss_dict}."
                )

            self.log(self.cur_iter, total_loss=loss_value)
            if len(metrics_dict) > 1:
                self.log(self.cur_iter, **metrics_dict)

    def train_on_iter(self) -> None:
        """Train one iteration.

        .. Note::

            Standard PyTorch LR scheduler is epoch-based and called at the end of epoch.
            However, our scheduler is iteration-based, so it should be called after every iteration.

        Subclass :class:`Code.Trainer` and implement your :meth:`train_one_iter`
        to do something fancier.
        """
        iter_start_time = time.perf_counter()
        lr_this_iter = self.lr

        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        batch = next(self._data_iter)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        with torch.autocast(device_type=self.autocast_type, enabled=self._enable_amp):
            self.loss_dict = self.model(batch, self.device)

        if isinstance(self.loss_dict, torch.Tensor):
            self.loss_dict = {"total_loss": self.loss_dict}

        ###########################
        # 3. Log Metrics
        ###########################
        self._log_iter_metrics(self.loss_dict, data_time, time.perf_counter() - iter_start_time, lr_this_iter)

    def _train_one_epoch(self) -> None:
        self.model.train()
        # for _, p in self.model.named_parameters():
        #     p.requires_grad = False
        # for _, p in self.model.head.named_parameters():
        # for _, p in self.model.cls_head.named_parameters():
        #     p.requires_grad = True
        for self.inner_iter in range(self.epoch_len):
            self._call_hooks("before_iter")
            self.train_on_iter()
            self._call_hooks("after_iter")
        self._data_iter = iter(self.data_loader)

    def train(self, load_checkpoint: str = None) -> None:
        """Start training."""
        # self.model.cls_head = torch.nn.Sequential(torch.nn.BatchNorm1d(self.model.cls_head.in_features, affine=False, eps=1e-6), self.model.cls_head)
        self._prepare_for_training()
        self._set_to_device()
        if load_checkpoint is not None:
            self.load_checkpoint(path=load_checkpoint)

        logger.info(f"Start training from epoch {self.start_epoch}")
        self._call_hooks("before_train")
        for self.epoch in range(self.start_epoch, self.max_epochs):
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
        self._call_hooks("after_train")