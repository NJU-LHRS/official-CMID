import time
import logging
import ml_collections
import pytorch_lightning as pl
import torch.distributed as dist
from optimizer import build_optimizer
from .pretrain_model import MoCoBasedDDP, MoCoBased
from typing import Dict, Any, Sequence, Optional
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
logger = logging.getLogger("train")


def get_rank() -> int:
    """Return the rank of the current process in the current process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Return if the current process is the master process or not."""
    return get_rank() == 0


class PLPretrainModel(pl.LightningModule):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(PLPretrainModel, self).__init__()
        self.model = MoCoBasedDDP(config)
        self.config = config

    def configure_optimizers(self):
        opt = build_optimizer(self.model, self.config)

        if self.config.name == "cosine":
            max_warmup_steps = (self.config.warmup_epochs)
            max_scheduler_steps = (self.trainer.max_epochs)
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    opt,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.config.min_lr if self.config.warmup_epochs > 0 else self.config.lr,
                    eta_min=self.config.min_lr,
                ),
                "interval": "epoch",
                "frequency": 1,
            }

        return [opt], [scheduler]

    def forward(self, X):
        return self.model(X, self.device)

    def on_train_epoch_start(self):
        """Updates the current epoch in DINO's loss object."""
        self.model.class_loss.epoch = self.current_epoch

    def training_step(self, X):
        result_dict = self(X)

        if self.trainer.global_step % 100 == 0 and is_main_process():
            process_string = f"Epoch: [{self.trainer.current_epoch}][{self.trainer.global_step}/{self.trainer.estimated_stepping_batches - 1}]"

            space = " " * 2
            iter_time = time.perf_counter() - self.iter_time

            loss_strings = [
                f"{key}: {his_buf.detach().cpu().numpy():.4g}"
                for key, his_buf in result_dict.items()
                if "loss" in key or "acc" in key
            ]
            logger.info(
                "{process}{losses}{iter_time}".format(
                    process=process_string,
                    losses=space + "  ".join(loss_strings) if loss_strings else "",
                    iter_time=space + f"iter_time: {iter_time:.4f}" if iter_time is not None else ""
                )
            )

        return result_dict["total_loss"]

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.iter_time = time.perf_counter()

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.model.momentum_pairs
            for mp in momentum_pairs:
                self.model.ema.update(*mp)
            # log tau momentum
            self.log("tau", self.model.ema.cur_tau, prog_bar=True, on_step=True)
            # update tau
            self.model.ema.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step