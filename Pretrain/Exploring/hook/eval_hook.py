import torch
import logging
from .hookbase import HookBase
from ..utils import MetricStroge, accuracy_at_k
logger = logging.getLogger("train")


class EvalHook(HookBase):
    """Run an evaluation function periodically.

    It is executed every ``period`` epochs and after the last epoch.
    """

    def __init__(self, period: int):
        """
        Args:
            period (int): The period to run ``eval_func``. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_func (callable): A function which takes no arguments, and
                returns a dict of evaluation metrics.
        """
        super(EvalHook, self).__init__()
        self._period = period
        self.max_acc = None

    @torch.no_grad()
    def _eval_func(self):
        self.trainer.model_or_module.eval()
        metric = MetricStroge(window_size=len(self.trainer.eval_loader))

        for idx, batch in enumerate(self.trainer.eval_loader):
            images = batch[0]
            target = batch[-1].to(self.trainer.device, non_blocking=True)

            with torch.autocast(device_type=self.trainer.autocast_type, enabled=self.trainer._enable_amp):
                result = self.trainer.model(batch, self.trainer.device, return_loss=False, eval=True)

            acc1, acc5 = accuracy_at_k(result, target, top_k=(1, 5))
            metric.update(acc1=acc1.detach().cpu().numpy()[0], smooth=False)
            metric.update(acc5=acc5.detach().cpu().numpy()[0], smooth=False)


        acc1 = metric["acc1"].global_avg
        acc5 = metric["acc5"].global_avg

        logger.info("Epoch: {0}\tacc@1: {1:.4f}\tacc@5: {2:.4f}".format(
            self.trainer.epoch, acc1, acc5
        ))

        if self.max_acc is None:
            self.max_acc = acc1
        else:
            if acc1 > self.max_acc:
                self.max_acc = acc1
                self.trainer.save_checkpoint(f"best_ckpt_{str(self.trainer.epoch)}_{str(acc1)}.pth")

        logger.info(f'Max accuracy: {self.max_acc:.2f}%')

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._eval_func()