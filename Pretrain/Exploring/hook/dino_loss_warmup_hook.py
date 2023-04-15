from .hookbase import HookBase


class DINOLossWarmUp(HookBase):
    def __init__(self,
                 warmup_epoch: int):
        super(DINOLossWarmUp, self).__init__()
        self.warmup_epoch = warmup_epoch

    def before_epoch(self) -> None:
        if self.trainer.epoch >= self.warmup_epoch:
            self.trainer.model_or_module.class_loss.epoch = self.trainer.epoch - self.warmup_epoch