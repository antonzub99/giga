from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class NoOpScheduler(LRScheduler):
    """Scheduler that maintains constant learning rate."""

    def __init__(self, optimizer: Optimizer):
        """Initialize scheduler with optimizer.

        Args:
            optimizer: The optimizer whose learning rate should be maintained.
        """
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> list[float]:
        """Return the current learning rates.

        Returns:
            List of learning rates for each parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch: int = None) -> None:
        """Step the scheduler.

        Args:
            epoch: The current epoch. Not used in this case.
        """
        pass
