from ..core import Optimizer, Scheduler


class LinearLR(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_lr: float | None,
        end_lr: float | None,
        total_steps: int | None,
        rate: float | None,
    ) -> None:
        """
        Possible combinations of parameters:
        1. start_lr, end_lr, total_steps
        2. start_lr, end_lr, rate
        3. start_lr, total_steps, rate
        4. end_lr, total_steps, rate
        5. start_lr, rate (assumes infinite total steps)
        """
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.rate = rate

        if (
            start_lr is not None
            and end_lr is not None
            and total_steps is not None
            and rate is not None
        ):
            raise ValueError(
                "Parameter mismatch. Only three of the parameters start_lr, end_lr, total_steps, rate should be provided (or two: start_lr, rate)."
            )

        # Case 1
        if start_lr is not None and end_lr is not None and total_steps is not None:
            self.rate = (end_lr - start_lr) / total_steps
        # Case 2
        elif start_lr is not None and end_lr is not None and self.rate is not None:
            self.total_steps = int((end_lr - start_lr) / self.rate)
        # Case 3
        elif start_lr is not None and total_steps is not None and self.rate is not None:
            self.end_lr = start_lr + total_steps * self.rate
        # Case 4
        elif end_lr is not None and total_steps is not None and self.rate is not None:
            self.start_lr = end_lr - total_steps * self.rate
        # Case 5
        elif start_lr is not None and self.rate is not None:
            self.total_steps = float("inf")

    def step(self, *args, **kwargs) -> None:
        if self.steps < self.total_steps:
            new_lr = self.start_lr + self.rate * self.steps
            self.optimizer.set_lr(new_lr)
        self.steps += 1


class ExponentialLR(Scheduler):
    pass


class CosineAnnealingLR(Scheduler):
    pass
