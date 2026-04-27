from ..core import Optimizer, Scheduler
import numpy as np


class LinearLR(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_lr: float | None = None,
        end_lr: float | None = None,
        total_steps: int | None = None,
        rate: float | None = None,
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
        elif start_lr is not None and end_lr is not None and rate is not None:
            self.total_steps = int((end_lr - start_lr) / rate)
        # Case 3
        elif start_lr is not None and total_steps is not None and rate is not None:
            self.end_lr = start_lr + total_steps * rate
        # Case 4
        elif end_lr is not None and total_steps is not None and rate is not None:
            self.start_lr = end_lr - total_steps * rate
        # Case 5
        elif start_lr is not None and self.rate is not None:
            self.total_steps = float("inf")

    def step(self, *args, **kwargs) -> None:
        if self.steps < self.total_steps:
            new_lr = self.start_lr + self.rate * self.steps
            self.optimizer.set_param("lr", new_lr)
        self.steps += 1


class ExponentialLR(Scheduler):
    """Decays the learning rate by a multiplicative factor each step.

    Computes the learning rate as:
        lr = start_lr * gamma^steps

    Any three of start_lr, end_lr, total_steps, gamma can be provided
    to fully define the schedule. Alternatively, only start_lr and gamma
    can be provided for an infinite decay.

    Args:
        optimizer: The optimizer whose learning rate should be scheduled.
        start_lr: Initial learning rate.
        end_lr: Final learning rate after total_steps.
        total_steps: Number of steps over which to decay.
        gamma: Multiplicative factor applied each step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_lr: float | None = None,
        end_lr: float | None = None,
        total_steps: int | None = None,
        gamma: float | None = None,
    ) -> None:
        """
        Possible combinations of parameters:
        1. start_lr, end_lr, total_steps
        2. start_lr, end_lr, gamma
        3. start_lr, total_steps, gamma
        4. end_lr, total_steps, gamma
        5. start_lr, gamma (assumes infinite total steps)
        """
        super().__init__(optimizer)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.gamma = gamma

        if (
            start_lr is not None
            and end_lr is not None
            and total_steps is not None
            and gamma is not None
        ):
            raise ValueError(
                "Parameter mismatch. Only three of the parameters start_lr, end_lr, total_steps, gamma should be provided (or two: start_lr, gamma)."
            )

        # Case 1
        if start_lr is not None and end_lr is not None and total_steps is not None:
            self.gamma = (end_lr / start_lr) ** (1.0 / total_steps)
        # Case 2
        elif start_lr is not None and end_lr is not None and gamma is not None:
            self.total_steps = int(round(np.log(end_lr / start_lr) / np.log(gamma)))
        # Case 3
        elif start_lr is not None and total_steps is not None and gamma is not None:
            self.end_lr = start_lr * (gamma**total_steps)
        # Case 4
        elif end_lr is not None and total_steps is not None and gamma is not None:
            self.start_lr = end_lr / (gamma**total_steps)
        # Case 5
        elif start_lr is not None and gamma is not None:
            self.total_steps = float("inf")
        else:
            raise ValueError(
                "Invalid parameter combination for ExponentialLR. "
                "Provide exactly three of start_lr, end_lr, total_steps, gamma (or two: start_lr, gamma)."
            )

    def step(self, *args, **kwargs) -> None:
        if self.steps < self.total_steps:
            new_lr = self.start_lr * (self.gamma**self.steps)
            self.optimizer.set_param("lr", new_lr)
        self.steps += 1


class CosineAnnealingLR(Scheduler):
    """Sets the learning rate using cosine annealing with warm restarts.

    The learning rate follows:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t_cur / T_i))

    where t_cur is the number of steps since the last restart and T_i is the
    length of the current period. After each period expires, t_cur resets to 0
    and T_i is multiplied by T_mult.

    Args:
        optimizer: The optimizer whose learning rate should be scheduled.
        T_0: Number of steps in the first restart period.
        T_mult: Factor by which the period length is multiplied after each restart. Default is 1 (period length never changes).
        lr_min: Minimum learning rate. Default is 0.
        lr_max: Peak learning rate at the start of each period. If None, the optimizer's current learning rate is used. If provided, the optimizer's learning rate is set to this value immediately.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        lr_min: float = 0.0,
        lr_max: float | None = None,
    ) -> None:
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.lr_min = lr_min
        self.lr_max = lr_max if lr_max is not None else optimizer.lr
        self.T_cur = 0
        self.T_i = T_0

        if lr_max is not None:
            optimizer.set_param("lr", lr_max)

    def step(self, *args, **kwargs) -> None:
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * self.T_cur / self.T_i)
        )
        self.optimizer.set_param("lr", lr)
        self.T_cur += 1
        self.steps += 1

        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
