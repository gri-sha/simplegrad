from ..core import Optimizer, Scheduler


class ReduceLROnPlateauLR(Scheduler):
    """Reduce learning rate when a metric has stopped improving.

    After each call to :meth:`step`, this scheduler compares the provided metric
    against the best observed value. When the metric has not improved for
    ``patience`` consecutive steps, the learning rate is reduced by ``factor``.
    This allows the optimizer to escape plateaus and continue converging.

    Args:
        optimizer: The optimizer whose learning rate should be scheduled.
        factor: Multiplicative factor applied to the learning rate when a plateau
            is detected. For example, 0.1 reduces the lr by a factor of 10.
        patience: Number of consecutive steps with no improvement before the
            learning rate is reduced. Default is 10.
        min_lr: Lower bound on the learning rate. Can be a scalar or a list of
            floats for each parameter group. Default is 0.
        threshold: Minimum change in the monitored quantity to qualify as an
            improvement. Default is 1e-4.
        threshold_mode: One of ``'rel'`` or ``'abs'``. In ``'rel'`` mode, an
            improvement is detected when ``best * (1 + threshold)`` is exceeded
            (for ``maximize_metric=True``) or when ``best * (1 - threshold)``
            is exceeded (for ``maximize_metric=False``). In ``'abs'`` mode, the
            threshold is added/subtracted directly from the best value.
            Default is ``'rel'``.
        cooldown: Number of steps to wait after a learning rate reduction before
            monitoring resumes. During cooldown, the scheduler ignores
            improvements and only counts steps toward exiting cooldown.
            Default is 0.
        maximize_metric: If ``True``, the scheduler treats higher metric values
            as better (e.g., for accuracy). If ``False``, lower values are better
            (e.g., for loss). Default is ``False``.
        verbose: If ``True``, a message is printed each time the learning rate
            is reduced. Default is ``False``.
        eps: Minimum decay applied to lr. If the difference between new and old
            lr is smaller than eps, the update is ignored. Default is 1e-8.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float,
        patience: int = 10,
        min_lr: float = 0.0,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        maximize_metric: bool = False,
        verbose: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(optimizer)
        if factor >= 1.0:
            raise ValueError("factor must be less than 1.0")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError("threshold_mode must be 'rel' or 'abs'")

        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.maximize_metric = maximize_metric
        self.verbose = verbose
        self.eps = eps

        self.best = None
        self.num_bad_steps = 0
        self.cooldown_remaining = 0

    def step(self, metric: float) -> None:
        """Update the learning rate based on the provided metric.

        Args:
            metric: The current value of the monitored metric (e.g., validation
                loss). Unlike other schedulers, this method requires
                a metric value, not an epoch number.
        """
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.steps += 1
            return

        if self.best is None:
            self.best = metric
        else:
            if self._is_improvement(metric):
                self.best = metric
                self.num_bad_steps = 0
            else:
                self.num_bad_steps += 1

        if self.num_bad_steps >= self.patience:
            new_lr = self._compute_new_lr()
            if new_lr < self.optimizer.lr - 1e-12:
                self.optimizer.set_param("lr", new_lr)
                if self.verbose:
                    print(f"ReduceLROnPlateauLR: reducing learning rate to {new_lr:.4e}")
                self.cooldown_remaining = self.cooldown
            self.num_bad_steps = 0
            self.best = metric

        self.steps += 1

    def _is_improvement(self, metric: float) -> bool:
        if self.maximize_metric:
            if self.threshold_mode == "rel":
                return metric > self.best * (1 + self.threshold)
            else:
                return metric > self.best + self.threshold
        else:
            if self.threshold_mode == "rel":
                return metric < self.best * (1 - self.threshold)
            else:
                return metric < self.best - self.threshold

    def _compute_new_lr(self) -> float:
        new_lr = self.optimizer.lr * self.factor
        return max(new_lr, self.min_lr)
