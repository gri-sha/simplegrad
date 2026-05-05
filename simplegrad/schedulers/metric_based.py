from ..core import Optimizer, Scheduler


class ReduceLROnPlateauLR(Scheduler):
    """Reduce learning rate when a metric has stopped improving.

    After each call to :meth:`step`, this scheduler compares the provided metric
    against the best observed value. When the metric has not improved for
    ``patience`` consecutive steps, the learning rate is reduced by ``factor``.
    This allows the optimizer to escape plateaus and continue converging.
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
