"""AdamW optimizer (Adam with decoupled weight decay)."""

from ..core import Optimizer, Module
from ..core.devices import get_backend


class AdamW(Optimizer):
    """AdamW optimizer: Adam with decoupled weight decay regularization.

    Standard Adam folds L2 regularization into the gradient before computing
    the adaptive scaling, which means the effective regularization strength
    varies with the gradient magnitude. AdamW decouples weight decay from the
    gradient update, applying it directly to the parameters. This produces
    better-regularized models, especially when using adaptive learning rates.

    Update rule (``maximize=False``)::

        m_t = beta_1 * m_{t-1} + (1 - beta_1) * grad
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * grad^2
        m_hat = m_t / (1 - beta_1^t)
        v_hat = v_t / (1 - beta_2^t)
        param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

    The weight-decay term ``lr * weight_decay * param`` is applied after the
    bias-corrected Adam step, keeping it independent of the gradient scaling.

    When ``maximize=True`` the Adam component of the update is negated (the
    weight-decay component always pulls parameters toward zero regardless).

    Supports parameter groups, allowing different hyperparameters per group:

        >>> optimizer = AdamW(
        ...     lr=1e-3,
        ...     param_groups=[
        ...         {"params": model.encoder},
        ...         {"params": model.decoder, "lr": 1e-4, "weight_decay": 0.0},
        ...     ],
        ... )

    Args:
        model: The model whose parameters to optimize (single-group shorthand).
        lr: Default learning rate. Defaults to 1e-3.
        beta_1: Default exponential decay for the first moment. Defaults to 0.9.
        beta_2: Default exponential decay for the second moment. Defaults to 0.999.
        eps: Default numerical stability constant. Defaults to 1e-8.
        weight_decay: Decoupled weight decay coefficient. Defaults to 0.
        maximize: If True, maximizes the objective instead of minimizing it.
            Defaults to False.
        param_groups: List of parameter group dicts with optional per-group
            overrides for ``lr``, ``beta_1``, ``beta_2``, ``eps``,
            ``weight_decay``, and ``maximize``.
    """

    def __init__(
        self,
        model: Module | None = None,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0,
        maximize: bool = False,
        param_groups: list[dict] | None = None,
    ) -> None:
        super().__init__(
            lr, model, param_groups,
            beta_1=beta_1, beta_2=beta_2, eps=eps,
            weight_decay=weight_decay, maximize=maximize,
        )

        self.moments1 = {
            (group["label"], name): get_backend(param.device).zeros_like(param.values)
            for group in self.param_groups
            for name, param in group["params"].items()
        }
        self.moments2 = {
            (group["label"], name): get_backend(param.device).zeros_like(param.values)
            for group in self.param_groups
            for name, param in group["params"].items()
        }

    def step(self) -> None:
        """Apply one AdamW update step to all parameters across all groups.

        Applies the decoupled weight decay after the bias-corrected Adam update.
        Uses the hyperparameters stored in each parameter group.

        Raises:
            ValueError: If any parameter gradient is None (forgot to call backward).
        """
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            for name, param in group["params"].items():
                if param.grad is None:
                    raise ValueError(
                        f"Gradient for {name} is None. Did you forget to call backward()?"
                    )
                key = (group["label"], name)

                # Update biased first moment estimate
                self.moments1[key] = beta_1 * self.moments1[key] + (1 - beta_1) * param.grad

                # Update biased second raw moment estimate
                self.moments2[key] = beta_2 * self.moments2[key] + (1 - beta_2) * (param.grad**2)

                # Compute bias-corrected estimates
                m_hat = self.moments1[key] / (1 - beta_1**self.step_count)
                v_hat = self.moments2[key] / (1 - beta_2**self.step_count)

                xp = get_backend(param.device)
                adam_update = lr * m_hat / (xp.sqrt(v_hat) + eps)
                if maximize:
                    param.values += adam_update
                else:
                    param.values -= adam_update

                # Decoupled weight decay — always reduces parameter magnitude
                param.values -= lr * weight_decay * param.values

    def state(self) -> dict:
        """Return the full optimizer state, including hyperparameters and moment estimates.

        Returns:
            Dict with ``step_count`` and a ``param_groups`` list. Each group
            entry contains its label, all hyperparameters, and ``moments1`` /
            ``moments2`` dicts mapping parameter names to their current arrays.
        """
        return {
            "step_count": self.step_count,
            "param_groups": [
                {
                    "label": group["label"],
                    "lr": group["lr"],
                    "beta_1": group["beta_1"],
                    "beta_2": group["beta_2"],
                    "eps": group["eps"],
                    "weight_decay": group["weight_decay"],
                    "maximize": group["maximize"],
                    "moments1": {
                        name: self.moments1[(group["label"], name)].copy()
                        for name in group["params"]
                    },
                    "moments2": {
                        name: self.moments2[(group["label"], name)].copy()
                        for name in group["params"]
                    },
                }
                for group in self.param_groups
            ],
        }
