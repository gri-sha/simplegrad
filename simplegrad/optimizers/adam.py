"""Adam optimizer."""

from ..core import Optimizer, Module
from ..core.devices import get_backend


class Adam(Optimizer):
    """Adam optimizer with bias-corrected moment estimates.

    Update rule::

        m_t = beta_1 * m_{t-1} + (1 - beta_1) * grad
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * grad^2
        m_hat = m_t / (1 - beta_1^t)
        v_hat = v_t / (1 - beta_2^t)
        param -= lr * m_hat / (sqrt(v_hat) + eps)

    Supports parameter groups, allowing different ``lr``, ``beta_1``,
    ``beta_2``, and ``eps`` per group. Pass a list of dicts to ``param_groups``,
    each with a ``"params"`` key (a Module or a ``dict[str, Tensor]``) and
    optional per-group overrides:

        >>> optimizer = Adam(
        ...     lr=1e-3,
        ...     param_groups=[
        ...         {"params": model.encoder},
        ...         {"params": model.decoder, "lr": 1e-4, "beta_1": 0.8},
        ...     ],
        ... )

    Args:
        model: The model whose parameters to optimize (single-group shorthand).
        lr: Default learning rate. Defaults to 1e-3.
        beta_1: Default exponential decay for the first moment. Defaults to 0.9.
        beta_2: Default exponential decay for the second moment. Defaults to 0.999.
        eps: Default numerical stability constant. Defaults to 1e-8.
        param_groups: List of parameter group dicts with optional per-group
            overrides for ``lr``, ``beta_1``, ``beta_2``, and ``eps``.
    """

    def __init__(
        self,
        model: Module | None = None,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        param_groups: list[dict] | None = None,
    ) -> None:
        super().__init__(lr, model, param_groups, beta_1=beta_1, beta_2=beta_2, eps=eps)

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
        """Apply one Adam update step to all parameters across all groups.

        Uses the ``lr``, ``beta_1``, ``beta_2``, and ``eps`` stored in each
        parameter group, so different groups may use different hyperparameters.

        Raises:
            ValueError: If any parameter gradient is None (forgot to call backward).
        """
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
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

                # Compute bias-corrected first moment estimate
                m_hat = self.moments1[key] / (1 - beta_1**self.step_count)

                # Compute bias-corrected second raw moment estimate
                v_hat = self.moments2[key] / (1 - beta_2**self.step_count)

                # Update parameters
                xp = get_backend(param.device)
                param.values -= lr * m_hat / (xp.sqrt(v_hat) + eps)

    def state(self) -> dict:
        """Return the full optimizer state, including hyperparameters and moment estimates.

        The returned dict contains enough information to resume training from
        an exact checkpoint. Moment arrays are nested inside each group entry
        so the structure mirrors ``param_groups``.

        Returns:
            Dict with ``step_count`` and a ``param_groups`` list. Each group
            entry contains its ``label``, hyperparameters (``lr``, ``beta_1``,
            ``beta_2``, ``eps``), and ``moments1`` / ``moments2`` dicts mapping
            parameter names to their current first and second moment arrays.
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
