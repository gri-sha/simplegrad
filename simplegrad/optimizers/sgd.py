"""Stochastic Gradient Descent optimizer with optional momentum."""

from ..core import Optimizer, Module
from ..core.devices import get_backend


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

    Update rule (with momentum)::

        v_t = momentum * v_{t-1} - lr * (1 - dampening) * grad
        param += v_t

    Supports parameter groups, allowing different ``lr``, ``momentum``, and ``dampening`` per group.
    Pass a list of dicts to ``param_groups``, each with a ``"params"`` key (a Module or a ``dict[str, Module]``)
    and optional per-group overrides:

        >>> optimizer = SGD(
        ...     lr=0.01,
        ...     momentum=0.9,
        ...     param_groups=[
        ...         {"params": model.encoder},
        ...         {"params": model.decoder, "lr": 0.001},
        ...     ],
        ... )

    Args:
        model: The model whose parameters to optimize (single-group shorthand).
        lr: Default learning rate. Defaults to 0.01.
        momentum: Default momentum factor. 0 disables momentum. Defaults to 0.
        dampening: Default dampening applied to the gradient. Defaults to 0.
        param_groups: List of parameter group dicts with optional per-group overrides for ``lr``, ``momentum``, and ``dampening``.

    Raises:
        TypeError: If ``model`` is provided but is not a Module.
    """

    def __init__(
        self,
        model: Module | None = None,
        lr: float = 0.01,
        momentum: float = 0,
        dampening: float = 0,
        param_groups: list[dict] | None = None,
    ) -> None:
        if model is not None and not isinstance(model, Module):
            raise TypeError("model must be a Module")

        super().__init__(lr, model, param_groups, momentum=momentum, dampening=dampening)

        self.velocities = {
            (group["label"], name): get_backend(param.device).zeros_like(param.values)
            for group in self.param_groups
            for name, param in group["params"].items()
        }

    def step(self) -> None:
        """Apply one SGD update step to all parameters across all groups.

        Uses the ``lr``, ``momentum``, and ``dampening`` stored in each parameter group,
        so different groups may use different hyperparameters.

        Raises:
            ValueError: If any parameter gradient is None (forgot to call backward).
        """
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            for name, param in group["params"].items():
                if param.grad is None:
                    raise ValueError(
                        f"Gradient for {name} is None. Did you forget to call backward()?"
                    )
                key = (group["label"], name)
                self.velocities[key] = (
                    momentum * self.velocities[key] - lr * (1 - dampening) * param.grad
                )
                param.values += self.velocities[key]

    def state(self) -> dict:
        """Return the full optimizer state, including hyperparameters and velocities.

        The returned dict contains enough information to resume training from an exact checkpoint.
        Velocities are nested inside each group entry so the structure mirrors ``param_groups``.

        Returns:
            Dict with ``step_count`` and a ``param_groups`` list.
            Each group entry contains its ``label``, hyperparameters (``lr``,``momentum``, ``dampening``),
            and a ``velocities`` dict mapping parameter names to their current velocity arrays.
        """
        return {
            "step_count": self.step_count,
            "param_groups": [
                {
                    "label": group["label"],
                    "lr": group["lr"],
                    "momentum": group["momentum"],
                    "dampening": group["dampening"],
                    "velocities": {
                        name: self.velocities[(group["label"], name)].copy()
                        for name in group["params"]
                    },
                }
                for group in self.param_groups
            ],
        }
