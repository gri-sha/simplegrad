"""Abstract base class for all optimizers."""

from .module import Module
from .devices import get_backend


class Optimizer:
    """Base class for all optimizers.

    Subclasses must implement `step()` to define the parameter update rule.
    """

    def __init__(
        self,
        lr: float,
        model: Module | None = None,
        param_groups: list[dict] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the optimizer.

        Accepts parameters in one of two forms:

        - ``model``: a single Module; all its parameters form one group labelled
          ``"default"``.
        - ``param_groups``: a list of dicts, each with a ``"params"`` key (a
          Module or a ``dict[str, Tensor]``) and optional per-group overrides.

        All keyword arguments (``lr`` and any subclass-specific hyperparameters
        such as ``momentum``, ``beta_1``, etc.) become the defaults for every
        group. A value supplied inside a group dict overrides the corresponding
        default for that group only.

        Args:
            lr: Default learning rate.
            model: Model whose parameters to optimize (single-group shorthand).
            param_groups: Explicit list of parameter group dicts.
            **kwargs: Additional default hyperparameters forwarded by subclasses.

        Raises:
            ValueError: If both ``model`` and ``param_groups`` are provided,
                neither is provided, or ``lr`` is None.
        """
        if model is not None and param_groups is not None:
            raise ValueError("Cannot specify both model and param_groups. Choose one.")
        if model is None and param_groups is None:
            raise ValueError("Either model or param_groups must be provided.")
        if lr is None:
            raise ValueError("Learning rate (lr) must be provided.")

        self.lr = lr
        self.step_count = 0
        self.param_groups = self._resolve_param_groups(lr, model, param_groups, kwargs)

    def _resolve_param_groups(
        self,
        lr: float,
        model: Module | None,
        param_groups: list[dict] | None,
        extra_defaults: dict,
    ) -> list[dict]:
        """Normalize constructor inputs into the internal param_groups format.

        Each resolved group is a dict containing at minimum:
        ``{"label": str, "params": dict[str, Tensor], "lr": float, ...}``.
        Per-group values take precedence over constructor defaults.

        Args:
            lr: Default learning rate.
            model: Optional single Module (shorthand for one default group).
            param_groups: Optional list of raw group dicts.
            extra_defaults: Subclass hyperparameter defaults passed via kwargs.

        Returns:
            Resolved list of parameter group dicts.
        """
        all_defaults = {"lr": lr, **extra_defaults}

        if model is not None:
            return [{"label": "default", "params": model.parameters(), **all_defaults}]

        resolved = []
        for i, group in enumerate(param_groups):
            group = dict(group)
            name = group.pop("label", f"group_{i}")
            params_spec = group.pop("params")

            if isinstance(params_spec, Module):
                params = params_spec.parameters()
            else:
                params = dict(params_spec)

            # all_defaults first so per-group values override them
            resolved.append({"label": name, "params": params, **all_defaults, **group})

        return resolved

    def zero_grad(self) -> None:
        """Zero gradients for all parameters across all groups."""
        for group in self.param_groups:
            for param in group["params"].values():
                xp = get_backend(param.device)
                param.grad = xp.zeros_like(param.values)

    def step(self):
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError("step() method is not implemented.")

    def reset_step_count(self) -> None:
        """Reset the internal step counter to zero."""
        self.step_count = 0

    def set_param(self, key: str, value, group: str | None = None) -> None:
        """Set a hyperparameter value for one or all parameter groups.

        This is the general-purpose setter for any hyperparameter stored in
        ``param_groups`` (e.g. ``"lr"``, ``"momentum"``, ``"beta_1"``).
        When updating ``"lr"`` across all groups, ``self.lr`` is also kept in
        sync for scheduler compatibility.

        Args:
            key: Name of the hyperparameter to update (e.g. ``"lr"``).
            value: New value to assign.
            group: Label of the group to update. If None, all groups are updated.

        Raises:
            KeyError: If ``key`` does not exist in the targeted group(s).

        Example:
            >>> optimizer.set_param("lr", 1e-4)
            >>> optimizer.set_param("momentum", 0.99, group="encoder")
        """
        for g in self.param_groups:
            if group is None or g["label"] == group:
                if key not in g:
                    raise KeyError(f"Hyperparameter '{key}' not found in group '{g['label']}'.")
                g[key] = value
        if key == "lr" and group is None:
            self.lr = value

    def state(self) -> dict:
        """Return the current optimizer state.

        Returns:
            Dict with ``step_count`` and a ``param_groups`` list, each entry
            containing the group label and its current learning rate.
            Subclasses may override this to include additional state.
        """
        return {
            "step_count": self.step_count,
            "param_groups": [{"label": g["label"], "lr": g["lr"]} for g in self.param_groups],
        }
