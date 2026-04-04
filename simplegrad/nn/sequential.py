"""Sequential container for chaining modules."""

from ..core import Tensor, Module


class Sequential(Module):
    """A sequential container that applies modules in the order they are passed.

    Args:
        *modules: Any number of Module instances to chain together.

    Example:
        ```python
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        output = model(x)
        ```
    """

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        """Pass input through each module in sequence.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after all modules have been applied.
        """
        for module in self.modules:
            x = module(x)
        return x

    def __str__(self):
        res = "Sequential(\n"
        for module in self.modules:
            res += f"  {str(module)},\n"
        res += ")"
        return res
