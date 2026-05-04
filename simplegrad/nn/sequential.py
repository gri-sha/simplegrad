"""Sequential container for chaining modules."""

from ..core import Tensor, Module


class Sequential(Module):
    """A sequential container that applies modules in the order they are passed.

    Args:
        *modules: Any number of Module instances to chain together.

    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through each module in sequence."""
        for module in self.modules:
            x = module(x)
        return x

    def __str__(self):
        res = "Sequential(\n"
        for module in self.modules:
            res += f"  {str(module)},\n"
        res += ")"
        return res
