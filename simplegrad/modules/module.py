from simplegrad.core.tensor import Tensor


class Module:
    def __init__(self):
        self.label = self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement 'forward'")

    # # TODO rewrite too long representations
    # def __repr__(self):
    #     attrs = ", ".join(
    #         f"{k}={v!r}" for k, v in self.__dict__.items() if k != "label"
    #     )
    #     return f"{self.label}({attrs})"

    def _add_to_graph(self, label=None):
        pass

    def parameters(self):
        """
        Recursively collects all trainable parameters (Tensors) from this module and its child modules.

        Returns:
            list: A list of Tensor objects that are trainable parameters.
        """
        params = []
        for attr_value in self.__dict__.values():
            if isinstance(attr_value, Tensor):
                # Direct tensor attribute is a parameter
                params.append(attr_value)
            elif isinstance(attr_value, Module):
                # Recursively collect parameters from child modules
                params.extend(attr_value.parameters())
            elif isinstance(attr_value, (list, tuple)):
                # Handle collections of modules or tensors
                for item in attr_value:
                    if isinstance(item, Tensor):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def named_parameters(self, prefix=""):
        """
        Recursively collects all trainable parameters with their hierarchical names.

        Args:
            prefix (str): Prefix for parameter names (used for nested modules).

        Returns:
            list: A list of tuples (name, tensor) for each parameter.
        """
        named_params = []
        for attr_name, attr_value in self.__dict__.items():
            if attr_name == "label":
                continue
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(attr_value, Tensor):
                named_params.append((full_name, attr_value))
            elif isinstance(attr_value, Module):
                named_params.extend(attr_value.named_parameters(prefix=full_name))
            elif isinstance(attr_value, (list, tuple)):
                for i, item in enumerate(attr_value):
                    item_name = f"{full_name}[{i}]"
                    if isinstance(item, Tensor):
                        named_params.append((item_name, item))
                    elif isinstance(item, Module):
                        named_params.extend(item.named_parameters(prefix=item_name))
        return named_params

    def modules(self):
        """
        Recursively yields all modules including this one.

        Yields:
            Module: Each module in the hierarchy.
        """
        yield self
        for attr_value in self.__dict__.values():
            if isinstance(attr_value, Module):
                yield from attr_value.modules()
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, Module):
                        yield from item.modules()
