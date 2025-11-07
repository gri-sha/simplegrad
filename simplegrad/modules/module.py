import graphviz
from simplegrad.core.tensor import Tensor


class Module:
    def __init__(self):
        self.label = self.__class__.__name__
        self._parameters = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement 'forward'")

    def parameters(self, force_refresh=False):
        if not self._parameters or force_refresh:
            params = self._get_parameters()
            self._parameters = params
        return self._parameters

    def _get_parameters(self, prefix=""):
        params = {}
        for attr_name, attr_value in self.__dict__.items():
            if attr_name == "label" or attr_name == "_parameters":
                continue

            key = f"{prefix}.{attr_name}" if prefix else attr_name

            if isinstance(attr_value, Tensor):
                params[key] = attr_value
            elif isinstance(attr_value, Module):
                module_params = attr_value._get_parameters(prefix=key)
                params.update(module_params)
            elif isinstance(attr_value, (list, tuple)):
                for i, item in enumerate(attr_value):
                    item_key = f"{key}.{i}"
                    if isinstance(item, Tensor):
                        params[item_key] = item
                    elif isinstance(item, Module):
                        module_params = item._get_parameters(prefix=item_key)
                        params.update(module_params)

        return params
