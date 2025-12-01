import numpy as np
from simplegrad.core.tensor import Tensor
from simplegrad.dtypes import convert_to_dtype


class Module:
    def __init__(self) -> None:
        self.label = self.__class__.__name__
        self._parameters = {}

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Subclasses must implement 'forward'")

    def parameters(self, force_refresh: bool = False) -> dict[str, Tensor]:
        if not self._parameters or force_refresh:
            params = self._get_parameters()
            self._parameters = params
        return self._parameters
    
    @staticmethod
    def _init_param(shape, label, k, dtype):
        limit = np.sqrt(k)
        data = np.random.uniform(-limit, limit, size=shape)
        return Tensor(convert_to_dtype(array=data, dtype=dtype), label=label)

    def _get_parameters(self, prefix: str = "") -> dict[str, Tensor]:
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

    def __str__(self) -> str:
        return f"{self.label} Module"

    def summary(self) -> None:
        print(f"Parameters of {self.label}\n")
        print(f"{'Parameter':<20} {'Shape':<15} {'Trainable Parameters':<40}")
        print("-" * 60)
        params = self.parameters()
        tot = 0
        for name, tensor in params.items():
            tr = np.prod(tensor.values.shape)
            tot += tr
            print(f"{name:<20} {str(tensor.values.shape):<15} {str(tr):<40}")
        print("-" * 60)
        print(f"Total trainable parameters: {tot}")
