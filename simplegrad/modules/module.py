class Module:
    def __init__(self):
        self.label = self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement 'forward'")

    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items() if k != "label")
        return f"{self.label}({attrs})"