class Function:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        if args or kwargs:
            return instance.apply(*args, **kwargs)
        return instance
    
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)
    
    def apply(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the apply method")
    
    def __repr__(self):
        return f"NonLabeledFunction()"