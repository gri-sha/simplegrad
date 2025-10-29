class Function:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__init__()
        if args:
            # Direct call: create instance and apply immediately
            instance._stored_kwargs = {}
            return instance.apply(*args, **kwargs)
        else:
            # Instantiation: create instance and store kwargs
            instance._stored_kwargs = kwargs
            return instance
    
    def __init__(self, **kwargs):
        # Initialize _stored_kwargs if not already set by __new__
        # if not hasattr(self, '_stored_kwargs'):
        #     self._stored_kwargs = {}
        pass
    
    def __call__(self, *args, **kwargs):
        # Check for conflicts: if any key in stored_kwargs is also in kwargs, raise error
        for key in self._stored_kwargs:
            if key in kwargs:
                raise ValueError(
                    f"Parameter '{key}' is already set for this function instance and cannot be overridden."
                )
        # Combine stored kwargs with new kwargs
        combined_kwargs = {**self._stored_kwargs, **kwargs}
        return self.apply(*args, **combined_kwargs)
    
    def apply(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the apply method")
    
    def __repr__(self):
        return f"NonLabeledFunction()"
