class HyperParameters:
    """
    Stores hyperparameters as both dictionary entries and instance attributes.
    """
    def __init__(self, **kwargs):
        self.params = {}
        # Set each parameter as an attribute.
        for key, value in kwargs.items():
            self.params[key] = value
            setattr(self, key, value)

    def update(self, **kwargs):
        # Update parameters and attributes.
        for key, value in kwargs.items():
            self.params[key] = value
            setattr(self, key, value)

    def to_dict(self):
        # Return a dictionary representation.
        return dict(self.params)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"
    
    def __getstate__(self):
        # Return state for pickling.
        return self.__dict__

    def __setstate__(self, state):
        # Restore state from pickled data.
        self.__dict__.update(state)