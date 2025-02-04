class HyperParameters:
    """
    A hyper-parameters class that stores parameters both as dictionary
    entries and as attributes of the instance.
    """
    def __init__(self, **kwargs):
        # Create an internal dictionary to store the parameters.
        self.params = {}
        # Loop over each key-value pair and set it as an attribute.
        for key, value in kwargs.items():
            self.params[key] = value
            setattr(self, key, value)

    def update(self, **kwargs):
        """
        Update the hyper-parameters with provided keyword arguments.
        This updates both the internal dictionary and the instance attributes.
        """
        for key, value in kwargs.items():
            self.params[key] = value
            setattr(self, key, value)


    def to_dict(self):
        """
        Return a dictionary representation of the hyper-parameters.
        """
        return dict(self.params)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"