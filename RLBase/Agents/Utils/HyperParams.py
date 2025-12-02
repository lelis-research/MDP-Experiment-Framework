from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class HyperParameters:
    """
    Flexible hyperparameter container.
    Accepts arbitrary kwargs, stores them in params, and mirrors them to attributes.
    Shared instances can be mutated at runtime and changes are visible to all holders.
    """
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        object.__setattr__(self, "params", dict(kwargs))
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def update(self, **kwargs):
        self.params.update(kwargs)
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def to_dict(self):
        return dict(self.params)

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.params})"
    
    def __repr__(self):
        cls = self.__class__.__name__
        if not self.params:
            return f"{cls}()"

        # Pretty multi-line formatting
        lines = [f"{cls}("]
        for k in sorted(self.params.keys()):
            v = self.params[k]
            lines.append(f"  {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)
    

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
