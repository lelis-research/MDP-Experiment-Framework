import importlib.util
import sys

def load_config(path):
    spec = importlib.util.spec_from_file_location("user_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# This function shouldn't really be here but I didn't want to create a new file for just this one
def fmt_wrap(name, params):
    # wrappers and their params to a string
    if not params:
        return name
    # turn {"agent_view_size":9} → "agent_view_size-9", etc.
    kv = "_".join(f"{k}-{v}" for k, v in params.items())
    return f"{name}({kv})"