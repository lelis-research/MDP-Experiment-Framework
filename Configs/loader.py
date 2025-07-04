import importlib.util
import sys

def load_config(path):
    spec = importlib.util.spec_from_file_location("user_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg