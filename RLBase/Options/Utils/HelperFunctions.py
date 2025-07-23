from ...loaders import load_option

import torch

def save_options_list(options_lst, file_path=None):
    """
    Serialize a list of BaseOption-derived objects to disk.
    """
    # 1) extract each option’s checkpoint dict (but don’t write individual files)
    checkpoint = [option.save(file_path=None) for option in options_lst]
    # 2) dump the whole list
    if file_path is not None:
        torch.save(checkpoint, file_path)
    return checkpoint


def load_options_list(file_path, checkpoint=None) -> list:
    """
    Load a list of options from a single file.
    """
    if checkpoint is None:
        # 1) load the list of checkpoint dicts
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # 2) reconstruct each option
    options_lst = [load_option(None, checkpoint=ckpt) for ckpt in checkpoint]
    return options_lst