import torch
import torch.nn as nn
from typing import List

class Permute(nn.Module):
    """
    Permute tensor dimensions.

    Example:
      dims=[0, 3, 2, 1] turns (B,C,H,W) -> (B,W,H,C) if input is 4D.
    """
    def __init__(self, dims: List[int], contiguous: bool = True):
        super().__init__()
        if not isinstance(dims, (list, tuple)) or len(dims) == 0:
            raise ValueError(f"Permute requires non-empty dims list, got: {dims}")
        self.dims = list(dims)
        self.contiguous = bool(contiguous)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != len(self.dims):
            raise ValueError(
                f"Permute expected input rank {len(self.dims)} (dims={self.dims}), "
                f"but got tensor with rank {x.dim()} and shape {tuple(x.shape)}"
            )
        y = x.permute(*self.dims)
        return y.contiguous() if self.contiguous else y