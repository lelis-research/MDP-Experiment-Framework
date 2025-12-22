import numpy as np
import torch
from typing import Sequence, TypeVar, Optional
from gymnasium.utils import seeding

T = TypeVar("T")

class RandomGenerator:
    _np_random: Optional[np.random.Generator] = None

    # ---------- RNG core ----------
    def set_seed(self, seed: int | None = None):
        self.seed = seed
        
        if seed is None:
            # still seed numpy generator nondeterministically via gymnasium helper
            self._np_random, self._np_random_seed = seeding.np_random(None)
            return

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._np_random, self._np_random_seed = seeding.np_random(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random(None)
        return self._np_random

    def get_rng_state(self):
        return self.np_random.bit_generator.state

    def set_rng_state(self, state_dict):
        self.np_random.bit_generator.state = state_dict

    # ---------- scalar helpers ----------
    def _rand_int(self, low: int, high: int) -> int:
        return int(self.np_random.integers(low, high))

    def _rand_float(self, low: float, high: float) -> float:
        return float(self.np_random.uniform(low, high))

    def _rand_bool(self) -> bool:
        return bool(self.np_random.integers(0, 2))

    # ---------- fast index-based helpers ----------
    def _rand_index(self, n: int) -> int:
        """Fast random index in [0, n)."""
        return int(self.np_random.integers(0, n))

    def _rand_indices(self, n: int, size: int, replace: bool = True) -> np.ndarray:
        """Fast batch of indices in [0, n)."""
        if replace:
            return self.np_random.integers(0, n, size=size)
        return self.np_random.choice(n, size=size, replace=False)

    # ---------- element sampling (requires Sequence) ----------
    def _rand_elem(self, seq: Sequence[T]) -> T:
        """Pick random element from an indexable sequence (NO list() copy)."""
        return seq[self._rand_index(len(seq))]

    def _rand_subset(self, seq: Sequence[T], num_elems: int, replace: bool = False) -> list[T]:
        """
        Sample a subset of elements from an indexable sequence.
        For replay buffers, consider using _rand_indices and index directly.
        """
        n = len(seq)
        if num_elems > n and not replace:
            raise ValueError("num_elems must be <= len(seq) when replace=False")

        idxs = self._rand_indices(n, num_elems, replace=replace)
        # still a Python list comprehension; acceptable for small num_elems,
        # but for max speed return idxs and gather elsewhere.
        return [seq[i] for i in idxs]

    def _rand_permutation(self, n: int) -> np.ndarray:
        return self.np_random.permutation(n)

    # ---------- optional: fast normal noise ----------
    def _randn(self, size, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        return self.np_random.normal(loc=mean, scale=std, size=size)