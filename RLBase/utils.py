
import numpy as np
import torch
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.utils import seeding
T = TypeVar("T")

class RandomGenerator:
    _np_random = None
    
    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        n = len(lst)
        assert num_elems <= n

        # Sample indices without replacement using our RNG
        idxs = self.np_random.choice(n, size=num_elems, replace=False)

        # Gather elements
        return [lst[i] for i in idxs]

    def _rand_permutation(self, n: int):
        return self.np_random.permutation(n)
    
    
    def set_seed(self, seed: int | None = None):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self._np_random, self._np_random_seed = seeding.np_random(seed)

    @property
    def np_random(self) -> np.random.Generator:
        """Return internal RNG, creating one if necessary."""
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    
    def get_rng_state(self):
        """Return RNG state dict (can be pickled or saved)."""
        return self.np_random.bit_generator.state

    def set_rng_state(self, state_dict):
        """Restore RNG from a saved state dict."""
        self.np_random.bit_generator.state = state_dict
