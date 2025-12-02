"""
Base buffer primitives for RL agents.

These classes are framework-agnostic: they store transitions and expose a small API
so agents can plug in different buffer behaviors (uniform replay, prioritized replay, HER).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List, Sequence, Tuple, TypeVar

from ..utils import RandomGenerator

T = TypeVar("T")


class BaseBuffer(RandomGenerator):
    """Simple FIFO buffer with optional capacity."""

    def __init__(self, capacity: int | None = None):
        super().__init__()
        self.capacity = capacity
        self._data: Deque[T] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def is_full(self) -> bool:
        return self.capacity is not None and len(self._data) >= self.capacity

    def clear(self) -> None:
        self._data.clear()

    def add(self, item: T) -> None:
        self._data.append(item)

    def extend(self, items: Iterable[T]) -> None:
        self._data.extend(items)

    def all(self) -> List[T]:
        return list(self._data)

    def sample(self, batch_size: int) -> List[T]:
        """Override in subclasses if sampling is supported."""
        raise NotImplementedError("sample is not implemented for BaseBuffer")


class ReplayBuffer(BaseBuffer):
    """Uniform replay buffer with optional capacity and random sampling."""

    def sample(self, batch_size: int) -> List[T]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size > len(self._data):
            raise ValueError(f"batch_size {batch_size} exceeds buffer size {len(self._data)}")
        return self._rand_subset(self._data, batch_size)


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=5)
    buf.set_seed(42)
    
    for i in range(5):
        buf.add(f"t{i}")
    print("Buffer contents:", buf.all())
    batch = buf.sample(3)
    print("Sampled:", batch)
