"""
Prioritized replay buffer (proportional PER, Schaul et al. 2015).

- Storage: a sum-tree holds priorities and transitions; leaves store priorities, internal nodes store sums for O(log N) add/update/sample.
- Add: add(transition, priority=None) inserts with p = (|priority| + eps)^alpha (defaults to current max priority), updates max_priority.
- Sample: sample(batch_size) splits total priority into segments, samples one value per segment, traverses the tree to retrieve (idx, p, data).
          Computes prob = p / total, importance weight = (prob * N)^(-beta), normalized by min weight. Returns (batch, idxs, weights).
- Update: update_priorities(idxs, priorities) rewrites leaf priorities (with alpha/eps) and refreshes max_priority.

Usage: add transitions with a TD-error priority, sample batches (keep idxs/weights for loss scaling), then update_priorities with new TD errors after learning.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, TypeVar

from ..utils import RandomGenerator

T = TypeVar("T")


class SumTree:
    """Binary sum-tree for priorities."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self.tree = [0.0] * (2 * capacity)
        self.data: List[T] = [None] * capacity  # type: ignore
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[1]

    def add(self, p: float, data: T) -> int:
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, T]:
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer(RandomGenerator):
    """Proportional prioritized replay buffer."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.001,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(self, transition: T, priority: float | None = None) -> None:
        p = self.max_priority if priority is None else (abs(priority) + self.epsilon) ** self.alpha
        self.tree.add(p, transition)
        self.max_priority = max(self.max_priority, p)

    def sample(self, batch_size: int) -> Tuple[List[T], List[int], List[float]]:
        if batch_size > len(self):
            raise ValueError(f"batch_size {batch_size} exceeds buffer size {len(self)}")
        segment = self.tree.total() / batch_size
        samples: List[T] = []
        idxs: List[int] = []
        weights: List[float] = []

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        min_prob = min(self.tree.tree[self.tree.capacity : self.tree.capacity + self.tree.size]) / self.tree.total()
        min_weight = (min_prob * len(self)) ** (-self.beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = self._rand_float(a, b)
            idx, p, data = self.tree.get(s)
            prob = p / self.tree.total()
            weight = (prob * len(self)) ** (-self.beta)
            weight /= min_weight  # normalize

            samples.append(data)
            idxs.append(idx)
            weights.append(weight)

        return samples, idxs, weights

    def update_priorities(self, idxs: Sequence[int], priorities: Sequence[float]) -> None:
        for idx, p in zip(idxs, priorities):
            priority = (abs(p) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    per = PrioritizedReplayBuffer(capacity=8)
    # Add transitions with default priority
    for i in range(8):
        per.add({"id": i})
    # Sample and update priorities
    batch, idxs, weights = per.sample(4)
    print("Sampled ids:", [b["id"] for b in batch])
    print("Idxs:", idxs)
    print("Weights:", [round(w, 3) for w in weights])
    # Pretend TD errors = 0.5 for all
    per.update_priorities(idxs, [0.5] * len(idxs))
    print("Updated priorities for sampled items.")
