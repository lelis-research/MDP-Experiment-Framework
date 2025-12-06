# Buffers

Pluggable replay buffers for agents. Each buffer exposes a minimal API; agents can swap implementations without changing training code.

What a buffer should support:
- Construction with optional `capacity` (FIFO truncation when set).
- Core methods from the base: `add(item)`, `extend(items)`, `sample(batch_size)` (implement in subclass), `__len__`, `is_full()`, `clear()`, `all()`. Buffers inherit `RandomGenerator` for reproducible sampling.
- Extra hooks for specific behaviors (e.g., `update_priorities` for PER).

Current buffers:
- `ReplayBuffer` (Base.py): uniform FIFO replay with random `sample(batch_size)`.
- `PrioritizedReplayBuffer` (Prioritized.py): proportional PER (sum-tree) with importance weights; methods: `add(transition, priority=None)`, `sample(batch_size)` → `(batch, idxs, weights)`, `update_priorities(idxs, priorities)`.
- `HERBuffer` (HER.py): Hindsight Experience Replay; store full episodes via `add_episode(episode)`, sample relabeled transitions via `sample(batch_size)` using strategies `future` or `final` and `k` relabels per transition.

Usage examples:
```python
from RLBase.Buffers import ReplayBuffer, PrioritizedReplayBuffer, HERBuffer

# Uniform replay
rb = ReplayBuffer(capacity=10000)
rb.add(transition)
batch = rb.sample(32)

# Prioritized replay
per = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)
per.add(transition, priority=td_error)
batch, idxs, weights = per.sample(32)
per.update_priorities(idxs, new_td_errors)

# HER (transitions must include achieved_goal/desired_goal)
her = HERBuffer(capacity=5000, strategy="future", k=4)
her.add_episode(episode_transitions)
relabeled_batch = her.sample(32)
```

Designing a new buffer:
1) Subclass `BaseBuffer` (or `RandomGenerator` if you need custom storage) and implement `sample(batch_size)` plus any extra methods the agent will call.
2) Respect capacity semantics (FIFO trimming) and raise clear errors when sampling beyond available data.
3) Document required transition fields (e.g., goals for HER) so agents can supply compatible data. Keep sampling reproducible via the RNG helpers.
