# Buffers

Pluggable replay buffers for agents. Select the buffer type that matches your algorithm; all expose small, explicit APIs to keep agents decoupled from buffer internals.

Contents
- `Base.py`: `BaseBuffer` (FIFO storage) and `ReplayBuffer` (uniform sampling).
- `Prioritized.py`: `PrioritizedReplayBuffer` with proportional sampling and importance weights (sum-tree implementation).
- `HER.py`: `HERBuffer` for Hindsight Experience Replay; stores episodes and relabels goals on sample (`future` or `final` strategy).
- `__init__.py`: exports all buffer types.

Usage examples
```python
from RLBase.Buffers import ReplayBuffer, PrioritizedReplayBuffer, HERBuffer

# Uniform replay
buf = ReplayBuffer(capacity=10000)
buf.add(transition)
batch = buf.sample(32)

# Prioritized replay
per = PrioritizedReplayBuffer(capacity=10000)
per.add(transition, priority=td_error)
batch, idxs, weights = per.sample(32)
per.update_priorities(idxs, new_td_errors)

# HER (assumes dict observations with achieved_goal/desired_goal)
her = HERBuffer(capacity=5000, strategy="future", k=4)
her.add_episode(episode_transitions)
relabeled_batch = her.sample(32)
```

Notes
- `BaseBuffer`/`ReplayBuffer` use a deque; sampling beyond current size raises a clear error.
- `PrioritizedReplayBuffer.sample` returns transitions, tree indices, and importance weights; call `update_priorities` after computing new TD errors.
- `HERBuffer` expects complete episodes; it trims oldest episodes when capacity is exceeded. Relabeling creates `k` variants plus the original for each sampled step.
