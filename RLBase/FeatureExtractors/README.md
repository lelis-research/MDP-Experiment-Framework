# Feature Extractors

Purpose: map environment observations to model-ready features. Except for the tabular case, extractors return a **dict of tensors**; agents know which keys to route into their networks. Tabular uses hashable state tuples instead of a dict.

Core base API (what every extractor provides):
- `__init__(observation_space, device="cpu", allowed_spaces=...)` (from `BaseFeature`): validates Gymnasium spaces (Box, Discrete, MultiDiscrete, Dict), stores device, and seeds RNG via `RandomGenerator`.
- `__call__(observation)`: takes a **batched** observation and returns feature outputs. For function-approximation extractors, this must be a dict of tensors keyed by names the agent expects (e.g., `{"x": tensor}` or nested dicts for structured obs). Tabular is the exception: it returns a list of hashable states.
- `features_dict` property: describes the feature shapes or dims keyed the same way as the output dict (informational for network builders).
- Optional helpers: `reset(seed)`, `update()`, `save(file_path)`, `load(file_path, checkpoint=None)`.

Shipped extractors:
- `FlattenFeature` (`FlattenExtractor.py`): flattens supported spaces into a single vector `{"x": tensor}`.
- `OneHotFlattenFeature` (`OneHotFlattenFeatureExtractor.py`): integer spaces one-hot encoded, flattened into `{"x": tensor}`.
- `OneHotKeepDimFeature` (`OneHotKeepDimExtractor.py`): integer spaces one-hot encoded while keeping spatial/structural dims; Dict inputs yield a dict of tensors with matching keys.
- `TabularFeature` (`TabularExtractor.py`): flattens to a list of hashable tuples for exact tabular lookups (no output dict).

Designing a new feature extractor:
1) Subclass `BaseFeature` and call `super().__init__(observation_space, device=..., allowed_spaces=...)` to enforce supported Gymnasium spaces.
2) Define `self._features_dict` to describe your output structure (e.g., `{"x": flat_dim}` or per-key shapes for Dict inputs). Implement the `features_dict` property to return it.
3) Implement `__call__(observation)` to accept batched observations and return a dict of tensors on the chosen device. Keep keys consistent with what the agent/network expects. Preserve structure for Dict observations when appropriate.
4) Register it with `@register_feature_extractor` (from `RLBase.registry`) so it can be discovered by name.
5) For tabular-style exact lookups you can follow `TabularFeature`, but for function approximation the output should remain a dict of tensors.

What to ensure when adding one:
- Output is batched, on the right device, and returned as a dict (unless explicitly implementing a tabular-style extractor).
- `features_dict` matches the keys/shapes of the returned tensors so networks can wire inputs correctly.
- Handle nested `Dict` observations if your environments use them, keeping key structure intact.

Usage example:
```python
from RLBase.FeatureExtractors import FlattenFeature
extractor = FlattenFeature(env.observation_space, device="cuda")
feat = extractor(obs_batch)          # {"x": torch.FloatTensor}
print(extractor.features_dict)       # e.g., {"x": 128}
```
