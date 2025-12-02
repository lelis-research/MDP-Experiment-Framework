# Feature Extractors

Purpose: map environment observations to model-ready tensors, carrying a `features_dict` that describes the output shapes. All extractors inherit from `BaseFeature` and can be registered via `RLBase.registry.register_feature_extractor`.

Contents
- `Base.py`: `BaseFeature` validates supported Gymnasium spaces (Box, Discrete, MultiDiscrete, Dict), holds device, RNG helpers, save/load stubs.
- `FlattenExtractor.py`: `FlattenFeature` flattens observations into a single vector under key `"x"`. Supports Discrete, MultiDiscrete, Box, and Dict (concats sub-flats).
- `TabularExtractor.py`: `TabularFeature` flattens and returns hashable arrays (tuples) for tabular agents; output key `"x"`.
- `__init__.py`: re-exports registered extractors.

Usage
```python
from RLBase.FeatureExtractors import FlattenFeature
extractor = FlattenFeature(env.observation_space, device="cpu")
features = extractor(obs_batch)              # expects batch dimension
feat_dict = extractor.features_dict          # e.g., {"x": flat_dim}
```

Adding a new extractor
1) Subclass `BaseFeature` in this package. Validate supported spaces in `__init__` (call `super().__init__(..., allowed_spaces=...)`), set `self._features_dict`, and implement `__call__` to accept batched observations.
2) Decorate with `@register_feature_extractor` from `RLBase.registry` so it becomes discoverable.
3) Keep outputs batched and return tensors/arrays on the right device if needed. Update `features_dict` keys to match what downstream networks expect.
