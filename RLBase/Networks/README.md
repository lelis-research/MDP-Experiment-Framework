# Networks

Graph-based network builder plus presets and initializers.

Structure
- `NetworkFactory.py`: `NetworkGen` builds a graph from layer configs; `prepare_network_config` fills missing shape fields. `_build_module` supports `conv2d`, `linear`, `batchnorm2d`, `maxpool2d`, `flatten`, activations (`relu`, `leakyrelu`, `sigmoid`, `tanh`), and graph ops (`input`, `concat`, `add`, `identity`). Module-backed layers **must** supply `init_params` for weight init.
- `LayerInit.py`: common initializers (`kaiming_*`, `xavier_*`, `orthogonal`, `lecun_normal`, `dirac`, `trunc_normal`) and `apply_init(name=..., ...)` dispatcher.
- `Presets.py`: ready-made configs (`CONV_MERGE_SMALL`, `CONV_MERGE_DEEP`, `MLP_MEDIUM`) using the required `init_params` format. Includes a small MLP/CNN smoke test under `__main__`.
- `__init__.py`: exports factory, presets, and init helpers.

Config basics
- Each layer is a dict with `type`, `id`, optional `from` (list of source ids). For module-backed layers, include `init_params` like `{"name": "kaiming_normal", "nonlinearity": "relu", "mode": "fan_out"}`. Supply other hyperparameters per layer type (`out_channels`, `kernel_size`, etc.).
- `prepare_network_config(config, input_dims, output_dim=None)` auto-fills shapes (e.g., `in_channels`, `in_features`, final `out_features`).
- Instantiate with `NetworkGen(layer_descriptions=...)` and call with keyword inputs matching `input_key` fields.

Example
```python
from RLBase.Networks import NetworkGen, prepare_network_config, CONV_MERGE_SMALL

cfg = prepare_network_config(CONV_MERGE_SMALL,
                             input_dims={"img": (3, 16, 16), "dir_carry": 10},
                             output_dim=10)
model = NetworkGen(cfg)
logits = model(img=batch_images, dir_carry=batch_vectors)
```

Extending
- Add new layer types by handling them in `_build_module` (construct the module or return None for graph ops) and teaching `prepare_network_config` any needed shape bookkeeping.
- Add new presets in `Presets.py` and expose them via `NETWORK_PRESETS`.

Notes
- Shape inference assumes `(C,H,W)` for conv branches and feature dims for vectors. Graph ops expect compatible shapes (`concat` supports either flattened vectors or `(C,H,W)` along dim=1).
- Initializers: ensure `init_params["name"]` is provided; missing names will error in `apply_init`.
