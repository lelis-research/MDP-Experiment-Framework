# Networks

Graph-based network builder (`NetworkGen`) with presets and initialization helpers.

How to build a network config (full example):
```python
from RLBase.Networks import NetworkGen, prepare_network_config

layer_config = [
    # image branch
    {"type": "input",  "id": "x_img", "input_key": "img"},
    {"type": "conv2d", "id": "conv1", "from": "x_img",
     "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,
     "init_params": {"name": "kaiming_normal", "nonlinearity": "relu", "mode": "fan_out"}},
    {"type": "relu",   "id": "relu1", "from": "conv1"},
    {"type": "maxpool2d", "id": "pool1", "from": "relu1", "kernel_size": 2, "stride": 2},
    {"type": "flatten","id": "flat",  "from": "pool1"},

    # vector branch
    {"type": "input",  "id": "x_vec", "input_key": "vec"},

    # merge branches
    {"type": "concat", "id": "merged", "from": ["flat", "x_vec"], "dim": 1, "flatten": True},

    # MLP trunk + head
    {"type": "linear", "id": "fc1", "from": "merged", "out_features": 128,
     "init_params": {"name": "xavier_uniform", "gain": 1.0}},
    {"type": "leakyrelu", "id": "fc1_act", "from": "fc1", "negative_slope": 0.01},
    {"type": "linear", "id": "out", "from": "fc1_act", "out_features": "num_actions",
     "init_params": {"name": "orthogonal", "gain": 1.0, "bias_const": 0.0}},
]

cfg = prepare_network_config(layer_config,
                             input_dims={"img": (3, 32, 32), "vec": 16},
                             output_dim=10)   # sets last linear out_features when placeholder used
model = NetworkGen(cfg)
logits = model(img=batch_images, vec=batch_vectors)
```
Config rules:
- Each layer dict needs `type` and `id` (auto-filled if omitted). `from` can be a string or list of source ids; defaults to previous node for non-`input` nodes.
- `input` nodes require `input_key` to match the forward kwarg.
- Module-backed types must include `init_params` with at least `{"name": "<init_name>"}`; extra fields depend on the initializer.
- `prepare_network_config` infers `in_channels`/`in_features`/`num_features` and final `out_features` when given `input_dims` and `output_dim`.

Supported layer/ops:
- Modules: `conv2d`, `maxpool2d`, `flatten`, `linear`, `batchnorm2d`.
- Activations: `relu`, `leakyrelu`, `sigmoid`, `tanh`.
- Graph ops (no module): `input`, `concat` (optionally flatten sources, concat along `dim`), `add` (elementwise sum), `identity`.
- Shapes: conv/BN assume `(C,H,W)`, vectors are ints; `concat` without `flatten` only supports `(C,H,W)` along channel dim.

Initializations (in `LayerInit.py`, via `init_params["name"]`):
- `orthogonal`, `kaiming_normal`, `kaiming_uniform`, `xavier_uniform`, `xavier_normal`, `lecun_normal`, `dirac`, `trunc_normal`.
- Tunables: `bias_const`, `gain`, `std`, `nonlinearity`, `mode` (fan_in/fan_out).

Presets (`Presets.py`):
- `CONV_MERGE_SMALL` / `CONV_MERGE_DEEP`: image + vector merge conv trunks.
- `MLP_MEDIUM`: 3-layer MLP.
- All are registered in `NETWORK_PRESETS`; pass through `prepare_network_config` before instantiation.

Adding functionality:
- New init: implement a function in `LayerInit.py` and register it in `INIT_REGISTRY` so `apply_init` can dispatch to it.
- New layer type: extend `_build_module` (construct and optionally initialize), and update `prepare_network_config` to compute shapes for that type. If it is a pure graph op, return `None` in `_build_module` and handle behavior in `NetworkGen.forward`.
- New preset: define a list of layer dicts in `Presets.py` using the same schema, add it to `NETWORK_PRESETS`.

Notes:
- Always provide `init_params` for module-backed layers; missing names will error in `apply_init`.
- The last returned tensor is from the final config node. Multi-input models require kwargs matching `input_key`s.
