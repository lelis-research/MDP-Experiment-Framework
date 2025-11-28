import torch
import torch.nn as nn
import math
import numpy as np
from typing import Any, Dict, List, Union, Optional

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    if getattr(layer, "bias", None) is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ---- New: small factory for module-creating layers ----
def _build_module(layer_cfg: Dict[str, Any]) -> Optional[nn.Module]:
    t = layer_cfg.get("type", "").lower()
    if t == "conv2d":
        m = nn.Conv2d(
            in_channels=layer_cfg["in_channels"],
            out_channels=layer_cfg["out_channels"],
            kernel_size=layer_cfg["kernel_size"],
            stride=layer_cfg.get("stride", 1),
            padding=layer_cfg.get("padding", 0),
        )
        return layer_init(m, std=layer_cfg.get("std", np.sqrt(2)),
                          bias_const=layer_cfg.get("bias_const", 0.0))
    elif t == "maxpool2d":
        return nn.MaxPool2d(
            kernel_size=layer_cfg["kernel_size"],
            stride=layer_cfg.get("stride", layer_cfg["kernel_size"]),
            padding=layer_cfg.get("padding", 0),
        )
    elif t == "flatten":
        return nn.Flatten()
    elif t == "linear":
        m = nn.Linear(
            in_features=layer_cfg["in_features"],
            out_features=layer_cfg["out_features"],
        )
        return layer_init(m, std=layer_cfg.get("std", np.sqrt(2)),
                          bias_const=layer_cfg.get("bias_const", 0.0))
    elif t == "batchnorm2d":
        return nn.BatchNorm2d(layer_cfg["num_features"])
    elif t == "relu":
        return nn.ReLU(inplace=layer_cfg.get("inplace", False))
    elif t == "leakyrelu":
        return nn.LeakyReLU(
            negative_slope=layer_cfg.get("negative_slope", 0.01),
            inplace=layer_cfg.get("inplace", False),
        )
    elif t == "sigmoid":
        return nn.Sigmoid()
    elif t == "tanh":
        return nn.Tanh()
    # Non-module ops handled inside forward: input, concat, add, identity
    elif t in ("input", "concat", "add", "identity"):
        return None
    else:
        raise ValueError(f"Unsupported layer type: {t}")

class NetworkGen(nn.Module):
    """
    Graph-based network generator with multi-input support and mid-network merges.
    Each config item can have:
      - type: str (conv2d, maxpool2d, flatten, linear, batchnorm2d,
                   relu, leakyrelu, sigmoid, tanh, input, concat, add, identity)
      - id:   str unique name for this node's output
      - from: str or list[str] of source node ids (default: previous node)
      - For 'input': provide input_key (e.g., 'x1', 'x2') to read from forward(**inputs)
      - For 'concat': optional dim (default 1), flatten(bool, default False)
    """
    def __init__(self, layer_descriptions: List[Dict[str, Any]]):
        super().__init__()
        self.config: List[Dict[str, Any]] = []
        self.layers_dict = nn.ModuleDict()

        # Normalize and create modules where needed
        for i, raw in enumerate(layer_descriptions):
            cfg = dict(raw)  # shallow copy
            if "type" not in cfg:
                raise ValueError(f"Layer at index {i} missing 'type'")
            cfg["type"] = cfg["type"].lower()
            if "id" not in cfg:
                cfg["id"] = f"n{i}"

            # Normalize 'from'
            if "from" in cfg:
                if isinstance(cfg["from"], str):
                    cfg["from"] = [cfg["from"]]
                elif not isinstance(cfg["from"], list):
                    raise ValueError(f"'from' must be str or list[str], got {type(cfg['from'])}")
            # build module if applicable
            mod = _build_module(cfg)
            if mod is not None:
                self.layers_dict[cfg["id"]] = mod

            self.config.append(cfg)

    def forward(self, *args, **kwargs):
        """
        Pass named inputs (e.g., model(x1=..., x2=...)).
        Returns the output of the LAST node.
        """
       
        
        if args and not kwargs:
            # If positional inputs are given, map them to the input nodes by order
            input_keys = [cfg.get("input_key", "x") for cfg in self.config if cfg["type"] == "input"]
            assert len(args) == len(input_keys), f"Expected {len(input_keys)} inputs, got {len(args)}"
            inputs = {k: v for k, v in zip(input_keys, args)}
        else:
            inputs = kwargs  # Already named inputs like forward(img=..., dir_carry=...)
            
        cache: Dict[str, torch.Tensor] = {}
        last_id: Optional[str] = None

        for cfg in self.config:
            t = cfg["type"]
            nid = cfg["id"]
            src_ids = cfg.get("from")

            # Default 'from' is previous node if not provided
            if src_ids is None:
                if last_id is None and t != "input":
                    raise ValueError(f"First non-input node '{nid}' needs a 'from'")
                src_ids = [] if t == "input" else [last_id]

            # Resolve sources
            src_tensors: List[torch.Tensor] = []
            for sid in src_ids:
                if sid in cache:
                    src_tensors.append(cache[sid])
                elif sid is None:
                    # Allow None if e.g., first node is 'input' with no sources
                    pass
                else:
                    raise KeyError(f"Unknown source id '{sid}' for node '{nid}'")

            # Execute
            if t == "input":
                key = cfg.get("input_key", None)
                if key is None:
                    # Default: 'x' to keep backward compatible single-input use
                    key = "x"
                if key not in inputs:
                    raise KeyError(f"Missing input '{key}' for input node '{nid}'")
                out = inputs[key]

            elif t == "concat":
                if len(src_tensors) < 2:
                    raise ValueError(f"'concat' node '{nid}' needs at least 2 sources")
                if cfg.get("flatten", False):
                    src_tensors = [s.flatten(1) for s in src_tensors]
                dim = cfg.get("dim", 1)
                out = torch.cat(src_tensors, dim=dim)

            elif t == "add":
                if len(src_tensors) < 2:
                    raise ValueError(f"'add' node '{nid}' needs at least 2 sources")
                out = src_tensors[0]
                for s in src_tensors[1:]:
                    out = out + s

            elif t == "identity":
                if len(src_tensors) != 1:
                    raise ValueError(f"'identity' node '{nid}' expects exactly 1 source")
                out = src_tensors[0]

            else:
                # module-backed op
                if len(src_tensors) != 1:
                    raise ValueError(f"Node '{nid}' of type '{t}' expects a single source")
                mod = self.layers_dict[nid]
                out = mod(src_tensors[0])

            cache[nid] = out
            last_id = nid

        return cache[last_id]



def prepare_network_config(config, input_dims: dict, output_dim: int | None = None):
    """
    input_dims: mapping from input_key -> shape
      - image-like: (C, H, W)
      - vector-like: int (feature dim)

    Fills in common shape fields (e.g., conv in_channels, batchnorm2d num_features,
    linear in_features after flatten/concat). For tricky cases, still specify explicitly.
    """
    import math

    def is_tuple_shape(x): return isinstance(x, (tuple, list)) and len(x) == 3
    def feat(x): return x if isinstance(x, int) else math.prod(x)  # flatten to features

    # We track shapes per node id. Shapes are either int (features) or (C,H,W)
    shapes = {}
    updated = [dict(n) for n in config]

    def conv2d_out(ch, h, w, k, s=1, p=0):
        H = (h + 2*p - k) // s + 1
        W = (w + 2*p - k) // s + 1
        return H, W
    
    # very last layer's output dim
    if output_dim is not None:
        last = updated[-1]
        if last["type"].lower() == "linear":
            last["out_features"] = output_dim

    for i, layer in enumerate(updated):
        layer_type  = layer["type"].lower()
        layer_id = layer.get("id", f"n{i}")
        layer["id"] = layer_id

        src = layer.get("from")
        if isinstance(src, str): src = [src]

        if layer_type == "input":
            if layer['input_key'] not in input_dims:
                raise KeyError(f"Missing input_dims for '{layer['input_key']}'")
            shapes[layer_id] = input_dims[layer['input_key']]

        elif layer_type == "conv2d":
            assert src and len(src) == 1
            s = shapes[src[0]]
            if not is_tuple_shape(s):
                raise ValueError(f"conv2d '{layer_id}' needs (C,H,W) but got {s}")
            C, H, W = s
            if "in_channels" not in layer:
                layer["in_channels"] = C
            k = layer["kernel_size"]; st = layer.get("stride", 1); p = layer.get("padding", 0)
            H2, W2 = conv2d_out(C, H, W, k, st, p)
            shapes[layer_id] = (layer["out_channels"], H2, W2)

        elif layer_type == "maxpool2d":
            assert src and len(src) == 1
            s = shapes[src[0]]
            if not is_tuple_shape(s):
                raise ValueError(f"maxpool2d '{layer_id}' needs (C,H,W) but got {s}")
            C, H, W = s
            k = layer["kernel_size"]; st = layer.get("stride", k); p = layer.get("padding", 0)
            H2, W2 = conv2d_out(C, H, W, k, st, p)
            shapes[layer_id] = (C, H2, W2)

        elif layer_type == "batchnorm2d":
            assert src and len(src) == 1
            s = shapes[src[0]]
            if not is_tuple_shape(s):
                raise ValueError(f"batchnorm2d '{layer_id}' needs (C,H,W) but got {s}")
            C, _, _ = s
            if "num_features" not in layer:
                layer["num_features"] = C
            shapes[layer_id] = s

        elif layer_type == "flatten":
            assert src and len(src) == 1
            s = shapes[src[0]]
            shapes[layer_id] = feat(s)

        elif layer_type == "linear":
            assert src and len(src) == 1
            s = shapes[src[0]]
            fin = feat(s)
            if "in_features" not in layer:
                layer["in_features"] = fin
            shapes[layer_id] = layer["out_features"]

        elif layer_type in ("relu", "leakyrelu", "sigmoid", "tanh", "identity"):
            assert src and len(src) == 1
            shapes[layer_id] = shapes[src[0]]

        elif layer_type == "concat":
            assert src and len(src) >= 2
            if layer.get("flatten", False):
                total = sum(feat(shapes[sid]) for sid in src)
                shapes[layer_id] = total
            else:
                dim = layer.get("dim")
                # Simple case: all sources are (C,H,W) and concat along C
                all_img = all(is_tuple_shape(shapes[sid]) for sid in src)
                if not all_img or dim != 1:
                    raise ValueError(f"concat '{layer_id}' without flatten only supports (C,H,W) along dim=1")
                C = sum(shapes[sid][0] for sid in src)
                H = shapes[src[0]][1]; W = shapes[src[0]][2]
                if any(shapes[sid][1:] != (H,W) for sid in src):
                    raise ValueError(f"concat '{layer_id}' H,W mismatch")
                shapes[layer_id] = (C, H, W)

        elif layer_type == "add":
            assert src and len(src) >= 2
            base = shapes[src[0]]
            if any(shapes[sid] != base for sid in src[1:]):
                raise ValueError(f"add '{layer_id}' shape mismatch: {[shapes[s] for s in src]}")
            shapes[layer_id] = base

        else:
            raise ValueError(f"Unsupported layer type in graph prep: {layer_type}")    

    return updated

# ----------------- Example usage -----------------
if __name__ == '__main__':
    """
    Example: x1 is an image (B,1,28,28). x2 is a vector (B,16).
    Pipeline:
      x1 -> conv -> relu -> pool -> conv -> relu -> pool -> flatten -> (id: flat)
      x2 -> linear(16->64) -> relu -> (id: meta64)
      concat(flat, meta64) along dim=1 (auto-flatten enabled for safety)
      -> linear(32*7*7 + 64 -> 128) -> relu -> linear(128 -> 10)
    """
    layer_config = [
        # image branch
        {"type": "input",  "id": "x_img", "input_key": "x1"},
        {"type": "conv2d", "id": "conv1", "from": "x_img", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu",   "id": "conv1_relu", "from": "conv1"},
        
        {"type": "flatten","id": "flat","from": "conv1_relu"},

        # vector branch
        {"type": "input",  "id": "x_dir", "input_key": "x2"},
        
        # merge
        {"type": "concat", "id": "merged", "from": ["x_dir", "flat"], "dim": 1, "flatten": True},
        
        {"type": "linear", "id": "l1",  "from": "merged", "out_features": 64},
        {"type": "relu",   "id": "l1_relu", "from": "l1"},

        #head
        {"type": "linear", "id": "out", "from": "l1_relu", "in_features": 64}
    ]
    
    layer_config = prepare_network_config(layer_config, input_dims={"x1": [20, 9, 9], "x2": 16}, output_dim=7)
    model = NetworkGen(layer_config)
    print(model)

    dummy_x1 = torch.randn(4, 20, 9, 9)
    dummy_x2 = torch.randn(4, 16)
    y = model(x1=dummy_x1, x2=dummy_x2)
    print("Output shape:", y.shape)  # (4, 10)