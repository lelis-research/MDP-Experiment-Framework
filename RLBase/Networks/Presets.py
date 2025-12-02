"""
Reusable network presets for the graph-based builder.

These configs are dictionaries understood by NetworkGen/prepare_network_config.
Keep them simple and composable; users can copy/modify as needed.
"""

# --- Shared presets ---
CONV_MERGE_SMALL = [
    # image branch
    {"type": "input",  "id": "x_img", "input_key": "img"},
    {"type": "conv2d", "id": "conv1", "from": "x_img", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,
     "init_params": {"name": "kaiming_normal", "nonlinearity": "relu", "mode": "fan_out"}},
    {"type": "relu",   "id": "conv1_relu", "from": "conv1"},

    {"type": "flatten","id": "flat","from": "conv1_relu"},

    # vector branch
    {"type": "input",  "id": "x_dir", "input_key": "dir_carry"},

    # merge
    {"type": "concat", "id": "merged", "from": ["x_dir", "flat"], "dim": 1, "flatten": True},

    {"type": "linear", "id": "l1",  "from": "merged", "out_features": 64,
     "init_params": {"name": "xavier_uniform", "gain": 1.0}},
    {"type": "relu",   "id": "l1_relu", "from": "l1"},

    # head
    {"type": "linear", "id": "out", "from": "l1_relu", "in_features": 64,
     "init_params": {"name": "orthogonal", "gain": 1.0}},
]

CONV_MERGE_DEEP = [
  # image branch (img: N,C,H,W where C = num_bits one-hot planes)
  {"type":"input",  "id":"x_img", "input_key":"img"},
  {"type":"conv2d", "id":"conv1", "from":"x_img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
   "init_params": {"name": "kaiming_normal", "nonlinearity": "relu", "mode": "fan_out"}},
  {"type":"relu",   "id":"relu1", "from":"conv1"},
  {"type":"conv2d", "id":"conv2", "from":"relu1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
   "init_params": {"name": "kaiming_normal", "nonlinearity": "relu", "mode": "fan_out"}},
  {"type":"relu",   "id":"relu2", "from":"conv2"},
  {"type":"flatten","id":"flat",  "from":"relu2"},

  # vector branch (dir/carry one-hots concatenated beforehand; e.g., 4 + 11 + 6 = 21)
  {"type":"input",  "id":"x_dir", "input_key":"dir_carry"},

  # merge
  {"type":"concat", "id":"merged", "from":["flat","x_dir"], "dim":1, "flatten":True},

  # trunk
  {"type":"linear", "id":"fc1",   "from":"merged", "out_features":128,
   "init_params": {"name": "xavier_uniform", "gain": 1.0}},
  {"type":"relu",   "id":"relu_fc1","from":"fc1"},

  # Q head
  {"type":"linear", "id":"out",   "from":"relu_fc1", "out_features":"num_actions",
   "init_params": {"name": "orthogonal", "gain": 1.0}},
]

MLP_MEDIUM = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1",   "from":"x", "out_features":64,
     "init_params": {"name": "kaiming_uniform", "nonlinearity": "relu", "mode": "fan_in"}},
    {"type":"relu",   "id":"relu_fc1", "from":"fc1"},

    {"type":"linear", "id":"fc2",   "from":"relu_fc1", "in_features":64, "out_features":64,
     "init_params": {"name": "kaiming_uniform", "nonlinearity": "relu", "mode": "fan_in"}},
    {"type":"relu",   "id":"relu_fc2", "from":"fc2"},

    {"type":"linear", "id":"fc3",   "from":"relu_fc2", "in_features":64,
     "init_params": {"name": "orthogonal", "gain": 1.0}},
]

# Public registry
NETWORK_PRESETS = {
    "conv1": CONV_MERGE_SMALL,
    "conv2": CONV_MERGE_DEEP,
    "mlp1": MLP_MEDIUM,

}


if __name__ == "__main__":
    """
    Tiny smoke tests:
      1) Train the MLP preset on synthetic data.
      2) Train the small Conv+MLP preset on synthetic images + dummy vector input.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from .NetworkFactory import NetworkGen, prepare_network_config
    except ImportError:
        # Allow running as a standalone script
        from NetworkFactory import NetworkGen, prepare_network_config

    torch.manual_seed(0)
    feature_dim = 64
    num_classes = 5

    # Synthetic supervised dataset
    x = torch.randn(512, feature_dim)
    y = torch.randint(0, num_classes, (512,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    # Build model from preset
    cfg = prepare_network_config(MLP_MEDIUM, input_dims={"x": feature_dim}, output_dim=num_classes)
    model = NetworkGen(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(x=batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")

    # CNN smoke test
    img_shape = (3, 16, 16)
    x_img = torch.randn(512, *img_shape)
    y_img = torch.randint(0, num_classes, (512,))
    img_loader = DataLoader(TensorDataset(x_img, y_img), batch_size=32, shuffle=True)

    conv_cfg = prepare_network_config(CONV_MERGE_SMALL, input_dims={"img": img_shape, "dir_carry": num_classes}, output_dim=num_classes)
    conv_model = NetworkGen(conv_cfg)
    conv_criterion = torch.nn.CrossEntropyLoss()
    conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr=1e-3)

    # Use a one-hot of the label as the vector branch input to exercise concat
    for epoch in range(200):
        for batch_x, batch_y in img_loader:
            dir_vec = torch.nn.functional.one_hot(batch_y, num_classes).float()
            conv_optimizer.zero_grad()
            logits = conv_model(img=batch_x, dir_carry=dir_vec)
            loss = conv_criterion(logits, batch_y)
            loss.backward()
            conv_optimizer.step()
        print(f"[CNN] epoch {epoch+1}: loss={loss.item():.4f}")
