
conv_network_1 = [
    {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1},
    {"type": "relu"},
    {"type": "conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1},
    {"type": "relu"},
    {"type": "flatten"},
    {"type": "linear", "out_features": 512},
    {"type": "relu"},
    {"type": "linear", "in_features": 512}
]
fc_network_1 = [
    {"type": "linear", "out_features": 64},
    {"type": "relu"},
    {"type": "linear", "in_features": 64},
]

fc_network_2 = [
    {"type": "linear", "out_features": 64},
    {"type": "tanh"},
    {"type": "linear", "in_features": 64, "out_features":64},
    {"type": "tanh"},
    {"type": "linear", "in_features": 64, "std":0.01}
]

linear_network_1 = [
    {"type": "linear"}
]

NETWORKS = {
    "fc_network_1": fc_network_1,
    "fc_network_2": fc_network_2,
}