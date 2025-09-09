"""
You can define your neural network with any of the below known layers. 
Or you can define a new one in the NetworkGenerator.py 

Known Layers:
    conv2d: "in_channels", "out_channels", "kernel_size", "stride", "padding"
    maxpool2d: "kenel_size", "stride", "padding"
    flatten:
    linear: "in_features", "outfeatures"
    batchnorm2d: "num_features"
    relu: "inplace"
    leakyrelu: "negative_slope", "inplace"
    sigmoid: 
    tanh:
    
"""

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

conv_network_2 = [
    {"type": "conv2d", "in_channels": 3,  "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2},
    {"type": "relu"},

    {"type": "conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 7, "stride": 3, "padding": 2},
    {"type": "relu"},

    {"type": "conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "relu"},

    {"type": "flatten"},

    {"type": "linear", "out_features": 512},
    {"type": "relu"},
    
    {"type": "linear", "in_features": 512},
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
fc_network_relu = [
    {"type": "linear", "out_features": 64},
    {"type": "relu"},
    {"type": "linear", "in_features": 64, "out_features":64},
    {"type": "relu"},
    {"type": "linear", "in_features": 64, "std":0.01}
]

linear_network_1 = [
    {"type": "linear"}
]

NETWORKS = {
    "fc_network_1": fc_network_1,
    "fc_network_2": fc_network_2,
    "fc_network_tanh": fc_network_2,
    "fc_network_relu": fc_network_relu,
    "conv_network_1": conv_network_1,
    "conv_network_2": conv_network_2,
}