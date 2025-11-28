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

conv1_network = [
        # image branch
        {"type": "input",  "id": "x_img", "input_key": "img"},
        {"type": "conv2d", "id": "conv1", "from": "x_img", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu",   "id": "conv1_relu", "from": "conv1"},
        
        {"type": "flatten","id": "flat","from": "conv1_relu"},

        # vector branch
        {"type": "input",  "id": "x_dir", "input_key": "dir_carry"},
        
        # merge
        {"type": "concat", "id": "merged", "from": ["x_dir", "flat"], "dim": 1, "flatten": True},
        
        {"type": "linear", "id": "l1",  "from": "merged", "out_features": 64},
        {"type": "relu",   "id": "l1_relu", "from": "l1"},

        #head
        {"type": "linear", "id": "out", "from": "l1_relu", "in_features": 64}
    ]


conv2_network = [
  # image branch (img: N,C,H,W where C = num_bits one-hot planes)
  {"type":"input",  "id":"x_img", "input_key":"img"},
  {"type":"conv2d", "id":"conv1", "from":"x_img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1},
  {"type":"relu",   "id":"relu1", "from":"conv1"},
  {"type":"conv2d", "id":"conv2", "from":"relu1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1},
  {"type":"relu",   "id":"relu2", "from":"conv2"},
  {"type":"flatten","id":"flat",  "from":"relu2"},

  # vector branch (dir/carry one-hots concatenated beforehand; e.g., 4 + 11 + 6 = 21)
  {"type":"input",  "id":"x_dir", "input_key":"dir_carry"},

  # merge
  {"type":"concat", "id":"merged", "from":["flat","x_dir"], "dim":1, "flatten":True},

  # trunk
  {"type":"linear", "id":"fc1",   "from":"merged", "out_features":128},
  {"type":"relu",   "id":"relu_fc1","from":"fc1"},

  # Q head
  {"type":"linear", "id":"out",   "from":"relu_fc1", "out_features":"num_actions"}
]

linear1_network = [
    {"type":"input",  "id":"x", "input_key":"x"},
    
    {"type":"linear", "id":"fc1",   "from":"x", "out_features":64},
    {"type":"relu",   "id":"relu_fc1", "from":"fc1"},
    
    {"type":"linear", "id":"fc2",   "from":"relu_fc1", "in_features":64, "out_features":64},
    {"type":"relu",   "id":"relu_fc2", "from":"fc2"},
    
    {"type":"linear", "id":"fc3",   "from":"relu_fc2", "in_features":64},
]
NETWORKS = {
    "conv1_network": conv1_network,
    "conv2_network": conv2_network,
    "linear1_network": linear1_network,
}