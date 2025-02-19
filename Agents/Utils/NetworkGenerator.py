import torch
import torch.nn as nn


class NetworkGen(nn.Module):
    def __init__(self, layer_descriptions):
        """
        Generates a neural network based on a layer description list.
        
        Args:
            layer_descriptions (list): Each element is a dict specifying a layer.
                For example:
                    {"type": "conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}
                    {"type": "relu"}
                    {"type": "flatten"}
                    {"type": "linear", "in_features": 16*28*28, "out_features": 10}
        
        Returns:
            nn.Module: A neural network with a proper forward method.
        """
        super(NetworkGen, self).__init__()
        layers = []
        for config in layer_descriptions:
            layer_type = config.get('type', '').lower()
            if layer_type == 'conv2d':
                layers.append(nn.Conv2d(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', 1),
                    padding=config.get('padding', 0)
                ))
            elif layer_type == 'linear':
                layers.append(nn.Linear(
                    in_features=config['in_features'],
                    out_features=config['out_features']
                ))
            elif layer_type == 'relu':
                layers.append(nn.ReLU(inplace=config.get('inplace', False)))
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'batchnorm2d':
                layers.append(nn.BatchNorm2d(config['num_features']))
            elif layer_type == 'maxpool2d':
                layers.append(nn.MaxPool2d(
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', None),
                    padding=config.get('padding', 0)
                ))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        # Wrap layers in nn.Sequential for simplicity
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def prepare_network_config(config, input_dim, output_dim):
    """
    Prepares the network configuration by ensuring the first linear layer has 'in_features'
    and the last linear layer has 'out_features'.

    Args:
        config (list): List of dictionaries defining the network layers.
        input_dim (int): The input dimension for the first layer.
        output_dim (int): The output dimension for the last layer.

    Returns:
        list: Updated configuration list with complete parameters.
    """
    # Create a copy to avoid modifying the original list
    updated_config = [dict(layer) for layer in config]

    # Update first layer if it is linear and missing 'in_features'
    if updated_config[0]['type'].lower() == 'linear':
        if 'in_features' not in updated_config[0]:
            updated_config[0]['in_features'] = input_dim
    #TODO: Add for other layers

    # Update last layer if it is linear and missing 'out_features'
    if updated_config[-1]['type'].lower() == 'linear':
        if 'out_features' not in updated_config[-1]:
            updated_config[-1]['out_features'] = output_dim
    #TODO: Add for other layers

    return updated_config

# Example usage:
if __name__ == '__main__':
    # Define the network architecture using a list of layer configurations.
    layer_config = [
        {"type": "conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        {"type": "conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
        {"type": "flatten"},
        # Adjust in_features based on the input image size and pooling operations.
        {"type": "linear", "in_features": 32 * 7 * 7, "out_features": 10},
    ]

    # Create the model.
    model = NetworkGen(layer_config)
    print(model)

    # Test a forward pass with a dummy input (e.g., for a 28x28 grayscale image).
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print("Output shape:", output.shape)