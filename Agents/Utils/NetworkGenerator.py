import torch
import torch.nn as nn
import math

class NetworkGen(nn.Module):
    def __init__(self, layer_descriptions):
        """
        Generates a neural network based on a list of layer descriptions.
        
        Args:
            layer_descriptions (list): Each element is a dict specifying a layer.
                Supported layer types (with complete arguments) include:
                
                - Conv2d:
                  {"type": "conv2d", "in_channels": <int>, "out_channels": <int>,
                   "kernel_size": <int or tuple>, "stride": <int or tuple>, "padding": <int or tuple>}
                
                - MaxPool2d:
                  {"type": "maxpool2d", "kernel_size": <int or tuple>,
                   "stride": <int or tuple>, "padding": <int or tuple>}
                
                - Flatten:
                  {"type": "flatten"}
                
                - Linear:
                  {"type": "linear", "in_features": <int>, "out_features": <int>}
                
                - BatchNorm2d:
                  {"type": "batchnorm2d", "num_features": <int>}
                
                - Activation layers:
                  {"type": "relu", "inplace": <bool>}
                  {"type": "leakyrelu", "negative_slope": <float>, "inplace": <bool>}
                  {"type": "sigmoid"}
                  {"type": "tanh"}
        
        Returns:
            nn.Module: A neural network with the specified layers arranged in order.
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
            elif layer_type == 'maxpool2d':
                layers.append(nn.MaxPool2d(
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', config['kernel_size']),
                    padding=config.get('padding', 0)
                ))
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'linear':
                layers.append(nn.Linear(
                    in_features=config['in_features'],
                    out_features=config['out_features']
                ))
            elif layer_type == 'batchnorm2d':
                layers.append(nn.BatchNorm2d(config['num_features']))
            elif layer_type == 'relu':
                layers.append(nn.ReLU(inplace=config.get('inplace', False)))
            elif layer_type == 'leakyrelu':
                layers.append(nn.LeakyReLU(
                    negative_slope=config.get('negative_slope', 0.01),
                    inplace=config.get('inplace', False)
                ))
            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_type == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        
        # Wrap layers in an nn.Sequential container.
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def prepare_network_config(config, input_dim, output_dim):
    """
    Prepares the network configuration by ensuring the first linear layer has 'in_features'
    and the last linear layer has 'out_features'.

    Args:
        config (list): List of dictionaries defining the network layers.
        input_dim (int or tuple): The input dimension for the first layer.
        output_dim (int): The output dimension for the last layer.

    Returns:
        list: Updated configuration list with complete parameters.
    """
    assert isinstance(input_dim, int) or (isinstance(input_dim, tuple) and len(input_dim) == 3), \
        "input_dim must be an int or a tuple of three dimensions (C, H, W)"
    assert isinstance(output_dim, int), "output_dim must be an int"

    # Create a copy to avoid modifying the original list
    updated_config = [dict(layer) for layer in config]
    current_shape = input_dim
    for i, layer in enumerate(updated_config):
        layer_type = layer.get('type', '').lower()

        if layer_type == 'conv2d':
            # Set in_channels if missing using the channel from current_shape.
            if 'in_channels' not in layer:
                if isinstance(current_shape, (tuple, list)):
                    layer['in_channels'] = current_shape[0]
                else:
                    raise ValueError("Expected current_shape as tuple (C, H, W) for conv2d layer.")
            # Get parameters (assumed to be ints; you can extend to tuples if needed)
            kernel_size = layer['kernel_size']
            stride = layer.get('stride', 1)
            padding = layer.get('padding', 0)
            # current_shape should be (C, H, W)
            if not isinstance(current_shape, (tuple, list)):
                raise ValueError("Expected current_shape as tuple (C, H, W) before conv2d layer.")
            C_in, H_in, W_in = current_shape
            # Calculate new spatial dimensions using the convolution formula.
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1
            current_shape = (layer['out_channels'], H_out, W_out)

        elif layer_type == 'maxpool2d':
            kernel_size = layer['kernel_size']
            stride = layer.get('stride', kernel_size)  # default stride equals kernel size
            padding = layer.get('padding', 0)
            if not isinstance(current_shape, (tuple, list)):
                raise ValueError("Expected current_shape as tuple (C, H, W) before maxpool2d layer.")
            C, H_in, W_in = current_shape
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1
            current_shape = (C, H_out, W_out)
        
        elif layer_type == 'flatten':
            # Flatten the current shape to a single dimension.
            if isinstance(current_shape, (tuple, list)):
                current_shape = math.prod(current_shape)
            # else, assume it's already an int.
        
        elif layer_type == 'linear':
            # If the current_shape is still a tuple, flatten it.
            if isinstance(current_shape, (tuple, list)):
                current_shape = math.prod(current_shape)
            # Set 'in_features' if missing.
            if 'in_features' not in layer:
                layer['in_features'] = current_shape
            # Propagate the dimension if 'out_features' is given.
            if 'out_features' in layer:
                current_shape = layer['out_features']
            # Otherwise, leave current_shape unchanged (it might be the final layer)
        
        elif layer_type == 'batchnorm2d':
            # For batchnorm2d, if 'num_features' is missing, use the channel count.
            if 'num_features' not in layer:
                if isinstance(current_shape, (tuple, list)):
                    layer['num_features'] = current_shape[0]
                else:
                    raise ValueError("Expected current_shape as tuple (C, H, W) for batchnorm2d layer.")
            # current_shape remains unchanged.
        
        elif layer_type in ['relu', 'leakyrelu', 'sigmoid', 'tanh']:
            # Activation layers do not change the shape.
            pass
        
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    # After processing, if the last layer is linear and missing 'out_features', set it to output_dim.
    if updated_config[-1]['type'].lower() == 'linear':
        if 'out_features' not in updated_config[-1]:
            updated_config[-1]['out_features'] = output_dim
            current_shape = output_dim  # update current_shape to reflect final output

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