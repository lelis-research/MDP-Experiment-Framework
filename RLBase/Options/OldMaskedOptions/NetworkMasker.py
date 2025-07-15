import torch
import torch.nn as nn

class NetworkMasker(nn.Module):
    def __init__(self, original_net: nn.Module, mask_dict: dict):
        """
        Wraps an existing network so that, during the forward pass,
        the outputs of specified activation layers are masked.
        
        Args:
            net: The original network. Must have only one sequential attribute called network
            mask_dict: A dictionary mapping layer indices as strings to their mask.
        """
        super(NetworkMasker, self).__init__()
        self.network = original_net.network # Assuming original_net has one sequential module called network
        assert self.is_mask_valid(mask_dict), "Mask layers are not in the network"
        self.mask_dict = mask_dict

    def is_mask_valid(self, mask_dict):
        layers_names = []
        for name, module in self.network.named_children():
            layers_names.append(name)
        for layer in mask_dict:
            if layer not in layers_names+['input']:
                return False
        return True

    def forward(self, x):
        '''
        1: active
        -1: deactive
        0: part of the program
        '''
        # Assume the network is a Sequential.
        
        if 'input' in self.mask_dict:
            # mask the input 
            mask = self.mask_dict['input']
            
            if not isinstance(mask, torch.Tensor):
                mask_tensor = torch.tensor(mask, dtype=x.dtype, device=x.device)
            else:
                mask_tensor = mask.to(x.device)
            
            if mask_tensor.dim() == 1:
                if x.dim() == 2:
                    # linear input (batch, features)
                    mask_tensor = mask_tensor.unsqueeze(0).expand(x.size(0), -1)
                elif x.dim() >2:
                    # image input (batch, channels, H, W)
                    mask_tensor = mask_tensor.view(1, -1, 1, 1).expand(x.size(0), -1, x.size(2), x.size(3))
                else:
                    raise ValueError("layer dimension for the mask is unknown")
                    
            x[mask_tensor == 1] = 1 #input 1 is active
            x[mask_tensor == -1] = 0 #input 0 is inactive
       
        for name, module in self.network.named_children():
            output = module(x)
            if name in self.mask_dict:
                # get masking with the correct dimensions
                mask = self.mask_dict[name]
                if not isinstance(mask, torch.Tensor):
                    mask_tensor = torch.tensor(mask, dtype=x.dtype, device=x.device)
                else:
                    mask_tensor = mask.to(x.device)
                
                # Adjust mask shape based on output dimensions.
                if mask_tensor.dim() == 1:
                    if output.dim() == 2:
                        # For linear layers: (batch, features) -> assume mask is per feature
                        # Reshape mask to (1, features) and expand to (batch, features)
                        mask_tensor = mask_tensor.unsqueeze(0).expand(output.size(0), -1)
                    elif output.dim() > 2:
                        # For conv layers: (batch, channels, H, W) -> assume mask is per channel.
                        # Reshape mask to (1, channels, 1, 1) and expand to (batch, channels, H, W)
                        mask_tensor = mask_tensor.view(1, -1, 1, 1).expand(output.size(0), -1, output.size(2), output.size(3))
                    else:
                        raise ValueError("layer dimension for the mask is unknown")
                

                # Depending on the activation type, adjust the output.
                if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU): # ReLU and LeakyReLU Masking
                    output = output.clone()
                    output[mask_tensor == 1] = x[mask_tensor == 1]
                    output[mask_tensor == -1] = 0.0
                
                elif isinstance(module, nn.Sigmoid): # Sigmoid Masking
                    output = output.clone()
                    output[mask_tensor == 1] = 1.0
                    output[mask_tensor == -1] = 0.0
                
                elif isinstance(module, nn.Tanh): # Tanh Masking
                    output = output.clone()
                    output[mask_tensor == 1] = 1.0
                    output[mask_tensor == -1] = -1.0
                else:
                    raise ValueError("The layer is not a known activation layer")
            x = output
        return x
    
    @staticmethod
    def maskable_layers(network):
        """
        Returns a dictionary mapping incremental keys (as strings) to maskable activation layers 
        and their output sizes.

        Only activation layers (ReLU, LeakyReLU, Sigmoid, Tanh) are considered maskable.
        The output size is inferred from the most recent layer with a defined output size 
        (e.g. from a Linear or Conv2d layer via the attributes 'out_features' or 'out_channels').

        The returned dictionary maps keys (e.g. "0", "1", ...) to dictionaries with:
            - 'layer': the activation layer module,
            - 'size': the inferred output size for that activation.

        Returns:
            dict: A mapping from incremental string keys to dictionaries with keys 'layer' and 'size'.
        """
        maskable_layers = {}
        

        # Retrieve the underlying sequential module.
        seq_net = network.network

        last_output_size = None

        # Mask for the input
        if hasattr(seq_net[0], 'in_features'):
            last_output_size = seq_net[0].in_features
        elif hasattr(seq_net[0], 'in_channels'):
            last_output_size = seq_net[0].in_channels
        maskable_layers['input'] = {
            'layer': None,
            'size': last_output_size
        }

        # Iterate through layers to detect activation functions.
        for idx, module in enumerate(seq_net):
            # Update the most recent output size if the module defines it.
            if hasattr(module, 'out_features'):
                last_output_size = module.out_features
            elif hasattr(module, 'out_channels'):
                last_output_size = module.out_channels

            # Only activation layers are maskable.
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                if last_output_size is None:
                    # If we haven't encountered a preceding layer with a size, skip this activation.
                    continue
                maskable_layers[str(idx)] = {
                    'layer': module,
                    'size': last_output_size
                }
        return maskable_layers
