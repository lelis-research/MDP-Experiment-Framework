import torch
import torch.nn as nn

class MaskedNetwork(nn.Module):
    def __init__(self, original_net: nn.Module, mask_dict: dict):
        """
        Wraps an existing network so that, during the forward pass,
        the outputs of specified activation layers are masked.
        
        Args:
            net: The original network.
            mask_dict: A dictionary mapping layer indices (as strings) or full names
                       to their mask.
        """
        super(MaskedNetwork, self).__init__()
        self.network = original_net.network # Assuming original_net has one sequential module called network
        self.mask_dict = mask_dict

    def forward(self, x):
        # Assume the network is a Sequential for simplicity.
        # This example shows how to handle Sequential models.
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
