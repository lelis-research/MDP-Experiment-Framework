import torch
import torch.nn as nn

class NetworkMasker(nn.Module):
    def __init__(self, original_net: nn.Module): #, mask_dict: nn.ParameterDict):
        """
        Wraps an existing network so that, during the forward pass,
        the outputs of specified activation layers are masked.
        
        Args:
            net: The original network. Must have only one sequential attribute called network
            mask_dict: A dictionary mapping layer indices as strings to their mask.
        """
        super(NetworkMasker, self).__init__()
        self.network = original_net.network # Assuming original_net has one sequential module called network


    def forward(self, x, mask_dict):
        '''
        0: active
        1: deactive
        2: part of the program
        '''
        # Assume the network is a Sequential.
        if 'input' in mask_dict:
            # mask the input 
            mask = mask_dict['input'].to(x.device) # (3, ...)
                
            probs = torch.softmax(mask, dim=0) # (3, ...)
            idxs = probs.argmax(dim=0, keepdim=True)             # (1, ...)
            hard = torch.zeros_like(probs).scatter_(0, idxs, 1)  # (3, ...)
            
            # differentiable trick
            m = hard.detach() - probs.detach() + probs # (3, ...)
            p_act, p_deact, p_prog = m.unbind(0) # each (..., )
            
            # linear input (batch, features)
            # image input (batch, channels, H, W)
            shape = [1, -1] + [1] * (x.dim() - 2)
            p_act   = p_act.view(*shape).expand_as(x)
            p_deact = p_deact.view(*shape).expand_as(x)
            p_prog  = p_prog.view(*shape).expand_as(x)
            
            active     = torch.ones_like(x)       # force 1
            deactivate = torch.zeros_like(x)      # force 0
            program    = x                        # leave as-is

            x = p_act * active + p_deact * deactivate + p_prog * program
            
        for name, module in self.network.named_children():
            output = module(x)
            if name in mask_dict:
                # get masking with the correct dimensions
                mask = mask_dict[name].to(x.device)
                
                probs = torch.softmax(mask, dim=0) # (3, ...)
                idxs = probs.argmax(dim=0, keepdim=True)             # (1, ...)
                hard = torch.zeros_like(probs).scatter_(0, idxs, 1)  # (3, ...)
                
                # differentiable trick
                m = hard.detach() - probs.detach() + probs # (3, ...)
                p_act, p_deact, p_prog = m.unbind(0) # each (..., )
                
                # linear input (batch, features)
                # image input (batch, channels, H, W)
                shape = [1, -1] + [1] * (output.dim() - 2)
                p_act   = p_act.view(*shape).expand_as(output)
                p_deact = p_deact.view(*shape).expand_as(output)
                p_prog  = p_prog.view(*shape).expand_as(output)
                

                # Depending on the activation type, adjust the output.
                if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU): # ReLU and LeakyReLU Masking
                    active     = x
                    deactivate = torch.zeros_like(output)
                    program    = output
        
                elif isinstance(module, nn.Sigmoid): # Sigmoid Masking
                    active     = torch.ones_like(output)   # force 1
                    deactivate = torch.zeros_like(output)  # force 0
                    program    = output                    # normal sigmoid
                
                elif isinstance(module, nn.Tanh): # Tanh Masking
                    active     = torch.ones_like(output)    # force +1
                    deactivate = -torch.ones_like(output)   # force -1
                    program    = output                     # normal tanh
                else:
                    raise ValueError("The layer is not a known activation layer")

                output = p_act * active + p_deact * deactivate + p_prog * program

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
