import h5py
import torch
import json
from networks.siren import SIRENPytorch

def load_model_pytorch(model, param_config, params):
    # Load the data from HDF5 file
    """
    with h5py.File(path, 'r') as file:
        # Decode the JSON string to get the parameter configuration
        param_config = json.loads(file['param_config'][0].decode('utf-8'))
        # Get the flattened parameter array
        flat_params = file['params'][:]
    """

    # Assuming you have a function to match the parameter shapes and names in PyTorch
    flat_params = params[:]
    params = rename_and_unflatten(param_config, flat_params)
    model.load_state_dict(params)

def rename_and_unflatten(param_config, flat_params):
    new_state_dict = {}
    offset = 0
    for name, shape in param_config:
        num_elements = torch.prod(torch.tensor(shape))
        param_values = torch.tensor(flat_params[offset:offset + num_elements]).reshape(shape)

        if 'kernel' in name:
            # Transpose weights to match PyTorch's expectation
            param_values = param_values.T
        
        # Convert JAX names to PyTorch names
        if 'kernel_net' in name:
            # Assuming the format is weight_net_x.linear.weight or weight_net_x.linear.bias
            parts = name.split('.')
            layer_num = parts[0].split('_')[2]  # This extracts 'x' from weight_net_x
            param_type = 'weight' if 'kernel' == parts[2] else 'bias'
            new_name = f"layers.{layer_num}.linear.{param_type}"
            new_state_dict[new_name] = param_values
        offset += num_elements

        if 'output_linear' in name:
            parts = name.split('.')
            param_type = 'weight' if 'kernel' == parts[1] else 'bias'
            new_name = f"output_linear.{param_type}"
            new_state_dict[new_name] = param_values

    
    return new_state_dict




