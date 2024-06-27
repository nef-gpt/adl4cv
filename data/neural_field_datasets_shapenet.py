from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
from vector_quantize_pytorch  import VectorQuantize

from animation.util import backtransform_weights
from networks.mlp_models import MLP3D
from os.path import join

from utils import get_default_device

mlp_kwargs = {
    "out_size": 1,
    "hidden_neurons": [128, 128, 128],
    "use_leaky_relu": False,
    "input_dims": 3,
    "multires": 4,
    "include_input": True,
    "output_type": "occ",
}



class ShapeNetDataset(Dataset):
    def __init__(
        self, mlps_folder, transform=None
    ):
        self.mlps_folder = mlps_folder
        self.transform = transform
        self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        self.device = torch.device(get_default_device())

    def __getitem__(self, index):
        file = join(self.mlps_folder, self.mlp_files[index])
        out = torch.load(file, map_location=self.device)
        y = "plane"

        if self.transform:
            out, y = self.transform(out, y)

        return out, y

    def __len__(self):
        return len(self.mlp_files)
    
class ModelTransform3D(torch.nn.Module):
    def __init__(self, weights_dict: dict = mlp_kwargs):
        super().__init__()
        self.weights_dict = weights_dict
    def forward(self, state_dict, y=None):
        model = MLP3D(**self.weights_dict)
        model.load_state_dict(state_dict)
        return model, y
    
class FlattenTransform3D(torch.nn.Module):  
    def forward(self, state_dict, y=None):
        weights = torch.cat(
            [
                state_dict[key].flatten()
                for key in state_dict.keys()
            ]
        )
        return weights, y
    
# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class TokenTransform3D(nn.Module):
    def __init__(self, vq: VectorQuantize):
        super().__init__()
        self.flatten = FlattenTransform3D()
        self.vq = vq
        self.eval()

    def forward(self, weights_dict, y):
        # Apply min-max normalization
        weigths, y = self.flatten(weights_dict, y)
        with torch.no_grad():
            _x, indices, _commit_loss = self.vq(weigths.unsqueeze(-1), freeze_codebook=True)
            return indices, y

    def backproject(self, indices):
        return self.vq.get_codes_from_indices(indices)


class ModelTransform3DFromTokens(torch.nn.Module):
    def __init__(self, vq: VectorQuantize, weights_dict: dict = mlp_kwargs):
        super().__init__()
        self.weights_dict = weights_dict
        self.vq = vq
        self.token_transform = TokenTransform3D(vq)

    def forward(self, indices, y=None):
        weights = self.token_transform.backproject(indices)
        model = MLP3D(**self.weights_dict)

        prototyp = model.state_dict()

        model.load_state_dict(backtransform_weights(weights.flatten().unsqueeze(0), prototyp))

        return model, y
    

# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class WeightTransform3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y):
        # Apply min-max normalization
        state_dict[f"layers.0.weight"]
        weights = torch.stack(state_dict[f"layers.0.weight"], state_dict[f"layers.1.weight"], state_dict[f"layers.2.weight"], torch.transpose(state_dict[f"layers.3.weight"], 0, 1))
        return weights, y

    def backproject(self, indices):
        return self.vq.get_codes_from_indices(indices)