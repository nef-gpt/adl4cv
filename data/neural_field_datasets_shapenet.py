from torch.utils.data import Dataset
import numpy as np
import os
import torch

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
            out, y = self.transform(out)

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
