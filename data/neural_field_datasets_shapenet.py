from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

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
    def __init__(self, mlps_folder, transform=None):
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
        weights = torch.cat([state_dict[key].flatten() for key in state_dict.keys()])
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
            _x, indices, _commit_loss = self.vq(
                weigths.unsqueeze(-1), freeze_codebook=True
            )
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

        model.load_state_dict(
            backtransform_weights(weights.flatten().unsqueeze(0), prototyp)
        )

        return model, y


# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class WeightTransform3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y):
        # Apply min-max normalization
        state_dict[f"layers.0.weight"]
        weights = torch.stack(
            state_dict[f"layers.0.weight"],
            state_dict[f"layers.1.weight"],
            state_dict[f"layers.2.weight"],
            torch.transpose(state_dict[f"layers.3.weight"], 0, 1),
        )
        return weights, y

    def backproject(self, indices):
        return self.vq.get_codes_from_indices(indices)


def get_neuron_mean_n_std(dataset: ShapeNetDataset):
    all_weights = torch.stack([sample[0] for sample in dataset])
    neuron_count = all_weights.shape[1]
    means = torch.stack([all_weights[:, i, :].mean(dim=0) for i in range(neuron_count)])
    stds = torch.stack([all_weights[:, i, :].std(dim=0) for i in range(neuron_count)])
    return means, stds


# Transform that uses vq, first flatten data, then use vq for quantization and return indices and label
class AllWeights3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y):
        # Apply min-max normalization
        all_weights = torch.cat(
            (
                state_dict[f"layers.0.weight"].T,
                state_dict[f"layers.1.weight"].T,
                state_dict[f"layers.2.weight"].T,
                state_dict[f"layers.3.weight"],
                state_dict[f"layers.0.bias"].unsqueeze(0),
                state_dict[f"layers.1.bias"].unsqueeze(0),
                state_dict[f"layers.2.bias"].unsqueeze(0),
            )
        )
        return all_weights, state_dict[f"layers.3.bias"]


class ImageTransform3D(nn.Module):
    """
    Transforms a model to a 2D image

    Padding is applied to the weights to make them square
    """

    def __init__(self):
        super().__init__()

    def forward(self, state_dict, y):
        # Store reference to original state_dict (for inverse)
        self.original_state_dict = state_dict

        # Apply min-max normalization
        cat = torch.cat([state_dict[key].view(-1) for key in state_dict], dim=0)
        cat = torch.cat([cat, torch.zeros(128 - cat.shape[0] % 128)])
        cat = cat.view(1, 128, -1)
        return cat, y

    def inverse(self, cat, model_dict=None):
        # flatten
        cat = cat.view(-1)
        if model_dict is None:
            model_dict = self.original_state_dict
        i = 0
        for key in model_dict:
            model_dict[key] = cat[i : i + model_dict[key].numel()].view(
                model_dict[key].shape
            )
            i += model_dict[key].numel()
        return model_dict
