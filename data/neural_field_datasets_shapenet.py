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

outliers = [
    16,
    31,
    47,
    53,
    90,
    184,
    201,
    222,
    223,
    282,
    315,
    342,
    344,
    377,
    435,
    438,
    469,
    508,
    513,
    527,
    531,
    598,
    675,
    691,
    723,
    766,
    794,
    828,
    830,
    909,
    959,
    992,
    1015,
    1045,
    1091,
    1097,
    1208,
    1243,
    1290,
    1395,
    1476,
    1500,
    1503,
    1504,
    1514,
    1528,
    1546,
    1548,
    1557,
    1601,
    1644,
    1658,
    1718,
    1735,
    1756,
    1783,
    1815,
    1860,
    1913,
    1963,
    1979,
    2047,
    2071,
    2094,
    2187,
    2222,
    2242,
    2273,
    2307,
    2314,
    2341,
    2376,
    2388,
    2409,
    2489,
    2509,
    2552,
    2562,
    2570,
    2576,
    2590,
    2651,
    2662,
    2678,
    2696,
    2733,
    2747,
    2767,
    2776,
    2778,
    2782,
    2802,
    2804,
    2854,
    2858,
    2872,
    2893,
    2940,
    2948,
    2953,
    2958,
    2964,
    2980,
    2986,
    3018,
    3020,
    3043,
    3057,
    3082,
    3098,
    3099,
    3105,
    3157,
    3160,
    3168,
    3169,
    3228,
    3264,
    3286,
    3308,
    3320,
    3341,
    3342,
    3360,
    3378,
    3390,
    3399,
    3414,
    3472,
    3480,
    3483,
    3504,
    3525,
    3569,
    3591,
    3699,
    3721,
    3748,
    3765,
    3857,
    3877,
    3891,
    3912,
    3914,
    3944,
    3948,
    3967,
    4002,
    4011,
    4016,
    4042,
    221,
    301,
    576,
    610,
    632,
    748,
    1311,
    1421,
    1905,
    2017,
    2798,
]

outliers_set = set(outliers)


class ShapeNetDataset(Dataset):
    def __init__(self, mlps_folder, transform=None):
        self.mlps_folder = mlps_folder

        if hasattr(transform, "__iter__"):
            self.transform = transform
        else:
            self.transform = [transform]

        self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        self.mlp_files = [
            file for i, file in enumerate(self.mlp_files) if i not in outliers_set
        ]

        self.device = torch.device(get_default_device())
        self.mlp_kwargs = mlp_kwargs

    def __getitem__(self, index):
        file = join(self.mlps_folder, self.mlp_files[index])
        out = torch.load(file, map_location=self.device)
        y = "plane"

        for t in self.transform:
            out, y = t(out, y)

        return out, y

    def __len__(self):
        return len(self.mlp_files)


class ModelTransform3D(torch.nn.Module):
    def __init__(self, weights_dict: dict = mlp_kwargs):
        super().__init__()
        self.weights_dict = weights_dict

    def forward(self, state_dict, y=None):
        if "state_dict" in state_dict:
            model = MLP3D(**state_dict["model_config"])
            model.load_state_dict(state_dict["state_dict"])
        else:
            model = MLP3D(**self.weights_dict)
            model.load_state_dict(state_dict)
        return model, y


class FlattenTransform3D(torch.nn.Module):
    def forward(self, state_dict, y):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
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


def get_total_mean_n_std(dataset: ShapeNetDataset, dim=128, norm_over_dim=None):
    all_weights = torch.stack([sample[0] for sample in dataset])
    if norm_over_dim:
        return all_weights.view(-1, dim).mean(dim=norm_over_dim), all_weights.view(
            -1, dim
        ).std(dim=norm_over_dim)
    else:
        return all_weights.view(-1, dim).mean(), all_weights.view(-1, dim).std()


class ZScore3D(nn.Module):
    def __init__(self, neuron_means, neuron_stds):
        super().__init__()
        self.neuron_means = neuron_means
        self.neuron_stds = neuron_stds

    def forward(self, neuron_weights, last_bias):
        # Apply min-max normalization
        normalized_neurons = (neuron_weights - self.neuron_means) / self.neuron_stds
        return normalized_neurons, last_bias

    def reverse(self, neuron_weights, last_bias):
        # Revers min-max normalization
        normalized_neurons = neuron_weights * self.neuron_stds + self.neuron_means
        return normalized_neurons, last_bias


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
