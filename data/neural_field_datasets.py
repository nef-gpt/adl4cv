from data.base_dataset import BaseDataset
import logging
import json
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from collections import defaultdict
from pathlib import Path
import os
import torch
import torch.nn as nn
from networks.mlp_models import MLP3D

from networks.naive_rq_ae import RQAutoencoder
from vector_quantize_pytorch import VectorQuantize


def generate_splits(data_path, save_path, name="mnist_splits.json", val_size=5000):
    save_path = Path(save_path) / name
    inr_path = Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))
    for p in list(inr_path.glob("mnist_png_*/**/*.pth")):

        if "training" in p.as_posix():
            s = "train"
        else:
            s = "test"

        data_split[s]["path"].append((os.getcwd() / p).as_posix())
        data_split[s]["label"].append(p.parent.parent.stem.split("_")[-2])

    # val split
    val_size = val_size
    train_indices, val_indices = train_test_split(
        range(len(data_split["train"]["path"])), test_size=val_size
    )
    data_split["val"]["path"] = [data_split["train"]["path"][v] for v in val_indices]
    data_split["val"]["label"] = [data_split["train"]["label"][v] for v in val_indices]

    data_split["train"]["path"] = [
        data_split["train"]["path"][v] for v in train_indices
    ]
    data_split["train"]["label"] = [
        data_split["train"]["label"][v] for v in train_indices
    ]

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, test size: {len(data_split['test']['path'])}"
    )

    print(save_path)
    with open(save_path, "w") as file:
        json.dump(data_split, file)



# OUR DATASET 
class MnistNeFDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        type: str = "unconditioned",
        quantized: bool = False,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        download_url: str = None,
        force_download: bool = False,
        fixed_label: Optional[int] = None,
    ):
        """Initialize the Dataset object.

        Args:
            path (Union[str, Path]): The path to the directory containing the dataset files.
            start_idx (Union[int, float], optional): The starting index or fraction of the dataset to use. Defaults to 0.0.
            end_idx (Union[int, float], optional): The ending index or fraction of the dataset to use. Defaults to 1.0.
            data_prefix (str, optional): The prefix of the dataset files. Defaults to "".
            data_keys (List[str], optional): The list of keys to load from the dataset files. Defaults to None, which loads all keys.
            transform (Optional[Union[Callable, Dict[str, Callable]]], optional): The transformation function to apply to the loaded data.
                Defaults to None.
        """

        super().__init__(path, download_url=download_url, force_download=force_download)

        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f"Path {path.absolute()} does not exist"
        assert path.is_dir(), f"Path {path.absolute()} is not a directory"

        self.dataset = json.load(open(Path(path) / "overview.json", "r"))
        self.transform = transform
        self.type = type + " quantized" if quantized else type
        self.fixed_label = fixed_label

        self.length, self.mapping = self.calculate_length_and_mapping()

    def min_max(self):
        mins = []
        maxs = []
        for i in range(len(self.dataset)):
            weights, _ = self.__getitem__(i)
            mins.append(weights.min())
            maxs.append(weights.max())
        min = torch.Tensor(mins).min()
        max = torch.Tensor(maxs).max()
        return min, max
            
    def calculate_length_and_mapping(self):
        # check for entry label for calculation
        if self.fixed_label is not None:
            keys = [
                    (i, k)
                    for i, k in enumerate(self.dataset[self.type].keys())
                    if self.dataset[self.type][k]["label"] == self.fixed_label
                ]
            
            return len(keys), [i for i, k in keys]
        else:
            return len(self.dataset[self.type].keys()), list(range(len(self.dataset[self.type].keys())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        target = torch.load(
            self.dataset[self.type][list(self.dataset[self.type].keys())[self.mapping[idx]]]["output"], map_location=torch.device("cpu")
        )

        label = self.dataset[self.type][list(self.dataset[self.type].keys())[self.mapping[idx]]]["label"]

        if self.fixed_label is not None:
            assert label == self.fixed_label, f"Label {label} does not match fixed label {self.fixed_label}"

        result = (target, label)
        if self.transform:
            result = self.transform(*result)

        return result

# Taken from neural-field-arena
class DWSNetsDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        split: str = "train",
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        download_url: str = None,
        force_download: bool = False,
    ):
        """Initialize theDataset object.

        Args:
            path (Union[str, Path]): The path to the directory containing the dataset files.
            start_idx (Union[int, float], optional): The starting index or fraction of the dataset to use. Defaults to 0.0.
            end_idx (Union[int, float], optional): The ending index or fraction of the dataset to use. Defaults to 1.0.
            data_prefix (str, optional): The prefix of the dataset files. Defaults to "".
            data_keys (List[str], optional): The list of keys to load from the dataset files. Defaults to None, which loads all keys.
            transform (Optional[Union[Callable, Dict[str, Callable]]], optional): The transformation function to apply to the loaded data.
                Defaults to None.
        """

        super().__init__(path, download_url=download_url, force_download=force_download)

        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f"Path {path.absolute()} does not exist"
        assert path.is_dir(), f"Path {path.absolute()} is not a directory"

        self.split = split
        self.transform = transform
        self.path = path
        if os.path.isfile(path / "mnist_splits.json") is not True:
            generate_splits(path, path, name="mnist_splits.json")

        self.dataset = json.load(open(Path(path) / "mnist_splits.json", "r"))

    def __len__(self):
        return len(self.dataset[self.split]["path"])

    def __getitem__(self, idx):
        target = torch.load(
            self.dataset[self.split]["path"][idx], map_location=torch.device("cpu")
        )

        label = int(self.dataset[self.split]["label"][idx])

        result = (target, label)
        if self.transform:
            result = self.transform(*result)

        return result


class ModelTransform(nn.Module):
    def forward(self, weights_dict, y):
        model = MLP3D(**weights_dict["model_config"])
        model.load_state_dict(weights_dict["state_dict"])
        return model, y


def quantize_model(model, vq):
    # Quantize and replace each parameter
    for name, param in model.named_parameters():
        shape = param.shape
        flattened_param = param.flatten()
        quantized_params, _, _ = vq(flattened_param.unsqueeze(-1))
        param.data = quantized_params.squeeze(-1).view(shape)
    return model


class QuantizeTransform(nn.Module):
    def __init__(self, vq: VectorQuantize):
        super().__init__()
        self.vq = vq
        self.model_transform = ModelTransform()
    def forward(self, weights_dict, y):
        model, y = self.model_transform(weights_dict, y)
        for param in model.parameters():
            param.requires_grad = False

        for param in self.vq.parameters():
            param.requires_grad = True

        return quantize_model(model, self.vq), y



class FlattenTransform(nn.Module):
    def forward(self, weights_dict, y):
        weights = torch.cat(
            [weights_dict["state_dict"][key].flatten() for key in weights_dict["state_dict"].keys()]
        )
        return weights, y

"""
class ModelTransform(nn.Module):
    def __init__(self, model: RQAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, weights, y):
        # Apply min-max normalization
        _x, indices, _commit_loss = self.model.encode_to_cb(weights)
        codes = self.model.vq.get_codes_from_indices(indices)
        x = torch.cat(
            (
                torch.Tensor([self.model.codebook_size]),
                indices,
                torch.Tensor([self.model.codebook_size + 1]),
            )
        )
        return x, y
"""

class LayerOneHotTransform(nn.Module):
    def forward(self, weights_dict, y):
        # one hot encoding of the layer id, should be a tensor of shape (num_of_flattened_entries, num_layers)
        layer_index = []
        for key in weights_dict.keys():
            layer = int(key.split(".")[1])
            layer_index += [layer] * weights_dict[key].numel()
        one_hot = nn.functional.one_hot(torch.tensor(layer_index))
        return one_hot, y


class BiasFlagTransform(nn.Module):
    def forward(self, weights_dict, y):
        bias_flag = []
        for key in weights_dict.keys():
            bias = int(key.split(".")[2] == "bias")
            bias_flag += [bias] * weights_dict[key].numel()
        return torch.tensor(bias_flag).unsqueeze(-1), y


# class PositionEncodingTransform(nn.Module):
#     def __init__(self, sigma: float = 1.0, m: int = 10):
#         super().__init__()
#         self.pos_enc = PositionalEncoding(sigma=sigma, m=m)

#     def forward(self, weights, y):
#         positions = torch.arange(1, weights.size(0) + 1, dtype=torch.float32)
#         weights = self.pos_enc(positions).reshape((weights.size(0), -1))
#         return weights, y


class TokenTransform(nn.Module):
    def __init__(self, model: RQAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, weights, y):
        # Apply min-max normalization
        _x, indices, _commit_loss = self.model.encode_to_cb(weights)
        codes = self.model.vq.get_codes_from_indices(indices)
        x = torch.cat(
            (
                torch.Tensor([self.model.codebook_size]),
                indices,
                torch.Tensor([self.model.codebook_size + 1]),
            )
        )
        return x, y


class MinMaxTransform(nn.Module):
    def __init__(self, min_value: float = -0.3587, max_value: float = 0.4986):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, weights, y):
        # Apply min-max normalization
        weights = (weights - self.min_value) / (self.max_value - self.min_value)
        return weights, y
    
    def reverse(self, normalized_weights):
        # Reverse the min-max normalization
        original_weights = normalized_weights * (self.max_value - self.min_value) + self.min_value
        return original_weights
    
class FlattenMinMaxTransform(torch.nn.Module):
  def __init__(self, min_max: tuple = None):
    super().__init__()
    self.flatten = FlattenTransform()
    if min_max:
      self.minmax = MinMaxTransform(*min_max)
    else:
      self.minmax = MinMaxTransform()

  def forward(self, x, y):
    x, _ = self.flatten(x, y)
    x, _ = self.minmax(x, y)
    return x, y
