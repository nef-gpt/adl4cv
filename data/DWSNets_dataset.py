from data.base_dataset import BaseDataset
import logging
import json
from sklearn.model_selection import train_test_split
from rff.layers import PositionalEncoding
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from collections import defaultdict
from pathlib import Path
import os
import torch
import torch.nn as nn

from networks.naive_rq_ae import RQAutoencoder



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

        if self.transform:
            target, label = self.transform(target, label)

        return target, label


class FlattenTransform(nn.Module):
    def forward(self, weights_dict, y):
        weights = torch.cat(
            [weights_dict[key].flatten() for key in weights_dict.keys()]
        )
        return weights.unsqueeze(-1), y


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


class PositionEncodingTransform(nn.Module):
    def __init__(self, sigma: float = 1.0, m: int = 10):
        super().__init__()
        self.pos_enc = PositionalEncoding(sigma=sigma, m=m)

    def forward(self, weights, y):
        positions = torch.arange(1, weights.size(0) + 1, dtype=torch.float32)
        weights = self.pos_enc(positions).reshape((weights.size(0), -1))
        return weights, y
    
class TokenTransform(nn.Module):
    def __init__(self, model: RQAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, weights, y):
        # Apply min-max normalization
        _x, indices, _commit_loss = self.model.encode_to_cb(weights)
        codes = self.model.vq.get_codes_from_indices(indices)
        x = torch.cat((torch.Tensor([self.model.codebook_size]), indices, torch.Tensor([self.model.codebook_size + 1])))
        return x, y
    

class MinMaxTransformer(nn.Module):
    def __init__(self, min_value: float = -0.3587, max_value: float = 0.4986):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, weights, y):
        # Apply min-max normalization
        weights = (weights - self.min_value) / (self.max_value - self.min_value)
        return weights, y
