from torch.utils.data import Dataset
import numpy as np
import os
import torch

from networks.mlp_models import MLP3D
from os.path import join

def get_mlp(mlp_kwargs):
    return MLP3D(**mlp_kwargs)


class WeightDataset(Dataset):
    def __init__(
        self, mlps_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mlps_folder = mlps_folder
        self.condition = cfg.transformer_config.params.condition
        files_list = list(os.listdir(mlps_folder))
        blacklist = {}
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))
        if object_names is None:
            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        else:
            self.mlp_files = []
            for file in list(os.listdir(mlps_folder)):
                # Excluding black listed shapes
                if cfg.filter_bad and file.split("_")[1] in blacklist:
                    continue
                # Check if file is in corresponding split (train, test, val)
                # In fact, only train split is important here because we don't use test or val MLP weights
                if ("_" in file and (file.split("_")[1] in object_names or (
                        file.split("_")[1] + "_" + file.split("_")[2]) in object_names)) or (file in object_names):
                    self.mlp_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.mlp_kwargs = mlp_kwargs
        self.cfg = cfg
        if "first_weight_name" in cfg and cfg.first_weight_name is not None:
            self.first_weights = self.get_weights(
                torch.load(os.path.join(self.mlps_folder, cfg.first_weight_name))
            ).float()
        else:
            self.first_weights = torch.tensor([0])

    def get_weights(self, state_dict):
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        prev_weights = weights.clone()

        if self.cfg.jitter_augment:
            weights += np.random.uniform(0, 1e-3, size=weights.shape)

        if self.transform:
            weights = self.transform(weights)
        # We also return prev_weights, in case you want to do permutation, we store prev_weights to sanity check later
        return weights, prev_weights

    def __getitem__(self, index):
        file = self.mlp_files[index]
        dir = join(self.mlps_folder, file)
        if os.path.isdir(dir):
            path1 = join(dir, "checkpoints", "model_final.pth")
            path2 = join(dir, "checkpoints", "model_current.pth")
            state_dict = torch.load(path1 if os.path.exists(path1) else path2)
        else:
            state_dict = torch.load(dir, map_location=torch.device("cpu"))

        weights, weights_prev = self.get_weights(state_dict)

        return weights.float(), weights_prev.float(), weights_prev.float()

    def __len__(self):
        return len(self.mlp_files)