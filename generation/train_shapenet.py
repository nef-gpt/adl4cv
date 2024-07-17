# Description: Overfit neural field networks on the MNIST dataset
import os
import sys
from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm

# from utils import get_default_device

try:
    set_start_method("spawn")
except RuntimeError:
    pass


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data.pointcloud_dataset import PointCloud
from networks.mlp_models import MLP3D
from torchvision import datasets, transforms
from PIL import Image
import json
import numpy as np
import torch
import generation.train_shapenet_nef as training
import wandb
import time
from torch.utils.data import DataLoader, Dataset
from munch import DefaultMunch
import argparse

config = {}
config["unconditioned"] = {}
config["pretrained"] = {}

files = [file for file in os.listdir("./datasets/02691156_pc")]
file_unconditioned = ["5d7c2f1b6ed0d02aa4684be4f9cb3c1d.npy"]


from vector_quantize_pytorch import VectorQuantize
from data.nef_mnist_dataset import quantize_model

model_config = {
    "out_size": 1,
    "hidden_neurons": [32, 32, 32, 32],
    "use_gelu": True,
    "input_dims": 3,
    "multires": 2,
    "include_input": True,
    "output_type": "sdf",
}

# settings
idx_range = None  # Srange(0, 2500)S
save_during_epochs = None  # 1
skip_existing_models = False
skip_unconditioned = False

config_file = "./datasets/shapenet_nef_2/overview.json"
cpu_mode = False
device = torch.device("cpu" if cpu_mode else "cuda")  # get_default_device()

print("Using device", device)


def load_config():
    # create config file if it does not exist
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(
                {
                    "pretrained": {},
                    "unconditioned": {},
                },
                f,
            )
    with open(config_file, "r") as f:
        return json.load(f)


def update_config(config, entry):
    """
    Takes a config dictionary and updates the config object
    based on a fit_single_batch entry.

    First match the type of the entry, then use the idx to set the config
    """

    # try to reload the config first
    config = load_config()

    if entry["type"] == "pretrained":
        config["pretrained"][entry["idx"]] = entry
    else:
        config["unconditioned"][entry["idx"]] = entry

    return config


def save_config(config):
    with open(config_file, "w") as f:
        json.dump(config, f)


def fit_single_batch(i: int, init_model_path=None, batch_size=2 * 4096):

    # Create dataset and dataloader
    if init_model_path:
        files = [file for file in os.listdir("./datasets/02691156_pc")]
    else:
        files = file_unconditioned

    dataset = PointCloud("./datasets/02691156_pc/" + files[i], batch_size, strategy="")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = MLP3D(**model_config)

    if init_model_path:
        model.load_state_dict(torch.load(init_model_path)["state_dict"])

    model.to(device)

    # loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    loss_fn = torch.nn.functional.mse_loss

    subfoldername = "pretrained" if init_model_path else "unconditioned"
    foldername = f"./datasets/shapenet_nef_2/{subfoldername}"

    train_config = {
        "epochs": 300,
        "lr": 1e-2,
        "steps_til_summary": 100,
        "epochs_til_checkpoint": 100,
        "model_dir": "datasets/shapenet_nef_2",
        "double_precision": False,
        "clip_grad": False,
        "use_lbfgs": False,
        "loss_schedules": None,
        "filename": "{}/nef-{}".format(foldername, files[i].split(".")[0]),
    }

    cfg = {
        "scheduler": {
            "type": "adaptive",
            "step_size": 30,
            "gamma": 0.1,
            "min_lr": 1e-5,
            "patience": 1000,
            "patience_adaptive": 10,
            "factor": 0.8,
            "threshold": 0.0,
        },
        "strategy": "not_continue",
        "mlp_config": {"move": False},
    }

    # check if we already trained this model (eg. if filename exists), then skip training and just return object
    if (
        os.path.exists(train_config["filename"] + "_model_final.pth")
        and skip_existing_models
    ):
        print("Model already trained, skipping")
        return {
            "file-prefix": train_config["filename"],
            "output": train_config["filename"] + "_model_final.pth",
            "label": "plane",
            "loss": 0,
            "type": "pretrained" if init_model_path else "unconditioned",
            "init_model": init_model_path if init_model_path else "None",  # "None
            "idx": i,
        }

    # init wandb
    wandb.init(
        project="nefs",
        name="shapenet-"
        + str(i)
        + "-"
        + ("pretrained" if init_model_path else "unconditioned")
        + "-run-"
        + time.strftime("%Y-%m-%d-%H-%M-%S"),
        config=model_config | train_config | cfg,
    )

    total_loss, output_name = training.train(
        model,
        train_dataloader=dataloader,
        loss_fn=loss_fn,
        **train_config,
        cfg=DefaultMunch.fromDict(cfg),
        wandb=wandb,
        model_config=model_config,
        summary_fn=None,
        save_epoch_interval=save_during_epochs,
        device=device,
        disable_tqdm=False,
        l2_loss_lambda=5.0 if init_model_path is None else 0.0,
    )

    return {
        "file-prefix": train_config["filename"],
        "output": output_name,
        "label": "plane",
        "loss": total_loss,
        "type": "pretrained" if init_model_path else "unconditioned",
        "init_model": init_model_path if init_model_path else "None",  # "None
        "idx": i,
    }


# load config
config = load_config()


def train_unconditioned_single(i):
    if (idx_range is not None) and (i not in idx_range):
        return

    global config

    print(f"Training image {i} and no pretrained model")
    entry = fit_single_batch(i, None)
    # update config
    config = update_config(config, entry)
    save_config(config)


def train_unconditioned():

    for i in range(1):
        train_unconditioned_single(i)


def lookup_pretrained(config):
    # config[unconditioned] is a dictionary with keys as indices and values as dictionaries
    for key, entry in config["unconditioned"].items():
        return entry
    return None


def train_pretrained_single(i):
    if (idx_range is not None) and (i not in idx_range):
        print(f"Invalid index {i}")
        return

    global config
    pretrained_entry = lookup_pretrained(config)
    print(f"Training image {i} with pretrained model {pretrained_entry['output']}")
    entry = fit_single_batch(i, init_model_path=pretrained_entry["output"])

    # update config
    # config = update_config(config, entry)
    # save_config(config)


def train_pretrained(backward=cpu_mode):

    for i in tqdm(range(int(len(files) / 2))):
        if backward:
            i = len(files) - i - 1
        train_pretrained_single(i)


def main():
    # ensure save config is called in the end
    try:
        if not skip_unconditioned:
            print("Training unstructured")
            train_unconditioned()
        print("Training structured")
        train_pretrained()
    finally:
        save_config(config)


if __name__ == "__main__":
    main()
