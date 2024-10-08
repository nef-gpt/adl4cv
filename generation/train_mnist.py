# Description: Overfit neural field networks on the MNIST dataset
import os
import sys
from torch.multiprocessing import Pool, Process, set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from networks.mlp_models import MLP3D
from torchvision import datasets, transforms
from PIL import Image
import json
import numpy as np
import torch
import generation.train_mnist_nef as training
import wandb
import time
from torch.utils.data import DataLoader, Dataset
from munch import DefaultMunch
import argparse

config = {}
config["unconditioned"] = {}
config["pretrained"] = {}

from vector_quantize_pytorch import VectorQuantize
from data.nef_mnist_dataset import quantize_model

vq = VectorQuantize(
    dim=1,
    codebook_size=1024,  # codebook size
    decay=0.65,  # the exponential moving average decay, lower means the dictionary will change faster
    commitment_weight=1.0,  # the weight on the commitment loss
)


class MNISTNeRFDataset(Dataset):
    def __init__(self, image):
        self.image = image
        self.coords = self._generate_coords()

    def _generate_coords(self):
        h, w = self.image.shape
        coords = np.array([(x, y) for x in range(w) for y in range(h)])
        return coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        pixel_value = self.image[y, x]
        # Normalize coordinates to [0, 1]
        x_norm = x / self.image.shape[1]
        y_norm = y / self.image.shape[0]
        return torch.tensor([x_norm, y_norm], dtype=torch.float32), torch.tensor(
            pixel_value, dtype=torch.float32
        )


# Create neural field network
# out_size=1, hidden_neurons=[16, 16], use_leaky_relu=True, input_dims=2
model_config = {
    "out_size": 1,
    "hidden_neurons": [16, 16],
    "use_leaky_relu": False,
    "output_type": "logits",  # "
    "input_dims": 2,
    "multires": 4,
    "weight_init": lambda tensor: torch.nn.init.xavier_uniform_(
        tensor, gain=torch.nn.init.calculate_gain("relu")
    ),
    "include_input": False,
}

# settings
only_label = None  # can also be None
idx_range = [57980, 57981, 57983, 57985]  # can also be None
save_during_epochs = None  # 1
skip_existing_models = True
skip_unconditioned = True
multi_process = False
quantization = False

config_file = "./datasets/mnist-nerfs/overview.json"
device = torch.device("cuda")  # get_default_device()

print("Using device", device)


def load_config():
    # create config file if it does not exist
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(
                {
                    "pretrained": {},
                    "unconditioned": {},
                    "pretrained quantized": {},
                    "unconditioned quantized": {},
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
    if quantization:
        if entry["type"] == "pretrained":
            config["pretrained quantized"][entry["idx"]] = entry
        else:
            config["unconditioned quantized"][entry["idx"]] = entry
    else:
        if entry["type"] == "pretrained":
            config["pretrained"][entry["idx"]] = entry
        else:
            config["unconditioned"][entry["idx"]] = entry

    return config


def save_config(config):
    with open(config_file, "w") as f:
        json.dump(config, f)


def fit_single_batch(image: Image.Image, label: int, i: int, init_model_path=None):
    # image is a (28, 28) tensor
    image = transforms.functional.pil_to_tensor(image).squeeze(0) / 255.0
    # we need to construct a dataloader that maps x, y -> image[x, y] to train a neural field network
    # where x, y are coordinates in the image (but between 0 and 1)

    # Create dataset and dataloader
    dataset = MNISTNeRFDataset(image)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLP3D(**model_config)

    if quantization:
        model = quantize_model(model, vq)

    if init_model_path:
        model.load_state_dict(torch.load(init_model_path)["state_dict"])

    model.to(device)

    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    subfoldername = "pretrained" if init_model_path else "unconditioned"
    if quantization:
        subfoldername += "_quantized"
    foldername = f"./datasets/mnist-nerfs/{subfoldername}"

    train_config = {
        "epochs": 250,
        "lr": 4e-3 if init_model_path is None else 4e-3,
        "steps_til_summary": 100,
        "epochs_til_checkpoint": 100,
        "model_dir": "mnist-nerfs",
        "double_precision": False,
        "clip_grad": False,
        "use_lbfgs": False,
        "loss_schedules": None,
        "filename": "{}/image-{}".format(foldername, i),
    }

    cfg = {
        "scheduler": {
            "type": "adaptive",
            "step_size": 30,
            "gamma": 0.1,
            "min_lr": 1e-5,
            "patience": 100,
            "patience_adaptive": 10,
            "factor": 0.95,
            "threshold": 0.00001,
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
            "label": label,
            "loss": 0,
            "type": "pretrained" if init_model_path else "unconditioned",
            "init_model": init_model_path if init_model_path else "None",  # "None
            "idx": i,
        }

    # init wandb
    wandb.init(
        project="nerfs",
        name="image-"
        + str(i)
        + "-"
        + ("quantized-" if quantization else "")
        + ("pretrained" if init_model_path else "unconditioned")
        + "-run-"
        + time.strftime("%Y-%m-%d-%H-%M-%S"),
        config=model_config | train_config | cfg,
    )

    if quantization:
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
            vq=vq,
        )
    else:
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
        )

    return {
        "file-prefix": train_config["filename"],
        "output": output_name,
        "label": label,
        "loss": total_loss,
        "type": "pretrained" if init_model_path else "unconditioned",
        "quantization": quantization,
        "init_model": init_model_path if init_model_path else "None",  # "None
        "idx": i,
    }


mnist = datasets.MNIST("mnist-data", train=True, download=True)
# load config
config = load_config()


def train_unconditioned_single(i, data):
    if (idx_range is not None) and (i not in idx_range):
        return

    image, label = data

    if only_label is not None and label != only_label:
        return

    global config

    print(f"Training image {i} with label {label} and no pretrained model")
    entry = fit_single_batch(image, label, i, None)
    # update config
    config = update_config(config, entry)
    save_config(config)


def train_unconditioned():
    if multi_process:
        pool = Pool(8)
        pool.starmap(train_unconditioned_single, enumerate(mnist))
    else:
        for i, data in enumerate(mnist):
            train_unconditioned_single(i, data)


def lookup_pretrained(label, config):
    # config[unconditioned] is a dictionary with keys as indices and values as dictionaries
    for key, entry in (
        config["unconditioned quantized"].items()
        if quantization
        else config["unconditioned"].items()
    ):
        # if entry["label"] == label:
        return entry
    return None


def train_pretrained_single(i, data):
    if (idx_range is not None) and (i not in idx_range):
        return
    image, label = data

    if only_label is not None and label != only_label:
        return
    global config
    pretrained_entry = lookup_pretrained(label, config)
    print(
        f"Training image {i} with label {label} and pretrained model {pretrained_entry['output']}"
    )
    entry = fit_single_batch(image, label, i, pretrained_entry["output"])

    # update config
    config = update_config(config, entry)
    save_config(config)


def train_pretrained():
    if multi_process:
        pool = Pool(8)
        pool.starmap(train_pretrained_single, enumerate(mnist))
    else:
        for i, data in enumerate(mnist):
            train_pretrained_single(i, data)


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
