# Description: Overfit neural field networks on the MNIST dataset
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from networks.mlp_models import MLP3D
from torchvision import datasets, transforms
from PIL import Image
import json
import numpy as np
import torch
import loss_functions
import train_nerf as training
import wandb
import time
from torch.utils.data import DataLoader, Dataset
from munch import DefaultMunch


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
}

config_file = "./datasets/mnist-nerfs/overview.json"


def load_config():
    with open(config_file, "r") as f:
        return json.load(f)


def update_config(config, entry):
    """
    Takes a config dictionary and updates the config object
    based on a fit_single_batch entry.

    First match the type of the entry, then use the idx to set the config
    """
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

    if init_model_path:
        model.load_state_dict(torch.load(init_model_path))

    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    foldername = (
        f"./datasets/mnist-nerfs/{"pretrained" if init_model_path else "unconditioned"}"
    )

    train_config = {
        "epochs": 250,
        "lr": 4e-4 if init_model_path is None else 4e-3,
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

    # init wandb
    wandb.init(
        project="nerfs",
        name="image-" + str(i) + "-run-" + time.strftime("%Y-%m-%d-%H-%M-%S"),
        config=model_config | train_config | cfg,
    )

    total_loss = training.train(
        model,
        train_dataloader=dataloader,
        loss_fn=loss_fn,
        **train_config,
        cfg=DefaultMunch.fromDict(cfg),
        wandb=wandb,
        summary_fn=None,
        save_epoch_interval=save_during_epochs,
    )

    return {
        "file": train_config["filename"],
        "label": label,
        "loss": total_loss,
        "type": "pretrained" if init_model_path else "unconditioned",
        "init_model": init_model_path if init_model_path else "None",  # "None
        "idx": i,
    }


mnist = datasets.MNIST("mnist-data", train=True, download=True)
# load config
config = load_config()

# settings
only_label = 5  # can also be None
range = range(0, 100)  # can also be None
save_during_epochs = 1


def train_unconditioned():
    for i, data in enumerate(mnist):

        if (range is not None) and (i not in range):
            continue

        image, label = data

        if only_label is not None and label != only_label:
            continue

        entry = fit_single_batch(image, label, i, None)
        # update config
        config = update_config(config, entry)


def lookup_pretrained(label):
    for entry in config["unconditioned"]:
        if entry["label"] == label:
            return entry
    return None


def train_pretrained():
    for i, data in enumerate(mnist):
        if (range is not None) and (i not in range):
            continue
        image, label = data

        if only_label is not None and label != only_label:
            continue

        entry = fit_single_batch(image, label, i, lookup_pretrained(label)["file"])
        # update config
        config = update_config(config, entry)
    pass


def main():
    print("Training unstructured")
    train_unconditioned()
    print("Training structured")
    train_pretrained()

    save_config(config)


if __name__ == "__main__":
    main()
