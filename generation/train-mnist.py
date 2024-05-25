# Description: Overfit neural field networks on the MNIST dataset

from networks.mlp_models import MLP3D
from torchvision import datasets, transforms
from PIL import Image
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


def fit_single_batch(image: Image.Image, label: int, i: int):
    # image is a (28, 28) tensor
    image = transforms.functional.pil_to_tensor(image).squeeze(0) / 255.0
    # we need to construct a dataloader that maps x, y -> image[x, y] to train a neural field network
    # where x, y are coordinates in the image (but between 0 and 1)

    # Create dataset and dataloader
    dataset = MNISTNeRFDataset(image)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Create neural field network
    # out_size=1, hidden_neurons=[16, 16], use_leaky_relu=True, input_dims=2
    model_config = {
        "out_size": 1,
        "hidden_neurons": [16, 8],
        "use_leaky_relu": True,
        "output_type": "logits",  # "
        "input_dims": 2,
    }
    model = MLP3D(**model_config)

    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    train_config = {
        "epochs": 800,
        "lr": 1e-3,
        "steps_til_summary": 100,
        "epochs_til_checkpoint": 100,
        "model_dir": "mnist-nerfs",
        "double_precision": False,
        "clip_grad": False,
        "use_lbfgs": False,
        "loss_schedules": None,
        "filename": "mnist-nerfs-unstructured-{}".format(i),
    }

    cfg = {
        "scheduler": {
            "type": "adaptive",
            "step_size": 30,
            "gamma": 0.1,
            "min_lr": 1e-5,
            "patience": 50,
            "patience_adaptive": 10,
            "factor": 0.8,
            "threshold": 0,
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

    training.train(
        model,
        train_dataloader=dataloader,
        loss_fn=loss_fn,
        **train_config,
        cfg=DefaultMunch.fromDict(cfg),
        wandb=wandb,
        summary_fn=None,
    )


def main():

    mnist = datasets.MNIST("mnist-data", train=True, download=True)

    for i, data in enumerate(mnist):
        image, label = data
        fit_single_batch(image, label, i)


if __name__ == "__main__":
    main()
