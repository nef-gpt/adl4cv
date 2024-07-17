import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import os
import numpy as np

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import cv2
import sys
from typing import List, Tuple, Union
from tqdm import tqdm
from PIL import Image
import glob
from networks.mlp_models import MLP3D
from animation.animation_util import (
    make_coordinates,
    ensure_folder_exists,
    reconstruct_image,
    get_vmin_vmax,
    get_model_difference,
)
from animation.animation_util import (
    make_coordinates,
    ensure_folder_exists,
    reconstruct_image,
    get_vmin_vmax,
    get_model_difference,
)
from utils.hd_utils import render_image
from utils.visualization3d import model_to_mesh

model_config = {
    "out_size": 1,
    "hidden_neurons": [16, 16],
    "use_leaky_relu": False,
    "output_type": "logits",
    "input_dims": 2,
    "multires": 4,
}


def save_colorbar(save_path: str, vmin: torch.Tensor, vmax: torch.Tensor):
    fig, ax = plt.subplots(
        figsize=(6, 1)
    )  # Adjusted figsize for horizontal orientation
    fig.subplots_adjust(bottom=0.5)  # Adjusted the subplot for horizontal orientation

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    cmap = plt.cm.viridis

    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal"
    )

    cb1.set_label("", color="white")
    cb1.ax.xaxis.set_tick_params(color="white")
    cb1.outline.set_edgecolor((0, 0, 0, 0))

    # Set the color of the tick labels
    plt.setp(plt.getp(cb1.ax.axes, "xticklabels"), color="white")

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()


def state_dict_to_min_max(state_dict):
    weights = []

    for key, value in state_dict.items():
        weights.append(value)

    vmax = torch.Tensor([w.max() for w in weights]).max()
    vmin = torch.Tensor([w.min() for w in weights]).min()

    return vmin, vmax


def get_figure(
    model: torch.nn.Module,
    vmin: torch.Tensor,
    vmax: torch.Tensor,
    comparison_model: torch.nn.Module = None,
):

    # Generate random data for demonstration
    mesh, sdf = model_to_mesh(model, res=512 + 256)
    image = render_image(mesh)

    state_dict = model.state_dict()
    W1 = state_dict["layers.0.weight"]
    W2 = state_dict["layers.1.weight"]
    W3 = state_dict["layers.2.weight"]
    W4 = state_dict["layers.3.weight"]
    W5 = state_dict["layers.4.weight"]
    b1 = state_dict["layers.0.bias"].unsqueeze(-1)
    b2 = state_dict["layers.1.bias"].unsqueeze(-1)
    b3 = state_dict["layers.2.bias"].unsqueeze(-1)
    b4 = state_dict["layers.3.bias"].unsqueeze(-1)

    # Set the x-size of the whole figure
    figure_x_size = 10

    # Create the figure and axes using gridspec for better layout control
    fig = plt.figure(figsize=(figure_x_size, figure_x_size * 3))
    gs = gridspec.GridSpec(
        6, 2, height_ratios=[28, 16, 16, 16, 16, 0.5], width_ratios=[18, 1]
    )

    gs.update(wspace=0.1, hspace=0.3)

    # First row: Image
    ax_image = plt.subplot(gs[0, 0])
    ax_image.set_xlim((500, 3000))
    ax_image.set_ylim((4250, 1250))
    ax_image.imshow(image, cmap="gray", aspect="equal")
    ax_image.axis("off")

    # Second row: W1 and b1
    ax_W1 = plt.subplot(gs[1, 0])
    ax_W1.imshow(W1, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_W1.axis("off")

    ax_b1 = plt.subplot(gs[1, 1])
    ax_b1.imshow(b1, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_b1.axis("off")

    # Third row: W2 and b2
    ax_W2 = plt.subplot(gs[2, 0])
    ax_W2.imshow(W2, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_W2.axis("off")

    ax_b2 = plt.subplot(gs[2, 1])
    ax_b2.imshow(b2, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_b2.axis("off")

    # Fourth row: W3 and b3
    ax_W3 = plt.subplot(gs[3, 0])
    ax_W3.imshow(W3, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_W3.axis("off")

    ax_b3 = plt.subplot(gs[3, 1])
    ax_b3.imshow(b3, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_b3.axis("off")

    # Fifth row: W4 and b4
    ax_W4 = plt.subplot(gs[4, 0])
    ax_W4.imshow(W4, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_W4.axis("off")

    ax_b4 = plt.subplot(gs[4, 1])
    ax_b4.imshow(b4, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_b4.axis("off")

    # Sixth row: W5 and b5
    ax_W5 = plt.subplot(gs[5, 0])
    ax_W5.imshow(W5, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_W5.axis("off")

    # Save the figure
    plt.rcParams.update(
        {
            "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
            "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
        }
    )

    return fig


if __name__ == "__main__":

    model_dict = torch.load(
        "./datasets/shapenet_nef_2/unconditioned/nef-5d7c2f1b6ed0d02aa4684be4f9cb3c1d_model_final.pth"
    )
    files = [file for file in os.listdir("./datasets/shapenet_nef_2/pretrained/")]
    files = [
        "nef-51ebcde47b4c29d81a62197a72f89474_model_final.pth",
        "nef-56ba815f883279b462b600da24e0965_model_final.pth",
        "nef-e5abd988cb34ed9cdc82b8fee1057b30_model_final.pth",
        "nef-f97fa7329969bcd0ebf1d9fd44798b9b_model_final.pth",
    ]

    model_config = model_dict["model_config"]
    model = MLP3D(**model_config)
    model.load_state_dict(model_dict["state_dict"])

    vmin, vmax = state_dict_to_min_max(model.state_dict())
    vmins = [vmin]
    vmaxs = [vmax]

    for file in files:
        model_dict_vminmax = torch.load("./datasets/shapenet_nef_2/pretrained/" + file)
        model_config_vminmax = model_dict_vminmax["model_config"]
        model_vminmax = MLP3D(**model_config_vminmax)
        model_vminmax.load_state_dict(model_dict_vminmax["state_dict"])
        vmin, vmax = state_dict_to_min_max(model_vminmax.state_dict())
        vmins.append(vmin)
        vmaxs.append(vmax)

    vmin = torch.Tensor(vmins).min()
    vmax = torch.Tensor(vmaxs).max()

    # fig = get_figure(model, vmin, vmax)
    # fig.savefig("./submissions/poster/shapenet_training_visualization/unconditioned_plane.png")

    print("saved unconditioned")

    files = [
        "nef-56ba815f883279b462b600da24e0965_model_final.pth",
        "nef-e5abd988cb34ed9cdc82b8fee1057b30_model_final.pth",
        "nef-f97fa7329969bcd0ebf1d9fd44798b9b_model_final.pth",
    ]

    for file in files:
        model_dict = torch.load("./datasets/shapenet_nef_2/pretrained/" + file)
        model_config = model_dict["model_config"]
        model = MLP3D(**model_config)
        model.load_state_dict(model_dict["state_dict"])
        fig = get_figure(model, vmin, vmax)
        fig.savefig(
            f"./submissions/poster/shapenet_training_visualization/{file.split(".")[0]}.png"
        )
        print(f"saved {file.split(".")[0]}")
