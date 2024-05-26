import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Tuple, Union
from networks.mlp_models import MLP3D
import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

mnist = datasets.MNIST("mnist-data", train=True, download=True)


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (0, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def visualize_neural_field(idx: int):
    # Configuration
    image_size = (28, 28)
    # Get dataset elements
    image, label = mnist[idx]

    # Initialize and load the INR model
    model_config = {
        "out_size": 1,
        "hidden_neurons": [16, 16],
        "use_leaky_relu": False,
        "output_type": "logits",  # "
        "input_dims": 2,
        "multires" : 4,
    }
    model = MLP3D(**model_config)

    model.load_state_dict(
        torch.load(
            "mnist-nerfs/mnist-nerfs-unstructured-{}_model_final.pth".format(idx)
        )
    )

    # Generate image using the INR model
    input_coords = make_coordinates(image_size, 1)
    with torch.no_grad():
        reconstructed_image = model(input_coords)
        reconstructed_image = torch.sigmoid(reconstructed_image)
        reconstructed_image = reconstructed_image.view(*image_size, -1)
        reconstructed_image = reconstructed_image.permute(2, 0, 1)

    reconstructed_tensor = reconstructed_image.squeeze(0)

    # Plotting the tensors as heatmaps in grayscale
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(image, cmap="gray", aspect="auto")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")

    axes[1].imshow(reconstructed_tensor, cmap="gray", aspect="auto")
    axes[1].set_title("Reconstructed")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Y-axis")

    plt.colorbar(axes[0].imshow(image, cmap="gray", aspect="auto"), ax=axes[0])
    plt.colorbar(
        axes[1].imshow(reconstructed_tensor, cmap="gray", aspect="auto"), ax=axes[1]
    )
    plt.show()


def main():
    idx = 0
    visualize_neural_field(idx)


if __name__ == "__main__":
    main()
