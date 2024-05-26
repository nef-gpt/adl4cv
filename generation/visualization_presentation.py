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
from collections import OrderedDict
from typing import List, Tuple, Union

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from networks.mlp_models import MLP3D

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

def reconstruct_image(model: torch.nn.Module, image_size: tuple = (28, 28)):

    input_coords = make_coordinates(image_size, 1)
     # Generate image using the INR model
    with torch.no_grad():
        reconstructed_image = model(input_coords)
        reconstructed_image = torch.sigmoid(reconstructed_image)
        reconstructed_image = reconstructed_image.view(*image_size, -1)
        reconstructed_image = reconstructed_image.permute(2, 0, 1)

    return reconstructed_image.squeeze(0).numpy()

def state_dict_to_min_max(state_dict: OrderedDict):
    weights = []

    for key, value in state_dict.items():
        weights.append(value)

    vmax = torch.Tensor([w.max() for w in weights]).max()
    vmin = torch.Tensor([w.min() for w in weights]).min()

    return vmin, vmax


def get_vmin_vmax(image_idx: int, num_epochs: int, model_config: dict):
    vmins = []
    vmaxs = []
    for epoch in range(num_epochs):
        path = os.path.dirname(os.path.abspath(__file__))
        
        model_path = path + f"/mnist-nerfs/recording/mnist-nerfs-unstructured-{image_idx}_{epoch}_model_final.pth"
        assert os.path.exists(model_path), f"File {model_path} does not exist"

        vmin, vmax = state_dict_to_min_max(torch.load(model_path))
        vmins.append(vmin)
        vmaxs.append(vmax)

    vmax = torch.Tensor([v.max() for v in vmaxs]).max()
    vmin = torch.Tensor([v.min() for v in vmins]).min()

    return vmin, vmax

def get_figure(model: torch.nn.Module, vmin: torch.Tensor, vmax: torch.Tensor):

    # Generate random data for demonstration
    image = reconstruct_image(model)
    state_dict = model.state_dict()
    W1 = state_dict['layers.0.weight']
    W2 = state_dict['layers.1.weight']
    W3 = state_dict['layers.2.weight']
    b1 = state_dict['layers.0.bias'].unsqueeze(-1)
    b2 = state_dict['layers.1.bias'].unsqueeze(-1)
    b3 = state_dict['layers.2.bias'].unsqueeze(-1)

    vmax = torch.Tensor((W1.max(), W2.max(), W3.max(), b1.max(), b2.max(), b3.max())).max()
    vmin = torch.Tensor((W1.min(), W2.min(), W3.min(), b1.min(), b2.min(), b3.min())).min()


    # Set the x-size of the whole figure
    figure_x_size = 10

    # Create the figure and axes using gridspec for better layout control
    fig = plt.figure(figsize=(figure_x_size, figure_x_size * 3))
    gs = gridspec.GridSpec(6, 3, height_ratios=[28, 16, 16, 1, 1, 28], width_ratios=[18, 1, 1])
    gs.update(wspace=0.1, hspace=0.3)

    # First row: Image
    ax_image = plt.subplot(gs[0, 0])
    ax_image.imshow(image, cmap='gray', aspect='equal')
    ax_image.axis('off')

    # Second row: W1 and b1
    ax_W1 = plt.subplot(gs[1, 0])
    im1 = ax_W1.imshow(W1, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    ax_W1.axis('off')

    ax_b1 = plt.subplot(gs[1, 1])
    ax_b1.imshow(b1, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    ax_b1.axis('off')

    # Third row: W2 and b2
    ax_W2 = plt.subplot(gs[2, 0])
    ax_W2.imshow(W2, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    ax_W2.axis('off')

    ax_b2 = plt.subplot(gs[2, 1])
    ax_b2.imshow(b2, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    ax_b2.axis('off')

    # Fourth row: W3 and b3
    ax_W3 = plt.subplot(gs[3, 0])
    ax_W3.imshow(W3, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    ax_W3.axis('off')

    ax_b3 = plt.subplot(gs[3, 1])
    ax_b3.imshow(b3, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    ax_b3.axis('off')


    # Save the figure
    plt.rcParams.update(
        {
            "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
            "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
        }
    )
    """plt.savefig('visualization.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    """

    return fig



def visualize_learning_process(image_idx: int, num_epochs: int, model_config: dict):

    frames_dir = "frames"

    vmin, vmax = get_vmin_vmax(image_idx, num_epochs, model_config)


    for epoch in range(num_epochs):
        path = os.path.dirname(os.path.abspath(__file__))
        model = MLP3D(**model_config)
        
        model_path = path + f"/mnist-nerfs/recording/mnist-nerfs-unstructured-{image_idx}_{epoch}_model_final.pth"
        assert os.path.exists(model_path), f"File {model_path} does not exist"

        model.load_state_dict(torch.load(model_path))

        fig = get_figure(model, vmin, vmax)

        frame_path = os.path.join(frames_dir, f"frame_{epoch:03d}.png")
        plt.savefig(frame_path, format="png")
        plt.close(fig)

    # Compile the images into a video
    frame_paths = [
        os.path.join(frames_dir, f"frame_{epoch:03d}.png")
        for epoch in range(num_epochs)
        if os.path.exists(os.path.join(frames_dir, f"frame_{epoch:03d}.png"))
    ]
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape
    
    video_path = "learning_process.mp4"
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {video_path}")


def main():
    model_config = {
        "out_size": 1,
        "hidden_neurons": [16, 16],
        "use_leaky_relu": False,
        "output_type": "logits",
        "input_dims": 2,
        "multires": 4,
    }

    image_idx = 0  # Change this to visualize a different image from the dataset
    num_epochs = 200  # Number of epochs or models saved
    visualize_learning_process(image_idx, num_epochs, model_config)


if __name__ == "__main__":
    main()

