import sys
import os
from typing import List, Tuple, Union
import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import cv2


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from networks.mlp_models import MLP3D

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


def visualize_learning_process(image_idx: int, num_epochs: int):
    # Configuration
    image_size = (28, 28)
    # Get dataset elements
    image, label = mnist[image_idx]

    # Initialize model configuration
    model_config = {
        "out_size": 1,
        "hidden_neurons": [16, 16],
        "use_leaky_relu": False,
        "output_type": "logits",
        "input_dims": 2,
        "multires": 4,
    }
    model = MLP3D(**model_config)
    input_coords = make_coordinates(image_size, 1)

    # Create a directory to store the frames
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    for epoch in range(num_epochs):
        path = os.path.dirname(os.path.abspath(__file__))
        
        model_path = path + f"/mnist-nerfs/recording_pretrained/mnist-nerfs-structured-{image_idx}_{epoch}_model_final.pth"
        
        assert os.path.exists(model_path), f"File {model_path} does not exist"

        model.load_state_dict(torch.load(model_path))

        # Generate image using the INR model
        with torch.no_grad():
            reconstructed_image = model(input_coords)
            reconstructed_image = torch.sigmoid(reconstructed_image)
            reconstructed_image = reconstructed_image.view(*image_size, -1)
            reconstructed_image = reconstructed_image.permute(2, 0, 1)

        reconstructed_tensor = reconstructed_image.squeeze(0).numpy()

        # Save the frame as a PNG image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image, cmap="gray", aspect="auto")
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("X-axis")
        axes[0].set_ylabel("Y-axis")

        axes[1].imshow(reconstructed_tensor, cmap="gray", aspect="auto")
        axes[1].set_title(f"Reconstructed (Epoch {epoch})")
        axes[1].set_xlabel("X-axis")
        axes[1].set_ylabel("Y-axis")

        plt.tight_layout()

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
    
    video_path = "learning_process_pretrained.mp4"
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {video_path}")


def main():
    image_idx = 35  # Change this to visualize a different image from the dataset
    num_epochs = 200  # Number of epochs or models saved
    visualize_learning_process(image_idx, num_epochs)


if __name__ == "__main__":
    main()
