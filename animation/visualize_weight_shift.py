import torch
import glob
import matplotlib.pyplot as plt
import numpy as np

idx = 65

sample_0 = torch.load("./datasets/mnist-nerfs/unconditioned/image-0_model_final.pth")
structured_sample = torch.load(
    f"./datasets/mnist-nerfs/pretrained/image-{idx}_model_final.pth"
)
unstructured_sample = torch.load(
    f"./datasets/mnist-nerfs/unconditioned/image-{idx}_model_final.pth"
)

layer = 1


structure_error = (
    sample_0[f"layers.{layer}.weight"] - structured_sample[f"layers.{layer}.weight"]
).abs()
unstructure_error = (
    sample_0[f"layers.{layer}.weight"] - unstructured_sample[f"layers.{layer}.weight"]
).abs()

vmin = torch.Tensor([unstructure_error.min(), structure_error.min()]).min()
vmax = torch.Tensor([unstructure_error.max(), structure_error.max()]).max()


def plot_heatmaps(array1, array2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first heatmap
    im1 = axes[0].imshow(array1, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("Heatmap 1")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    fig.colorbar(im1, ax=axes[0])

    # Plot the second heatmap
    im2 = axes[1].imshow(array2, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("Heatmap 2")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Y-axis")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


# Call the function with your arrays
plot_heatmaps(structure_error, unstructure_error)
