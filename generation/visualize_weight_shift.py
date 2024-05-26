import torch
import glob
import matplotlib.pyplot as plt
import numpy as np

idx = 11

sample_0 = torch.load("mnist-nerfs/unstructured/mnist-nerfs-unstructured-0_model_final.pth")
structured_sample = torch.load(f"mnist-nerfs/structured/mnist-nerfs-structured-{idx}_model_final.pth")
unstructured_sample = torch.load(f"mnist-nerfs/unstructured/mnist-nerfs-unstructured-{idx}_model_final.pth")

print(sample_0['layers.0.weight'].size())

structure_error = (sample_0['layers.2.weight'] - structured_sample['layers.2.weight']).abs()
unstructure_error = (sample_0['layers.2.weight'] - unstructured_sample['layers.2.weight']).abs()

def plot_heatmaps(array1, array2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first heatmap
    im1 = axes[0].imshow(array1, cmap='viridis', aspect='auto')
    axes[0].set_title('Heatmap 1')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')
    fig.colorbar(im1, ax=axes[0])

    # Plot the second heatmap
    im2 = axes[1].imshow(array2, cmap='viridis', aspect='auto')
    axes[1].set_title('Heatmap 2')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

# Call the function with your arrays
plot_heatmaps(structure_error, unstructure_error)


"""
sample_0 = torch.cat(
            [sample_0[key].flatten() for key in sample_0.keys()]
        )

errors_structured = []
errors_unstructured = []


for filename in glob.glob("mnist-nerfs/unstructured/*.pth"):    
    idx = filename.split("-")[4].split("_")[0]

    if idx == "0":
        continue

    structured_sample = torch.load(f"mnist-nerfs/structured/mnist-nerfs-structured-{idx}_model_final.pth") 
    structured_sample = torch.cat(
                [structured_sample[key].flatten() for key in structured_sample.keys()]
            )
    unstructured_sample = torch.load(f"mnist-nerfs/unstructured/mnist-nerfs-unstructured-{idx}_model_final.pth")
    unstructured_sample = torch.cat(
                [unstructured_sample[key].flatten() for key in unstructured_sample.keys()]
            )

    error_structured = (sample_0 - structured_sample).abs().sum()
    error_unstructured = (sample_0 - unstructured_sample).abs().sum()

    errors_structured.append(error_structured)
    errors_unstructured.append(error_unstructured)

errors_structured = torch.Tensor(errors_structured)
errors_unstructured = torch.Tensor(errors_unstructured)


"""
