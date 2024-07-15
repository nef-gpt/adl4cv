import os

import torch
from data.neural_field_datasets_shapenet import ImageTransform3D, ShapeNetDataset
import matplotlib.pyplot as plt



image_transform = ImageTransform3D()
dataset_pretrained = ShapeNetDataset(os.path.join("./", "datasets", "shapenet_nefs", "pretrained"), transform=image_transform)

dataset_unconditioned = ShapeNetDataset(os.path.join("./", "datasets", "shapenet_nefs", "unconditioned"), transform=image_transform)

images = torch.cat([dataset_unconditioned[0][0], torch.cat([dataset_pretrained[i][0] for i in range(5)])])

print(images.shape)


# Convert the tensor to a NumPy array
tensor_np = images.cpu().numpy()

num_slices = tensor_np.shape[0]
fig, axes = plt.subplots(num_slices, 1, figsize=(15, 3 * num_slices))

for i in range(num_slices):
    ax = axes[i]
    slice_to_visualize = tensor_np[i]  # Shape [32, 116]
    im = ax.imshow(slice_to_visualize, aspect='auto', cmap='viridis')
    ax.set_title(f'Slice {i}')
    ax.set_xlabel('Dimension 2')
    ax.set_ylabel('Dimension 1')

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.show()