from animation.animation_util import reconstruct_image
from networks.mlp_models import MLP3D
from data.nef_mnist_dataset import (
    FlattenMinMaxTransform,
    MnistNeFDataset,
    FlattenTransform,
    MinMaxTransform,
)

import os
from PIL import Image

# Config for this script

dir_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_root_ours = os.path.join(dir_path, "adl4cv", "datasets", "mnist-nerfs")
dataset_kwargs = {"fixed_label": 5, "type": "pretrained"}

# Load Dataset
dataset = MnistNeFDataset(data_root_ours, **dataset_kwargs)


def store_ground_truth(idx: int):
    state_dict, label = dataset[idx]

    # Initialize model configuration
    nef_config = {
        "out_size": 1,
        "hidden_neurons": [16, 16],
        "use_leaky_relu": False,
        "output_type": "logits",
        "input_dims": 2,
        "multires": 4,
    }
    # regression_config = RegressionTransformerConfig(n_embd=32, block_size=592, n_head=8, n_layer=16)

    nef = MLP3D(**nef_config)
    nef.load_state_dict(state_dict)

    reconstructed_tensor = reconstruct_image(nef)

    # Save the image (from PIL)
    img = Image.fromarray((reconstructed_tensor * 255.0).astype("uint8"))
    img.save(f"./animation/ground_truth_{idx}.png")


def main():
    for i in range(4):
        store_ground_truth(i)


if __name__ == "__main__":
    main()
