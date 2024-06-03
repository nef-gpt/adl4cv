#!/usr/bin/env python

import torch
from PIL import Image

from animation.util import reconstruct_image
from networks.mlp_models import MLP3D

def visualize_nef(file_path: str):
    # Load the model from the file path
    loaded = torch.load(file_path)
    nef_config = loaded["model_config"]
    nef = MLP3D(**nef_config)
    nef.load_state_dict(loaded["state_dict"])

    # Reconstruct the image using the model
    reconstructed_tensor = reconstruct_image(nef)

    # Convert the reconstructed image to a PIL image
    img = Image.fromarray((reconstructed_tensor * 255.0).astype("uint8"))

    return img

# main function to load from cli arguments and display the image
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a NEF model")
    parser.add_argument("model_path", type=str, help="Path to the NEF model")

    args = parser.parse_args()
    img = visualize_nef(args.model_path)


if __name__ == "__main__":
    main()