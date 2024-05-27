import sys
import os
from typing import List, Tuple, Union
import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import cv2
from networks.regression_transformer import RegressionTransformerConfig, RegressionTransformer
from data.neural_field_datasets import MnistNeFDataset, MinMaxTransform, FlattenTransform
from animation.util import ensure_folder_exists, backtransform_weights, reconstruct_image
from tqdm import tqdm
from PIL import Image



# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from networks.mlp_models import MLP3D

class FlattenMinMaxTransform(torch.nn.Module):
  def __init__(self, min_max: tuple = None):
    super().__init__()
    self.flatten = FlattenTransform()
    if min_max:
      self.minmax = MinMaxTransform(*min_max)
    else:
      self.minmax = MinMaxTransform()

  def forward(self, x, y):
    x, _ = self.flatten(x, y)
    x, _ = self.minmax(x, y)
    return x, y
  
dir_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_root_ours = os.path.join(dir_path, "adl4cv", "datasets", "mnist-nerfs")

dataset_wo_min_max = MnistNeFDataset(data_root_ours, type="pretrained", transform=FlattenTransform())
min_ours, max_ours = dataset_wo_min_max.min_max()
dataset_with_transform = MnistNeFDataset(data_root_ours, type="pretrained", transform=FlattenMinMaxTransform((min_ours, max_ours)))
dataset_no_transform = MnistNeFDataset(data_root_ours, type="pretrained")


def visualize_learning_process(image_idx: int, num_iters: int, foldername: str, video_name: str = "learning_process", fps=10, iter_step=iter_step):

    # Initialize model configuration
    nef_config = {
        "out_size": 1,
        "hidden_neurons": [16, 16],
        "use_leaky_relu": False,
        "output_type": "logits",
        "input_dims": 2,
        "multires": 4,
    }
    regression_config = RegressionTransformerConfig(n_embd=16, block_size=592, n_head=4, n_layer=8)

    transformer = RegressionTransformer(regression_config)
    nef = MLP3D(**nef_config)

    # Create a directory to store the frames
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    original_dict = dataset_no_transform[image_idx][0]
    sample = dataset_with_transform[image_idx][0]

    with tqdm(total=num_iters/iter_step) as pbar:

        for iter in range(0, num_iters, iter_step):
            
            model_path = f"./models/{foldername}/iter_{iter}.pt"
            assert os.path.exists(model_path), f"File {model_path} does not exist"

            transformer.load_state_dict(torch.load(model_path)["model"])

            X, Y = (sample[:regression_config.block_size].unsqueeze(-1).unsqueeze(0), sample[1: 1 + regression_config.block_size].unsqueeze(-1).unsqueeze(0))

            pred, _ = transformer(X, Y)

            minmax_transformer = MinMaxTransform(min_value=min_ours, max_value=max_ours)
            pred_flattened = minmax_transformer.reverse(torch.cat((X[0][0].unsqueeze(0), pred[0])).unsqueeze(0))

            # Backtransform weights
            reconstructed_dict = backtransform_weights(pred_flattened, original_dict)

            nef.load_state_dict(reconstructed_dict)
            reconstructed_tensor = reconstruct_image(nef)



            # Save the frame as a PNG image
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))

            axes.imshow(reconstructed_tensor, cmap="gray", aspect="equal")
            axes.axis('off')

            plt.tight_layout()

            frame_path = os.path.join(frames_dir, f"frame_iter_{iter:03d}.png")
            plt.savefig(frame_path, format="png")
            plt.close(fig)
            pbar.update(1)
            pbar.set_description(
                    "Finished creating frame for epoch %d"
                    % (iter)
                )

    # Compile the images into a video
    frame_paths = [
        os.path.join(frames_dir, f"frame_iter_{iter:03d}.png")
        for iter in range(0, num_iters, iter_step)
        if os.path.exists(os.path.join(frames_dir, f"frame_iter_{iter:03d}.png"))
    ]
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape
    
    video_path = "./animation/" + foldername + "/" + video_name + ".mp4"
    gif_path = "./animation/" + foldername + "/" + video_name + ".gif"
    ensure_folder_exists("./animation/" + foldername)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {video_path}")

    images = [Image.open(frame_path).convert("RGBA") for frame_path in frame_paths]

    # Save as GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1/fps,  # duration in milliseconds
        loop=0,
        transparency=0,  # specify the transparency color (usually 0 for black)
        disposal=2  # ensures that each frame is replaced, not drawn on top of the previous one
    )
    print(f"GIF saved to {gif_path}")


def main():
    image_idx = 0  # Change this to visualize a different image from the dataset
    num_iters = 10000  # Number of epochs or models saved

    foldername = "training_sample_0"

    for image_idx in range(1):
        visualize_learning_process(image_idx, num_iters, foldername, video_name=f"image_{image_idx}")


if __name__ == "__main__":
    main()
