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
from animation.util import make_coordinates, ensure_folder_exists, reconstruct_image, get_vmin_vmax, get_model_difference

model_config = {
        "out_size": 1,
        "hidden_neurons": [16, 16],
        "use_leaky_relu": False,
        "output_type": "logits",
        "input_dims": 2,
        "multires": 4,
    }

def save_colorbar(save_path: str, vmin: torch.Tensor, vmax: torch.Tensor):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
        cax=ax, orientation='horizontal'
    )
    
    cb1.set_label('Colorbar')
    plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_figure(model: torch.nn.Module, vmin: torch.Tensor, vmax: torch.Tensor, comparison_model: torch.nn.Module = None):

    # Generate random data for demonstration
    image = reconstruct_image(model)
    if comparison_model:
        state_dict = comparison_model.state_dict()
        W1 = state_dict['layers.0.weight']
        W2 = state_dict['layers.1.weight']
        W3 = state_dict['layers.2.weight']
        b1 = state_dict['layers.0.bias'].unsqueeze(-1)
        b2 = state_dict['layers.1.bias'].unsqueeze(-1)
        b3 = state_dict['layers.2.bias'].unsqueeze(-1)
    else:
        state_dict = model.state_dict()
        W1 = state_dict['layers.0.weight']
        W2 = state_dict['layers.1.weight']
        W3 = state_dict['layers.2.weight']
        b1 = state_dict['layers.0.bias'].unsqueeze(-1)
        b2 = state_dict['layers.1.bias'].unsqueeze(-1)
        b3 = state_dict['layers.2.bias'].unsqueeze(-1)

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
    ax_W1.imshow(W1, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
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

    return fig



def visualize_learning_process(image_idx: int, num_epochs: int, model_config: dict, foldername: str, 
                               video_name: str = "learning_process", v: tuple = None, fps: int = 10,
                               comparison_model: torch.nn.Module = None):

    frames_dir = "frames"
    frames = []

    if v:
        vmin, vmax = v
    else:
        vmin, vmax = get_vmin_vmax(image_idx, num_epochs, foldername)

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            model = MLP3D(**model_config)
            
            model_path = "{}/image-{}".format(foldername, image_idx) + f"_model_epoch_{epoch}.pth"
            if os.path.exists(model_path) is False:
                print(f"File {model_path} does not exist")
                return

            model.load_state_dict(torch.load(model_path))

            if comparison_model:
                model_difference = MLP3D(**model_config)
                get_model_difference(model, comparison_model, model_difference)

                fig = get_figure(model, vmin, vmax, model_difference)
            else:
                fig = get_figure(model, vmin, vmax)

            frame_path = os.path.join(frames_dir, f"frame_{epoch:03d}.png")
            plt.savefig(frame_path, format="png")

            plt.close(fig)

            pbar.update(1)
            pbar.set_description(
                    "Finished creating frame for epoch %d"
                    % (epoch)
                )

    # Compile the images into a video
    frame_paths = [
        os.path.join(frames_dir, f"frame_{epoch:03d}.png")
        for epoch in range(num_epochs)
        if os.path.exists(os.path.join(frames_dir, f"frame_{epoch:03d}.png"))
    ]
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape

    
    video_path = video_name + ".mp4"
    gif_path = video_name + ".gif"
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        frames.append(frame)
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


def create_one_video():

    image_idx = 1721  # Change this to visualize a different image from the dataset
    num_epochs = 250  # Number of epochs or models saved
    subfoldername = "unconditioned" #"pretrained"
    foldername = (
        f"./datasets/mnist-nerfs/{subfoldername}"
    )
    visualize_learning_process(image_idx, num_epochs, model_config, foldername, f"./animation/{subfoldername}/presentation_{subfoldername}_{image_idx}")

def create_all_videos():
    for filename in glob.glob("./datasets/mnist-nerfs/unconditioned/*_final.pth"):

        image_idx = filename.split("_")[-3].split("-")[-1]  # Change this to visualize a different image from the dataset
        num_epochs = 250  # Number of epochs or models saved
        subfoldername = "unconditioned" #"pretrained"
        foldername = (
            f"./datasets/mnist-nerfs/{subfoldername}"
        )
        visualize_learning_process(image_idx, num_epochs, model_config, foldername, f"./animation/{subfoldername}/presentation_{subfoldername}_{image_idx}")

    for filename in glob.glob("./datasets/mnist-nerfs/pretrained/*_final.pth"):

        image_idx = filename.split("_")[-3].split("-")[-1]  # Change this to visualize a different image from the dataset
        num_epochs = 250  # Number of epochs or models saved
        subfoldername = "pretrained" #"pretrained"
        foldername = (
            f"./datasets/mnist-nerfs/{subfoldername}"
        )
        visualize_learning_process(image_idx, num_epochs, model_config, foldername, f"./animation/{subfoldername}/presentation_{subfoldername}_{image_idx}")

def compare_different_runs(image_idxs: List[int], num_epoch: int, model_config: dict, subfoldernames: List[str], comparison_model: torch.nn.Module = None):
    vmins = []
    vmaxs = []

    # find max and min of v
    for idx in image_idxs:
        for subfoldername in subfoldernames:
            if comparison_model:
                vmin, vmax = get_vmin_vmax(idx, num_epoch,  f"./datasets/mnist-nerfs/{subfoldername}", comparison_model.state_dict())
            else:
                vmin, vmax = get_vmin_vmax(idx, num_epoch,  f"./datasets/mnist-nerfs/{subfoldername}")
            vmins.append(vmin)
            vmaxs.append(vmax)
    vmax = torch.Tensor([v.max() for v in vmaxs]).max()
    vmin = torch.Tensor([v.min() for v in vmins]).min()

    #vmax = 1
    #vmin = -1

    if comparison_model:
        save_folder = f"./animation/comparison_with_comparison_model_{'_'.join(map(str, image_idxs))}"
    else:
        save_folder = f"./animation/comparison_{'_'.join(map(str, image_idxs))}"
    
    for idx in image_idxs:
        for subfoldername in subfoldernames:
            foldername =  f"./datasets/mnist-nerfs/{subfoldername}"
            video_name = save_folder + f"/{subfoldername}_{idx}"

            ensure_folder_exists(save_folder)
            visualize_learning_process(idx, num_epoch, model_config, foldername,  video_name=video_name, v= (vmin, vmax), comparison_model=comparison_model)
    
    save_colorbar(save_folder + "/colorbar.png", vmin, vmax)

if __name__ == "__main__":
    comparison_model = MLP3D(**model_config)
            
    model_path = "./datasets/mnist-nerfs/unconditioned/image-0_model_final.pth"
    assert os.path.exists(model_path), f"File {model_path} does not exist"

    comparison_model.load_state_dict(torch.load(model_path))
    compare_different_runs([11, 35, 47, 65], 200, model_config, ["pretrained"], comparison_model)
    

