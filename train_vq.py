import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from data.neural_field_datasets import (
    MnistNeFDataset,
    FlattenTransform,
    ModelTransform,
    QuantizeTransform,
)
from animation.util import reconstruct_image
import matplotlib.pyplot as plt
from tqdm import tqdm


import os

from utils import get_default_device


def cat_weights(dataset, n=None):
    if n is None:
        n = len(dataset)

    weights_cat = torch.Tensor()
    print("Concatentationg the weights")
    for j in tqdm(range(0, n)):
        weights_1d_conditioned = dataset[j][0].numpy()
        weights_cat = torch.cat((weights_cat, torch.tensor(weights_1d_conditioned)), 0)

    return weights_cat


def find_best_vq(
    input, kmeans_init, kmean_iters, codebook_size=2**8 - 11, trials=25, training_iters=1000, track_performance = False
):
    vq_lowest_loss = None
    lowest_loss = np.inf

    vq_parameters = []

    vq_config = {
        "dim": 1,
        "codebook_size": codebook_size,  #   2**8 - 11,
        "decay": 0.8,
        "commitment_weight": 1.0,
        "kmeans_init": kmeans_init,
        "kmeans_iters": kmean_iters,
    }

    input = input.to(get_default_device())

    
    if kmeans_init:
        print("Performing kMean for Vector Quantize")
        for _ in tqdm(range(trials)):
            vq = None
            if get_default_device() == "cuda":
                torch.cuda.empty_cache()

            vq = VectorQuantize(**vq_config).to(get_default_device())
            weights_quantized, indices, loss = vq(input)
            if loss < lowest_loss:
                lowest_loss = loss
                vq_lowest_loss = vq
    else:
        vq_lowest_loss = VectorQuantize(**vq_config).to(get_default_device())

    vq = vq_lowest_loss
    losses = []

    print("Training Vector Quantize")
    for _ in tqdm(range(training_iters)):
        weights_quantized, indices, loss = vq(input)
        vq_parameters.append(vq.state_dict())
        losses.append(loss.item())
    return vq, vq_config, losses, vq_parameters

def save_vq_dict(path: str, vq: VectorQuantize, vq_config: dict):
    torch.save(
        {
            "state_dict": vq.state_dict(),
            "vq_config": vq_config,
        },
        path,
    )


def main():
    dir_path = os.path.dirname(os.path.abspath(os.getcwd()))
    dataset_path = os.path.join(dir_path, "adl4cv", "datasets", "mnist-nerfs")
    dataset = MnistNeFDataset(
        dataset_path, type="pretrained", fixed_label=5, transform=FlattenTransform()
    )

    weights = cat_weights(dataset, n=len(dataset))
    vq, vq_config, losses, vq_parameters = find_best_vq(dataset[0][0].unsqueeze(-1), False, 0, training_iters=2000)

    #vq, vq_config, losses = find_best_vq(weights.unsqueeze(-1), 10, training_iters=2000)
    #save_vq_dict(os.path.join(dir_path, "adl4cv", "models", "vqs", "vq_mnist_with_all_5_conditioned_n_501.pt"), vq, vq_config)

    plt.plot(losses)
    plt.show()

    """
    quantized_dataset_conditioned = MnistNeFDataset(
        dataset_path,
        type="pretrained",
        fixed_label=3,
        transform=QuantizeTransform(vq.to(get_default_device)),
    )
    no_quantized_dataset_conditioned = MnistNeFDataset(
        dataset_path, type="pretrained", fixed_label=3, transform=ModelTransform()
    )

    idx = -3
    model_quantized_conditioned = quantized_dataset_conditioned[idx][0]
    model_unquantized_conditioned = no_quantized_dataset_conditioned[idx][0]

    # Plotting the tensors as heatmaps in grayscale
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    image_quantized_conditioned = reconstruct_image(model_quantized_conditioned)
    image_unquantized_conditioned = reconstruct_image(model_unquantized_conditioned)

    # image_quantized_conditioned[18][7] = 1

    axes[0].imshow(image_quantized_conditioned, cmap="gray", aspect="auto")
    axes[1].imshow(image_unquantized_conditioned, cmap="gray", aspect="auto")

    axes[0].set_title("Image using quantized weights")
    axes[1].set_title("Image using unquantized weights")
    """


if __name__ == "__main__":
    main()

