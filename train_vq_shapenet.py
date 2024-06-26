import copy
from matplotlib import animation
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from torchvision import datasets, transforms
from data.neural_field_datasets import (
    MnistNeFDataset,
    FlattenTransform,
    ModelTransform,
    QuantizeTransform,
)
from animation.util import reconstruct_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import os

from data.neural_field_datasets_shapenet import FlattenTransform3D, ShapeNetDataset
from utils import get_default_device

dir_path = os.path.dirname(os.path.abspath(os.getcwd()))
dataset_path = os.path.join(dir_path, "adl4cv", "datasets", "mnist-nerfs")
# plt.style.use("dark_background")


def cat_weights(dataset, n=None):
    if n is None:
        n = len(dataset)

    print("Concatentationg the weights")
    weights_cat = torch.cat([dataset[j][0] for j in tqdm(range(0, n))])

    return weights_cat


def find_best_vq(
    input,
    kmean_iters=0,
    batch_size=1028,
    vec_dim=1,
    codebook_size=2**8 - 11,
    threshold_ema_dead_code=0,
    trials=1,
    training_iters=1000,
    track_process=False,
    model=None
):
    if model:
        

    vq_lowest_loss = None
    lowest_loss = np.inf

    assert input.shape[0] % vec_dim == 0, "Shape does not match."

    input = torch.reshape(
        input[: input.shape[0] - (input.shape[0] % (batch_size * vec_dim))],
        (-1, batch_size, vec_dim),
    )

    vq_config = {
        "dim": vec_dim,
        "codebook_size": codebook_size,  #   2**8 - 11,
        "decay": 0.8,
        "commitment_weight": 1.0,
        "kmeans_init": True if kmean_iters > 0 else False,
        "kmeans_iters": kmean_iters,
        "threshold_ema_dead_code": threshold_ema_dead_code,
    }

    input = input.to(get_default_device())

    if kmean_iters:
        print("Performing kMean for Vector Quantize")
        for _ in tqdm(range(trials)):
            vq = None
            if get_default_device() == "cuda":
                torch.cuda.empty_cache()

            vq = VectorQuantize(**vq_config).to(get_default_device())

            batch = input[0]
            weights_quantized, indices, loss = vq(batch)

            if loss < lowest_loss:
                lowest_loss = loss
                vq_lowest_loss = vq
    else:
        vq_lowest_loss = VectorQuantize(**vq_config).to(get_default_device())

    vq = vq_lowest_loss
    losses = []
    vq_parameters = []

    print("Training Vector Quantize")
    for _ in tqdm(range(training_iters)):
        if track_process:
            vq_parameters.append(copy.deepcopy(vq.state_dict()))
        for i in tqdm(range(input.shape[0])):
            batch = input[i]
            weights_quantized, indices, loss = vq(batch)
            losses.append(loss.item())

    if track_process:
        vq_parameters.append(vq.state_dict())
    return vq, vq_config, losses, vq_parameters


def save_vq_dict(
    path: str, vq: VectorQuantize, vq_config: dict, loss: list, vq_parameter: dict
):
    torch.save(
        {
            "state_dict": vq.state_dict(),
            "vq_config": vq_config,
            "loss": loss,
            "vq_parameter": vq_parameter,
        },
        path,
    )

    # Define the transform to convert the images to tensors


def train_on_shape_net(
    weights,
    vocab_size,
    batch_size=32768,
    dim=17,
    kmean_iters=1,
    threshold_ema_dead_code=0,
):

    if not os.path.exists(
        f"./models/vq_search_results/vq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth"
    ):

        vq, vq_config, loss, vq_parameters = find_best_vq(
            weights,
            kmean_iters=kmean_iters,
            codebook_size=vocab_size,
            batch_size=batch_size,
            training_iters=1,
            vec_dim=dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

        save_vq_dict(
            f"./models/vq_search_results/vq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth",
            vq,
            vq_config,
            loss,
            vq_parameters,
        )

        label = f"Vocab size: {vocab_size}, Dim: {dim}, Batch Size: {batch_size}, Treshhold: {threshold_ema_dead_code}, kmean iters: {kmean_iters},"

        plt.plot(loss, label=label)
    else:
        print(
            f"./models/vq_search_results/vq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth already exists"
        )


def main():
    vocab_sizes = [256, 512, 1024, 2048]

    dims = [17]  # [1, 17]
    batch_sizes = [2**15, 2**16, 2**17]
    threshold_ema_dead_codes = [0, 1, 2, 4, 8]
    kmean_iters_list = [0, 1, 2, 4, 8]

    dataset = ShapeNetDataset(
        "./datasets/plane_mlp_weights", transform=FlattenTransform3D()
    )
    weights = cat_weights(dataset, n=len(dataset))

    for vocab_size in vocab_sizes:
        for dim in dims:
            for batch_size in batch_sizes:
                for threshold_ema_dead_code in threshold_ema_dead_codes:
                    for kmean_iters in kmean_iters_list:
                        print(
                            f"Current vocab size: {vocab_size}, dim: {dim}, batch size: {batch_size}, threshold: {threshold_ema_dead_code}, kmean iters: {kmean_iters}"
                        )
                        train_on_shape_net(
                            weights,
                            vocab_size,
                            batch_size=batch_size,
                            dim=dim,
                            kmean_iters=kmean_iters,
                            threshold_ema_dead_code=threshold_ema_dead_code,
                        )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
