import copy
from matplotlib import animation
from data.hdf5_dataset import HDF5Dataset
from torch.utils.data import DataLoader

import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from tqdm import tqdm
from PIL import Image

import os

from data.neural_field_datasets_shapenet import (
    FlattenTransform3D,
    ImageTransform3D,
    ShapeNetDataset,
)
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
):

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


def find_best_vq_hdf5(
    dataloader,
    kmean_iters=0,
    batch_size=1028,
    vec_dim=1,
    codebook_size=2**8 - 11,
    threshold_ema_dead_code=0,
    trials=1,
    training_iters=1000,
    track_process=False,
):
    dataloader = DataLoader(hdf5_dataset, batch_size=batch_size, shuffle=True)

    vq_lowest_loss = None
    lowest_loss = np.inf

    vq_config = {
        "dim": vec_dim,
        "codebook_size": codebook_size,  #   2**8 - 11,
        "decay": 0.8,
        "commitment_weight": 1.0,
        "kmeans_init": True if kmean_iters > 0 else False,
        "kmeans_iters": kmean_iters,
        "threshold_ema_dead_code": threshold_ema_dead_code,
    }

    if kmean_iters:
        print("Performing kMean for Vector Quantize")
        for _ in tqdm(range(trials)):
            for batch in dataloader:
                batch = batch.to(get_default_device())
                vq = None
                if get_default_device() == "cuda":
                    torch.cuda.empty_cache()

                vq = VectorQuantize(**vq_config).to(get_default_device())
                weights_quantized, indices, loss = vq(batch)

                if loss < lowest_loss:
                    lowest_loss = loss
                    vq_lowest_loss = vq

                break
    else:
        vq_lowest_loss = VectorQuantize(**vq_config).to(get_default_device())

    vq = vq_lowest_loss
    losses = []
    vq_parameters = []

    print("Training Vector Quantize")
    for _ in tqdm(range(training_iters)):
        if track_process:
            vq_parameters.append(copy.deepcopy(vq.state_dict()))
        for batch in tqdm(dataloader):
            batch = batch.to(get_default_device())
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
    hdf5=True,
):

    save_path = f"./models/vq_search_results/vq_model_hdf5_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth"

    if not (os.path.exists(save_path)):

        if hdf5:
            vq, vq_config, loss, vq_parameters = find_best_vq_hdf5(
                weights,
                kmean_iters=kmean_iters,
                codebook_size=vocab_size,
                batch_size=batch_size,
                training_iters=1,
                vec_dim=dim,
                threshold_ema_dead_code=threshold_ema_dead_code,
            )
        else:
            vq, vq_config, loss, vq_parameters = find_best_vq(
                weights,
                kmean_iters=kmean_iters,
                codebook_size=vocab_size,
                batch_size=batch_size,
                training_iters=1,
                vec_dim=dim,
                threshold_ema_dead_code=threshold_ema_dead_code,
            )
        if hdf5:
            save_vq_dict(
                f"./models/vq_search_results/vq_model_hdf5_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth",
                vq,
                vq_config,
                loss,
                vq_parameters,
            )
        else:
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
        loss = torch.load(save_path)["loss"]
        label = f"Vocab size: {vocab_size}, Dim: {dim}, Batch Size: {batch_size}, Treshhold: {threshold_ema_dead_code}, kmean iters: {kmean_iters},"

        plt.plot(loss, label=label)
        print(
            f"./models/vq_search_results/vq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth already exists"
        )


def main(
    weights,
    dims=[17],
    vocab_sizes=[1024],
    batch_sizes=[2**15],
    threshold_ema_dead_codes=[0],
    kmean_iters_list=[0],
):

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
                            hdf5=False,
                        )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # dataset = ShapeNetDataset("./datasets/plane_mlp_weights", transform=FlattenTransform3D())
    # weights = cat_weights(dataset, n=len(dataset))

    # Instantiate the dataset and dataloader
    # hdf5_dataset = HDF5Dataset("datasets/plane_mlp_weights.h5", "dataset")
    shapenet = ShapeNetDataset(
        "./datasets/plane_mlp_weights", transform=ImageTransform3D()
    )
    shapenet_all = torch.stack(
        [shapenet[i][0] for i in range(len(shapenet))], dim=0
    ).squeeze(1)
    dataset = shapenet_all[
        : (shapenet_all.flatten().shape[0] - shapenet_all.flatten().shape[0] % 3)
    ]

    main(
        dataset,
        dims=[3],
        vocab_sizes=[1048],
        batch_sizes=[2**16],
        threshold_ema_dead_codes=[0],
        kmean_iters_list=[1],
    )
