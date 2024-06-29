import copy
from matplotlib import animation
from data.hdf5_dataset import HDF5Dataset
from torch.utils.data import DataLoader

import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
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


def find_best_rq(
    input,
    kmean_iters=0,
    batch_size=1028,
    vec_dim=1,
    codebook_size=2**8 - 11,
    threshold_ema_dead_code=0,
    trials=1,
    training_iters=1000,
    track_process=False,
    num_quantizers=1
):
        
    rq_lowest_loss = None
    lowest_loss = np.inf

    assert input.shape[0] % vec_dim == 0, "Shape does not match."

    input = torch.reshape(
        input[: input.shape[0] - (input.shape[0] % (batch_size * vec_dim))],
        (-1, batch_size, vec_dim),
    )

    rq_config = {
        "dim": vec_dim,
        "codebook_size": codebook_size,  #   2**8 - 11,
        "decay": 0.8,
        "commitment_weight": 1.0,
        "kmeans_init": True if kmean_iters > 0 else False,
        "kmeans_iters": kmean_iters,
        "threshold_ema_dead_code": threshold_ema_dead_code,
        "num_quantizers": num_quantizers,
    }

    input = input.to(get_default_device())

    if kmean_iters:
        print("Performing kMean for Vector Quantize")
        for _ in tqdm(range(trials)):
            rq = None
            if get_default_device() == "cuda":
                torch.cuda.empty_cache()

            rq = VectorQuantize(**rq_config).to(get_default_device())

            batch = input[0]
            weights_quantized, indices, loss = rq(batch)

            if loss < lowest_loss:
                lowest_loss = loss
                rq_lowest_loss = rq
    else:
        rq_lowest_loss = VectorQuantize(**rq_config).to(get_default_device())

    rq = rq_lowest_loss
    losses = []
    rq_parameters = []

    print("Training Vector Quantize")
    for _ in tqdm(range(training_iters)):
        if track_process:
            rq_parameters.append(copy.deepcopy(rq.state_dict()))
        for i in tqdm(range(input.shape[0])):
            batch = input[i]
            weights_quantized, indices, loss = rq(batch)
            losses.append(loss.item())

    if track_process:
        rq_parameters.append(rq.state_dict())
    return rq, rq_config, losses, rq_parameters


def find_best_rq_hdf5(
    dataloader,
    kmean_iters=0,
    batch_size=1028,
    vec_dim=1,
    codebook_size=2**8 - 11,
    threshold_ema_dead_code=0,
    trials=1,
    training_iters=1000,
    track_process=False,
    num_quantizers=1,
):
    dataloader = DataLoader(hdf5_dataset, batch_size=batch_size, shuffle=True)

        
    rq_lowest_loss = None
    lowest_loss = np.inf

    rq_config = {
        "dim": vec_dim,
        "codebook_size": codebook_size,  #   2**8 - 11,
        "decay": 0.8,
        "commitment_weight": 1.0,
        "kmeans_init": True if kmean_iters > 0 else False,
        "kmeans_iters": kmean_iters,
        "threshold_ema_dead_code": threshold_ema_dead_code,
        "num_quantizers": num_quantizers,
    }


    if kmean_iters:
        print("Performing kMean for Vector Quantize")
        for _ in tqdm(range(trials)):
            for batch in dataloader:
                rq = None
                if get_default_device() == "cuda":
                    torch.cuda.empty_cache()

                rq = VectorQuantize(**rq_config).to(get_default_device())
                weights_quantized, indices, loss = rq(batch)

                if loss < lowest_loss:
                    lowest_loss = loss
                    rq_lowest_loss = rq
                
                break
    else:
        rq_lowest_loss = VectorQuantize(**rq_config).to(get_default_device())

    rq = rq_lowest_loss
    losses = []
    rq_parameters = []

    print("Training Vector Quantize")
    for _ in tqdm(range(training_iters)):
        if track_process:
            rq_parameters.append(copy.deepcopy(rq.state_dict()))
        for batch in tqdm(dataloader):
            weights_quantized, indices, loss = rq(batch)
            losses.append(loss.item())

    if track_process:
        rq_parameters.append(rq.state_dict())
    return rq, rq_config, losses, rq_parameters


def save_rq_dict(
    path: str, rq: VectorQuantize, rq_config: dict, loss: list, rq_parameter: dict
):
    torch.save(
        {
            "state_dict": rq.state_dict(),
            "rq_config": rq_config,
            "loss": loss,
            "rq_parameter": rq_parameter,
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
    hdf5 = False,
    num_quantizers=1,
):

    if not os.path.exists(
        f"./models/rq_search_results/rq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}.pth"
    ):

        if hdf5:
            rq, rq_config, loss, rq_parameters = find_best_rq(
                weights,
                kmean_iters=kmean_iters,
                codebook_size=vocab_size,
                batch_size=batch_size,
                training_iters=1,
                vec_dim=dim,
                threshold_ema_dead_code=threshold_ema_dead_code,
                num_quantizers=num_quantizers
            )
        else:
            rq, rq_config, loss, rq_parameters = find_best_rq_hdf5(
                weights,
                kmean_iters=kmean_iters,
                codebook_size=vocab_size,
                batch_size=batch_size,
                training_iters=1,
                vec_dim=dim,
                threshold_ema_dead_code=threshold_ema_dead_code,
                num_quantizers=num_quantizers
            )

        save_rq_dict(
            f"./models/rq_search_results/rq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}_num_quantizers_{num_quantizers}.pth",
            rq,
            rq_config,
            loss,
            rq_parameters,
        )

        label = f"Vocab size: {vocab_size}, Dim: {dim}, Batch Size: {batch_size}, Treshhold: {threshold_ema_dead_code}, kmean iters: {kmean_iters}, num quantizers: {num_quantizers}"

        plt.plot(loss, label=label)
    else:
        print(
            f"./models/rq_search_results/rq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}_num_quantizers_{num_quantizers}.pth already exists"
        )


def main(weights, dims =[17], vocab_sizes=[1024], batch_sizes = [2**15], threshold_ema_dead_codes = [0], kmean_iters_list = [0]):

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
    # dataset = ShapeNetDataset("./datasets/plane_mlp_weights", transform=FlattenTransform3D())
    # weights = cat_weights(dataset, n=len(dataset))
    
    # Instantiate the dataset and dataloader
    hdf5_dataset = HDF5Dataset('datasets/plane_mlp_weights.h5', 'dataset')
    main(hdf5_dataset, dims=[128], vocab_sizes=[1024], batch_sizes=[2**19], threshold_ema_dead_codes=[1], kmean_iters_list=[1])
