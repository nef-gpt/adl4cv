import copy
from matplotlib import animation
from data.hdf5_dataset import HDF5Dataset
from torch.utils.data import DataLoader

import torch
import numpy as np
from vector_quantize_pytorch import ResidualVQ, kmeans
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

from data.neural_field_datasets_shapenet import AllWeights3D, FlattenTransform3D, ShapeNetDataset
from utils import get_default_device

def uniform_init(*shape):
    t = torch.empty(shape)
    torch.nn.init.kaiming_uniform_(t)
    return t



def find_best_rq_dataset(
    dataset,
    kmean_iters=0,
    batch_size=1028,
    vec_dim=1,
    codebook_size=2**8 - 11,
    threshold_ema_dead_code=0,
    trials=1,
    training_iters=1000,
    track_process=False,
    num_quantizers=1,
    use_init=True
):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
        
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
        "ema_update": False,
        "learnable_codebook": True,
        "in_place_codebook_optimizer": torch.optim.Adam,
        "learnable_codebook": True,
        "use_cosine_sim": True
    }


    if kmean_iters:
        print("Performing kMean for Vector Quantize")
        for _ in tqdm(range(trials)):
            for batch in dataloader:
                batch = batch[0].view(-1, vec_dim)
                rq = None
                if get_default_device() == "cuda":
                    torch.cuda.empty_cache()

                rq = ResidualVQ(**rq_config).to(get_default_device())
                
                if use_init:
                    pass

                    
                
                weights_quantized, indices, loss = rq(batch)

                if loss[0][-1] < lowest_loss:
                    lowest_loss = loss[0][-1]
                    rq_lowest_loss = rq
                
                break
            
    else:
        rq_lowest_loss = ResidualVQ(**rq_config).to(get_default_device())
        if use_init:
            rq_lowest_loss.layers[0]._codebook.initted = torch.Tensor([1.0])
            embed = uniform_init(1, codebook_size, vec_dim)
            embed[0, :dataset[0][0].shape[0], :] = dataset[0][0]
            rq_lowest_loss.layers[0]._codebook.embed.data.copy_(embed)
            rq_lowest_loss.layers[0]._codebook.embed_avg.data.copy_(embed.clone())
            rq_lowest_loss.layers[0]._codebook.cluster_size.data.copy_(torch.zeros(1, codebook_size))
            rq_lowest_loss.layers[0]._codebook.initted.data.copy_(torch.Tensor([True]))
    
    for layer in rq_lowest_loss.layers:
        for g in layer.in_place_codebook_optimizer.param_groups:
            g['lr'] = 5e-1

    rq = rq_lowest_loss
    losses = []
    rq_parameters = []

    print("Training Vector Quantize")
    for _ in tqdm(range(training_iters)):
        if track_process:
            rq_parameters.append(copy.deepcopy(rq.state_dict()))
        for batch in tqdm(dataloader):
            batch = batch[0].view(-1, vec_dim)
            weights_quantized, indices, loss = rq(batch)
            print(loss)
            losses.append(loss[0][-1].item())
        for layer in rq_lowest_loss.layers:
            for g in layer.in_place_codebook_optimizer.param_groups:
                g['lr'] = g['lr'] * 8e-1

    if track_process:
        rq_parameters.append(rq.state_dict())
    return rq, rq_config, losses, rq_parameters


def save_rq_dict(
    path: str, rq: ResidualVQ, rq_config: dict, loss: list, rq_parameter: dict
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
    num_quantizers=1,
    use_init=True,
    training_iters=1,
    force=False
):   
    path = f"./models/rq_search_results/learnable_rq_model_dim_{dim}_vocab_{vocab_size}_batch_size_{batch_size}_threshold_ema_dead_code_{threshold_ema_dead_code}_kmean_iters_{kmean_iters}_num_quantizers_{num_quantizers}_use_init_{use_init}.pth"
    label = f"Vocab size: {vocab_size}, Dim: {dim}, Batch Size: {batch_size}, Treshhold: {threshold_ema_dead_code}, kmean iters: {kmean_iters}, num quantizers: {num_quantizers}, use_init: {use_init}"
   
    if not os.path.exists(path) or force:

        rq, rq_config, loss, rq_parameters = find_best_rq_dataset(
            weights,
            kmean_iters=kmean_iters,
            codebook_size=vocab_size,
            batch_size=batch_size,
            training_iters=training_iters,
            vec_dim=dim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            num_quantizers=num_quantizers,
            use_init=use_init,
        )

        save_rq_dict(
            path,            
            rq,
            rq_config,
            loss,
            rq_parameters,
        )
        
    else:
        print(f"{path} already exists")
        loss = torch.load(path)["loss"]

    plt.plot(loss, label=label)


def main(weights, dims =[17], vocab_sizes=[1024], batch_sizes = [2**15], threshold_ema_dead_codes = [0], kmean_iters_list = [0], num_quantizers_list=[1], use_inits=[True], force=False, training_iters=1):

    for vocab_size in vocab_sizes:
        for dim in dims:
            for batch_size in batch_sizes:
                for threshold_ema_dead_code in threshold_ema_dead_codes:
                    for kmean_iters in kmean_iters_list:
                        for num_quantizers in num_quantizers_list:
                            for use_init in use_inits:
                                print(
                                    f"Current vocab size: {vocab_size}, dim: {dim}, batch size: {batch_size}, threshold: {threshold_ema_dead_code}, kmean iters: {kmean_iters}, num quantizers: {num_quantizers}, use init: {use_init}"
                                )
                                train_on_shape_net(
                                    weights,
                                    vocab_size,
                                    batch_size=batch_size,
                                    dim=dim,
                                    kmean_iters=kmean_iters,
                                    threshold_ema_dead_code=threshold_ema_dead_code,
                                    num_quantizers=num_quantizers,
                                    use_init=use_init,
                                    training_iters=training_iters,
                                    force=force
                                    
                                )
    plt.legend()

    plt.show()
   

if __name__ == "__main__":
    # dataset = ShapeNetDataset("./datasets/plane_mlp_weights", transform=FlattenTransform3D())
    # weights = cat_weights(dataset, n=len(dataset))
    
    # Instantiate the dataset and dataloader
    dataset = ShapeNetDataset("./datasets/plane_mlp_weights", transform=AllWeights3D())
    #main(dataset, dims=[128], vocab_sizes=[287], batch_sizes=[512], threshold_ema_dead_codes=[0, 2, 4, 8, 16], kmean_iters_list=[0], num_quantizers_list=[4], use_inits=[True], training_iters=10, force=True)
    #main(dataset, dims=[128], vocab_sizes=[287], batch_sizes=[512], threshold_ema_dead_codes=[0], kmean_iters_list=[1], num_quantizers_list=[4], use_inits=[False], training_iters=10)
    #main(dataset, dims=[128], vocab_sizes=[512, 1048], batch_sizes=[512, 1024, 2048], threshold_ema_dead_codes=[0], kmean_iters_list=[0], num_quantizers_list=[1, 2, 3, 4], use_inits=[False], training_iters=3)
    main(dataset, dims=[128], vocab_sizes=[287], batch_sizes=[1024], threshold_ema_dead_codes=[0], kmean_iters_list=[0], num_quantizers_list=[2], use_inits=[True], training_iters=20, force=False)


