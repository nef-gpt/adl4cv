
import os
import torch
from networks.nano_gpt import GPT
from training.mnist_classifier_score import compute_mnist_score
from utils import get_default_device
from vector_quantize_pytorch import VectorQuantize
from tqdm import tqdm
import numpy as np




def scores_from_path(
        path: str, 
        num_iters: int = 1,
        batch_size: int = 128,
        top_ks: int = [None],
        temperatures: float = [1.0],
        ):
    with torch.no_grad():
        model_dict = torch.load(path)

        model = GPT(model_dict["model_args"])
        model.to(device=get_default_device())
        model.load_state_dict(model_dict["model"])
        model.eval()


        vq = VectorQuantize(**model_dict["vq_config"])
        vq.load_state_dict(model_dict["vq_state_dict"])
        vq.to(device=get_default_device())
        vq.eval()

        results = {}

        for top_k in top_ks:
            results[str(top_k)] = {}

            for temperature in temperatures:

                acc, loss = compute_mnist_score(model, vq, get_default_device(), model_dict["token_dict"], num_iters=num_iters, batch_size=batch_size, top_k=top_k, temperature=temperature)

                results[str(top_k)][str(temperature)] = {
                    "acc": acc,
                    "loss": loss,
                    }
                print(results)
    
        return results


def main():
    path = "./models/token_transformer/N_ALL_5M_LARGE_GOOD.pth"
    results = scores_from_path(path, top_ks=range(3, 10), temperatures=np.arange(0.5, 1.5, 0.1))
    
    torch.save(results, "performance_models.pth")

if __name__ == "__main__":
    main()




            