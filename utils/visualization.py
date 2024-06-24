from animation.util import backtransform_weights, reconstruct_image
from data.neural_field_datasets import MnistNeFDataset
from networks.mlp_models import MLP3D
from networks.nano_gpt import GPT
from vector_quantize_pytorch import VectorQuantize
import os
from PIL import Image
import torch


def generate_neural_field(
    model: GPT,
    vq: VectorQuantize,
    sos: int,
    condition: list[int],
    device,
    top_k=None,
    temperature=1.0,
) -> MLP3D:
    seed = torch.zeros((len(condition), 2)).long()
    seed[:, 0] = sos
    seed[:, 1] = torch.Tensor(condition).long()

    novel_tokens = model.generate(
        seed.to(device), model.config.block_size, temperature=temperature, top_k=top_k
    )[:, 2:]
    novel_tokens = novel_tokens.to("cpu")
    novel_weights = vq.get_codes_from_indices(
        (novel_tokens.clamp(0, vq.codebook_size - 1))
    )

    dataset_no_transform = MnistNeFDataset(
        os.path.join(
            os.path.dirname(os.path.abspath(os.getcwd())),
            "adl4cv",
            "datasets",
            "mnist-nerfs",
        )
    )
    original_dict = dataset_no_transform[0][0]

    def make_model(i):
        reconstructed_dict = backtransform_weights(
            novel_weights[i].unsqueeze(0), original_dict["state_dict"]
        )

        mlp3d = MLP3D(**original_dict["model_config"])
        mlp3d.load_state_dict(reconstructed_dict)
        return mlp3d

    return [make_model(i) for i in range(len(condition))]
