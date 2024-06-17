from animation.util import backtransform_weights, reconstruct_image
from data.neural_field_datasets import MnistNeFDataset
from networks.mlp_models import MLP3D
from networks.nano_gpt import GPT
from vector_quantize_pytorch import VectorQuantize
import os
from PIL import Image
import torch


def generate_neural_field(
    model: GPT, vq: VectorQuantize, start_tokens: list[int]
) -> MLP3D:
    novel_tokens = model.generate(
        start_tokens,
        model.config.block_size - len(start_tokens),
        temperature=1.0,
        top_k=None,
    )[:, len(start_tokens) :]
    novel_tokens = novel_tokens.unsqueeze(-1).to("cpu")
    novel_weights = vq.get_codes_from_indices((novel_tokens))

    dataset_no_transform = MnistNeFDataset(os.path.join(os.path.dirname(os.path.abspath(os.getcwd()))root, "datasets", "mnist-nerfs"))
    original_dict = dataset_no_transform[0][0]

    reconstructed_dict = backtransform_weights(novel_weights, original_dict["state_dict"])

    mlp3d = MLP3D(**original_dict["model_config"])
    mlp3d.load_state_dict(reconstructed_dict)

    return mlp3d
