from random import randrange
from animation.animation_util import reconstruct_image
from networks.nano_gpt import GPT
import torch
import torchvision.transforms as transforms
from utils import decorator_timer
from utils.visualization import generate_neural_field
from vector_quantize_pytorch import VectorQuantize
from torchvision.transforms import Normalize
from PIL import Image

normalizer = Normalize(mean=0.45, std=0.22)

import matplotlib.pyplot as plt

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("fxmarty/resnet-tiny-mnist")
mnist_classifier = AutoModelForImageClassification.from_pretrained(
    "fxmarty/resnet-tiny-mnist"
)
mnist_classifier.eval()


@torch.no_grad()
# @decorator_timer
def compute_mnist_score(
    model: GPT,
    vq: VectorQuantize,
    device,
    special_tokens: dict[str, int],
    num_iters: int = 1,
    batch_size: int = 64,
    top_k: int = None,
    temperature: float = 1.0,
):
    """
    Compute the average cross entropy loss when using the model to generate neural fields
    and then get images from them and classify them with a pretrained MNIST classifier.

    special tokens "0" to "9" have to exist in the special_tokens dict and SOS
    """

    # asserts
    assert all(
        [str(i) in special_tokens.keys() for i in range(10)]
    ), "Special tokens must have keys 0 to 9"
    assert "SOS" in special_tokens.keys(), "SOS token must exist in special tokens"

    # main loop
    total_loss = 0
    acc = 0
    batch_size = 64
    for i in range(num_iters):
        # generate neural field

        # randomly select a special token

        label = str(i % 10)
        model.eval()
        mlp3ds = generate_neural_field(
            model,
            vq,
            special_tokens["SOS"],
            [special_tokens[str(i % 10)] for i in range(batch_size)],
            device,
            top_k=top_k,
            temperature=temperature,
        )

        # get image
        for i, mlp3d in enumerate(mlp3ds):
            image = reconstruct_image(mlp3d)
            image = Image.fromarray(image)

            image = transforms.ToTensor()(image).unsqueeze(0)
            image = normalizer(image)

            # classify image
            label = mnist_classifier(image).logits
            predicted = torch.zeros_like(label)
            predicted[:, label.argmax()] = 1

            gt = torch.zeros_like(label)
            gt[:, i % 10] = 1

            loss = torch.nn.functional.cross_entropy(label, gt)

            acc += torch.dot(predicted.flatten(), gt.flatten())
            total_loss += loss

    return acc.item() / (num_iters * batch_size), total_loss.item() / (
        num_iters * batch_size
    )
