from animation.util import reconstruct_image
from networks.nano_gpt import GPT
import torch
import torchvision.transforms as transforms
from utils.animation import generate_neural_field
from vector_quantize_pytorch import VectorQuantize
from PIL import Image

# compute metrics
mnist_classifier = torch.hub.load("pytorch/vision:v0.10.0", "mnist", pretrained=True)
mnist_classifier.eval()


@torch.no_grad()
def compute_mnist_score(
    model: GPT,
    vq: VectorQuantize,
    special_tokens: dict[str, int],
    num_images: int = 1000,
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
    for i in range(num_images):
        # generate neural field

        # randomly select a special token
        model.eval()
        mlp3d = generate_neural_field(
            model, vq, [special_tokens["SOS"], special_tokens[str(i % 10)]]
        )

        # get image
        image = reconstruct_image(mlp3d)
        image = Image.fromarray((image * 255).astype("uint8"))
        image = transforms.ToTensor()(image).unsqueeze(0)

        # classify image
        label = mnist_classifier(image).argmax().item()

        # compute loss
        loss = -torch.log_softmax(mlp3d(image), dim=1)[0, label]
        total_loss += loss

    pass
