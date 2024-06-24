

import os
import torch
from tqdm import tqdm

from data.neural_field_datasets import MnistNeFDataset, ModelTransform
from networks.nano_gpt import GPT
from utils import get_default_device
from vector_quantize_pytorch import VectorQuantize
from einops import rearrange

from animation.util import reconstruct_image_correct_and_not_luis_bs
from utils.metrics import compute_all_metrics
from utils.visualization import generate_neural_field



label = 3
skip_gt = True
skip_novel = False
model_dict = torch.load("./models/token_transformer/proper_tokens.pt")

device = get_default_device()
model = GPT(model_dict["model_args"])#model_dict
model.to(device=device)
model.load_state_dict(model_dict["model"])
model.eval()

vq = VectorQuantize(**model_dict["vq_config"])
vq.load_state_dict(model_dict["vq_state_dict"])
vq.eval()

dir_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_root = os.path.join(dir_path, "adl4cv")

kwargs = {
"type": "pretrained",
"fixed_label": label,
}

cb_size = model_dict["vq_config"]["codebook_size"]
token_dict = {
    "SOS": cb_size + 0,
    "0": cb_size + 10,
    "1": cb_size + 9,
    "2": cb_size + 8,
    "3": cb_size + 7,
    "4": cb_size + 6,
    "5": cb_size + 5,
    "6": cb_size + 4,
    "7": cb_size + 3,
    "8": cb_size + 2,
    "9": cb_size + 1
}


dataset = MnistNeFDataset(os.path.join(data_root, "datasets", "mnist-nerfs"), transform=ModelTransform(), **kwargs)
dataset_length = len(dataset)

if not skip_gt:

  print("generating images from nerfs")

  dataset_length = len(dataset)
  # generate images from nerfs and store in tensor
  imgs = torch.zeros(dataset_length, 1, 28, 28)
  for i, (neural_field, label) in tqdm(enumerate(dataset), "Neural Field", total=dataset_length):
    reconstructed = reconstruct_image_correct_and_not_luis_bs(neural_field)
    imgs[i] = rearrange(reconstructed, "... -> 1 ...")

    # display reconstructed using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(reconstructed.squeeze(0).numpy(), cmap="gray")
    plt.show()

  torch.save(imgs, "./models/metrics/mnist_nerf_images.pt")

if not skip_novel:
  print("Generating novel images")

  top_k_options = [None, 3, 5, 7, 10, 15]
  temperature_options = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.5]

  with tqdm(total=len(top_k_options)*len(temperature_options), desc="Main Generation") as upper_pbar:

    for top_k in top_k_options:
      for temperature in temperature_options:
        upper_pbar.set_description(f"Top K: {top_k}, Temperature: {temperature}")
        # generate images from nerfs and store in tensor
        dataset_length = 256

        # generate images from nerfs and store in tensor
        imgs = torch.zeros(dataset_length, 1, 28, 28)

        # enumerate over the dataset in batches
        batch_size = 32
        model.eval()
        with tqdm(total=dataset_length, desc="Neural Field") as pbar:
          for i in range(0, dataset_length, batch_size):
            models = generate_neural_field(model, vq, token_dict["SOS"], [token_dict[str(label)] for _ in range(batch_size)], device, top_k=top_k, temperature=temperature)
            for j, mlp3d in enumerate(models):
              if i+j >= dataset_length:
                break
              reconstructed = reconstruct_image_correct_and_not_luis_bs(mlp3d)
              imgs[i+j] = rearrange(reconstructed, "... -> 1 ...")
              pbar.update(1)

        torch.save(imgs, f"./models/metrics/mnist_novel_images_tk_{top_k}_temp_{temperature}.pt")
        upper_pbar.update(1)


# # Load the images
# imgs = torch.load("./models/metrics/mnist_nerf_images.pt")
# # only use the first 196 images
# imgs = imgs[:196]
# imgs_novel = torch.load("./models/metrics/mnist_novel_images_tk_3_temp_0.8.pt")


# print(compute_all_metrics(imgs_novel, imgs))