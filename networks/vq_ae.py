import torch
import math
import torch.nn as nn
import torch.optim as optim
from utils import get_default_device
from vector_quantize_pytorch import VectorQuantize
from dataclasses import asdict, dataclass


@dataclass
class VQAutoencoderConfig:
    dim_enc: tuple = (4, 2, 1)
    dim_dec: tuple = (1, 2, 4)
    with_vq: bool = True
    codebook_size: int = 1024  # codebook size
    activation: nn.Module = nn.ReLU(True)
    kmeans_init = False  # set to True
    kmeans_iters = 10  # number of kmeans iterations to calculate the centroids for the codebook on init


class VQAutoencoder(nn.Module):
    def __init__(self, config: VQAutoencoderConfig):
        super().__init__()
        self.codebook_size = config.codebook_size

        # Encoder
        if config.dim_enc is not None:
            assert (
                config.dim_enc[-1] == config.dim_dec[0]
            ), "Latent space dimension of encoder/decoder don't match"
            assert (
                len(config.dim_enc) > 1 and len(config.dim_dec) > 1
            ), "dimensions of decoder/encoder tuple must be bigger than 1"

            num_l = len(config.dim_enc) - 1
            self.encoder = nn.Sequential()
            for i in range(num_l - 1):
                self.encoder.add_module(
                    "encoder-linear-{}".format(i),
                    nn.Linear(config.dim_enc[i], config.dim_enc[i + 1]),
                )
                self.encoder.add_module(
                    f"encoder-{(config.activation._get_name())}-{i}", config.activation
                )

            self.encoder.add_module(
                "encoder-linear-{}".format(num_l - 1),
                nn.Linear(config.dim_enc[-2], config.dim_enc[-1]),
            )
        else:
            self.encoder = nn.Identity()

        # Decoder
        if config.dim_dec is not None:
            num_l = len(config.dim_dec) - 1
            self.decoder = nn.Sequential()
            for i in range(0, num_l - 1):
                self.decoder.add_module(
                    "decoder-linear-{}".format(i),
                    nn.Linear(config.dim_dec[i], config.dim_dec[i + 1]),
                )
                self.decoder.add_module(
                    f"decoder-{(config.activation._get_name())}-{i}", config.activation
                )

            self.decoder.add_module(
                "decoder-linear-{}".format(num_l - 1),
                nn.Linear(config.dim_dec[-2], config.dim_dec[-1]),
            )
        else:
            self.decoder = nn.Identity()

        if config.with_vq:
            self.vq = VectorQuantize(config.dim_dec[0], config.codebook_size, kmeans_init=config.kmeans_init, kmeans_iters=config.kmeans_iters).to(get_default_device())
        else:
            self.vq = None

    def encode(self, x):
        return self.encoder(x)

    def encode_to_cb(self, x):
        x = self.encode(x)
        return self.vq(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        if self.vq:
            x, _indices, _commit_loss = self.vq(x)
        x = self.decoder(x)
        return x
