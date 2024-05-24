import torch
import math
import torch.nn as nn
import torch.optim as optim
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from dataclasses import dataclass


@dataclass
class RQAutoencoderConfig:
    dim_enc: tuple = (4, 2, 1)
    dim_dec: tuple = (1, 2, 4)
    num_quantizers: int = 8  # specify number of quantizers
    codebook_size: int = 1024  # codebook size
    activation: nn.Module = nn.ReLU(True)
    vq_decay = 0.99  # decay for the vq layer
    kmeans_init = False  # set to True
    kmeans_iters = 10  # number of kmeans iterations to calculate the centroids for the codebook on init


class RQAutoencoder(nn.Module):
    def __init__(self, config: RQAutoencoderConfig):
        super().__init__()
        self.codebook_size = config.codebook_size

        assert (
            config.dim_enc[-1] == config.dim_dec[0]
        ), "Latent space dimension of encoder/decoder don't match"
        assert (
            len(config.dim_enc) > 1 and len(config.dim_dec) > 1
        ), "dimensions of decoder/encoder tuple must be bigger than 1"

        # Encoder
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

        # Decoder
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

        self.vq = ResidualVQ(
            decay=config.vq_decay,
            dim=config.dim_enc[-1],
            shared_codebook=True,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            kmeans_init=config.kmeans_init,
            kmeans_iters=config.kmeans_iters,
        )

        # ResidualVQ(
        #     dim=config.dim_enc[-1],
        #     shared_codebook=True,
        #     num_quantizers=config.num_quantizers,
        #     codebook_size=config.codebook_size
        # )

    def encode(self, x):
        return self.encoder(x)

    def encode_to_cb(self, x):
        x = self.encode(x)
        return self.vq(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x, _indices, _commit_loss = self.vq(x)
        x = self.decoder(x)
        return x
