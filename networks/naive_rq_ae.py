import torch
import torch.nn as nn
import torch.optim as optim
from vector_quantize_pytorch import VectorQuantize
from dataclasses import dataclass


@dataclass
class RQAutoencoderConfig:
    dim_l: tuple = (4, 2, 1)
    size_cb: int = 512
    decay_cb: float = 0.8
    commitment_weight: float = 1.0


class RQAutoencoder(nn.Module):
    def __init__(self, config: RQAutoencoderConfig):
        super().__init__()

        self.num_l = len(config.dim_l - 1)
        # Encoder
        self.encoder = nn.Sequential()
        for i in range(self.num_l - 1):
            self.encoder.add_module(
                "encoder-layer-{}".format(i),
                nn.Linear(config.dim_l[i], config.dim_l[i + 1]),
            )
            self.encoder.add_module("encoder-relu-{}".format(i), nn.ReLU(True))

        self.encoder.add_module(
            "encoder-linear-{}".format(self.num_l),
            nn.Linear(config.dim_l[-2], config.dim_l[-1]),
        )

        # Decoder
        self.decoder = nn.Sequential()
        for i in range(1, self.num_l):
            self.decoder.add_module(
                "decoder-layer-{}".format(i),
                nn.Linear(config.dim_l[-i], config.dim_l[-(i + 1)]),
            )
            self.decoder.add_module("decoder-relu-{}".format(i), nn.ReLU(True))

        self.decoder.add_module(
            "decoder-layer-{}".format(self.num_l),
            nn.Linear(config.dim_l[1], config.dim_l[0]),
        )

        self.vq = VectorQuantize(
            dim=config.dim_l[-1],
            codebook_size=config.size_cb,  # codebook size
            decay=config.decay_cb,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=config.commitment_weight,  # the weight on the commitment loss
        )

    def forward(self, x):
        x = self.encoder(x)
        x, _indices, _commit_loss = self.vq(x)
        x = self.decoder(x)
        return x
