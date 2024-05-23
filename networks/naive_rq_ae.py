import torch
import torch.nn as nn
import torch.optim as optim
from vector_quantize_pytorch import ResidualVQ
from dataclasses import dataclass


@dataclass
class RQAutoencoderConfig:
    dim_l: tuple = (4, 2, 1)
    num_quantizers: int = 8      # specify number of quantizers
    codebook_size: int = 1024    # codebook size



class RQAutoencoder(nn.Module):
    def __init__(self, config: RQAutoencoderConfig):
        super().__init__()

        self.num_l = len(config.dim_l) - 1
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

        self.vq = ResidualVQ(
            dim=config.dim_l[-1],
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size
        )

    def encode(self, x):
        return self.encoder(x)
    
    def encode_to_cb(self, x):
        x = self.encode(x)
        return self.vq(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        #x, _indices, _commit_loss = self.vq(x)
        x = self.decoder(x)
        return x
