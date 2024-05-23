import torch
import torch.nn as nn
import torch.optim as optim
from vector_quantize_pytorch import VectorQuantize
from dataclasses import dataclass


@dataclass
class RegressionTransformerConfig:
    dropout: float = 0.0
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 1
    n_embd: int = 1  # 768
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class RQ_Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(nn.Linear(4, 2), nn.ReLU(True), nn.Linear(2, 1))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(True),
            nn.Linear(2, 4),
        )
        self.vq = VectorQuantize(
            dim=256,
            codebook_size=512,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.0,  # the weight on the commitment loss
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x
