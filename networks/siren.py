import torch
import torch.nn as nn
import numpy as np

class SirenLayerPytorch(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=True):
        super(SirenLayerPytorch, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            c = 1 if self.is_first else 6 / self.omega_0 ** 2
            w_std = (1 / c) ** 0.5
            self.linear.weight.uniform_(-w_std, w_std)
            self.linear.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.linear(x)
        return torch.sin(self.omega_0 * out)

class SIRENPytorch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, omega_0):
        super(SIRENPytorch, self).__init__()
        layers = [SirenLayerPytorch(input_dim, hidden_dim, omega_0, is_first=True)]
        layers += [SirenLayerPytorch(hidden_dim, hidden_dim, omega_0, is_first=False) for _ in range(num_layers - 2)]
        self.layers = nn.Sequential(*layers)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_linear(x)
