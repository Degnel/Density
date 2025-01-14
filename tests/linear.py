import torch
import torch.nn as nn
from math import sqrt

class LinearLayer(nn.Module):
    def __init__(self, dim, bias=True):
        super(LinearLayer, self).__init__()
        self.dim = dim

        std = sqrt(2/dim)

        # std = 0.0001

        # Matrix
        self.weight = nn.Parameter(torch.randn(dim, dim) * std)

        # Optionnel : biais
        self.bias = nn.Parameter(torch.randn(dim)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out