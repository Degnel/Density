import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, dim, rank, bias=True):
        super(LoRALayer, self).__init__()
        self.dim = dim
        self.rank = rank

        # Matrices LoRA
        self.down_proj = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.up_proj = nn.Parameter(torch.randn(rank, dim) * 0.01)

        # Optionnel : biais
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        # Application de la réduction puis de la projection
        lora_out = x @ self.down_proj @ self.up_proj
        if self.bias is not None:
            lora_out += self.bias
        return lora_out
