import torch.nn as nn
from n_diagonal import NDiagonalLayer
from lora import LoRALayer


class DeepNetwork(nn.Module):
    def __init__(
        self, layer_type="fully_connected", dim=10, depth=1, rank=1, bias=True
    ):
        super(DeepNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = nn.ReLU()

        for _ in range(depth):
            if layer_type == "n_diagonal":
                self.layers.append(NDiagonalLayer(dim, rank, bias))
            elif layer_type == "LoRA":
                self.layers.append(LoRALayer(dim, rank, bias))
            elif layer_type == "fully_connected":
                self.layers.append(nn.Linear(dim, dim, bias))
            else:
                raise ValueError(f"Type de couche non supporté : {layer_type}")

            self.norms.append(nn.LayerNorm(dim))

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            skip = x
            x = layer(x)
            x = self.activation(x)
            x = x + skip
            x = norm(x)
        return x
