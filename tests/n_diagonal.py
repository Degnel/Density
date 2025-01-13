import torch
import torch.nn as nn


class NDiagonalLayer(nn.Module):
    def __init__(self, dim, rank, bias=True):
        super(NDiagonalLayer, self).__init__()
        assert rank > 0, "Le nombre de diagonales doit être positif."

        self.dim = dim
        self.rank = rank

        # Matrice de poids limitée à n diagonales
        self.diagonal_weights = nn.Parameter(torch.randn(dim) * 0.01)
        self.lower_weights = [
            nn.Parameter(torch.randn(dim - (i + 1)) * 0.0001) for i in range(rank)
        ]
        self.upper_weights = [
            nn.Parameter(torch.randn(dim - (i + 1)) * 0.0001) for i in range(rank)
        ]

        # Optionnel : biais
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        weigths = torch.zeros((self.dim, self.dim))
        weigths = weigths + torch.diag(self.diagonal_weights)
        for i, (lower_weights, upper_weights) in enumerate(
            zip(self.lower_weights, self.upper_weights)
        ):
            weigths = weigths + torch.diag(lower_weights, diagonal=-(i + 1))
            weigths = weigths + torch.diag(upper_weights, diagonal=i + 1)

        out = x @ weigths
        if self.bias is not None:
            out += self.bias
        return out
