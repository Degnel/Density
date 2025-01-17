import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Cette classe implémente l'attention multi-tête d'OpenAI (et non celle du papier originel 'Attention is all you need'). 
La différence principale est que les tête d'attention sont sommées au lieu d'être concaténées. Cela permet n'implique donc pas de respecter la contrainte d_model%n_heads = 0.
L'autre différence, est qu'il n'y a pas de couche linéaire tradictionnellement appelée 'O' (pour 'output') appliquée à la fin du mécanisme d'attention.
"""


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        cubic=False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.cubic = cubic

        self.Q = nn.Linear(d_model, d_model * n_heads, False)
        self.K = nn.Linear(d_model, d_model * n_heads, False)
        self.V = nn.Linear(d_model, d_model * n_heads, False)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # Calculer Q, K, V et les diviser en têtes
        q, k, v = self.Q(x), self.K(x), self.V(x)

        query = self._reshape_to_batches(q)
        key = self._reshape_to_batches(k)
        value = self._reshape_to_batches(v)

        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if self.cubic:
            attention = F.softmax(scores, dim=-1)
        else:
            attention = scores
        y = attention.matmul(value)

        y = y.reshape(batch_size, self.n_heads, seq_len, self.d_model)
        y = y.sum(dim=1)

        return y

    def _reshape_to_batches(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: input tensor with shape (batch_size, seq_len, d_model*n_heads)

        Returns:
        Reshaped tensor with shape (batch_size*n_heads, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.n_heads, self.d_model)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.n_heads, seq_len, self.d_model)
        )
