import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, split=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections linéaires pour Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, False)
        self.fc_out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)
        self.split = split

        if split:
            # self.activation = nn.ReLU()
            self.activation = nn.Tanh()

    def forward(self, query, key, value):
        """
        query, key, value : Tensor de taille (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.size()

        # Calculer Q, K, V et les diviser en têtes
        qkv = self.qkv_proj(query).chunk(3, dim=-1)
        query, key, value = [
            x.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            for x in qkv
        ]

        if self.split:
            query = self.activation(query)
            key = self.activation(key)

        # Produit scalaire pour l'attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)

        # Appliquer les poids d'attention sur V
        attn_output = torch.matmul(attention, value)

        # Réassembler les têtes
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Projection finale
        return self.fc_out(attn_output)
