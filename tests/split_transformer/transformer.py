import torch.nn as nn
from tests.split_transformer.attention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1, split=False):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Liste des couches de l'encodeur
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, split)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, split=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, split)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        # Attention multi-têtes
        attn_output = self.self_attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Réseau feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x
