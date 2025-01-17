import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.cubic_transformer.attention import MultiHeadAttention
import torch.nn as nn
import torch
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        depth,
        dropout=0.1,
        cubic=False,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Liste des couches de l'encodeur
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    cubic,
                )
                for _ in range(depth)
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
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        cubic=False,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, cubic
        )

        self.cubic = cubic
        
        self.fc = nn.Linear(d_model, d_model)
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        
        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.previous_weights = None

    def forward(self, x):
        """
        x : Tensor de taille (batch_size, seq_len, d_model)
        """
        # Attention multi-têtes
        attn_output = self.self_attention(x)
        if self.cubic:
            x = attn_output + self.fc(x)
        else:  
            # x = self.layer_norm1(x + self.dropout(attn_output))
            x = F.normalize(x + self.dropout(attn_output), p=2, dim=-1)
            # x = F.normalize(x + attn_output, p=2, dim=-1)
            x = self.fc(x)

        # Réseau feed-forward
        # ff_output = self.fc_2(self.activation(self.fc_1(x)))
        ff_output = x
        # x = self.layer_norm2(x + self.dropout(ff_output))
        x = F.normalize(x + self.dropout(ff_output), p=2, dim=-1)
        # x = F.normalize(x + ff_output, p=2, dim=-1)
        # self.check_weights()
        return x

    def check_weights(self):
        current_weights = {name: param.clone() for name, param in self.named_parameters()}

        if self.previous_weights is not None:
            for name, param in current_weights.items():
                param = torch.round(torch.clamp(param, -1, 2))
                self.previous_weights[name] = torch.round(torch.clamp(self.previous_weights[name], -1, 2))
                pass
                # assert torch.equal(param, self.previous_weights[name]), f"Les poids pour {name} ont changé."

        self.previous_weights = current_weights