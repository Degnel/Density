from transformer import Transformer
from density.space import ArchitecturalSpace
from density.probabilistic_density import ArchitectureComparator
from torch import optim

"""
In this exemple we are comparing the OpenAI style Transformer achritecture with the mathematically simplest network allowing attention.
We have purposely remove one fully connected layer from the original architecture to ensure both architectures hold the same number of parameters.
"""

d_model = 6
seq_length = 5
n_heads = 1
d_ff = 6
max_depth = 4

# Create competing architectures
cubic_transformer_params = [
    {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "depth": i + 4,
        "cubic": True,
    }
    for i in range(max_depth)
]

transformer_params = [
    {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "depth": i + 4,
    }
    for i in range(max_depth)
]

# Create architectural spaces
epoch = [i+3 for i in range(max_depth)]

cubic_transformer_space = ArchitecturalSpace(
    (seq_length, d_model),
    "Cubic Transformer",
    Transformer,
    cubic_transformer_params,
    epoch=epoch,
    optimizer=optim.AdamW
)

transformer_space = ArchitecturalSpace(
    (seq_length, d_model),
    "Transformer",
    Transformer,
    transformer_params,
    epoch=epoch,
)

# Create comparator
comparator = ArchitectureComparator(cubic_transformer_space, transformer_space)

res = comparator.compare()
print(res)
comparator.plot("min")