import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from density.probabilistic_density import ArchitectureComparator
from density.space import ArchitecturalSpace
from tests.transformer import Transformer

"""
In this exemple we are comparing 2 transformers with the same general architecture
The only difference will be that one will apply an activation function before multiplying Q and K together
"""

# Create competing architectures
transformer_params = [
    {
        "d_model": 6,
        "n_heads": 3,
        "d_ff": 6,
        "num_layers": i+1,
    }
    for i in range(4)
]

split_transformer_params = [
    {
        "d_model": 6,
        "n_heads": 3,
        "d_ff": 6,
        "num_layers": i+1,
        "split": True,
    }
    for i in range(4)
]

# Create architectural spaces
transformer_space = ArchitecturalSpace(
    (5, 6), "transformers", Transformer, transformer_params
)

split_transformer_space = ArchitecturalSpace(
    (5, 6), "split_transformers", Transformer, split_transformer_params
)

# Create comparator
comparator = ArchitectureComparator(transformer_space, split_transformer_space)
res = comparator.compare()
print(res)
comparator.plot("min")
