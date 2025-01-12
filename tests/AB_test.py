import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from density.probabilistic_density import ArchitectureComparator
from density.space import ArchitecturalSpace
from tests.transformer import Transformer


# Create competing architectures
transformer_params = [{
    "d_model": 16,
    "n_heads": 4,
    "d_ff": 16,
    "num_layers": i + 1,
} for i in range(5)]

split_transformer_params = [{
    "d_model": 16,
    "n_heads": 4,
    "d_ff": 32,
    "num_layers": i + 1,
    "split": True,
} for i in range(5)]

# Create architectural spaces
transformer_space = ArchitecturalSpace(
    (10, 16), 
    "transformers", 
    Transformer, 
    transformer_params
)

split_transformer_space = ArchitecturalSpace(
    (10, 16), 
    "split_transformers", 
    Transformer, 
    split_transformer_params
)

# Create comparator
comparator = ArchitectureComparator(transformer_space, split_transformer_space)
res = comparator.compare()
print(res)
comparator.plot("min")