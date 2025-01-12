import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from density.probabilistic_density import ArchitectureComparator
from density.space import ArchitecturalSpace
from tests.transformer import Transformer


# Create competing architectures
transformers = [Transformer(16, 4, 16, i + 1) for i in range(5)]
split_transformers = [Transformer(16, 4, 16, i + 1, split=True) for i in range(5)]

# Create architectural spaces
transformer_space = ArchitecturalSpace((16, 10), "transformers", transformers)
split_transformer_space = ArchitecturalSpace(
    (16, 10), "split_transformers", split_transformers
)

# Create comparator
comparator = ArchitectureComparator(transformer_space, split_transformer_space)
res = comparator.compare()
print(res)
comparator.plot("min")
