import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from density.probabilistic_density import ArchitectureComparator
from density.space import ArchitecturalSpace
from tests.deep_nn import DeepNetwork

"""
In this exemple we are comparing the reduction of parameter count when using the usuall LoRA and a n-diagonal matrix instead
As the count of parameter does not scale the same way, we compare both architectures to an architecture with fully connected layers
"""

# Create competing architectures
n_diagonal_params = [
    {
        "layer_type": "n_diagonal",
        "dim": 5,
        "depth": 4,
        "rank": i + 1,
        "bias": True,
    }
    for i in range(5)
]

lora_params = [
    {
        "layer_type": "LoRA",
        "dim": 5,
        "depth": 4,
        "rank": i + 1,
        "bias": True,
    }
    for i in range(5)
]

fully_connected_params = [
    {
        "layer_type": "fully_connected",
        "dim": 5,
        "depth": 4,
        "bias": True,
    }
    for _ in range(5)
]


def compute_params(dim, depth, rank, bias):
    return depth * (dim + 2 * dim * rank - rank * (rank + 1) + dim * bias)


# Create architectural spaces
n_diagonal_space = ArchitecturalSpace(
    (5,),
    "N-Diagonal",
    DeepNetwork,
    n_diagonal_params,
    epoch=10,
    lr=0.01,
    automatic_mesurement_mode=None,
    mesurement=[compute_params(5, 4, i, True) for i in range(5)],
)

lora_space = ArchitecturalSpace(
    (5,),
    "LoRA",
    DeepNetwork,
    lora_params,
    epoch=10,
    lr=0.01,
    automatic_mesurement_mode="parameters",
)

fully_connected_space = ArchitecturalSpace(
    (5,),
    "Fully Connected",
    DeepNetwork,
    fully_connected_params,
    epoch=10,
    lr=0.01,
    automatic_mesurement_mode="parameters",
)

# Create comparator
comparator = ArchitectureComparator(n_diagonal_space, lora_space, fully_connected_space)
# comparator = ArchitectureComparator(lora_space, n_diagonal_space, fully_connected_space)
res = comparator.compare(100, 5)
print(res)
comparator.plot("min")
