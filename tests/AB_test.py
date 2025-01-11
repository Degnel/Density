from density.probabilistic_density import ArchitectureComparator
from density.space import ArchitecturalSpace

def main():
    # Create architectural spaces
    transformer_space = ArchitecturalSpace(input_size=(32, 32, 3))
    split_transformer_space = ArchitecturalSpace(input_size=(64, 64, 3))

    # Create an architecture comparator
    comparator = ArchitectureComparator(space_A, space_B)
    res = comparator.compare()
    print(res)
    comparator.plot("min")