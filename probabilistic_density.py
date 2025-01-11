import torch
from torch import nn
from space import ArchitecturalSpace

# Calculer la variance empirique afin de savoir quand s'arrêter (regarder pour quel paramètre c'est intéressant de le faire)

class ArchitectureComparator:
    def __init__(
            self, 
            space_A: ArchitecturalSpace, 
            space_B: ArchitecturalSpace,
            base_space: ArchitecturalSpace=None,
            criterion = nn.MSELoss(),
            law = torch.distributions.Normal(0, 1),
            iterations=100,
            sub_iterations=1,
            batch_size=1000,
        ):
        self.space_A = space_A
        self.space_B = space_B
        self.base_space = base_space
        self.criterion = criterion
        self.law = law
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.batch_size = batch_size

        assert space_A.input_size == space_B.input_size, "The input size of the two models must be the same"

        self.input_size = space_A.input_size

        try:
            test_tensor = torch.zeros_like(1, *self.input_size)
            A_output_size = space_A.architecture()(test_tensor).shape
            B_output_size = space_B.architecture()(test_tensor).shape

            assert A_output_size == B_output_size, "The output size of the two models must be the same"

            self.output_size = A_output_size

            if base_space is not None:
                assert self.input_size == base_space.input_size, "The input size of the two models must be the same"
                base_output_size = base_space.architecture()(test_tensor).shape
                assert self.output_size == base_output_size, "The output size of the two models must be the same"
        
        except:
            print("The input size is not correct")
            
    def compare(self, plot_mode=None):
        if self.base_space is None:
            # On essaie d'approximer pour tous les éléments dans l'espace A, les éléments dans l'espace B
            pass
        else:
            # On essaie d'approximer pour tous les éléments dans l'espace A, les éléments dans l'espace de base
            # De même pour l'espace B
            pass

        return

    def _compare(
        self, 
        target_architecture: ArchitecturalSpace, 
        source_architecture: ArchitecturalSpace
    ):
        minimum = torch.tensor([torch.inf]*self.iterations)
        mean = torch.zeros(self.iterations)
        for i in range(self.iterations):
            # Generate data
            mini_batch_count = self.batch_size//source_architecture.mini_batch_size
            shape = (mini_batch_count, source_architecture.mini_batch_size, *self.input_size)
            X = self.law.sample(shape)

            # Initilize target model
            target_model = target_architecture()
            target_model.eval()
            
            # Foward pass into target model
            target_output = target_model(X)

            # Initialize optimizer, grad_clamp and criterion
            optimizer = source_architecture.optimizer
            grad_clamp = source_architecture.grad_clamp
            criterion = self.criterion

            for j in range(self.sub_iterations):
                # Initialize source model
                source_model = source_architecture()
                source_model.train()
                
                # Train source model to fit target model
                loss = self.train_model(source_model, source_architecture.epoch, criterion, optimizer, grad_clamp, X, target_output)
                
                minimum[i] = min(minimum, loss)
                mean[i] += loss

            mean[i] /= self.sub_iterations
            
            # On keep track de min_A_B, min_B_A, mean_A_B, mean_B_A
        return minimum.mean(), mean.mean()


    def train_model(self, model, epochs, criterion, optimizer, grad_clamp, X, y):
        for epoch in range(epochs):
            for mini_batch in X:
                optimizer.zero_grad()
                output = model(mini_batch)
                loss = criterion(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clamp)
                optimizer.step()
                
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        
        return loss

    def plot(self, mode):
        # mode peut être égal à "min" ou "mean", par défaut on choisira "min"
        # Plot the comparison in terms of the number of params for each model
        # Le mode sert à choisir si l'on veut plot min_A_B, ... en fonction du nb de paramètres
        pass

    def get_densities(self):
        # Return the density of the comparison
        pass