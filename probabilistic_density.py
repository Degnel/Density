import torch
from torch import nn
from space import ArchitecturalSpace
import matplotlib.pyplot as plt


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

        assert len(space_A.architecture) == len(space_B.architecture), "The number of architectures must be the same in space A and B"
        self.count = len(space_A.architecture)

        try:
            test_tensor = torch.zeros_like(1, *self.input_size)
            A_output_size = space_A.architecture[0]()(test_tensor).shape
            B_output_size = space_B.architecture[0]()(test_tensor).shape

            assert A_output_size == B_output_size, "The output size of the two models must be the same"

            self.output_size = A_output_size

            if base_space is not None:
                assert self.input_size == base_space.input_size, "The input size of the two models must be the same"
                base_output_size = base_space.architecture[0]()(test_tensor).shape
                assert self.output_size == base_output_size, "The output size of the two models must be the same"
                assert len(base_space.architecture) == self.count
        
        except:
            print("The input size is not correct")

            
    def compare(self, plot_mode=None):
        self.min_A_fit = [None for _ in range(self.count)]
        self.mean_A_fit = [None for _ in range(self.count)]
        self.min_B_fit = [None for _ in range(self.count)]
        self.mean_B_fit = [None for _ in range(self.count)]

        for i in range(self.count):
            if self.base_space is None:
                self.min_A_fit[i], self.mean_A_fit[i] = self._fit_source_to_target(self.space_A.architecture[i], self.space_B.architecture[i])
                self.min_B_fit[i], self.mean_B_fit[i] = self._fit_source_to_target(self.space_B.architecture[i], self.space_A.architecture[i])
            else:
                self.min_A_fit[i], self.mean_A_fit[i] = self._fit_source_to_target(self.space_A.architecture[i], self.base_space.architecture[i])
                self.min_B_fit[i], self.mean_B_fit[i] = self._fit_source_to_target(self.space_B.architecture[i], self.base_space.architecture[i])

        if plot_mode is not None:
            self.plot(plot_mode)

        return

    def _fit_source_to_target(
        self, 
        source_architecture: ArchitecturalSpace,
        target_architecture: ArchitecturalSpace
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
            
        return minimum.mean().item(), mean.mean().item()


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

    def count_parameters(self, architecture):
        model = architecture()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def plot(self, mode):
        if mode not in ["min", "mean"]:
            raise ValueError("Mode must be 'min' or 'mean'")

        if mode == "min":
            values_A = self.min_A_fit
            values_B = self.min_B_fit
        elif mode == "mean":
            values_A = self.mean_A_fit
            values_B = self.mean_B_fit

        self.A_params = [self.count_parameters(arch) for arch in self.space_A.architecture]
        self.B_params = [self.count_parameters(arch) for arch in self.space_B.architecture]

        plt.figure(figsize=(10, 5))
        plt.plot(self.A_params, values_A, label=f'Architecture A ({mode})', marker='o')
        plt.plot(self.B_params, values_B, label=f'Architecture B ({mode})', marker='o')
        plt.xlabel('Number of Parameters')
        plt.ylabel(f'{mode.capitalize()} Value')
        plt.title(f'Comparison of {mode.capitalize()} Values for Architectures A and B')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_densities(self):
        # Return the density of the comparison
        pass