import torch
from torch import nn
from density.space import ArchitecturalSpace
import matplotlib.pyplot as plt

# Calculer la variance empirique afin de savoir quand s'arrêter (regarder pour quel paramètre c'est intéressant de le faire)

class ArchitectureComparator:
    def __init__(
        self,
        A_space: ArchitecturalSpace,
        B_space: ArchitecturalSpace,
        base_space: ArchitecturalSpace = None,
        criterion=nn.MSELoss(),
        law=torch.distributions.Normal(0, 1),
        iterations=100,
        sub_iterations=1,
        batch_size=1000,
    ):
        self.A_space = A_space
        self.B_space = B_space
        self.base_space = base_space
        self.criterion = criterion
        self.law = law
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.batch_size = batch_size

        assert (
            A_space.input_size == B_space.input_size
        ), "The input size of the two models must be the same"

        self.input_size = A_space.input_size

        assert len(A_space.parameters) == len(
            B_space.parameters
        ), "The number of architectures must be the same in space A and B"
        self.count = len(A_space.parameters)

        try:
            test_tensor = torch.zeros((1, *self.input_size))
            A_output_size = self._create_model(A_space, 0)(test_tensor).shape
            B_output_size = self._create_model(B_space, 0)(test_tensor).shape

            assert (
                A_output_size == B_output_size
            ), "The output size of the two models must be the same"

            self.output_size = A_output_size[1:]

            if base_space is not None:
                assert (
                    self.input_size == base_space.input_size
                ), "The input size of the two models must be the same"
                base_output_size = self._create_model(base_space, 0)(test_tensor).shape
                assert (
                    self.output_size == base_output_size[1:]
                ), "The output size of the two models must be the same"
                assert len(base_space.parameters) == self.count

        except Exception as e:
            print("The input size is not correct", e)

    def compare(self, plot_mode=None):
        self.min_A_fit = [None for _ in range(self.count)]
        self.mean_A_fit = [None for _ in range(self.count)]
        self.min_B_fit = [None for _ in range(self.count)]
        self.mean_B_fit = [None for _ in range(self.count)]

        for i in range(self.count):
            if self.base_space is None:
                self.min_A_fit[i], self.mean_A_fit[i] = self._fit_source_to_target(
                    self.A_space, self.B_space, i
                )
                self.min_B_fit[i], self.mean_B_fit[i] = self._fit_source_to_target(
                    self.B_space, self.A_space, i
                )
            else:
                self.min_A_fit[i], self.mean_A_fit[i] = self._fit_source_to_target(
                    self.A_space, self.base_space, i
                )
                self.min_B_fit[i], self.mean_B_fit[i] = self._fit_source_to_target(
                    self.B_space, self.base_space, i
                )

        if plot_mode is not None:
            self.plot(plot_mode)

        return self.min_A_fit, self.mean_A_fit, self.min_B_fit, self.mean_B_fit
    
    def _create_model(self, space: ArchitecturalSpace, index: int):
        return space.architecture(**space.parameters[index])

    def _fit_source_to_target(
        self,
        source_space: ArchitecturalSpace,
        target_space: ArchitecturalSpace,
        model_index: int,
    ):
        minimum = torch.tensor([torch.inf] * self.iterations)
        mean = torch.zeros(self.iterations)

        # Initialize epochs, grad_clamp and criterion
        epochs = source_space.epoch[model_index]
        grad_clamp = source_space.grad_clamp[model_index]
        criterion = self.criterion

        for i in range(self.iterations):
            # Generate data
            mini_batch_count = self.batch_size // source_space.mini_batch_size[model_index]
            mini_batch_size = source_space.mini_batch_size[model_index]
            shape = (
                mini_batch_count,
                mini_batch_size,
                *self.input_size,
            )
            X = self.law.sample(shape)
            X.detach()

            # Initilize target model
            target_model = self._create_model(target_space, model_index)
            target_model.eval()

            # Foward pass into target model
            with torch.no_grad():
                target_output = target_model(X.view(mini_batch_count*mini_batch_size, *self.input_size))
                target_output = target_output.view(mini_batch_count, mini_batch_size, *self.output_size)

            for _ in range(self.sub_iterations):
                # Initialize source model
                source_model = self._create_model(source_space, model_index)
                source_model.train()
                optimizer = source_space.optimizer(
                    source_model.parameters(), source_space.lr[model_index]
                )

                # Train source model to fit target model
                loss = self.train_model(
                    source_model,
                    epochs,
                    criterion,
                    optimizer,
                    grad_clamp,
                    X,
                    target_output,
                )

                minimum[i] = min(minimum, loss)
                mean[i] += loss

            mean[i] /= self.sub_iterations

        return minimum.mean().item(), mean.mean().item()

    def train_model(self, model, epochs, criterion, optimizer, grad_clamp, X, y):
        for epoch in range(epochs):
            for mini_batch, target in zip(X, y):
                optimizer.zero_grad()
                output = model(mini_batch)
                loss = criterion(output, target)
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

        self.A_params = [
            self.count_parameters(arch) for arch in self.A_space.architecture
        ]
        self.B_params = [
            self.count_parameters(arch) for arch in self.B_space.architecture
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(
            self.A_params,
            values_A,
            label=f"Architecture {self.A_space.name} ({mode})",
            marker="o",
        )
        plt.plot(
            self.B_params,
            values_B,
            label=f"Architecture {self.B_space.name} ({mode})",
            marker="o",
        )
        plt.xlabel("Number of Parameters")
        plt.ylabel(f"{mode.capitalize()} Value")
        plt.title(f"Comparison of {mode.capitalize()} Values for Architectures A and B")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_densities(self):
        # Return the density of the comparison
        pass