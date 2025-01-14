import torch
from torch import nn, optim
from density.space import ArchitecturalSpace
import matplotlib.pyplot as plt


class ArchitectureComparator:
    def __init__(
        self,
        A_space: ArchitecturalSpace,
        B_space: ArchitecturalSpace,
        base_space: ArchitecturalSpace = None,
        criterion=nn.MSELoss(),
        law=torch.distributions.Normal(0, 1),
    ) -> None:
        """
        Initialize the ArchitectureComparator.

        Parameters:
        - A_space (ArchitecturalSpace): The first architectural space.
        - B_space (ArchitecturalSpace): The second architectural space.
        - base_space (ArchitecturalSpace, optional): The base architectural space used for comparison.
        - criterion (nn.Module): Loss function used for training (default: nn.MSELoss).
        - law (torch.distributions.Distribution): Data distribution for sampling (default: Normal(0, 1)).
        """
        self.A_space = A_space
        self.B_space = B_space
        self.base_space = base_space
        self.criterion = criterion
        self.law = law

        assert (
            A_space.input_size == B_space.input_size
        ), "The input size of the two models must be the same"

        self.input_size = A_space.input_size

        assert len(A_space.parameters) == len(
            B_space.parameters
        ), "The number of architectures must be the same in space A and B"
        self.count = len(A_space.parameters)

        assert (
            A_space.automatic_mesurement_mode == B_space.automatic_mesurement_mode
            or A_space.automatic_mesurement_mode is None
            or B_space.automatic_mesurement_mode is None
        ), "The automatic mesurement mode must be the same in space A and B"

        if A_space.mesurement != B_space.mesurement:
            print(
                "Warning: The mesurements of space A and B are different, you may not compare both model on an equal footing"
            )

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

    def compare(
        self,
        max_iterations: int = 10,
        sub_iterations: int = 1,
        variance_threashold: float | None = None,
        plot_mode: str | None = None,
    ) -> tuple[list[float]]:
        """
        Compare architectures by fitting one to the other and evaluating performance.

        Parameters:
        - max_iterations (int): Maximum number of gradient descent iterations.
        - sub_iterations (int): Number of attempts of the source architecture to minimize error at each iteration.
        - variance_threashold (float, optional): Threshold to stop iterations based on variance.
        - plot_mode (str, optional): Plot comparison results; "min" or "mean".

        Returns:
        - tuple[list[float]]: Minimum and mean losses for architectures A and B.
        """

        self.max_iterations = max_iterations
        self.sub_iterations = sub_iterations
        if variance_threashold is None:
            self.variance_threashold = 0
        else:
            self.variance_threashold = variance_threashold

        self.min_A_fit = [None for _ in range(self.count)]
        self.mean_A_fit = [None for _ in range(self.count)]
        self.min_B_fit = [None for _ in range(self.count)]
        self.mean_B_fit = [None for _ in range(self.count)]

        for i in range(self.count):
            print(f"Fitting model {i+1} out of {self.count}")
            if self.base_space is None:
                print(f"{self.A_space.name} fits {self.B_space.name}")
                self.min_A_fit[i], self.mean_A_fit[i] = self._fit_source_to_target(
                    self.A_space, self.B_space, i
                )
                print(f"{self.B_space.name} fits {self.A_space.name}")
                self.min_B_fit[i], self.mean_B_fit[i] = self._fit_source_to_target(
                    self.B_space, self.A_space, i
                )
            else:
                print(f"{self.A_space.name} fits {self.base_space.name}")
                self.min_A_fit[i], self.mean_A_fit[i] = self._fit_source_to_target(
                    self.A_space, self.base_space, i
                )
                print(f"{self.base_space.name} fits {self.B_space.name}")
                self.min_B_fit[i], self.mean_B_fit[i] = self._fit_source_to_target(
                    self.B_space, self.base_space, i
                )

            if self.min_B_fit[i] > self.min_A_fit[i]:
                self.winnner = "A"
                print(f"Model {self.A_space.name} is better than {self.B_space.name}")
            else:
                self.winnner = "B"
                print(f"Model {self.B_space.name} is better than {self.A_space.name}")

            if self.mean_B_fit[i] > self.mean_A_fit[i]:
                if self.winnner == "A":
                    print(
                        f"Model {self.A_space.name} is better than {self.B_space.name} by any mean"
                    )
                else:
                    print(
                        f"However, model {self.A_space.name} shows better convergence in mean than {self.B_space.name}"
                    )
            else:
                if self.winnner == "B":
                    print(
                        f"Model {self.B_space.name} is better than {self.A_space.name} by any mean"
                    )
                else:
                    print(
                        f"However, model {self.B_space.name} shows better convergence in mean than {self.A_space.name}"
                    )

        if plot_mode is not None:
            self.plot(plot_mode)

        return self.min_A_fit, self.mean_A_fit, self.min_B_fit, self.mean_B_fit

    def _create_model(self, space: ArchitecturalSpace, index: int) -> nn.Module:
        """
        Create a model from a given architecture and a set of parameters.

        Parameters:
        - space (ArchitecturalSpace): The architectural space.
        - index (int): Index of the model within the space.

        Returns:
        - nn.Module: The created model.
        """

        return space.architecture(**space.parameters[index])

    def _fit_source_to_target(
        self,
        source_space: ArchitecturalSpace,
        target_space: ArchitecturalSpace,
        model_index: int,
    ) -> tuple[float]:
        """
        Fit a source model to match the behavior of a target model.

        Parameters:
        - source_space (ArchitecturalSpace): The source architectural space.
        - target_space (ArchitecturalSpace): The target architectural space.
        - model_index (int): Index of the model being compared.

        Returns:
        - tuple[float]: Mean and minimum losses for the source model fitting the target.
        """

        minimum = torch.tensor([torch.inf] * self.max_iterations)
        mean = torch.zeros(self.max_iterations)

        # Initialize epochs, grad_clamp and criterion
        epochs = source_space.epoch[model_index]
        grad_clamp = source_space.grad_clamp[model_index]
        criterion = self.criterion

        # We initialize mini_batch_count with both the target_space batch size and the source_space mini batch size
        # This allows us to take the information of the source space to improve convergence (as it is an important hyperparamter during the learning process of the source)
        # In the meantime, using the target space batch size allows us to know how much samples are need, if the target network has only one parameter, then the maximum degree of freedom its output is 1 (this is much more relevant thant taking the one of the source, but when compareing without a base space, we recommand to have similar mesurements for both the source and the target)
        mini_batch_count = (
            target_space.batch_size[model_index]
            // source_space.mini_batch_size[model_index]
        )
        mini_batch_size = source_space.mini_batch_size[model_index]
        shape = (
            mini_batch_count,
            mini_batch_size,
            *self.input_size,
        )

        for i in range(self.max_iterations):
            print(f"Iteration {i+1}/{self.max_iterations}")
            # Generate data
            X = self.law.sample(shape)
            X.detach()

            # Initilize target model
            target_model = self._create_model(target_space, model_index)
            target_model.eval()

            # Foward pass into target model
            with torch.no_grad():
                target_output = target_model(
                    X.view(mini_batch_count * mini_batch_size, *self.input_size)
                )
                target_output = target_output.view(
                    mini_batch_count, mini_batch_size, *self.output_size
                )

            for j in range(self.sub_iterations):
                print(f"Sub-iteration {j+1}/{self.sub_iterations}")
                # Initialize source model
                source_model = self._create_model(source_space, model_index)
                optimizer = source_space.optimizer(
                    source_model.parameters(), source_space.lr[model_index]
                )

                # Train source model to fit target model
                self.train_model(
                    source_model,
                    epochs,
                    criterion,
                    optimizer,
                    grad_clamp,
                    X,
                    target_output,
                )

                # Compute loss on the whole batch
                print("Computing score on the eval set...")
                loss = self.test_model(
                    source_model,
                    criterion,
                    X,
                    target_output,
                )

                minimum[i] = min(minimum[i], loss)
                mean[i] += loss

            mean[i] /= self.sub_iterations

            # Calculer la variance empirique afin de savoir quand s'arrêter
            min_var = torch.var(minimum, unbiased=True)
            mean_var = torch.var(mean, unbiased=True)
            max_var = max(min_var, mean_var)
            if max_var < self.variance_threashold:
                break

        return minimum.mean().item(), mean.mean().item()

    def test_model(
        self,
        model: nn.Module,
        criterion: nn.Module,
        X: list[torch.Tensor],
        y: list[torch.Tensor],
    ) -> float:
        """
        Test a model on a given dataset.

        Parameters:
        - model (nn.Module): The model to train.
        - criterion (nn.Module): Loss function.
        - X (list[torch.Tensor]): Input tensors.
        - y (list[torch.Tensor]): Target tensors.
        """

        model.eval()
        loss = 0
        for mini_batch, target in zip(X, y):
            output = model(mini_batch)
            loss += criterion(output, target)
        loss /= X.shape[0]
        print(f"Score on the whole set, loss: {loss}")
        return loss.item()

    def train_model(
        self,
        model: nn.Module,
        epochs: int,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        grad_clamp: float,
        X: list[torch.Tensor],
        y: list[torch.Tensor],
    ) -> None:
        """
        Train a model to minimize the loss between predicted and target outputs.

        Parameters:
        - model (nn.Module): The model to train.
        - epochs (int): Number of training epochs.
        - criterion (nn.Module): Loss function.
        - optimizer (optim.Optimizer): Optimizer for gradient updates.
        - grad_clamp (float): Maximum gradient value for clipping.
        - X (list[torch.Tensor]): Input tensors.
        - y (list[torch.Tensor]): Target tensors.
        """
        model.train()
        for epoch in range(epochs):
            for mini_batch, target in zip(X, y):
                optimizer.zero_grad()
                output = model(mini_batch)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clamp)
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def plot(self, mode: str) -> None:
        """
        Plot comparison results between architectures.

        Parameters:
        - mode (str): Plot type, "min" for minimum loss or "mean" for average loss.

        Raises:
        - ValueError: If the mode is not "min" or "mean".
        """

        if mode not in ["min", "mean"]:
            raise ValueError("Mode must be 'min' or 'mean'")

        if mode == "min":
            values_A = self.min_A_fit
            values_B = self.min_B_fit
        elif mode == "mean":
            values_A = self.mean_A_fit
            values_B = self.mean_B_fit

        plt.figure(figsize=(10, 5))
        plt.plot(
            self.A_space.mesurement,
            values_A,
            label=f"Architecture {self.A_space.name} ({mode})",
            marker="o",
        )
        plt.plot(
            self.B_space.mesurement,
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
        """
        Compute and return the density of the comparison.

        Returns:
        - To be implemented if mathematically cool.
        """
        pass
