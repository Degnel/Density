from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any


class ArchitecturalSpace:
    def __init__(
        self,
        input_size: tuple | torch.Size,
        name: str = None,
        architecture: nn.Module = None,
        parameters: Dict[str, Any] | list[Dict[str, Any]] | None = None,
        lr: float | list[float] = 0.001,
        epoch: int | list[int] = 3,
        batch_size: int | list[int] | None = None,
        automatic_batch_size_scale: float | None = 1.0,
        mesurement: float | list[float] | None = None,
        automatic_mesurement_mode: str | None = "information",
        mini_batch_size: int | list[int] = 16,
        optimizer=optim.AdamW,
        grad_clamp: int | list[int] = 1,
    ) -> None:
        """
        Initializes an instance of the ArchitecturalSpace class.

        Parameters:
        - input_size (tuple | torch.Size): The size of the input data.
        - name (str, optional): The name of the architectural space. Defaults to None.
        - architecture (nn.Module, optional): The neural network architecture. Defaults to None.
        - parameters (Dict[str, Any] | list[Dict[str, Any]] | None, optional): The parameters needed when initilizing the architecture. Defaults to None.
        - lr (float | list[float], optional): Learning rate(s) for the optimizer. Defaults to 0.001.
        - epoch (int | list[int], optional): Number of epochs for training. Defaults to 10.
        - batch_size (int | list[int] | None, optional): Batch size(s) for training. Defaults to None.
        - automatic_batch_size_scale (float | None, optional): Scale factor for automatic batch size calculation. Defaults to 10.0.
        - mesurement (float | list[float] | None, optional): Measurement(s) for the architecture. Defaults to None.
        - automatic_mesurement_mode (str | None, optional): Mode for automatic measurement calculation. Defaults to "information".
        - mini_batch_size (int | list[int], optional): Mini-batch size(s) for training. Defaults to 16.
        - optimizer (optional): Optimizer for training. Defaults to optim.AdamW.
        - grad_clamp (int | list[int], optional): Gradient clamp value(s). Defaults to 1.

        Returns:
        - None
        """

        assert (
            batch_size is not None or automatic_batch_size_scale is not None
        ), "Either batch_size or automatic_batch_size_scale must be defined"
        assert (
            mesurement is not None or automatic_mesurement_mode is not None
        ), "Either mesurement or automatic_mesurement_mode must be defined"

        if mesurement is None:
            assert automatic_mesurement_mode in [
                "information",
                "parameters",
            ], "automatic_mesurement_mode must be either 'information' or 'parameters'"

        self.name = name
        self.architecture = architecture
        self.parameters = parameters
        self.lr = lr
        self.epoch = epoch
        self.automatic_mesurement_mode = automatic_mesurement_mode
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.input_size = input_size
        self.grad_clamp = grad_clamp

        if type(parameters) is not list:
            self.parameters = [parameters]

        list_size = len(self.parameters)

        if automatic_mesurement_mode == "information":
            self.mesurement_method = self.count_information
        elif automatic_mesurement_mode == "parameters":
            self.mesurement_method = self.count_parameters
        else:
            self.mesurement_method = None

        if mesurement is None:
            self.mesurement = [
                self.mesurement_method(architecture(**params))
                for params in self.parameters
            ]
        else:
            self.mesurement = mesurement

        if automatic_batch_size_scale is None:
            self.batch_size = batch_size
        else:
            self.batch_size = [
                int(automatic_batch_size_scale * mesure) for mesure in self.mesurement
            ]

        for attr_name, attr_value in vars(self).items():
            if type(attr_value) is list:
                assert (
                    len(attr_value) == list_size
                ), "You should have as much elements in each list in your parameters"
            elif attr_name in [
                "lr",
                "epoch",
                "mini_batch_size",
                "grad_clamp",
                "batch_size",
                "mesurement",
            ]:
                setattr(
                    self, attr_name, [deepcopy(attr_value) for _ in range(list_size)]
                )

    def count_parameters(self, model: nn.Module) -> int:
        """
        Counts the number of trainable parameters in a given neural network architecture.

        Parameters:
        - model (nn.Module): The neural network architecture for which the parameters are to be counted.

        Returns:
        - int: The total number of trainable parameters in the architecture.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_information(self, model: nn.Module) -> int:
        """
        Calculate and return the count of information in bit.

        Returns:
            int: The number of bit needed to code the list of all trainable parameters.
        """
        total_bits = 0

        for p in model.parameters():
            if p.requires_grad:
                element_size_in_bytes = p.element_size()
                element_size_in_bits = element_size_in_bytes * 8
                total_bits += p.numel() * element_size_in_bits

        return total_bits
