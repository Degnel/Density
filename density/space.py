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
        epoch: int | list[int] = 10,
        batch_size: int | list[int] | None = None,
        automatic_batch_size_scale: float | None = 10.0,
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

        assert batch_size is not None or automatic_batch_size_scale is not None, "Either batch_size or automatic_batch_size_scale must be defined"
        assert mesurement is not None or automatic_mesurement_mode is not None, "Either mesurement or automatic_mesurement_mode must be defined"
        assert automatic_mesurement_mode in ["information", "parameters"], "automatic_mesurement_mode must be either 'information' or 'parameters'"

        self.name = name
        self.architecture = architecture
        self.parameters = parameters
        self.lr = lr
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.input_size = input_size
        self.grad_clamp = grad_clamp

        if type(parameters) is not list:
            self.parameters = [parameters]

        list_size = len(self.parameters)

        if automatic_mesurement_mode == "information":
            self.automatic_mesurement_mode = self.count_information
        elif automatic_mesurement_mode == "parameters":
            self.automatic_mesurement_mode = self.count_parameters
        else:
            self.automatic_mesurement_mode = None

        if automatic_mesurement_mode is None:
            self.mesurement = mesurement
        else:
            self.mesurement = [self.automatic_mesurement_mode(architecture(params)) for params in self.parameters]

        if automatic_batch_size_scale is None:
            self.batch_size = batch_size
        else:
            self.batch_size = [automatic_batch_size_scale * mesure for mesure in self.mesurement]

        for attr_name, attr_value in vars(self).items():
            if type(attr_value) is list:
                assert len(attr_value) == list_size
            elif attr_name in ["lr", "epoch", "mini_batch_size", "grad_clamp", "batch_size", "mesurement"]:
                setattr(
                    self, attr_name, [deepcopy(attr_value) for _ in range(list_size)]
                )

        assert batch_size is not None or automatic_batch_size_scale is not None, "Either batch_size or automatic_batch_size_scale must be defined"
        assert mesurement is not None or automatic_mesurement_mode is not None, "Either mesurement or automatic_mesurement_mode must be defined"
        assert automatic_mesurement_mode in ["information", "parameters"], "automatic_mesurement_mode must be either 'information' or 'parameters'"

        self.name = name
        self.architecture = architecture
        self.parameters = parameters
        self.lr = lr
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.input_size = input_size
        self.grad_clamp = grad_clamp

        # On obtient le nombre d'architectures
        if type(parameters) is not list:
            self.parameters = [parameters]

        list_size = len(self.parameters)

        # On calcule les mesures automatiquement si besoin
        if automatic_mesurement_mode == "information":
            self.automatic_mesurement_mode = self.count_information
        elif automatic_mesurement_mode == "parameters":
            self.automatic_mesurement_mode = self.count_parameters
        else:
            self.automatic_mesurement_mode = None

        if automatic_mesurement_mode is None:
            self.mesurement = mesurement
        else:
            self.mesurement = [self.automatic_mesurement_mode(architecture(params)) for params in self.parameters]

        # On compte mesure les réseau automatiquement si besoin
        if automatic_batch_size_scale is None:
            self.batch_size = batch_size
        else:
            self.batch_size = [automatic_batch_size_scale * mesure for mesure in self.mesurement]

        for attr_name, attr_value in vars(self).items():
            # On vérifie que s'il y a des listes alors elles ont toutes la même taille
            if type(attr_value) is list:
                assert len(attr_value) == list_size
            # On initialise les attributs qui peuvent être des listes en liste constantes
            elif attr_name in ["lr", "epoch", "mini_batch_size", "grad_clamp", "batch_size", "mesurement"]:
                setattr(
                    self, attr_name, [deepcopy(attr_value) for _ in range(list_size)]
                )

    def count_parameters(
            self, 
            architecture: nn.Module
        ) -> int:
        """
        Counts the number of trainable parameters in a given neural network architecture.

        Parameters:
        - architecture (nn.Module): The neural network architecture for which the parameters are to be counted.

        Returns:
        - int: The total number of trainable parameters in the architecture.
        """
        model = architecture()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def count_information(
            self, 
            architecture: nn.Module
        ) -> int:
        """
        Calculate and return the count of information in bit.

        Returns:
            int: The number of bit needed to code the list of all trainable parameters.
        """
        model = architecture()
        total_bits = 0

        for p in model.parameters():
            if p.requires_grad:
                element_size_in_bytes = p.element_size()
                element_size_in_bits = element_size_in_bytes * 8
                total_bits += p.numel() * element_size_in_bits

        return total_bits