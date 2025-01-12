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
        mini_batch_size: int | list[int] = 16,
        optimizer=optim.AdamW,
        loss=nn.MSELoss(),
        grad_clamp: int | list[int] = 1,
    ):
        self.name = name
        self.architecture = architecture
        self.parameters = parameters
        self.lr = lr
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.input_size = input_size
        self.grad_clamp = grad_clamp

        # On obtient le nombre d'architectures
        if type(parameters) is list:
            list_size = len(parameters)
        else:
            list_size = 1
            self.parameters = [parameters]

        for attr_name, attr_value in vars(self).items():
            # On vérifie que s'il y a des listes alors elles ont toutes la même taille
            if type(attr_value) is list:
                assert len(attr_value) == list_size
            # On initialise les attributs qui peuvent être des listes en liste constantes
            elif attr_name in ["lr", "epoch", "mini_batch_size", "grad_clamp"]:
                setattr(
                    self, attr_name, [deepcopy(attr_value) for _ in range(list_size)]
                )