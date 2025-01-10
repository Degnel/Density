import torch
import torch.optim as optim
import torch.nn as nn

class ArchitecturalSpace:
    def __init__(   
            self, 
            name=None,
            model=None,
            lr=0.001,
            epoch=10,
            batch_size=16,
            optimizer=optim.AdamW,
            loss=nn.MSELoss(),
        ):
        self.name = name
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss

    def __str__(self):
        return f"{self.name} - {self.model} - {self.lr} - {self.epoch} - {self.batch_size} - {self.optimizer} - {self.loss}"