from torch import nn
from space import ArchitecturalSpace

class ArchitectureComparator:
    def __init__(
            self, 
            modelA: ArchitecturalSpace, 
            modelB: ArchitecturalSpace,
            baseModel: ArchitecturalSpace=None,
            loss = nn.MSELoss(),
            law = nn.Normal(0, 1)
            ):
        self.modelA = modelA
        self.modelB = modelB
        self.baseModel = baseModel
        self.loss = loss
        self.law = law
        
    def compare(self, plot_mode=None):
        # Si on a un baseModel alors on s'en sert et les 2 doivent l'imiter
        # Compare the two models
        # Il nous faut génrer des données suivant la loi donnée (par défaut on utilise la normale)
        # On doit se servir de la forme d'entrée de chaque réseau (vérifier qu'il s'agit bien de la même)
        # Initiliser un model aléatoirement
        # On essaie de l'approximer avec l'autre
        # On recommence
        # On calcul le min et la mean
        # On keep track de min_A_B, min_B_A, mean_A_B, mean_B_A
        # On initialise le second réseau et on fait dans l'autre sens
        # Calculer la variance empirique afin de savoir quand s'arrêter (regarder pour quel paramètre c'est intéressant de le faire)
        pass

    def plot(self, mode):
        # Plot the comparison in terms of the number of params for each model
        # Le mode sert à choisir si l'on veut plot min_A_B, ... en fonction du nb de paramètres
        pass

    def get_densities(self):
        # Return the density of the comparison
        pass