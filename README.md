Pour comparer 2 architectures, il faut définir une distribution de probabilité sur les paramtres de chaque réseau.
Cette distribution doit être définie directement lors de la création du réseau dans la méthode init.
Afin de comparer 2 architectures de manière équitable, il faut s'assurer qu'il y a ait autant de paramètres pour chaque paire prise à indice égale dans les listes space.architecture.
Il n'y a pas ce problème lorsque l'on compare avec un réseau tier