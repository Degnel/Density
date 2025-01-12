Pour comparer 2 architectures, il faut définir une distribution de probabilité sur les paramtres de chaque réseau.
Cette distribution doit être définie directement lors de la création du réseau dans la méthode init.
Afin de comparer 2 architectures de manière équitable, il faut s'assurer qu'il y a ait autant de paramètres pour chaque paire prise à indice égale dans les listes space.architecture.
Il n'y a pas ce problème lorsque l'on compare avec un réseau tier.

Pour l'instant, il y a un bug du côté des modèles :
- soit on initialise dans le code du package, mais cela demande de connaître les paramètres à donner (il faudrait les passer en paramètres) -> en théorie c'est ce qu'il y a de mieux et de plus propre (en pratique c'est très lourd)
- soit on exige que seul le paramètre variant soit autorisé dans l'initilisation
- soit on n'autorise aucun paramètre lors de l'initialisation et on reset à chaque fois le NN