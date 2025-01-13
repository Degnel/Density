Pour comparer 2 architectures, il faut définir une distribution de probabilité sur les paramtres de chaque réseau.
Cette distribution doit être définie directement lors de la création du réseau dans la méthode init.
Afin de comparer 2 architectures de manière équitable, il faut s'assurer qu'il y a ait autant de paramètres pour chaque paire prise à indice égale dans les listes space.architecture.
Il n'y a pas ce problème lorsque l'on compare avec un réseau tier.
L'argument 'mesure' dans un espace sert à quantifier la 'taille' d'un réseau. Le mode 'parameter' pour l'argument 'automatic_mesurement_mode' permet de comptabiliser automatiquement le nombre paramètres apprenables dans le modèle. Le mode par défaut 'information' permet lui de prendre en compte également la précision des paramètres pour comptabiliser l'intégralité des bits néccessaires pour encoder l'information des poids apprenables. L'utilisateur pourra lui-même définir ses propres métriques en définissant manuellement l'argument 'mesure' pour l'intégralité des modèles de l'espace.

TODO:
In the next version it would be great to let the user define its training routine directly in the ArchitecturalSpace class