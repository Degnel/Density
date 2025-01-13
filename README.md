Pour comparer 2 architectures, il faut définir une distribution de probabilité sur les paramtres de chaque réseau.
Cette distribution doit être définie directement lors de la création du réseau dans la méthode init.
Afin de comparer 2 architectures de manière équitable, il faut s'assurer qu'il y a ait autant de paramètres pour chaque paire prise à indice égale dans les listes space.architecture.
Il n'y a pas ce problème lorsque l'on compare avec un réseau tier.
L'argument 'mesure' dans un espace sert à quantifier la 'taille' d'un réseau. Le mode 'parameter' pour l'argument 'automatic_mesurement_mode' permet de comptabiliser automatiquement le nombre paramètres apprenables dans le modèle. Le mode par défaut 'information' permet lui de prendre en compte également la précision des paramètres pour comptabiliser l'intégralité des bits néccessaires pour encoder l'information des poids apprenables. L'utilisateur pourra lui-même définir ses propres métriques en définissant manuellement l'argument 'mesure' pour l'intégralité des modèles de l'espace.


Tests :
- Le fichier split_transformers_test.py permet de comparer un transformers classique à un transformer appliquant une fonction d'activation juste avant le produit entre Q et K. Un effectue un test A/B entre les deux modèles.
- Le fichier n_diagonal_test.py compare l'usage d'une matrice n_diagonale, comparée à une matrice écrite sous la forme d'une LoRA. Le nombre de paramètre n'étant pas régoureusement identique dans les 2 architectures, on se propose de comparer ces dernière à une architecture tier qui englobe les 2 architectures. On compare donc à un réseau constitué de matrices de poids pleines.

TODO:
In the next version it would be great to let the user define its training routine directly in the ArchitecturalSpace class