# SpaceOrbitLAB — simulateur lunaire

Code source du projet SꙨL pour explorer la géométrie Soleil–Terre–Lune et les conditions d’éclairement et de visibilité autour du pôle Sud lunaire.

## Démarrage rapide

Prérequis : Python 3.12 ou une version compatible avec les dépendances du projet.

```powershell
python -m pip install -r SꙨL_Python/requirements.txt
python SꙨL_Python/NASA.py
```

Le lancement de l’interface 3D nécessite un environnement graphique compatible avec PyVista/VTK. Le programme peut aussi utiliser des données éphémérides SPICE et des services en ligne selon le scénario.

## Structure

- `SꙨL_Python/NASA.py` : application principale.
- `SꙨL_Python/SOL_Tools/` : calculs orbitaux, constantes, géométrie et outils graphiques utilisés par l’application.
- `SꙨL_Python/SOL_Tools/DATA/Images/` : textures de la Terre, de la Lune et du ciel, avec crédits dans `ASSETS.md`.
- `SꙨL_Python/requirements.txt` : bibliothèques Python requises.
- `SꙨL_Python/site_web/` : site informatif statique du projet.

## Auteurs et droits d’auteur

© 2026 Thierry Marcoux et Zacharie Robitaille Jean. Les avis de droits d’auteur présents dans les fichiers sources doivent être conservés.

Ce dépôt publie le code source pour consultation. Aucune licence de réutilisation ou de redistribution n’est accordée par ce README; contacter les auteurs pour les permissions.

## Statut

Le contenu correspond au code source disponible dans ce dépôt. Les dépendances sont listées, mais aucune matrice de validation ou de compatibilité multi-plateforme n’est fournie.
