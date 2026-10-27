# SꙨL — Site informatif du simulateur lunaire

Site statique en français présentant l’enjeu du pôle Sud lunaire, la vision du projet et un emplacement pour le futur téléchargement du simulateur `NASA.py`.

## Consulter le site

Ouvrir `index.html` dans un navigateur récent. Aucune installation, compilation, connexion réseau ou dépendance externe n’est nécessaire.

Pour utiliser un serveur local, depuis le dossier `site_web` :

```sh
python -m http.server 8000
```

Puis ouvrir http://localhost:8000. Arrêter le serveur avec Ctrl+C.

## Fichiers

- `index.html` : contenu, navigation, schéma, questions et section téléchargement.
- `styles.css` : mise en page responsive, couleurs, focus clavier et réduction des animations.
- `script.js` : menu mobile et schéma pédagogique interactif.
- `assets/moon.svg` : illustration conceptuelle originale, sans données topographiques.
- `assets/favicon.svg` : icône du site.

## Portée du contenu

Le contenu est fondé sur la description du projet fournie. Le fichier `NASA.py` n’était pas accessible lors de la création : ses fonctionnalités réelles et ses dépendances n’ont donc pas été vérifiées. Les étapes présentées décrivent les objectifs du projet, pas une liste de fonctions validées.

Le curseur du schéma déplace des astres fictifs avec des fonctions mathématiques de présentation. Il ne calcule ni éphémérides, ni relief, ni énergie disponible, ni fenêtres de communication. Les positions centrales des symboles déterminent le texte au-dessus/sous l’horizon. Ce schéma sert seulement à expliquer le principe.

## Ajouter le téléchargement plus tard

1. Préparer et tester une archive distribuable du simulateur avec sa documentation, ses dépendances et sa licence.
2. Créer `downloads/` dans `site_web`, puis y déposer l’archive, par exemple `simulateur-sol.zip`.
3. Dans `index.html`, remplacer le bouton désactivé de `.download-panel` par un lien réel :

```html
<a class="button download-button" href="downloads/simulateur-sol.zip" download>
  Télécharger le simulateur <span aria-hidden="true">↓</span>
</a>
```

4. Dans `styles.css`, déplacer `cursor: not-allowed` de `.download-button` vers `.download-button:disabled`.
5. Actualiser le statut « En préparation », le texte associé, la FAQ, les prérequis et les instructions d’installation. Vérifier les fonctions annoncées contre le code distribué.

## Vérifications manuelles

- Consulter la page sur ordinateur et mobile, vérifier l’absence de débordement horizontal.
- Parcourir les liens, le menu, le curseur et les questions au clavier.
- Vérifier l’ouverture/fermeture du menu mobile et la touche Échap.
- Déplacer le curseur : aux valeurs 0, 50 et 100, les statuts doivent suivre les positions des astres.
- Désactiver JavaScript : le contenu, la navigation et les questions doivent rester utilisables.
- Le téléchargement doit rester désactivé jusqu’à la mise à disposition du fichier réel.

Le site n’utilise ni cookies, ni suivi, ni police distante. Il ne présente aucune affiliation à la NASA et n’a pas été publié en ligne.
