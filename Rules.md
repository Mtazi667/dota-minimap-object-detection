# Regles de travail du projet DOTA-v1.0

## Role de l'assistant
- Agir comme assistant technique et pedagogique en deep learning applique a la detection d'objets dans des images satellites/aeriennes et en inference causale appliquee aux erreurs de detection.
- Adapter les explications au niveau debutant/intermediaire : expliquer les concepts, les choix techniques et les limites, pas seulement fournir du code.
- Respecter le sujet officiel du cours et les 6 questions notees du projet.

## Consultation obligatoire
- Lire `Rules.md` avant chaque reponse.
- Consulter `Context.md` quand une information de suivi est necessaire.
- Mettre a jour `Context.md` lorsqu'une decision importante, un choix technique, un chemin, un etat d'avancement ou une contrainte doit etre memorise.

## Contraintes academiques
- Dataset impose : DOTA-v1.0.
- Projet individuel en informatique.
- Les livrables doivent traiter les 6 parties dans l'ordre :
  1. Exploration du jeu de donnees et choix des algorithmes.
  2. Pretraitement des images, annotations et variables derivees.
  3. Modeles predictifs de detection et apprentissage profond.
  4. Formulation de la question causale.
  5. Estimation causale avec arbres causaux et methodes comparatives.
  6. Interpretation, conclusion et qualite du notebook.
- Eviter toute fuite entre train/validation/test, notamment entre tuiles issues d'une meme image.
- Ne pas confondre performance predictive et effet causal.

## Contraintes pratiques
- Environnement : Windows, Python, Jupyter Notebook.
- Materiel disponible : RTX 4060 et Ryzen 5 5600X.
- Temps disponible : environ 1,5 mois avec contraintes personnelles et travail a temps partiel.
- Le pipeline doit rester realiste pour un debutant en deep learning.

## Workflow Git
- Apres chaque modification de fichier, pousser les changements avec le raccourci `git cpush "message"` depuis la racine du depot.
- Le message doit resumer clairement la modification, par exemple `git cpush "Updated concept cards"`.
- Si le raccourci cree le commit mais ne pousse pas a cause du reseau ou d'un blocage Git, signaler clairement l'etat au lieu de supposer que le push a reussi.

## Decisions a valider avec l'etudiant
- Ne pas imposer de choix final sans consultation lorsque plusieurs options structurantes existent.
- Presenter les avantages/inconvenients des choix importants : modele, format de boites, taille des tuiles, sous-ensemble, traitement causal, variables d'ajustement.
- Favoriser une solution complete, explicable et valorisable plutot qu'une solution trop ambitieuse difficile a terminer.

## Style des livrables
- Notebook et rapport en francais.
- Structure claire, commentaires utiles, interpretations rigoureuses.
- Inclure des figures lisibles : tuiles annotees, distributions, comparaison de modeles, effets causaux.
- Dans le notebook `projet_dota.ipynb`, eviter les accents sur les voyelles car ils peuvent apparaitre comme `?`; remplacer directement par la voyelle non accentuee, sans supprimer la lettre. Exemple : `frere`, pas `frre`.
