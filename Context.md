# Contexte courant du projet

## Sujet
Projet de fin d'etudes en informatique :
Detection d'objets dans des images satellites et aeriennes, avec inference causale, apprentissage machine et apprentissage profond.

## Dataset
Dataset impose : DOTA-v1.0.

Arborescence locale :
- `C:\Users\Tazi\Desktop\School\Projet\Training Data\Images`
- `C:\Users\Tazi\Desktop\School\Projet\Training Data\LabelTxt`
- `C:\Users\Tazi\Desktop\School\Projet\Validation Data\Images`
- `C:\Users\Tazi\Desktop\School\Projet\Validation Data\LabelTxt`
- `C:\Users\Tazi\Desktop\School\Projet\Testing Images\Images`

Etat connu :
- Les donnees DOTA-v1.0 sont deja telechargees et organisees.
- Un notebook `projet_dota.ipynb` existe.
- `Subject.pdf` est present dans le repertoire du projet.
- Les dossiers et les nombres d'images/annotations ont deja ete verifies dans Jupyter par l'etudiant.

## Objectifs de l'etudiant
- Comprendre les concepts cles : detection d'objets, IoU, mAP, boites orientees, tuilage, causalite, erreurs de modele.
- Pouvoir expliquer le pipeline a un professeur ou a un recruteur.
- Produire un notebook et un rapport en francais, clairs et rigoureux.
- Obtenir un projet valorisable sur CV/LinkedIn.
- Rester dans un niveau de complexite realiste pour environ 1,5 mois de travail.

## Contraintes techniques
- Systeme : Windows.
- GPU : RTX 4060.
- CPU : Ryzen 5 5600X.
- Outils prevus : Python, Jupyter Notebook, PyTorch/torchvision, Ultralytics YOLO si pertinent, bibliotheques Python de causalite si pertinent.

## Points a verifier
- `Subject.pdf` a ete lu avec `pypdf` le 2026-06-09 : 4 pages, sujet confirme, bareme sur 100 points et 6 questions obligatoires.
- Choisir le format de detection principal : boites orientees conservees ou conversion en boites horizontales.
- Choisir un modele baseline et un modele deep moderne.
- Definir la question causale, le traitement, la sortie et les variables d'ajustement.

## Environnement Python
- `pypdf` a ete installe dans l'environnement Python utilisateur pour lire `Subject.pdf`.
- Des essais d'installation locale dans `.codex_deps`, `deps` et `deps_local` ont cree des dossiers non lisibles par le sandbox Windows; ne pas s'appuyer dessus pour le notebook.
- Verification de reprise 2026-06-10 : `nvidia-smi` detecte bien une RTX 4060, mais le Python terminal courant utilise `torch 2.11.0+cpu` sans CUDA. Avant l'entrainement profond, verifier l'environnement Jupyter/Python qui sera utilise pour avoir une version compatible GPU.

## Resume verifie du sujet officiel
- Langage autorise : Python uniquement.
- Fichier principal a soumettre : notebook `.ipynb` executable independamment, avec chemins relatifs et fonctions auxiliaires incluses dans le notebook.
- Rapport final a soumettre au format PDF.
- Le notebook doit indiquer le lien du dataset utilise et la date de telechargement.
- Utiliser au moins les ensembles entrainement et validation de DOTA-v1.0.
- Si un sous-ensemble est utilise, il doit etre stratifie par categorie et type de scene, avec justification.
- La separation entrainement/validation/test doit etre respectee; les images augmentees, recadrees ou dupliquees ne doivent jamais se retrouver dans plusieurs ensembles differents.

## Plan d'action propose
- Statut : valide comme direction generale le 2026-06-09.
- Organisation recommandee : avancer en 6 phases correspondant aux 6 questions du sujet, avec un axe parallele d'apprentissage.
- Priorite : construire rapidement un pipeline complet simple, puis l'ameliorer; eviter de commencer par une solution trop ambitieuse.
- Strategie probable a discuter : baseline simple + detecteur profond moderne; conserver les boites orientees si faisable avec YOLO-OBB, sinon convertir en boites horizontales avec justification claire.
- Apprentissage a integrer : a chaque phase, produire une courte fiche explicative en francais sur les concepts necessaires pour pouvoir les expliquer.

## Preferences de collaboration
- L'etudiant n'est pas nouveau en programmation, mais debute en machine learning/deep learning.
- L'assistant peut recommander les choix techniques quand l'etudiant n'est pas certain, mais doit expliquer ce qui se passe et pourquoi.
- Commencer par se familiariser avec les donnees avant l'entrainement de modeles.
- Construire d'abord une premiere version complete, detaillee et pedagogique du notebook, avec explications en profondeur. Quand le projet sera complet, creer ensuite une version plus concise et plus propre a proposer au professeur.
- Mode de travail prefere : apres chaque ajout important, l'etudiant lit et cherche a comprendre; il pose des questions si necessaire, puis demande principalement de poursuivre.
- Ajouter des commentaires dans le code quand un concept ou une operation risque d'etre nouveau pour l'etudiant; eviter les commentaires evidents.
- Travailler dans `projet_dota.ipynb`, remis a zero par l'etudiant.
- Des fichiers auxiliaires peuvent etre crees si cela aide l'organisation, mais le notebook soumis doit rester executable independamment et inclure les fonctions necessaires.
- Le code doit rester lisible et plausible pour un projet etudiant; les commentaires dans le code doivent surtout aider le lecteur du notebook, tandis que les explications detaillees se feront dans la conversation.
- Dans le notebook, eviter les accents sur les voyelles pour contourner le probleme d'affichage en `?`; garder simplement la voyelle non accentuee, meme si la grammaire devient moins correcte. Exemple : `frere`, pas `frre`.

## Avancement du notebook
- 2026-06-09 : `projet_dota.ipynb` a ete initialise avec 11 cellules.
- 2026-06-10 : ajout d'un bloc d'exploration detaille sur la repartition des classes, la comparaison train/validation, les objets difficiles, les tailles des boites, les tailles des images et une fonction de visualisation des boites orientees.
- 2026-06-10 : reprise dans un nouveau chat; `prompt.txt`, `Rules.md`, `Context.md`, `Subject.pdf` et `projet_dota.ipynb` ont ete relus/inspectes. Le notebook actuel contient 28 cellules, JSON valide, aucun caractere non ASCII, et les cellules de code passent sans erreur en execution non interactive.
- 2026-06-10 : ajout de la fin de la question 1 avec analyse de l'orientation des boites, discussion des algorithmes envisageables/inadaptes et risques methodologiques. Le notebook contient maintenant 35 cellules.
- Contenu actuel : titre, contexte, explication du format DOTA, imports, chemins relatifs, verification des dossiers, comptage des fichiers, lecture d'un fichier d'annotation, fonctions de parsing DOTA, analyses exploratoires principales, orientations des boites et discussion methodologique de la question 1.
- Verification executee hors notebook : les cellules de code passent sans erreur.
- Annotations chargees : 98 990 objets train, 28 853 objets validation, 127 843 objets au total.
- Orientation absolue des boites : mediane environ 47.17 degres en train et 52.28 degres en validation, ce qui confirme l'importance des boites orientees dans DOTA.
- Prochaine etape logique : commencer la question 2, donc le pretraitement, notamment le choix entre conserver les boites orientees ou convertir vers des boites horizontales selon le modele retenu.
