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

## Presentation intermediaire
- Presentation obligatoire le jeudi 9 juillet 2026 a midi.
- Duree attendue : 5 a 10 minutes.
- Contenu demande : presenter l'avancement du projet et expliquer en profondeur un algorithme d'apprentissage automatique utilise dans le projet.
- Besoin avant la fin du mercredi 8 juillet 2026 : creer une cheatsheet personnelle pour aider l'etudiant a presenter son avancement, son choix d'algorithme, les concepts essentiels et les reponses possibles aux questions.
- Algorithme recommande pour cette presentation : YOLO / YOLO-OBB, car il correspond directement a la detection d'objets dans DOTA et au choix provisoire de conserver les boites orientees.
- Preference de l'etudiant pour la cheatsheet : lisible sur telephone, tres complete, mais pas un texte a lire mot pour mot; utiliser des points de repere, phrases-clefs, questions/reponses et structure de presentation.
- L'etudiant souhaite presenter principalement le notebook et etre transparent sur l'usage de l'IA comme outil d'apprentissage, verification et comprehension.

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
- 2026-07-08 : debut de la question 2 dans `projet_dota.ipynb`. Ajout du pretraitement des annotations : encodage des classes, fusion avec les tailles d'images, coordonnees normalisees, representation horizontale type centre-largeur-hauteur, controle qualite et sous-ensemble `training_ready_objects`.
- Decision provisoire : conserver les boites orientees comme representation principale, mais preparer aussi les boites horizontales pour une baseline simple ou l'analyse des erreurs.
- Controle qualite du pretraitement : 0 boite avec largeur/hauteur non positive; 171 objets ont au moins une coordonnee normalisee hors `[0, 1]` et sont exclus provisoirement de `training_ready_objects`. Cela laisse 127 672 objets utilisables pour un premier export.
- 2026-07-08 : ajout d'un export provisoire YOLO-OBB dans le notebook. Le format utilise est `class_index x1 y1 x2 y2 x3 y3 x4 y4` avec coordonnees normalisees, conforme a la documentation Ultralytics consultee le 2026-07-08.
- Export YOLO-OBB verifie hors notebook : 1 407 fichiers label train pour 98 841 objets exportes; 456 fichiers label validation pour 28 831 objets exportes; fichier `prepared_data/dota_yolo_obb/dota_yolo_obb.yaml` cree.
- Les images ne sont pas encore liees/copiees dans `prepared_data/dota_yolo_obb/images/`; la cellule optionnelle `CREATE_IMAGE_LINKS = False` permet de le faire plus tard en utilisant d'abord des liens durs, puis des copies si necessaire.
- `.gitignore` ignore maintenant `prepared_data/`, car ces exports sont regenerables.
- 2026-07-08 : ajout d'une section pedagogique `Modele predictif envisage : YOLO-OBB` expliquant YOLO, YOLO-OBB, la difference avec les boites horizontales, les erreurs apprises pendant l'entrainement et pourquoi cet algorithme convient a DOTA. Cette section doit servir de base a la presentation du 2026-07-09.
- 2026-07-08 : creation de `cheatsheet_presentation_2026-07-09.md`, fiche personnelle complete et lisible sur telephone pour la presentation. Elle couvre l'avancement, le notebook a montrer, YOLO, YOLO-OBB, le pretraitement, l'usage de l'IA et les questions probables.
- 2026-07-09 : apres la presentation, ajout dans le notebook d'une verification de l'environnement d'entrainement et d'une cellule d'entrainement YOLO-OBB desactivee par defaut (`RUN_YOLO_OBB_TRAINING = False`).
- Verification environnement du 2026-07-09 : `nvidia-smi` voit la RTX 4060, mais le Python courant `C:\Python314\python.exe` utilise `torch 2.11.0+cpu`; `torch.cuda.is_available()` vaut `False`. Ultralytics 8.4.37 est installe, mais il faut forcer `YOLO_CONFIG_DIR` vers `.ultralytics_config` dans le projet pour eviter un probleme de permission avec `AppData\Roaming\Ultralytics`.
- La documentation Ultralytics consultee le 2026-07-09 presente les modeles OBB avec suffixe `-obb`, par exemple `yolo26n-obb.pt`, et le format label `class_index x1 y1 x2 y2 x3 y3 x4 y4`.
- Retour de presentation du 2026-07-09 : la presentation s'est bien passee, mais les questions du professeur ont revele un besoin prioritaire de preparation orale sur les concepts ML/causalite. Questions posees : schema causal `smoking`, `cancer`, `cancer gene`; explication de l'IoU dans le contexte de YOLO.
- Nouvelle priorite : identifier puis etudier les concepts omis ou insuffisamment maitrises avant de pousser trop loin l'entrainement, afin que l'etudiant puisse defendre le projet en presentiel sans dependance immediate a l'IA.
- 2026-07-09 : creation de `concepts_oraux_ml_causalite.md`, un audit des concepts a maitriser pour l'oral. Le fichier priorise les notions en detection, YOLO, causalite, apprentissage supervise, deep learning et methodologie experimentale.
- Preference d'etude mise a jour le 2026-07-09 : eviter les longues sections de type titre + bloc explicatif. Preferer des formats courts et actifs : question du prof, reponse en 20 secondes, intuition, mini-exemple, piege, lien avec DOTA, et lien explicite avec le notebook/pipeline actuel. Ne pas y passer trop de temps; garder un equilibre entre preparation orale et avancement du projet.
- Preference confirmee le 2026-07-11 : ne pas produire de gros blocs de concepts directement dans le chat. Les explications conceptuelles doivent etre ajoutees progressivement dans `concepts_oraux_ml_causalite.md`, par petits lots de qualite, sauf demande explicite contraire.
- 2026-07-09 : ajout dans `concepts_oraux_ml_causalite.md` d'un suivi rapide et des trois premieres cartes orales : IoU, true positive/false positive/false negative, precision/recall.
- 2026-07-11 : ajout progressif dans `concepts_oraux_ml_causalite.md` de trois cartes orales supplementaires : mAP, confidence score et NMS, avec lien explicite au pipeline DOTA/YOLO-OBB.
- 2026-07-11 : correction de l'ordre des cartes dans `concepts_oraux_ml_causalite.md` et ajout de deux cartes YOLO pratiques : loss YOLO et difference YOLO classique vs YOLO-OBB. Les blocs `Evaluation detection` et `YOLO pratique` ont maintenant un premier passage couvert, mais doivent etre pratiques oralement.
- Rythme retenu : couvrir 2 ou 3 cartes orales, puis revenir au projet si le bloc est suffisamment clair; continuer a donner progressivement ou on en est et ce qu'il reste.
- 2026-07-11 : ajout dans `projet_dota.ipynb` d'une section `Evaluation prevue du detecteur` apres la cellule d'entrainement preparee. Elle explique IoU, true positive/false positive/false negative, precision, recall et mAP avec deux petits exemples executables. Objectif : faire avancer la question 3 tout en renforcant les concepts oraux dans le contexte DOTA/YOLO-OBB.
- Verification du 2026-07-11 : notebook JSON valide, aucune source non ASCII, execution complete hors notebook sans erreur. Les exemples donnent IoU proche = 0.741, IoU eloignee = 0.0, precision/recall pour deux scenarios simples.
- 2026-07-11 : ajout dans `projet_dota.ipynb` d'une section `Preparation de la question causale`. Elle distingue prediction et causalite, definit traitement/outcome/variables d'ajustement dans le contexte DOTA, propose trois questions causales candidates et donne un DAG provisoire pour l'effet de la petite taille sur l'erreur du detecteur.
- Verification du 2026-07-11 : execution complete sans erreur. Variables candidates creees dans `causal_objects` : `small_object` environ 25.2%, `large_object` environ 25.0%, `diagonal_orientation` environ 25.3%. Outcome causal final encore indisponible tant que YOLO-OBB n'a pas produit de predictions/erreurs.
- Prochaine etape logique : soit creer les cartes orales causalite (DAG, confounder, mediator, collider), soit reprendre le projet technique en preparant les images YOLO et l'environnement CUDA.
