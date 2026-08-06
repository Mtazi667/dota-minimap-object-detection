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

## Etat final verifie le 2026-07-29
- Environnement GPU local valide : Python 3.14.3, torch 2.11.0+cu130, CUDA 13.0 et RTX 4060.
- Sous-ensemble experimental termine : 180 images train, 60 validation, 912 tuiles train, 245 validation, 15 classes dans les deux splits et zero overlap d'image source.
- Trois runs termines a 20 epochs : HBB-640, OBB-640 et OBB-1024. Le modele principal est OBB-1024.
- Validation finale OBB-1024 : precision 0.574, recall 0.290, F1 0.385, mAP50 0.274 et mAP50-95 0.186.
- Table causale finale : 1 646 objets uniques issus de 53 images. Traitement = premier quartile train de l'aire relative; outcome = meme classe, confiance >= 0.25 et IoU OBB >= 0.50.
- Ajustement final : classe, orientation, log ratio de forme, densite, source, GSD et position du centre dans la tuile. La fraction conservee est exclue pour eviter un ajustement sur une variable de selection ou post-traitement.
- AIPW principal : -0.368, IC 95 % bootstrap groupe [-0.530; -0.166]. Les estimations ponctuelles de sensibilite restent negatives; le clipping 0.02 donne un IC qui traverse zero.
- Arbre causal honnete limite a une division : effet holdout plus negatif pour les tuiles de plus de 16.5 objets, resultat exploratoire avec seulement 12 images dans la feuille dense.
- `projet_dota.ipynb` contient 95 cellules et les six questions. Verification clean-kernel : 48 cellules code executees, zero erreur, zero execution count manquant et zero voyelle accentuee dans les sources.
- Livrables finaux : `output/pdf/rapport_final_dota.pdf` (14 pages), `output/pdf/guide_etude_vol_dota.pdf` (66 pages), `output/study_pack/guide_etude_dota.html` et `output/notebook/projet_dota.html`.
- Pack vol : programme actif de 8 heures, cours detection/causalite, chapitre de resultats reels, exercices, corrige, 60 questions orales et 80 flashcards.
- Verification hors ligne : `scripts/verify_offline_ready.py` passe integralement, tests `7 passed`, rendu visuel de toutes les pages PDF controle.

## Lecture portable verifiee le 2026-07-30
- L'etudiant utilisera un autre ordinateur uniquement pour tirer le depot et lire les sorties deja calculees; aucune reexecution du notebook n'est attendue pendant le vol.
- `projet_dota.ipynb` a ete reexecute integralement avant export : 48 cellules code executees et zero sortie d'erreur.
- Le noyau enregistre est maintenant le noyau portable `python3` et aucun chemin utilisateur absolu ne reste dans le notebook.
- `output/notebook/projet_dota.html` est un lecteur autonome : figures integrees, aucune ressource distante active, aucun besoin de Python, Jupyter, DOTA ou poids YOLO.
- `PORTABLE_READING.md` et `scripts/verify_portable_reader.py` documentent et controlent ce mode lecture.

## Reprise compacte pour la soutenance le 2026-08-05
- Creation progressive d'un nouveau notebook `projet_dota_soutenance.ipynb` afin de conserver intacte la version complete existante.
- Style demande : texte court et naturel, niveau etudiant debutant en machine learning, sans accumulation de commentaires ou de methodes.
- Comparaison predictive retenue : Faster R-CNN avec boites horizontales contre YOLO26n-OBB-1024 avec boites orientees.
- La causalite reste obligatoire selon le sujet, mais sera presentee avec une estimation ajustee simple et un arbre causal peu profond.
- Le notebook sera construit une section a la fois. Seul le bloc 0 (introduction, environnement et chemins relatifs) est commence a cette date.
- Les dossiers DOTA sont actuellement places directement dans la racine du depot : `Training Data`, `Validation Data` et `Testing Images`.
- 2026-08-05 : ajout de la premiere petite sous-partie de la question 1 dans `projet_dota_soutenance.ipynb` avec le chargement des listes de fichiers, un tableau de comptage et une interpretation courte.
- Comptage obtenu : 1 411 images et 1 411 annotations train, 458 images et 458 annotations validation, puis 937 images test sans annotations fournies.
- Verification avec le noyau `dota-gpu` : 3 cellules de code executees, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. La suite de la question 1 n'est pas encore ajoutee.
- 2026-08-05 : ajout de la sous-partie 1.2 dans `projet_dota_soutenance.ipynb`. Une fonction simple charge les coordonnees, la classe et l'indicateur `difficult` de chaque objet DOTA.
- Le tableau par classe contient 98 990 objets train et 28 853 objets validation. Les 15 classes sont presentes dans les deux ensembles; `ship` et `small-vehicle` dominent nettement.
- Verification avec le noyau `dota-gpu` : 5 cellules de code executees, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. Les tailles, orientations, difficultes et visualisations ne sont pas encore traitees dans ce notebook compact.
- 2026-08-05 : question 1 terminee dans `projet_dota_soutenance.ipynb`. Ajout des tailles d'images, surfaces et orientations des objets, indicateur `difficult`, deux exemples OBB/HBB, choix des algorithmes et risques methodologiques.
- Statistiques principales : taille mediane proche de 1 800 x 1 800 pixels; aire mediane de 760 px2 en train et 766 px2 en validation; orientation mediane de 47.175 et 52.275 degres; objets difficiles a 5.55 % et 6.62 %.
- Comparaison retenue : Faster R-CNN avec boites horizontales contre YOLO26n-OBB-1024 avec boites orientees. L'arbre causal reste reserve a l'analyse ulterieure des erreurs de detection.
- Verification finale de la question 1 : 22 cellules dont 9 cellules de code executees avec `dota-gpu`, zero erreur, aucun chemin utilisateur absolu, aucune source non ASCII et deux figures controlees visuellement. La prochaine partie est la question 2.
- 2026-08-05 : question 2 terminee dans `projet_dota_soutenance.ipynb`. Le notebook charge les metadonnees, controle les annotations, recharge le jeu prepare, verifie les formats OBB/HBB et construit une table d'analyse par objet unique.
- Controle brut : 15 classes connues, zero aire ou HBB invalide, 149 objets train et 22 validation partiellement hors image. Ils sont clippes pendant le tuilage. Les objets `difficult` sont conserves et identifies.
- Sous-ensemble confirme : 180 images train, 60 validation, 912 tuiles train et 245 validation, 15 classes dans les deux ensembles et zero image source partagee. Tuiles de 1 024 pixels, pas train 824 et pas validation 1 024.
- Table d'analyse : 6 292 objets uniques, dont 1 646 objets validation issus de 53 images. La colonne `detection_correcte` reste vide jusqu'aux predictions de la question 3.
- Verification finale de la question 2 : 37 cellules dont 14 cellules de code executees sans erreur avec la RTX 4060; noyau enregistre `dota-gpu`, aucun chemin utilisateur absolu et aucune source non ASCII. La prochaine partie est la question 3.
- 2026-08-05 : apres avoir choisi le Python generique dans VS Code, le notebook de soutenance a lance `C:\Python314\python.exe` au lieu de l'environnement `.venv_gpu` et la premiere cellule est restee bloquee. Les metadonnees ont ete remises sur `dota-gpu` et l'etat execute restaure : comptes 1 a 14 et zero erreur. Dans VS Code, choisir `Jupyter Kernel > Python (DOTA GPU)`.
- 2026-08-05 : debut de la question 3 dans `projet_dota_soutenance.ipynb`. Ajout de la formulation predictive, du protocole Faster R-CNN HBB contre YOLO26n-OBB-1024, du chargeur de donnees HBB et de la construction de la baseline.
- Baseline retenue : Faster R-CNN MobileNetV3-FPN avec transfert COCO, entree interne 640 pixels, batch 2 et 8 epochs prevues. Ce backbone reste assez leger pour les 8 Go de la RTX 4060.
- Controle Faster R-CNN : 912 tuiles train, 245 validation, 16 sorties avec l'arriere-plan, 18 943 083 parametres entrainables et perte de test finie a 3.2761. Les poids Faster R-CNN ne sont pas encore entraines; les poids YOLO26n-OBB-1024 sont presents.
- Verification de cette sous-partie : 46 cellules dont 19 cellules de code executees avec `dota-gpu`, comptes 1 a 19, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. La prochaine sous-partie est l'entrainement et la sauvegarde de la baseline Faster R-CNN.
- 2026-08-05 : baseline Faster R-CNN entrainee pendant 8 epochs. Un batch compose seulement de tuiles vides produisait des pertes ROI `NaN`; les 801 tuiles train et 212 validation contenant des objets sont donc utilisees pour la perte. Les tuiles vides restent disponibles pour l'evaluation finale des faux positifs.
- Entrainement stabilise en precision normale avec SGD, taux initial 0.001, warm-up pendant la premiere epoch, decroissance du taux et controle des gradients. Aucun batch avec objets n'a ete ignore pendant le run final.
- Meilleur poids Faster R-CNN : epoch 4, perte train 0.4321 et perte validation 0.5151. Les 8 epochs ont pris environ 5.6 minutes. Poids sauvegarde dans `runs/dota_experiment_v1/faster_rcnn_mobilenet_hbb/best_model.pth` et historique dans `training_history.csv`.
- Verification apres reutilisation des poids : 51 cellules dont 22 cellules de code executees avec `dota-gpu`, comptes 1 a 22, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. Les poids Faster R-CNN et YOLO26n-OBB-1024 sont tous les deux presents. La prochaine sous-partie est l'evaluation predictive comparee.
- 2026-08-05 : question 3 terminee dans `projet_dota_soutenance.ipynb`. Les deux modeles sont charges depuis leurs meilleurs poids puis evalues sur les 245 tuiles de validation, avec un tableau global, une comparaison par classe et deux exemples visuels.
- Resultats Faster R-CNN HBB : precision 0.308, rappel 0.163, F1 0.213, mAP50 0.142, mAP50-95 0.060 et 17.1 ms par image. Resultats YOLO26n-OBB-1024 : precision 0.572, rappel 0.291, F1 0.386, mAP50 0.272, mAP50-95 0.185 et 7.9 ms par image.
- YOLO26n-OBB-1024 est conserve comme modele principal. Les predictions Faster R-CNN sont mises en cache dans `runs/dota_experiment_v1/faster_rcnn_mobilenet_hbb/validation_predictions.pt`; les mesures YOLO sont dans `outputs/analysis/yolo_obb_metrics.csv` et `yolo_obb_class_metrics.csv`.
- Verification finale de la question 3 : 62 cellules dont 29 cellules de code executees avec `dota-gpu`, comptes 1 a 29, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. La prochaine partie est la question 4 sur la formulation causale.
- 2026-08-06 : question 4 terminee dans `projet_dota_soutenance.ipynb`. La question distingue prediction et causalite, definit le traitement, l'outcome, la population, l'ATE, le DAG et les hypotheses d'identification.
- Traitement causal : objet tres petit selon le premier quartile train de l'aire relative de tuile, seuil 0.00050354. Outcome : prediction de meme classe, confiance >= 0.25 et IoU OBB >= 0.50.
- Population causale preparee : 1 646 objets uniques de validation issus de 53 images, avec 623 objets traites, 1 023 controles et 716 detections correctes. Le matching objet-prediction existant est controle contre les identifiants de validation.
- Le DAG retient la classe, le GSD, la source, l'orientation, le ratio de forme, la densite et la position dans la tuile comme variables pre-prediction. La confiance, l'IoU et la fraction conservee sont exclues de l'ajustement.
- Verification finale de la question 4 : 67 cellules dont 31 cellules de code executees avec `dota-gpu`, comptes 1 a 31, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. Tests : 7 passes. La prochaine partie est la question 5 avec une estimation ajustee simple et un arbre causal peu profond.
- 2026-08-06 : question 5 terminee dans `projet_dota_soutenance.ipynb`. La version compacte recalcule la difference brute, la g-computation, l'IPW et l'AIPW a partir des predictions de nuisance cross-fittees, puis construit des IC 95 % par bootstrap des images source.
- Resultats causaux : taux observe 30.7 % chez les objets tres petits contre 51.3 % chez les controles; difference brute -0.207, g-computation -0.170, IPW -0.620 et AIPW principal -0.368 avec IC 95 % [-0.530; -0.166].
- Diagnostic : 76.5 % des propensions sont dans [0.1, 0.9]. Les sensibilites ponctuelles restent negatives, mais le clipping 0.02 produit un IC qui traverse zero.
- Arbre causal honnete : une moitie des images choisit la division et l'autre estime les feuilles. Division unique a 16.5 objets par tuile; effet -0.131 pour les tuiles moins denses et -0.457, IC 95 % [-0.646; -0.221], pour les tuiles plus denses. La feuille dense ne contient que 12 images et reste exploratoire.
- Verification finale de la question 5 : 76 cellules dont 35 cellules de code executees avec `dota-gpu`, comptes 1 a 35, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. Tests : 7 passes. La prochaine partie est la question 6 avec interpretation generale, conclusion et limites.
- 2026-08-06 : question 6 terminee dans `projet_dota_soutenance.ipynb`. Ajout d'un tableau de synthese, des interpretations predictive et causale, des limites, des ameliorations possibles, des controles de reproductibilite et de la conclusion finale.
- Conclusion retenue : YOLO26n-OBB-1024 est le meilleur modele parmi les configurations testees, sans attribuer l'ecart aux seules boites orientees. L'effet AIPW de la tres petite taille reste interprete avec prudence comme un effet observationnel sous hypotheses.
- Limites explicites : sous-ensemble DOTA, comparaison de modeles non factorielle, absence de labels test locaux, desequilibre des classes, 53 images causales, sensibilite aux seuils et bootstrap sans reentrainement complet des nuisances.
- Les huit controles finaux passent : donnees presentes, splits sans fuite, 15 classes, poids disponibles, 245 tuiles evaluees, deux modeles compares, 1 646 objets causaux uniques et estimations completes.
- Etat final du notebook de soutenance : 83 cellules dont 37 cellules de code executees avec `dota-gpu`, comptes 1 a 37, zero erreur, aucun chemin utilisateur absolu et aucune source non ASCII. Les six questions sont presentes et les tests donnent 7 passes. Le notebook est techniquement termine; restent seulement les options d'export ou de preparation orale.

## Preparation de la presentation de 10 minutes le 2026-08-06
- Consigne du professeur : PowerPoint de deux pages maximum, choix de l'algorithme en 1 minute, fonctionnement en 4 ou 5 etapes en 3 minutes, formule mathematique en 3 minutes, code et resultats preliminaires en 2 minutes, pour 10 minutes au total.
- Angle retenu avec l'etudiant : YOLO26n-OBB-1024, formule de l'IoU orientee, comparaison avec Faster R-CNN HBB et demonstration d'inference sans entrainement en direct.
- Decision pedagogique : ne pas modifier le notebook principal. Le nouveau notebook `preparation_soutenance_yolo_obb.ipynb` concentre CNN, backbone/neck/head, YOLO-OBB, IoU, confiance, NMS, metriques, resultats, questions probables, chronometrage et checklist.
- La derniere cellule du notebook pedagogique est autonome. Elle charge directement le meilleur poids, utilise la tuile fixe `P0249__x0_y653.jpg`, choisit CUDA ou CPU et affiche les OBB. Test clean-kernel avec `dota-gpu` : 3 cellules de code executees, zero erreur, 16 detections et environ 0.62 seconde pour la cellule lors du controle.
- Image de secours : `outputs/presentation_assets/demo_p0249_yolo_obb.jpg`. Illustration IoU : `outputs/presentation_assets/iou_obb_diagram.png`.
- PowerPoint final : `outputs/soutenance_yolo_obb.pptx`, exactement deux diapositives. Diapositive 1 : justification et pipeline en cinq etapes. Diapositive 2 : IoU orientee, comparaison F1/mAP50-95, vitesse et limite de rappel.
- Fiche de repetition : `fiche_orale_soutenance_yolo_obb.md`.
- Verification : rendu individuel des deux diapositives controle visuellement, deux slides detectees, zero objet hors canvas, zero placeholder, archive PPTX valide et tests du projet `7 passed`.
- 2026-08-06 : creation de `script_soutenance_yolo_obb_10min.txt`, texte oral complet synchronise avec les deux diapositives et la cellule de demonstration. Le style reprend uniquement la structure du fichier de reference `C:\Users\Tazi\Downloads\script_soutenance_15min.txt` : sections chronometrees, paragraphes naturels, transitions et reperes temporels; aucun contenu du projet de reference n'est reutilise.
- Le script vise une fin vers 9 min 50 et contient un plan de secours court si l'inference ne peut pas etre relancee immediatement.
