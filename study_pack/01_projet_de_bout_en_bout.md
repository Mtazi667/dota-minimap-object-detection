# Le projet DOTA de bout en bout

## Le probleme en une phrase

Le projet doit localiser et classifier des objets dans de grandes images
aeriennes, puis etudier causalement si le fait qu'un objet soit tres petit
reduit sa probabilite d'etre correctement detecte.

Il contient donc deux problemes differents :

- probleme predictif : ou sont les objets et quelle est leur classe ?
- probleme causal : quel serait l'effet du traitement "objet tres petit" sur
  la detection, apres ajustement sur des differences observables pertinentes ?

Une bonne performance predictive ne prouve pas un effet causal.

## Le pipeline a savoir dessiner

```text
DOTA-v1.0 brut
    |
    +-- images train et validation separees
    +-- annotations OBB a 8 coordonnees
    +-- source, GSD, difficult
    |
    v
Audit et variables derivees
    |
    +-- aire, ratio de forme, orientation
    +-- densite, classe dominante, type de scene
    +-- controles de validite
    |
    v
Selection stratifiee par image source
    |
    +-- 180 images train
    +-- 60 images validation
    +-- 15 classes couvertes
    |
    v
Tuilage 1024 x 1024
    |
    +-- train : stride 824, augmentation de couverture
    +-- validation : stride 1024, evaluation moins redondante
    +-- fragments ambigus rejetes
    +-- tuiles negatives echantillonnees
    |
    +---------------------+
    |                     |
    v                     v
labels HBB             labels OBB
    |                     |
    v                     v
YOLO26n HBB            YOLO26n OBB
baseline               modele principal
    |                     |
    +----------+----------+
               |
               v
Evaluation validation
    |
    +-- precision, recall, mAP50, mAP50-95
    +-- AP par classe
    +-- predictions visuelles
    +-- performance par taille et orientation
               |
               v
Matching OBB prediction-verite terrain
               |
               v
Table causale unique par objet
    |
    +-- D : tres petit
    +-- Y : classe correcte et IoU >= 0.50
    +-- X : classe, orientation, densite, source, GSD,
            ratio de forme, position du centre dans la tuile
               |
               v
Difference brute, g-computation, IPW, AIPW
               |
               v
Arbre causal honnete + foret sur pseudo-outcomes
               |
               v
Interpretation, incertitude, sensibilite et limites
```

## Pourquoi utiliser un sous-ensemble ?

DOTA est volumineux et les images sont grandes. Un sous-ensemble permet :

- une experience executable sur une RTX 4060 de 8 Go ;
- plusieurs modeles et controles dans un temps realiste ;
- une analyse causale par objet sans plusieurs jours de calcul ;
- une reproduction pedagogique.

Ce choix est acceptable seulement si la selection est defendable. Ici elle
est faite au niveau de l'image, avec couverture des 15 classes et une strate
`classe dominante x densite`. Les classes rares restent rares : le projet ne
pretend pas que la stratification annule le desequilibre.

## Pourquoi selectionner avant de tuiler ?

Si on tuile d'abord puis qu'on repartit les tuiles au hasard, deux tuiles de la
meme image peuvent se retrouver dans train et validation. Elles partagent le
meme paysage, la meme texture, la meme source et parfois les memes objets.

Le modele pourrait reconnaitre le contexte plutot que generaliser. La metrique
de validation serait alors trop optimiste.

La bonne unite de separation est l'image source. Toutes ses tuiles restent dans
le meme split.

## Choix de tuilage

### Taille 1024

Une tuile de 1024 pixels :

- est beaucoup plus petite qu'une image DOTA complete ;
- conserve assez de contexte autour des objets ;
- reste compatible avec la memoire GPU ;
- permet ensuite un redimensionnement vers 640 ou 1024 pour YOLO.

Une tuile trop petite coupe plus d'objets et perd le contexte. Une tuile trop
grande reduit les objets minuscules lors du redimensionnement et augmente le
cout memoire.

### Stride 824 en train

Le chevauchement de 200 pixels augmente la probabilite qu'un objet proche d'un
bord apparaisse entier dans au moins une tuile. Cela enrichit le train, mais
peut dupliquer un objet. Le manifeste conserve `object_id` et
`tile_instances_for_object` pour rendre cette duplication visible.

### Stride 1024 en validation

La validation doit etre la moins redondante possible. Un stride sans
chevauchement est donc prefere. Une derniere position est ajoutee pour couvrir
le bord final de l'image, ce qui peut creer un petit chevauchement. Pour
l'analyse causale, une seule instance est choisie par objet original.

## Gestion des objets aux bords

Pour chaque objet, on calcule la fraction de son aire conservee dans la tuile :

- fraction >= 0.70 : objet garde ;
- fraction entre 0.20 et 0.70 : tuile rejetee car le fragment est ambigu ;
- fraction < 0.20 : fragment ignore.

Si un objet partiellement coupe est garde, son intersection avec la tuile est
approximee par une nouvelle boite orientee minimale. Les coordonnees sont
ensuite clippees et normalisees entre 0 et 1.

Ce choix reduit les labels trompeurs. Sa limite est qu'il elimine des tuiles et
modifie la distribution des objets proches des bords.

## Pourquoi garder des tuiles negatives ?

Une tuile negative ne contient aucun objet annote. Sans tuiles negatives, le
modele apprendrait surtout que toute image contient forcement un objet. Un
echantillon de negatives lui apprend le fond et aide a reduire les faux
positifs.

On ne garde pas toutes les tuiles vides parce qu'elles domineraient le dataset
et augmenteraient fortement le temps d'entrainement.

## Deux representations, memes images

### HBB

La boite horizontale est derivee du minimum et maximum des quatre coins :

```text
xmin = min(x1, x2, x3, x4)
xmax = max(x1, x2, x3, x4)
ymin = min(y1, y2, y3, y4)
ymax = max(y1, y2, y3, y4)
```

Le label YOLO contient classe, centre x, centre y, largeur et hauteur. Cette
representation est simple, mais englobe souvent du fond pour un objet diagonal.

### OBB

Le label contient la classe et les quatre coins normalises. Il respecte mieux
la geometrie DOTA et permet une IoU orientee.

### Pourquoi la comparaison est equitable

- memes images sources ;
- memes tuiles ;
- memes splits ;
- meme graine ;
- meme famille YOLO26n ;
- meme pre-entrainement COCO du backbone ;
- aucune initialisation OBB deja entrainee sur DOTA.

La tete OBB est differente par necessite. La comparaison mesure donc surtout
l'effet pratique de la representation et de la tete orientee dans ce protocole.

## Ce que fait chaque fichier de code

### `src/dota_pipeline.py`

- lit les annotations DOTA ;
- calcule les variables derivees ;
- selectionne les images ;
- decoupe et annote les tuiles ;
- produit simultanement HBB et OBB ;
- valide les labels et la separation.

### `scripts/prepare_experiment.py`

Point d'entree pour construire le dataset experimental.

### `scripts/train_models.py`

Entraine HBB ou OBB, protege les anciens runs, permet une reprise depuis
`last.pt` et enregistre les metadonnees de reproductibilite.

### `src/experiment_analysis.py`

- calcule l'IoU polygonale ;
- associe predictions et objets par classe ;
- choisit une instance unique par objet ;
- estime les effets par plusieurs methodes ;
- construit l'arbre et la foret d'effets.

### `scripts/evaluate_experiment.py`

Orchestre les evaluations, construit la table causale, genere les CSV, JSON et
figures finales.

### `scripts/verify_offline_ready.py`

Controle que l'environnement de vol contient tout le necessaire.

## Correspondance avec les six questions

| Question | Reponse du projet |
|---|---|
| Q1 Exploration | classes, difficult, dimensions, aires, orientations, visualisations et choix d'algorithmes |
| Q2 Pretraitement | selection stratifiee, tuilage, nettoyage, HBB/OBB, variables derivees et manifeste |
| Q3 Prediction | baseline HBB, modele OBB, metriques globales/par classe et exemples |
| Q4 Causalite | D, Y, X, DAG, hypotheses et estimand |
| Q5 Estimation | brut, g-computation, IPW, AIPW, arbre/foret, IC et sensibilite |
| Q6 Conclusion | interpretation separee, limites, figures et reproductibilite |

## Questions de rappel

Repondre sans regarder le texte :

1. Pourquoi le split doit-il preceder le tuilage ?
2. Pourquoi un stride plus petit est-il utilise en train ?
3. Pourquoi ne pas garder tous les fragments de bord ?
4. Pourquoi conserver des tuiles negatives ?
5. Pourquoi HBB est-il une baseline utile mais imparfaite ?
6. Pourquoi ne pas initialiser OBB avec des poids deja ajustes sur DOTA ?
7. Pourquoi une instance unique est-elle choisie pour l'analyse causale ?
8. Quelle colonne relie une instance tuilee a l'objet original ?
9. Quelle partie du pipeline produit l'outcome causal ?
10. Pourquoi un bon mAP ne suffit-il pas pour Q5 ?
