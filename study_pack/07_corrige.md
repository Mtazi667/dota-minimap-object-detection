# Corrige des exercices

## Detection

### D1

```text
union = 120 + 80 - 50 = 150
IoU = 50 / 150 = 0.333
```

La detection ne passe pas le seuil 0.50.

### D2

Une prediction ne peut valider qu'un objet et un objet ne doit recevoir qu'un
TP. La NMS peut supprimer B avant l'evaluation si les deux boites se
chevauchent assez. Sinon le matching associe normalement A a l'objet et B reste
non associee, donc peut compter comme FP.

### D3

```text
FP = 80 - 52 = 28
FN = 65 - 52 = 13
precision = 52 / 80 = 0.650
recall = 52 / 65 = 0.800
F1 = 2 * 0.65 * 0.80 / (0.65 + 0.80) = 0.717
```

### D4

En general, precision monte et recall baisse parce que moins de predictions
sont gardees. Sur un petit echantillon, la composition et l'ordre des scores
peuvent produire des variations non monotones de certaines statistiques.

### D5

- AP : resume la courbe precision-recall d'une classe.
- mAP : moyenne les AP des classes.
- mAP50 : matching avec seuil IoU 0.50.
- mAP50-95 : moyenne plusieurs seuils, donc localisation plus exigeante.
- AP par classe : revele les classes cachees par la moyenne.

### D6

La HBB contient plus de fond autour du rectangle diagonal. L'OBB suit son axe.
La HBB peut augmenter le chevauchement avec le fond et des voisins, degrader la
precision geometrique et rendre la NMS plus confuse. OBB peut quand meme moins
bien scorer si la tete orientee, plus complexe, est sous-entrainee ou si la
metrique orientee est plus stricte.

### D7

Le decalage represente la moitie de la largeur du petit objet mais seulement
2.5 % de celle du grand. La fraction d'intersection chute donc beaucoup plus
pour le petit objet.

### D8

Il faut verifier validation, pas seulement train. La loss n'est pas la mAP, un
overfitting est possible, les classes peuvent reagir differemment, et le
checkpoint a meilleure validation peut preceder la derniere epoque.

## Pipeline

### P1

Les tuiles partagent le meme paysage et parfois des objets. La validation
reconnait des motifs deja vus. Il faut d'abord repartir les images sources,
puis tuiler chaque split separement.

### P2

```text
position finale = 2500 - 1024 = 1476
```

Les positions sont `[0, 824, 1476]`. La derniere est forcee pour couvrir le
bord et chevauche la tuile precedente de `824 + 1024 - 1476 = 372` pixels.

### P3

- 95 % : garde ;
- 75 % : garde ;
- 45 % : zone ambigue, tuile rejetee ;
- 10 % : fragment ignore.

### P4

Les negatives enseignent le fond et reduisent les FP. Toutes les garder ferait
dominer le train par le fond. L'erreur directement concernee est le faux
positif.

### P5

La stratification augmente la couverture des classes et scenes. Elle ne cree
pas d'exemples pour une classe vraiment rare et ne garantit pas des comptes
egaux. Une validation artificiellement equilibree ne representerait plus la
population sans une definition explicite de la metrique cible.

### P6

Ce n'est pas une fuite si les deux tuiles restent train. La duplication change
le poids de l'objet dans la loss et doit etre tracee. En causalite, garder deux
instances donnerait plus de poids a certains objets et violerait l'unite
d'analyse.

### P7

Exemples : code, donnees/split, graine, versions Python/PyTorch/Ultralytics,
GPU/CUDA, poids de depart, architecture, epochs, batch, taille d'image,
optimiseur, patience, augmentations, seuils, checkpoint et temps.

### P8

OBB beneficie d'une information specifique a DOTA avant l'experience. Il faut
partir du meme pre-entrainement general, ici COCO, et ajuster chaque tete sur
les memes tuiles avec le meme split et la meme graine.

## Causalite

### C1

```text
Gene -> Smoking -> Cancer
  \----------------> Cancer
```

Traitement : smoking. Outcome : cancer. Confounder : gene. Pour l'effet total,
ajuster sur gene sous les hypotheses du DAG.

### C2

```text
Tres petit <- Classe -> Detection correcte
```

Les classes ont des tailles et difficultes differentes. Sans ajustement, leur
composition contamine la comparaison. L'ajustement compare des objets ayant des
profils de classe et autres covariables plus semblables.

### C3

Exemple : qualite des features extraites. Petite taille reduit l'information,
qui influence la detection. Ajuster dessus retire ce chemin et vise plutot un
effet direct.

### C4

```text
Petite taille -> Tuile selectionnee <- Scene complexe
```

Conditionner sur la selection peut rendre petite taille et complexite associees
dans l'echantillon retenu, ouvrant un chemin vers l'outcome.

### C5

Ajustements plausibles : classe, orientation, source, GSD, densite. A eviter :
confiance, IoU et feature interne car post-traitement/outcome ; aire exacte car
elle definit D dans ce protocole.

### C6

`Y(1)` est la detection si l'objet etait tres petit, `Y(0)` s'il ne l'etait
pas. L'objet n'a qu'une condition observee ; l'autre est contrefactuelle.

### C7

- Exchangeabilite : pas de confounding residuel apres X ; menace par flou non
  observe.
- Positivite : les deux traitements possibles pour chaque profil ; menace par
  classes toujours petites.
- Coherence : version de D suffisamment definie ; menace par plusieurs
  mecanismes de petitesse.
- Interference : un objet n'affecte pas l'autre ; menace par NMS.
- Absence de fuite : outcome hors train ; menace par tuiles de meme source.

### C8

```text
0.18 - 0.43 = -0.25
```

Difference de 25 points de pourcentage. Elle peut refleter classe, GSD,
orientation ou densite, pas seulement la taille.

### C9

0.52 a un meilleur support. A 0.03, un traite aurait un poids environ 33.3.
Le clipping borne le poids et la variance, mais ne fabrique pas de controles
comparables et ne corrige pas les confounders non observes.

### C10

Poids `1/0.20 = 5`. L'objet represente plusieurs profils similaires parce que
son traitement est rare. Des poids trop grands rendent l'estimation dominee par
quelques observations.

### C11

AIPW combine predictions d'outcome et propension, puis utilise les residus pour
corriger. Double robustesse couvre une mauvaise specification de l'un des deux
modeles sous conditions, pas les variables omises ou la mauvaise positivite.

### C12

Les predictions in-sample overfittent et donnent des pseudo-outcomes biaisés.
Grouper par image evite que des objets partageant le meme contexte traversent
les folds. C'est une logique hors echantillon.

### C13

La moitie structure choisit les divisions, la moitie estimation mesure les
feuilles. Cela evite d'utiliser le meme bruit deux fois. Faible profondeur et
grandes feuilles reduisent variance et decouvertes artificielles.

### C14

Le signe negatif est stable. L'amplitude depend du niveau d'exigence
geometrique. Conclusion : la penalite semble robuste en direction, mais sa
taille exacte est sensible a la definition de l'outcome.

## Sorties reelles

Les reponses R1 a R6 changent avec les runs. Elles doivent etre calculees
directement depuis `outputs/analysis/`. La bonne correction est une reponse qui
cite :

- la valeur exacte ;
- le denominteur ou l'effectif ;
- l'incertitude ;
- une explication plausible ;
- au moins une limite.

## Grille du mini-examen

Attribuer 2 points par item :

- 0 : absent ou faux ;
- 1 : idee correcte mais incomplete ;
- 2 : definition, lien DOTA et limite.

Score maximal : 30.

- 26-30 : solide ;
- 21-25 : defendable, quelques fragilites ;
- 15-20 : revoir les cartes prioritaires ;
- moins de 15 : reprendre le parcours reduit de 3 heures.
