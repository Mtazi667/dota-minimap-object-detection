# Resultats reels et interpretation defendable

Ce chapitre contient les chiffres finaux du projet. Il faut pouvoir les citer
sans les apprendre comme une liste isolee. Pour chaque valeur, savoir repondre
a quatre questions :

1. quelle population est mesuree ?
2. quelle definition est utilisee ?
3. quelle conclusion est permise ?
4. quelle limite empeche de generaliser trop vite ?

## 1. Protocole final en une minute

Le sous-ensemble contient :

- 180 images source pour l'entrainement ;
- 60 images source pour la validation ;
- 15 classes presentes dans les deux splits ;
- 912 tuiles train et 245 tuiles validation ;
- 6 515 annotations tuilees train et 1 972 validation ;
- zero image source commune entre train et validation.

Les images sont decoupees en tuiles de 1024 pixels. Le train utilise un stride
de 824 pixels et la validation un stride de 1024 pixels. Les fragments
conservant au moins 70 % de l'objet sont gardes. Les fragments ambigus entre
20 % et 70 % provoquent le rejet de la tuile concernee.

Trois detecteurs YOLO26n ont ete entraines pendant 20 epochs avec la graine 42 :

- une baseline HBB a 640 ;
- un OBB a 640 ;
- un OBB a 1024.

La validation finale autorise jusqu'a 1 000 detections par tuile. Les metriques
globales sont calculees par Ultralytics sur les 245 tuiles. L'outcome causal
utilise ensuite un seuil de confiance de 0.25 et une IoU orientee de 0.50.

## 2. Resultats predictifs globaux

| Modele | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference par image |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-HBB-640 | 0.451 | 0.283 | 0.347 | 0.257 | 0.145 | environ 3.0 ms |
| YOLO26n-OBB-640 | 0.256 | 0.299 | 0.276 | 0.239 | 0.146 | environ 5.3 ms |
| YOLO26n-OBB-1024 | 0.574 | 0.290 | 0.385 | 0.274 | 0.186 | environ 7.6 ms |

### Calculs a savoir refaire

Gain absolu de mAP50-95 de OBB-1024 par rapport a OBB-640 :

```text
0.186 - 0.146 = 0.040 point
```

Gain relatif :

```text
(0.186 / 0.146 - 1) x 100 = environ 27 %
```

Par rapport a HBB-640, le gain relatif de mAP50-95 est environ 28 %.

Le cout est une inference environ 44 % plus lente que OBB-640 :

```text
7.6 / 5.3 - 1 = environ 0.44
```

Les temps dependent du GPU et de l'etat de la machine. Ils decrivent ce run sur
RTX 4060, pas une constante universelle.

### Interpretation correcte

OBB-1024 est le meilleur modele de ce protocole selon mAP50-95. Son avantage
vient surtout d'une precision plus forte. Son recall reste proche de 0.29 :
environ sept objets sur dix ne contribuent donc pas a un rappel correct au
point de fonctionnement resume par la metrique globale.

La resolution 1024 aide probablement parce qu'un objet minuscule conserve plus
de pixels apres redimensionnement. Ce resultat soutient l'hypothese de
resolution, mais il ne permet pas d'affirmer que 1024 sera toujours optimal.

HBB-640 et OBB-640 ont presque la meme mAP50-95. Donc, dans cette experience,
conserver l'orientation sans augmenter la resolution n'a pas suffi a creer un
gain global clair. Il ne faut pas conclure que les OBB sont inutiles : ils
restent geometriquement plus fideles et sont necessaires a l'IoU orientee de
l'analyse causale.

## 3. Resultats par classe

Pour YOLO26n-OBB-1024, les meilleures AP50-95 sont :

| Rang | Classe | AP50-95 |
|---:|---|---:|
| 1 | tennis-court | 0.544 |
| 2 | plane | 0.384 |
| 3 | ship | 0.325 |
| 4 | large-vehicle | 0.249 |
| 5 | roundabout | 0.232 |

Les classes les plus faibles sont :

| Classe | AP50-95 |
|---|---:|
| helicopter | 0.000 |
| basketball-court | 0.032 |
| soccer-ball-field | 0.082 |
| harbor | 0.095 |

Ces differences combinent plusieurs mecanismes :

- nombre d'exemples ;
- taille apparente ;
- contraste ;
- densite de scene ;
- variabilite intra-classe ;
- facilite de localiser precisement la forme ;
- courte duree d'entrainement.

Une AP faible pour `helicopter` ne prouve pas que la classe est
intrinsequement impossible. La validation ne contient que tres peu
d'helicopteres et le modele a peu d'information pour apprendre et evaluer cette
classe.

Une AP elevee pour `tennis-court` est plausible parce que la forme est grande,
reguliere et visuellement distinctive. Cela reste une explication
interpretative, pas une estimation causale de la classe.

## 4. Taille, orientation et analyse descriptive

Les groupes de taille utilisent les quartiles train. Sur les 1 646 objets
validation uniques de la table causale :

| Groupe | Objets | Taux de detection correcte |
|---|---:|---:|
| tres petit | 623 | 0.307 |
| petit | 386 | 0.544 |
| moyen | 333 | 0.456 |
| grand | 304 | 0.536 |

Le groupe tres petit est clairement le plus faible. En revanche, les quatre
groupes ne suivent pas une relation parfaitement monotone. `petit` depasse
`moyen` et `grand` dans cette validation. La composition en classes, scenes et
orientations change entre groupes. Un tableau descriptif ne maintient pas les
autres variables constantes.

La heatmap `detection_by_size_orientation.png` affiche dans chaque case :

- le taux de detection correcte ;
- l'effectif `n`.

Il faut toujours lire les deux. Par exemple, une case a 0.80 avec peu
d'observations est moins stable qu'une case a 0.50 avec plusieurs centaines
d'objets.

Le groupe d'orientation 75-90 degres a un taux global eleve, mais il contient
beaucoup d'objets et une composition particuliere en classes. Ne pas dire :
"une forte orientation ameliore causalement la detection". Ce serait confondre
un contraste descriptif avec un effet causal.

## 5. Construction de la question causale

### Traitement D

```text
D = 1 si aire relative de l'objet <= 0.0005035400
```

Le seuil est le premier quartile des objets train uniques. Il represente
environ 0.050 % de la surface d'une tuile. Pour une tuile 1024 x 1024, cela
correspond a environ 528 pixels carres, soit un carre equivalent d'environ
23 x 23 pixels.

Cette equivalence aide l'intuition. Elle ne transforme pas tous les objets en
carres et ne remplace pas l'aire polygonale reelle.

### Outcome Y

```text
Y = 1 si une prediction OBB :
    - a la meme classe ;
    - a une confiance >= 0.25 ;
    - a une IoU orientee >= 0.50.
```

Le matching est un-a-un et maximise l'IoU dans chaque classe. Une prediction ne
peut donc pas sauver plusieurs objets.

### Population et estimand

- 1 646 objets originaux uniques ;
- 53 images source validation contenant ces objets ;
- 623 objets traites, soit 37.8 % ;
- ATE sur la probabilite de detection correcte.

La validation contient 60 images selectionnees, mais 53 contribuent a la table
causale finale. Les images sans objet unique eligible ne contribuent pas a cet
estimand.

### Variables d'ajustement

L'ensemble X contient :

- classe ;
- orientation absolue ;
- log du ratio de forme ;
- densite d'objets dans la tuile ;
- source d'image ;
- GSD ;
- position du centre de l'objet dans la tuile.

L'aire exacte est exclue parce qu'elle definit D. La fraction conservee est
exclue de l'ajustement final car elle est liee a la selection par tuilage et
peut se situer apres la taille dans le mecanisme etudie. La confiance et l'IoU
sont post-traitement ou definissent Y.

## 6. Contraste observe avant ajustement

| Groupe | Objets | Taux Y=1 |
|---|---:|---:|
| Tres petit D=1 | 623 | 0.307 |
| Controle D=0 | 1 023 | 0.513 |

Difference brute :

```text
0.307 - 0.513 = -0.207
```

Le groupe tres petit a donc environ 20.7 points de pourcentage de detection
correcte en moins dans les donnees observees. Cette difference n'est pas encore
un effet causal parce que les groupes different aussi en classe, GSD, densite
et autres covariables.

## 7. Estimateurs causaux compares

| Methode | Effet | IC 95 % bootstrap image |
|---|---:|---:|
| Difference brute | -0.207 | [-0.520 ; 0.052] |
| G-computation | -0.170 | [-0.260 ; -0.096] |
| IPW | -0.620 | [-0.860 ; -0.352] |
| AIPW doublement robuste | -0.368 | [-0.530 ; -0.166] |

### Phrase principale a apprendre

> Sous les hypotheses d'identification, la condition tres petite est associee
> a une diminution moyenne estimee de 36.8 points de pourcentage de la
> probabilite de detection correcte. L'intervalle bootstrap groupe a 95 % est
> [-53.0 ; -16.6] points et exclut zero dans la specification principale.

Le mot `associee` rappelle que le traitement n'est pas randomise. Le mot
`moyenne` rappelle qu'il s'agit d'un ATE dans la population analysee. Le mot
`points` evite de confondre difference absolue et baisse relative.

### Pourquoi les quatre valeurs different-elles ?

- la difference brute ne rend pas les groupes comparables ;
- la g-computation depend des modeles d'outcome ;
- l'IPW depend fortement des inverses de propension ;
- l'AIPW combine outcome et propension avec une correction residuelle.

L'IPW tres negatif et l'ecart entre estimateurs signalent une fragilite de
support ou de modelisation. Il serait incorrect de dire que l'AIPW "revele la
verite". Il fournit l'estimation principale selon le protocole choisi.

## 8. Positivite et propension

La part des objets dont la propension tronquee est entre 0.10 et 0.90 vaut
76.5 %. Cela montre un recouvrement reel, mais imparfait.

Details utiles :

- 205 objets atteignent la borne basse 0.05 ;
- 42 objets atteignent la borne haute 0.95 ;
- mediane de propension chez D=0 : 0.251 ;
- mediane de propension chez D=1 : 0.625.

Les profils traites et controles restent donc differents. Le clipping evite des
poids infinis, mais change aussi l'estimand pratique et introduit un compromis
biais-variance.

## 9. Analyse de sensibilite

| Specification | Effet AIPW | IC 95 % |
|---|---:|---:|
| IoU >= 0.40 | -0.374 | [-0.518 ; -0.190] |
| IoU >= 0.50 | -0.368 | [-0.519 ; -0.183] |
| IoU >= 0.60 | -0.332 | [-0.480 ; -0.141] |
| Clipping 0.02 | -0.274 | [-0.507 ; 0.085] |
| Clipping 0.10 | -0.388 | [-0.504 ; -0.246] |
| Graine nuisances 7 | -0.370 | [-0.535 ; -0.155] |
| Graine nuisances 99 | -0.366 | [-0.522 ; -0.177] |

Le signe reste negatif dans toutes les estimations ponctuelles. Les seuils IoU
et les graines des modeles nuisances donnent des amplitudes proches.

La specification de clipping 0.02 produit toutefois un intervalle qui traverse
zero. C'est important : la conclusion ponctuelle est stable en signe, mais la
force inferentielle depend du traitement des propensions extremes. Cette
nuance doit apparaitre dans une bonne presentation orale.

## 10. Arbre causal honnete

L'arbre est volontairement conservateur :

- une moitie des images choisit une seule division ;
- chaque feuille de structure doit etre grande ;
- l'autre moitie estime les effets des feuilles ;
- les intervalles reechantillonnent les images.

La division choisie est :

```text
nombre d'objets dans la tuile <= 16.5
```

Resultats sur la moitie holdout :

| Sous-groupe | Objets | Images | Effet | IC 95 % |
|---|---:|---:|---:|---:|
| Densite <= 16.5 | 296 | 20 | -0.131 | [-0.640 ; 0.474] |
| Densite > 16.5 | 795 | 12 | -0.457 | [-0.646 ; -0.221] |

L'effet semble plus negatif dans les tuiles denses. Le sous-groupe dense ne
contient cependant que 12 images source malgre 795 objets. Il ne faut donc pas
transformer cette coupure en loi universelle. Elle constitue une hypothese
d'heterogeneite a verifier sur davantage d'images.

Le mot `honnete` ne veut pas dire que le resultat est certainement vrai. Il
veut dire que les observations choisissant la coupure ne servent pas ensuite a
estimer l'effet de cette meme coupure.

## 11. Foret sur pseudo-outcomes

La foret predit une CATE a partir des scores AIPW dans des folds groupes par
image. Les predictions sont projetees dans [-1, 1], les bornes logiques d'un
effet sur un outcome binaire.

Resume de la distribution :

- mediane environ -0.258 ;
- premier quartile environ -0.418 ;
- troisieme quartile environ -0.132 ;
- 5e percentile environ -0.906 ;
- 95e percentile environ 0.287.

Cette foret est une approximation pedagogique d'une foret causale specialisee.
Les valeurs individuelles ne doivent pas etre presentees comme des effets
personnels vrais. Elles servent a visualiser une heterogeneite possible.

## 12. Ce qui est predictif et ce qui est causal

| Enonce | Type | Statut |
|---|---|---|
| OBB-1024 a mAP50-95 0.186 | Predictif | Mesure directe du protocole |
| `tennis-court` a la meilleure AP | Predictif | Mesure par classe |
| Les tres petits ont un taux observe de 0.307 | Descriptif | Association brute |
| AIPW vaut -0.368 | Causal sous hypotheses | Estimation ajustee |
| L'effet semble plus fort en scene dense | Heterogeneite causale exploratoire | A confirmer |

Un modele peut avoir une bonne AP pour `tennis-court` tout en ne repondant pas
a la question : "quel est l'effet d'etre tres petit ?". La mAP integre
localisation et classification. L'ATE compare des outcomes potentiels sous deux
conditions de taille, avec des hypotheses d'identification.

## 13. Figures : ordre de lecture

### `model_comparison.png`

1. identifier le meilleur mAP50-95 ;
2. comparer precision et recall ;
3. noter que F1 est dans le tableau global ;
4. rappeler le cout d'inference de 1024.

### `per_class_metrics.png`

1. regarder les ecarts de classe ;
2. ne pas ignorer les classes rares ;
3. comparer HBB/OBB sans generaliser hors protocole.

### `prediction_examples_yolo26n-obb-1024.jpg`

1. voir les OBB et confiances ;
2. reperer les tuiles sans prediction ;
3. relier ces faux negatifs au recall modeste.

### `detection_by_size_orientation.png`

1. lire le taux ;
2. lire l'effectif `n` ;
3. rappeler que la heatmap est descriptive.

### `causal_effects.png`

1. lire le signe ;
2. comparer les estimateurs ;
3. verifier si l'intervalle traverse zero ;
4. remarquer la dispersion IPW/AIPW.

### `propensity_overlap.png`

1. verifier qu'il existe du support commun ;
2. reperer les masses aux bornes 0.05 et 0.95 ;
3. expliquer pourquoi le clipping est necessaire mais pas gratuit.

### `causal_tree.png`

1. distinguer moitie structure et moitie holdout ;
2. lire le seuil de densite ;
3. comparer effets, intervalles et nombre d'images ;
4. conclure `exploratoire`.

## 14. Reponse orale de deux minutes

> J'ai d'abord separe les images DOTA au niveau source avant tout tuilage afin
> d'eviter une fuite entre tuiles proches. Sur 180 images train et 60
> validation, j'ai entraine trois YOLO26n pendant 20 epochs. Le meilleur est
> YOLO26n-OBB a 1024 avec une mAP50-95 de 0.186, contre environ 0.146 pour les
> deux modeles 640. Son recall reste toutefois modeste, environ 0.29, et les
> performances varient fortement par classe.
>
> La question causale est differente : quel est l'effet d'etre tres petit sur
> la probabilite d'une detection OBB correcte ? Le traitement utilise le
> premier quartile d'aire train et l'outcome exige meme classe, confiance 0.25
> et IoU orientee 0.50. Sur 1 646 objets uniques de 53 images, le taux observe
> est 0.307 chez les tres petits contre 0.513 chez les controles.
>
> Apres ajustement et cross-fitting groupe par image, l'AIPW principal estime
> -0.368, avec un intervalle bootstrap [-0.530 ; -0.166]. Les sensibilites
> restent negatives en valeur ponctuelle, mais le clipping 0.02 donne un
> intervalle traversant zero. L'arbre honnete suggere un effet plus negatif
> dans les tuiles denses, mais ce sous-groupe ne contient que 12 images. Je
> presente donc un signal coherent mais observationnel, limite par la
> positivite, les confounders non observes, le sous-ensemble et la definition
> imparfaite d'une intervention sur la taille.

## 15. Phrases interdites et corrections

Mauvais :

> OBB est toujours meilleur que HBB.

Correct :

> OBB-1024 obtient la meilleure mAP50-95 dans ce protocole ; OBB-640 et HBB-640
> sont presque egaux.

Mauvais :

> Les petits objets causent exactement 36.8 % d'erreurs en plus.

Correct :

> Sous les hypotheses d'identification, l'effet moyen AIPW est une baisse
> estimee de 36.8 points de probabilite, avec incertitude et sensibilite.

Mauvais :

> L'arbre prouve que la densite est la cause.

Correct :

> L'arbre suggere une heterogeneite selon la densite ; la division est
> exploratoire et repose sur peu d'images dans une feuille.

Mauvais :

> L'intervalle principal exclut zero, donc aucun biais n'est possible.

Correct :

> L'intervalle quantifie l'incertitude d'echantillonnage du score dans ce
> protocole ; il ne couvre pas les confounders non observes ni toutes les
> decisions de modelisation.

## 16. Grille de memorisation

Completer sans regarder :

| Element | Valeur a connaitre |
|---|---|
| Images train / validation |  |
| Tuiles train / validation |  |
| Meilleur modele |  |
| mAP50-95 du meilleur modele |  |
| Recall du meilleur modele |  |
| Nombre d'objets causaux |  |
| Nombre d'images causales |  |
| Seuil tres petit |  |
| Taux Y traite / controle |  |
| AIPW et IC |  |
| Recouvrement 0.1-0.9 |  |
| Coupure de l'arbre |  |
| Effet de la feuille dense |  |
| Sensibilite qui traverse zero |  |

Si une valeur est oubliee, ne pas inventer. Revenir au tableau correspondant
et reconstruire son sens.
