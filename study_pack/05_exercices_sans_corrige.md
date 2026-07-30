# Exercices sans corrige

Ne pas ouvrir `07_corrige.md` avant d'avoir ecrit ou prononce une reponse.

## Detection

### D1 - IoU

Deux boites de surface 120 et 80 ont une intersection de surface 50.

1. Calculer l'union.
2. Calculer l'IoU.
3. Dire si la detection passe un seuil IoU de 0.50.

### D2 - Deux boites pour un objet

Un objet reel `plane` recoit deux predictions `plane` :

- A : confiance 0.92, IoU 0.72 ;
- B : confiance 0.70, IoU 0.66.

Expliquer :

1. pourquoi les deux predictions ne doivent pas devenir deux TP ;
2. le role possible de la NMS ;
3. ce qui peut arriver a B apres le matching.

### D3 - Matrice de confusion de detection

Un modele produit 80 detections. Parmi elles, 52 sont des TP. Le dataset
contient 65 objets reels.

1. Calculer FP.
2. Calculer FN.
3. Calculer precision.
4. Calculer recall.
5. Calculer F1.

### D4 - Seuil de confiance

On augmente le seuil de confiance de 0.20 a 0.70.

1. Quel mouvement general attend-on pour precision ?
2. Quel mouvement general attend-on pour recall ?
3. Pourquoi ce n'est pas une loi absolue sur chaque petit echantillon ?

### D5 - mAP

Expliquer sans formule :

1. AP ;
2. mAP ;
3. mAP50 ;
4. mAP50-95 ;
5. pourquoi il faut aussi AP par classe.

### D6 - HBB et OBB

Dessiner un rectangle diagonal long et fin.

1. Dessiner sa HBB.
2. Dessiner son OBB.
3. Comparer la quantite de fond.
4. Expliquer un effet possible sur IoU et NMS.
5. Donner une raison pour laquelle OBB pourrait quand meme moins bien scorer
   dans une experience courte.

### D7 - Petite erreur, grand effet

Un petit objet mesure environ 8 x 8 pixels. La prediction est decalee de
4 pixels. Un grand objet mesure 160 x 160 pixels et subit le meme decalage.

Expliquer qualitativement pourquoi l'IoU du petit objet est plus affectee.

### D8 - Loss et metrique

Repondre a l'affirmation :

> La loss train a baisse, donc le modele final est forcement meilleur.

Donner au moins trois raisons pour lesquelles cette conclusion est insuffisante.

## Pipeline et donnees

### P1 - Fuite par tuiles

Une image DOTA est decoupee en 30 tuiles. On melange les 30 tuiles, puis on en
place 24 en train et 6 en validation.

1. Identifier la fuite.
2. Expliquer pourquoi la validation devient optimiste.
3. Proposer le bon ordre des operations.

### P2 - Positions de grille

Une dimension d'image vaut 2500 pixels, la tuile 1024 et le stride 824.

1. Les deux premieres positions sont 0 et 824.
2. Calculer la position finale qui couvre exactement le bord.
3. Expliquer pourquoi elle ne suit pas forcement le stride.
4. Dire si un chevauchement final existe.

### P3 - Fragments

Classer les cas :

- objet conserve a 95 % ;
- objet conserve a 75 % ;
- objet conserve a 45 % ;
- objet conserve a 10 %.

Utiliser les seuils 0.70 et 0.20 du pipeline et expliquer chaque decision.

### P4 - Tuiles negatives

1. Pourquoi ne pas supprimer toutes les tuiles negatives ?
2. Pourquoi ne pas garder toutes les tuiles negatives ?
3. Quel type d'erreur est directement concerne ?

### P5 - Stratification

La selection couvre les 15 classes et combine classe dominante et densite.

1. Pourquoi est-ce mieux qu'un simple tirage aleatoire ?
2. Pourquoi cela ne resout-il pas completement le desequilibre ?
3. Pourquoi la validation ne doit-elle pas etre artificiellement equilibree
   sans explication ?

### P6 - Objet duplique

Un objet apparait dans deux tuiles train chevauchantes.

1. Est-ce une fuite train-validation ?
2. Pourquoi faut-il quand meme le tracer ?
3. Pourquoi l'analyse causale choisit-elle une seule instance ?

### P7 - Reproductibilite

Donner au moins huit elements a enregistrer pour reproduire un run.

### P8 - Comparaison equitable

On compare :

- HBB initialise avec COCO ;
- OBB initialise avec un poids deja ajuste sur DOTA.

Expliquer le probleme et proposer un protocole plus equitable.

## Causalite

### C1 - Smoking, gene, cancer

Construire un DAG ou :

- le gene influence smoking ;
- le gene influence cancer ;
- smoking influence cancer.

1. Quel est le traitement ?
2. Quel est l'outcome ?
3. Quel est le confounder ?
4. Sur quoi faut-il ajuster pour l'effet total de smoking ?

### C2 - Confounder DOTA

Utiliser `classe`, `tres petit` et `detection correcte`.

1. Construire un backdoor path.
2. Expliquer le biais sans ajustement.
3. Expliquer ce que l'ajustement essaie de faire.

### C3 - Mediator

Proposer un mediator entre petite taille et detection. Expliquer pourquoi
l'ajuster ne donne plus l'effet total.

### C4 - Collider

Construire un collider avec `petite taille`, `scene complexe` et
`tuile selectionnee`. Expliquer pourquoi conditionner sur la selection peut
ouvrir un chemin.

### C5 - Variables post-traitement

Classer ces variables comme ajustement plausible ou variable a eviter dans le
modele principal :

- classe ;
- orientation ;
- score de confiance ;
- IoU ;
- source ;
- GSD ;
- feature interne du detecteur ;
- densite ;
- aire exacte lorsque le traitement est defini par l'aire.

### C6 - Outcomes potentiels

Pour un objet donne, definir `Y(1)` et `Y(0)`. Expliquer pourquoi les deux ne
sont jamais observes ensemble.

### C7 - Hypotheses

Donner une definition intuitive et une menace DOTA pour :

1. exchangeabilite conditionnelle ;
2. positivite ;
3. coherence ;
4. absence d'interference ;
5. absence de fuite.

### C8 - Difference brute

Taux de detection :

- tres petits : 0.18 ;
- controles : 0.43.

1. Calculer la difference brute.
2. L'exprimer en points de pourcentage.
3. Expliquer pourquoi elle n'est pas automatiquement causale.

### C9 - Score de propension

Deux objets ont des scores 0.03 et 0.52.

1. Lequel a le meilleur support commun ?
2. Pourquoi le premier peut creer un poids extreme ?
3. Que fait le clipping ?
4. Quelle limite le clipping ne resout-il pas ?

### C10 - IPW

Un objet traite a `e(X)=0.20`.

1. Quel est son poids IPW non stabilise ?
2. Pourquoi un traite avec faible propension recoit-il un grand poids ?
3. Quel est le danger ?

### C11 - AIPW

Expliquer les deux sources d'information combinees par AIPW. Definir
"doublement robuste" sans dire "toujours sans biais".

### C12 - Cross-fitting

1. Pourquoi ne pas predire les nuisances sur les memes objets qui les ont
   entrainees ?
2. Pourquoi grouper par image ?
3. Quelle ressemblance avec une prediction hors echantillon ?

### C13 - Arbre causal honnete

1. Quelle moitie choisit les divisions ?
2. Quelle moitie estime les effets des feuilles ?
3. Pourquoi separer ?
4. Pourquoi limiter profondeur et taille minimale ?

### C14 - Sensibilite

Le signe AIPW est negatif a IoU 0.40, 0.50 et 0.60, mais l'amplitude double a
0.60.

1. Quelle partie est stable ?
2. Quelle partie est sensible ?
3. Quelle conclusion prudente donner ?

## Analyse des sorties reelles

Ces exercices utilisent les fichiers dans `outputs/analysis/`.

### R1 - Comparaison globale

Ouvrir `model_comparison.csv`.

1. Quel modele a la meilleure mAP50 ?
2. Quel modele a la meilleure mAP50-95 ?
3. Quel est le gain ou la perte absolue ?
4. La resolution 1024 aide-t-elle ?
5. Proposer deux explications et une limite.

### R2 - Classes

Ouvrir `per_class_metrics.csv`.

1. Identifier les trois meilleures et trois pires classes du modele retenu.
2. Comparer avec les comptes de `class_coverage.csv`.
3. Donner un cas ou frequence et AP ne suivent pas exactement la meme tendance.

### R3 - Taille et orientation

Ouvrir `detection_by_size_orientation.png`.

1. Quel groupe semble le plus difficile ?
2. Les differences sont-elles monotones ?
3. Quels effectifs faudrait-il verifier avant de conclure ?

### R4 - Effet moyen

Ouvrir `causal_effect_estimates.csv`.

1. Comparer brut, g-computation, IPW et AIPW.
2. Le signe est-il stable ?
3. L'IC AIPW inclut-il zero ?
4. Ecrire une interpretation correcte en deux phrases.

### R5 - Positivite

Ouvrir `propensity_overlap.png`.

1. Les distributions se chevauchent-elles ?
2. Y a-t-il des zones presque sans controles ou traites ?
3. Quelle consequence pour la variance et la generalisation ?

### R6 - Heterogeneite

Ouvrir `causal_tree_subgroups.csv` et `cate_distribution.png`.

1. Quelle feuille a l'effet le plus negatif ?
2. Combien d'images et d'objets contient-elle ?
3. Quelle variable semble structurer les effets ?
4. Pourquoi ne faut-il pas transformer cette feuille en loi generale ?

## Mini-examen final

Temps conseille : 35 minutes, sans documents.

1. Dessiner le pipeline complet.
2. Definir IoU, precision, recall, AP et mAP.
3. Expliquer le choix OBB.
4. Expliquer le risque de fuite par tuile.
5. Justifier le sous-ensemble stratifie.
6. Definir D, Y, X et ATE.
7. Dessiner le DAG DOTA principal.
8. Distinguer confounder, mediator et collider.
9. Expliquer exchangeabilite et positivite.
10. Comparer difference brute, g-computation, IPW et AIPW.
11. Expliquer cross-fitting et bootstrap groupe.
12. Expliquer arbre causal honnete.
13. Donner les resultats reels principaux.
14. Donner cinq limites.
15. Conclure en 90 secondes sans confondre prediction et causalite.
