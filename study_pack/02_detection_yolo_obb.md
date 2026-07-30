# Detection, YOLO et boites orientees

## 1. Detection n'est pas classification

Un classifieur global produit une classe pour l'image entiere. Un detecteur
produit plusieurs objets :

```text
(classe, score de confiance, geometrie de boite)
```

Dans DOTA, une image peut contenir des centaines d'objets de plusieurs classes.
Un classifieur global perdrait leur position et leur nombre. Il est donc
inadapte a l'etat brut.

Reponse orale en 20 secondes :

> La classification dit principalement ce que contient une image. La detection
> doit aussi dire ou se trouve chaque objet, et DOTA peut en contenir beaucoup.
> Il faut donc predire plusieurs classes et plusieurs boites par image.

## 2. Representation d'une boite

### HBB

Une HBB reste alignee avec les axes. Elle est souvent representee par :

```text
(x_centre, y_centre, largeur, hauteur)
```

### OBB

Une OBB peut tourner. Dans les labels Ultralytics utilises ici :

```text
classe x1 y1 x2 y2 x3 y3 x4 y4
```

Les coordonnees sont normalisees par la taille de la tuile.

### Perte lors de la conversion

La HBB d'un objet diagonal couvre l'objet plus une zone de fond. Deux objets
orientes et proches peuvent aussi produire des HBB qui se chevauchent beaucoup,
meme si leurs OBB sont bien separees.

Cette perte peut diminuer la precision de localisation et rendre la NMS plus
agressive dans les scenes denses.

## 3. IoU

Formule :

```text
IoU = aire(intersection) / aire(union)
```

Proprietes :

- IoU = 1 : superposition parfaite ;
- IoU = 0 : aucune intersection ;
- bonne classe avec IoU faible : localisation incorrecte ;
- IoU ne mesure pas directement la confiance ou la calibration.

Pour OBB, l'intersection doit respecter la rotation. Calculer seulement l'IoU
des HBB autour de deux OBB peut surestimer leur chevauchement.

### Mini-calcul

Deux rectangles de surface 100 se chevauchent sur une surface 40 :

```text
union = 100 + 100 - 40 = 160
IoU = 40 / 160 = 0.25
```

Le piege classique est de diviser par la somme 200 sans retirer l'intersection
comptee deux fois.

## 4. Matching prediction-verite terrain

Il ne suffit pas de calculer chaque IoU independamment. Une prediction ne doit
pas valider plusieurs objets.

Le pipeline :

1. separe les objets par classe ;
2. calcule une matrice d'IoU ;
3. applique une association un-a-un qui maximise l'IoU totale ;
4. marque un objet correct si la classe est la meme, la confiance suffisante
   et l'IoU au-dessus du seuil.

Cette association evite de compter deux fois une unique detection.

## 5. TP, FP et FN

- TP : prediction associee a un objet de meme classe avec IoU suffisante.
- FP : prediction non associee correctement.
- FN : objet reel sans prediction correcte.

Une mauvaise classe produit generalement un FP pour la classe predite et un FN
pour la vraie classe.

Une boite de bonne classe mais mal placee produit aussi un FP et un FN selon la
regle d'evaluation.

## 6. Precision, recall et F1

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

Interpretation :

- precision : fiabilite des detections annoncees ;
- recall : couverture des objets reels ;
- F1 : compromis pour un seuil de confiance donne.

Le seuil de confiance change le compromis. Un seuil eleve elimine des
detections incertaines : souvent precision augmente et recall diminue.

## 7. AP et mAP

Pour une classe :

1. trier les predictions par confiance ;
2. faire varier le seuil ;
3. calculer les couples precision-recall ;
4. mesurer l'aire sous la courbe precision-recall : AP.

La mAP est la moyenne des AP sur les classes.

### mAP50

Une prediction est localement correcte a partir d'une IoU de 0.50. Cette
metrique est relativement tolerante.

### mAP50-95

On moyenne l'AP pour des seuils IoU de 0.50 a 0.95 par pas de 0.05. Elle
penalise plus fortement les boites imprecises.

Si mAP50 est nettement plus haute que mAP50-95, le modele reconnait parfois
l'objet mais la localisation precise reste difficile.

### Limite de la moyenne

Chaque classe contribue a la mAP, mais une seule valeur masque :

- les classes rares ;
- les objets minuscules ;
- les orientations extremes ;
- les scenes tres denses ;
- les sources d'image difficiles.

Il faut donc aussi etudier AP par classe et performance par sous-groupe.

## 8. Score de confiance

Le score sert a classer et filtrer les detections. Il n'est pas une garantie de
verite et n'est pas automatiquement une probabilite bien calibree.

Trois dimensions restent distinctes :

- classe correcte ;
- geometrie correcte ;
- confiance elevee.

Une prediction peut etre tres confiante et fausse.

## 9. NMS

La Non-Maximum Suppression retire les boites redondantes :

1. garder la boite la plus confiante ;
2. comparer son IoU aux autres ;
3. supprimer les boites trop chevauchantes ;
4. continuer.

Dans une scene dense, une NMS trop agressive peut supprimer un vrai voisin.
Avec OBB, une NMS orientee distingue mieux certains objets diagonaux proches.

## 10. YOLO en une vue

YOLO traite l'image dans un passage principal du reseau et produit de nombreux
candidats de detection.

### Backbone

Extrait des motifs visuels de complexite croissante :

- bords et textures dans les premieres couches ;
- formes et parties d'objets plus loin.

### Neck

Combine plusieurs echelles de features. C'est crucial pour DOTA :

- grandes structures comme les terrains ;
- petits vehicules de quelques pixels.

### Head

Produit les classes, scores et parametres de boites. La tete OBB doit aussi
representer l'orientation.

Reponse orale :

> Le backbone extrait les caracteristiques, le neck combine plusieurs echelles
> et la head transforme ces caracteristiques en classes, confiances et boites.
> Pour DOTA, le multi-echelle et la tete orientee sont particulierement utiles.

## 11. Loss et optimisation

La loss d'entrainement combine plusieurs erreurs, notamment :

- erreur de localisation ;
- erreur de classification ;
- erreur liee aux scores ;
- composante d'angle pour OBB.

La retropropagation calcule comment chaque poids contribue a cette erreur.
L'optimiseur met ensuite les poids a jour.

Attention :

- loss train basse ne garantit pas la generalisation ;
- mAP validation est une metrique d'evaluation, pas la loss ;
- une hausse de la performance train avec stagnation validation suggere un
  overfitting.

## 12. Epoch, batch et learning rate

### Epoch

Un passage approximatif sur tout le train. Davantage d'epochs donne plus
d'occasions d'apprendre, mais peut mener a l'overfitting.

### Batch

Nombre d'images traitees avant une mise a jour des poids. Un grand batch utilise
plus de memoire et donne un gradient souvent moins bruite.

### Learning rate

Amplitude des mises a jour :

- trop grand : apprentissage instable ;
- trop petit : apprentissage lent ou bloque ;
- scheduler : le fait varier pendant l'entrainement.

## 13. Transfert learning

Le backbone part de poids appris sur COCO. Il connait deja des motifs visuels
generaux, puis il est ajuste sur DOTA.

Dans cette experience, HBB et OBB utilisent le meme point de depart COCO autant
que leur architecture le permet. Aucun poids OBB deja ajuste sur DOTA n'est
utilise. Sinon OBB aurait un avantage experimental difficile a interpreter.

## 14. Augmentations

Les augmentations changent les exemples train sans toucher a la validation :

- couleur et luminosite ;
- translation et echelle ;
- mosaics ;
- retournement horizontal.

Elles peuvent ameliorer la generalisation. Elles doivent transformer
correctement les boites. Une augmentation visuelle dont les labels ne suivent
pas rend l'entrainement faux.

Une augmentation d'une image train reste train. Elle ne doit jamais migrer en
validation.

## 15. Petits objets

Les petits objets sont difficiles parce que :

- ils contiennent peu de pixels ;
- le redimensionnement les reduit encore ;
- leurs bords sont sensibles a une erreur de quelques pixels ;
- ils apparaissent souvent dans des zones denses ;
- une petite erreur absolue produit une forte baisse d'IoU.

Ameliorations possibles :

- augmenter `imgsz` ;
- utiliser des tuiles adaptees ;
- garder un neck multi-echelle ;
- entrainer plus longtemps ;
- sur-echantillonner avec prudence les scenes utiles ;
- analyser le seuil de confiance et `max_det`.

## 16. Desequilibre de classes

Un grand nombre de `small-vehicle` ne compense pas le manque de `helicopter`.
Une metrique globale peut etre dominee par les classes faciles ou frequentes.

Controles :

- couverture de toutes les classes ;
- AP par classe ;
- exemples d'erreurs pour classes rares ;
- limites explicites ;
- eventuellement ponderation ou sur-echantillonnage, sans fabriquer de
  validation artificielle.

## 17. Overfitting et underfitting

### Underfitting

- loss encore elevee ;
- train et validation faibles ;
- modele, resolution ou duree insuffisants.

### Overfitting

- train continue de s'ameliorer ;
- validation stagne ou baisse ;
- ecart train-validation grand.

### Early stopping

Arrete apres une periode sans amelioration. Il economise du calcul, mais ne
remplace pas l'analyse des courbes.

## 18. Pourquoi HBB peut parfois sembler meilleure

OBB est mieux adaptee geometriquement, mais elle doit apprendre un probleme plus
complexe. Avec peu d'epochs ou peu de donnees :

- la tete OBB a plus a apprendre ;
- l'angle ajoute une source d'erreur ;
- l'optimisation peut demander plus de temps ;
- la metrique orientee est plus stricte.

Une baseline HBB superieure dans un protocole court ne prouve donc pas que les
orientations sont inutiles. Il faut interpreter donnees, duree, resolution et
metrique ensemble.

## 19. Checklist d'analyse d'une courbe

Pour chaque run :

1. Combien d'epochs ont reellement fini ?
2. Quelle epoque a produit `best.pt` ?
3. Les losses train baissent-elles ?
4. mAP validation augmente-t-elle encore ?
5. Precision et recall sont-elles desequilibrees ?
6. Les classes rares ont-elles une AP presque nulle ?
7. Le meilleur modele est-il la derniere epoque ?
8. L'early stopping s'est-il active ?
9. La resolution et `max_det` sont-ils adaptes aux scenes denses ?
10. Les exemples visuels confirment-ils les chiffres ?

## 20. Questions de rappel rapide

1. Pourquoi une bonne classe ne suffit-elle pas pour un TP ?
2. Pourquoi l'union retire-t-elle l'intersection une fois ?
3. Comment une prediction peut-elle creer a la fois FP et FN ?
4. Que change le seuil de confiance ?
5. Quelle difference entre AP et mAP ?
6. Pourquoi mAP50-95 est-elle plus stricte ?
7. Quel est le risque d'une NMS forte dans un parking ?
8. Quel role joue le neck ?
9. Pourquoi le transfert learning aide-t-il ?
10. Pourquoi `imgsz=1024` peut aider les petits objets ?
11. Pourquoi peut-il aussi ralentir l'entrainement ?
12. Pourquoi OBB n'est-elle pas garantie de battre HBB ?
13. Quelle metrique montre surtout les objets manques ?
14. Pourquoi faut-il AP par classe ?
15. Comment reconnaitre un overfitting ?
