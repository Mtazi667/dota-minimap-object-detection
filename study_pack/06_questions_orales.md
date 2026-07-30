# Questions orales avec reponses courtes

Lire seulement la question, repondre, puis reveler la reponse.

## Projet et donnees

### 1. Quel est l'objectif du projet ?

Detecter et classifier des objets dans des images DOTA, puis estimer si la
condition tres petite affecte la probabilite de detection correcte. La premiere
partie est predictive, la seconde causale.

### 2. Pourquoi DOTA est-il difficile ?

Les images sont grandes, les objets varient fortement en taille, peuvent etre
minuscules, denses et orientes, et les 15 classes sont desequilibrees.

### 3. Que contient une annotation DOTA ?

Quatre coins, donc huit coordonnees, une classe et un indicateur `difficult`.
Le fichier contient aussi des metadonnees comme la source et parfois le GSD.

### 4. Pourquoi tuiler ?

Les images completes sont trop grandes pour un detecteur standard et les petits
objets seraient fortement reduits. Les tuiles rendent le calcul faisable et
preservent davantage de pixels par objet.

### 5. Quel est le principal risque du tuilage ?

La fuite si des tuiles d'une meme image source sont reparties entre train et
validation. Le split doit etre fait au niveau de l'image avant le tuilage.

### 6. Pourquoi 1024 pixels ?

C'est un compromis entre contexte, preservation des petits objets, nombre de
fragments et memoire GPU. Ce n'est pas une valeur universelle.

### 7. Pourquoi un chevauchement en train ?

Il augmente les chances qu'un objet proche du bord soit entier dans une autre
tuile et enrichit les positions vues pendant l'apprentissage.

### 8. Pourquoi moins de chevauchement en validation ?

Pour eviter une evaluation trop redondante et ne pas compter plusieurs fois les
memes objets. Une instance unique est ensuite gardee pour la causalite.

### 9. Pourquoi garder des tuiles negatives ?

Elles apprennent au modele a reconnaitre le fond et reduisent les fausses
alertes. Toutes les garder ferait dominer le fond et couterait trop cher.

### 10. Que signifie stratifier ici ?

Selectionner des images en preservant la couverture des classes et des types de
scene definis par classe dominante et densite, plutot que tirer aveuglement.

## Detection et YOLO

### 11. Detection ou classification ?

La classification donne surtout une classe globale. La detection donne
plusieurs classes et leurs localisations dans la meme image.

### 12. HBB ou OBB ?

HBB est alignee aux axes. OBB tourne avec l'objet. OBB correspond mieux aux
annotations et objets orientes de DOTA.

### 13. Pourquoi garder HBB ?

Elle fournit une baseline simple sur exactement les memes tuiles et permet de
quantifier ce que change la representation orientee.

### 14. Qu'est-ce que l'IoU ?

Le rapport entre l'aire d'intersection et l'aire d'union des boites. Elle mesure
la qualite geometrique de la localisation entre 0 et 1.

### 15. Quand une prediction est-elle un TP ?

Quand elle est associee a un objet reel, a la bonne classe, avec un score
suffisant et une IoU au-dessus du seuil choisi.

### 16. Difference FP/FN ?

FP est une alerte non correcte. FN est un objet reel manque. Une mauvaise
classification peut produire les deux simultanement.

### 17. Precision ?

Parmi les detections annoncees, proportion correcte : TP divise par TP plus FP.

### 18. Recall ?

Parmi les objets reels, proportion retrouvee : TP divise par TP plus FN.

### 19. AP ?

Aire sous la courbe precision-recall d'une classe quand le seuil de confiance
varie.

### 20. mAP ?

Moyenne des AP sur les classes. Elle resume le detecteur, mais doit etre
completee par les scores par classe et sous-groupe.

### 21. mAP50 contre mAP50-95 ?

mAP50 utilise un seuil IoU 0.50. mAP50-95 moyenne plusieurs seuils jusqu'a
0.95 et exige une localisation plus precise.

### 22. Score de confiance ?

Valeur utilisee pour classer et filtrer les detections. Elle n'est ni une
garantie de justesse ni une mesure directe de l'IoU.

### 23. NMS ?

Post-traitement qui garde une boite forte et supprime des boites redondantes
trop chevauchantes. Elle peut supprimer de vrais voisins en scene dense.

### 24. Backbone ?

Partie qui extrait les caracteristiques visuelles des pixels, des motifs simples
jusqu'aux representations plus semantiques.

### 25. Neck ?

Partie qui combine les features de plusieurs echelles, importante pour detecter
a la fois grands terrains et petits vehicules.

### 26. Head ?

Partie qui transforme les features en classes, scores et boites. La head OBB
represente aussi l'orientation.

### 27. Que minimise YOLO ?

Une loss combinant erreurs de localisation, classification et autres
composantes, avec une composante d'angle dans OBB.

### 28. Loss contre mAP ?

La loss guide l'optimisation sur train. La mAP evalue les detections sur
validation. Une loss plus basse ne garantit pas une meilleure mAP.

### 29. Transfert learning ?

Partir d'un backbone pre-entraine sur COCO puis l'ajuster sur DOTA. Cela
economise donnees et temps par rapport a un depart aleatoire.

### 30. Pourquoi les memes poids de depart ?

Pour comparer HBB et OBB sans donner a OBB l'avantage de poids deja ajustes sur
DOTA.

### 31. Pourquoi les petits objets sont-ils difficiles ?

Peu de pixels, perte au redimensionnement, bruit de fond, densite et forte
sensibilite de l'IoU a quelques pixels d'erreur.

### 32. Pourquoi augmenter `imgsz` ?

Le reseau voit davantage de pixels par petit objet. En contrepartie, le calcul
et la memoire augmentent et le batch doit parfois diminuer.

### 33. Overfitting ?

Le modele s'ameliore sur train mais ne generalise plus : validation stagne ou
baisse. On examine les courbes et utilise notamment l'early stopping.

### 34. Pourquoi AP par classe ?

La moyenne peut cacher une classe rare presque jamais detectee ou une classe
frequente tres facile.

## Causalite

### 35. Correlation contre causalite ?

La correlation decrit une difference observee. La causalite compare ce qui
arriverait sous deux conditions contrefactuelles, avec des hypotheses
d'identification.

### 36. Traitement retenu ?

Etre dans le premier quartile d'aire relative de tuile, seuil defini sur les
objets train uniques.

### 37. Outcome retenu ?

Une prediction OBB de meme classe, confiance au moins 0.25 et IoU au moins
0.50 sur la validation.

### 38. ATE ?

Moyenne de `Y(1)-Y(0)` dans la population d'objets analysee. Ici, une difference
de probabilite de detection correcte.

### 39. Confounder ?

Cause commune du traitement et de l'outcome. La classe peut influencer taille
typique et facilite de detection.

### 40. Mediator ?

Variable causee par le traitement qui transmet son effet. L'ajuster bloque une
partie de l'effet total.

### 41. Collider ?

Consequence commune de deux variables. Conditionner dessus peut ouvrir une
association artificielle.

### 42. Backdoor path ?

Chemin non causal entre traitement et outcome qui commence par une fleche
entrant dans le traitement, par exemple `D <- classe -> Y`.

### 43. Pourquoi ne pas ajuster sur l'aire exacte ?

Elle definit directement le traitement tres petit. L'inclure rendrait les
groupes presque deterministes et ne correspondrait pas a l'ajustement retenu.

### 44. Pourquoi ne pas ajuster sur la confiance ?

La confiance est produite apres le traitement par le detecteur et participe a
la definition de l'outcome. C'est une variable post-traitement.

### 45. Exchangeabilite ?

Apres ajustement sur X, les groupes seraient comparables quant aux outcomes
potentiels. Elle est non testable et menacee par le flou ou contraste non
observe.

### 46. Positivite ?

Pour chaque profil, il faut une probabilite non nulle d'etre traite et controle.
Des classes toujours minuscules violent localement cette hypothese.

### 47. Coherence ?

L'outcome observe sous le traitement recu correspond a l'outcome potentiel de
ce traitement. Elle demande une version suffisamment bien definie de "petit".

### 48. Interference ?

L'outcome d'un objet pourrait dependre d'autres objets via NMS, occlusion ou
`max_det`. Cela fragilise l'independance entre unites.

### 49. Difference brute ?

Taux moyen traite moins taux moyen controle. Simple mais confondu par les
differences de composition.

### 50. G-computation ?

Modeliser l'outcome sous D=1 et D=0 pour chaque X, calculer les deux predictions
et moyenner leur difference.

### 51. Score de propension ?

Probabilite de traitement conditionnelle aux covariables. Il sert a ajuster et
diagnostiquer le support, pas a supprimer les confounders non observes.

### 52. IPW ?

Pondere les objets par l'inverse de la probabilite de leur traitement observe.
Les scores extremes rendent l'estimation instable.

### 53. AIPW ?

Combine modele d'outcome et propension, puis corrige les predictions avec des
residus ponderes. Double robustesse ne signifie pas absence de toutes limites.

### 54. Cross-fitting ?

Les nuisances d'un objet sont predites par des modeles qui n'ont pas utilise
son fold. Les folds sont groupes par image pour limiter l'overfitting et la
dependance.

### 55. Pourquoi bootstrap par image ?

Les objets d'une meme image partagent contexte et conditions. Les
reechantillonner comme independants donnerait une incertitude trop petite.

### 56. Arbre causal honnete ?

Une partie des images choisit les divisions, l'autre estime les effets des
feuilles. Cela limite le biais de recherche de sous-groupes.

### 57. CATE ?

Effet moyen conditionnel a un profil X. Il sert a etudier l'heterogeneite,
mais les estimations extremes demandent de grands effectifs et validation.

### 58. Sensibilite ?

Refaire l'estimation avec d'autres seuils raisonnables, par exemple IoU ou
clipping, pour voir si signe et amplitude dependent d'un choix arbitraire.

### 59. Pourquoi le resultat reste-t-il prudent ?

Le traitement n'est pas randomise, des confounders manquent peut-etre, la
selection et l'interference existent, et la population est un sous-ensemble.

### 60. Conclusion type ?

Le detecteur fournit des performances predictives mesurees sur validation. La
partie causale estime ensuite un contraste sur les erreurs, sous des hypotheses
fortes. Les deux reponses sont complementaires mais non interchangeables.
