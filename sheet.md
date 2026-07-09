# Cheatsheet presentation - Projet DOTA / YOLO-OBB

Presentation : jeudi 9 juillet 2026, midi  
Duree : 5 a 10 minutes  
Objectif : presenter l'avancement + expliquer un algorithme ML en profondeur  
Algorithme principal : YOLO-OBB

---

## 0. Esprit de la presentation

Ne pas lire mot pour mot.

Utiliser cette fiche comme une carte :

- ou j'en suis ;
- pourquoi j'ai fait ces choix ;
- comment fonctionne YOLO-OBB ;
- comment je montre ca dans le notebook ;
- quoi repondre si le professeur pose des questions.

Angle principal :

> Mon projet porte sur la detection d'objets dans des images aeriennes avec DOTA-v1.0. J'ai commence par comprendre les donnees, puis j'ai prepare les annotations dans un format compatible avec YOLO-OBB, parce que DOTA annote les objets avec des boites orientees.

---

## 1. Structure rapide de ma presentation

### Plan 5 a 10 minutes

1. Probleme du projet
2. Dataset DOTA-v1.0
3. Exploration faite dans le notebook
4. Pourquoi les boites orientees sont importantes
5. Pretraitement et export YOLO-OBB
6. Explication de l'algorithme YOLO-OBB
7. Prochaines etapes
8. Usage de l'IA comme outil d'apprentissage

Si le temps est court :

- probleme ;
- DOTA ;
- exploration ;
- YOLO-OBB ;
- prochaines etapes.

---

## 2. Ou j'en suis dans le projet

### Avancement general

- Dataset DOTA-v1.0 telecharge et organise localement.
- Notebook structure en francais.
- Exploration du dataset terminee pour une premiere version.
- Annotations DOTA lues et transformees en tableau Pandas.
- Analyse des classes, tailles d'objets, dimensions d'images, objets difficiles.
- Visualisation d'images avec boites orientees.
- Analyse de l'orientation des objets.
- Pretraitement commence.
- Export provisoire des labels au format YOLO-OBB.
- Section pedagogique sur YOLO-OBB ajoutee.

### Chiffres importants

- Objets train : 98 990
- Objets validation : 28 853
- Objets total : 127 843
- Objets utilisables apres controle qualite : 127 672
- Objets exclus provisoirement : 171
- Labels YOLO-OBB train exportes : 1 407 fichiers
- Labels YOLO-OBB validation exportes : 456 fichiers
- Classes DOTA : 15

### Phrase simple

> A ce stade, je n'ai pas encore entraine le modele final. J'ai surtout construit la base propre : comprehension des donnees, verification des annotations, choix du format, et preparation des labels pour YOLO-OBB.

---

## 3. Ce que je montre dans le notebook

### Section 1 : exploration

Montrer :

- titre du notebook ;
- explication du format DOTA ;
- lecture d'un fichier d'annotation ;
- tableau `objects.head()` ;
- distribution des classes ;
- tailles des objets ;
- visualisation image + boites ;
- orientation des boites ;
- bilan de l'exploration.

Idee a dire :

> Je commence par comprendre la structure du dataset avant de choisir un modele. Pour DOTA, c'est essentiel, parce que les annotations ne sont pas seulement des classes : elles contiennent aussi la position et l'orientation des objets.

### Section 2 : pretraitement

Montrer :

- encodage des classes ;
- fusion avec les tailles d'images ;
- normalisation des coordonnees ;
- controle qualite ;
- `training_ready_objects` ;
- export YOLO-OBB ;
- fichier YAML.

Idee a dire :

> Le pretraitement transforme les annotations brutes DOTA en un format que le modele peut utiliser. Je garde les boites orientees, mais je prepare aussi des variables horizontales pour comparaison ou analyse d'erreurs.

### Section 3 : modele YOLO-OBB

Montrer :

- section "Modele predictif envisage : YOLO-OBB" ;
- "Comment fonctionne YOLO ?" ;
- "Difference entre YOLO classique et YOLO-OBB" ;
- "Pourquoi YOLO-OBB est adapte a ce projet".

Idee a dire :

> YOLO-OBB est coherent avec DOTA parce qu'il predit des objets avec des boites orientees, au lieu de forcer tous les objets dans des rectangles horizontaux.

---

## 4. Le probleme ML en une phrase

Ce n'est pas une classification simple.

Classification simple :

- entree : une image ;
- sortie : une seule classe.

Detection d'objets :

- entree : une image ;
- sortie : plusieurs objets ;
- pour chaque objet : classe + localisation.

Dans mon projet :

- entree : image aerienne/satellite ;
- sortie : objets comme `plane`, `ship`, `small-vehicle`, etc. ;
- localisation : boite orientee avec 4 coins.

Phrase utile :

> Le probleme est plus complexe qu'une classification, parce qu'une seule image peut contenir plusieurs objets de classes differentes, chacun avec sa position et son orientation.

---

## 5. DOTA-v1.0 : ce qu'il faut expliquer

DOTA est un dataset d'images aeriennes pour la detection d'objets.

Particularites importantes :

- images souvent grandes ;
- objets parfois tres petits ;
- objets dans toutes les orientations ;
- plusieurs objets par image ;
- classes desequilibrees ;
- annotations avec boites orientees.

Format d'une annotation DOTA :

```text
x1 y1 x2 y2 x3 y3 x4 y4 classe difficult
```

Explication :

- `x1 y1 ... x4 y4` : les 4 coins de la boite orientee ;
- `classe` : categorie de l'objet ;
- `difficult` : objet plus difficile a detecter ou annoter.

Phrase utile :

> La structure des annotations influence directement le choix du modele. Comme DOTA fournit des boites orientees, je privilegie un modele capable de les apprendre.

---

## 6. Pourquoi les boites orientees comptent

Boite horizontale :

- rectangle aligne avec l'image ;
- simple ;
- mais parfois beaucoup de fond inutile autour de l'objet.

Boite orientee :

- rectangle tourne ;
- suit mieux la forme de l'objet ;
- plus adaptee aux objets aeriennes : avions, bateaux, ponts, terrains.

Exemple oral :

> Si un avion est diagonal, une boite horizontale autour de lui contiendra beaucoup de fond. Une boite orientee suit mieux l'avion et donne une localisation plus precise.

Lien avec exploration :

- mediane orientation absolue train : environ 47 degres ;
- mediane orientation absolue validation : environ 52 degres ;
- donc beaucoup d'objets ne sont pas simplement horizontaux ou verticaux.

Phrase cle :

> L'analyse des orientations justifie techniquement le choix de YOLO-OBB.

---

## 7. YOLO : explication claire

YOLO = You Only Look Once.

Idee :

- le modele regarde l'image en une seule passe ;
- il produit directement les detections ;
- il predit des boites, des classes et des scores de confiance.

Sortie d'une detection :

- classe predite ;
- boite predite ;
- score de confiance ;
- parfois orientation dans le cas OBB.

Pourquoi "one stage" ?

- YOLO ne fait pas d'abord une liste de regions candidates comme certains anciens detecteurs ;
- il predit directement les objets depuis les caracteristiques de l'image.

Phrase simple :

> YOLO est un detecteur rapide parce qu'il transforme directement une image en une liste de detections, au lieu de resoudre plusieurs sous-problemes separes.

---

## 8. Architecture YOLO

Trois blocs principaux :

### Backbone

Role :

- extraire les caracteristiques visuelles ;
- detecter contours, textures, formes, motifs.

Dans mon contexte :

- apprendre des motifs visuels d'avions, bateaux, vehicules, terrains.

### Neck

Role :

- combiner les caracteristiques a plusieurs echelles.

Pourquoi important :

- DOTA contient de petits objets et de grands objets ;
- un vehicule peut etre minuscule ;
- un terrain ou un port peut etre grand.

### Head

Role :

- produire les predictions finales.

Sorties :

- classe ;
- position ;
- taille ;
- orientation ;
- score de confiance.

Phrase utile :

> Le backbone comprend l'image, le neck combine les informations a plusieurs echelles, et la head transforme ces informations en detections.

---

## 9. YOLO classique vs YOLO-OBB

YOLO classique :

```text
class x_center y_center width height
```

YOLO-OBB :

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Dans mon export :

- coordonnees normalisees entre 0 et 1 ;
- une ligne par objet ;
- un fichier `.txt` par image ;
- fichier YAML pour les chemins et les noms de classes.

Phrase utile :

> YOLO classique localise avec des rectangles horizontaux. YOLO-OBB localise avec des rectangles orientes, ce qui correspond mieux aux annotations originales de DOTA.

---

## 10. Ce que le modele apprend

Pendant l'entrainement, le modele compare :

- ses predictions ;
- les annotations reelles.

Il apprend a reduire plusieurs erreurs :

- erreur de classification : mauvaise classe ;
- erreur de localisation : boite mal placee ;
- erreur de taille : boite trop grande ou trop petite ;
- erreur d'orientation : angle incorrect ;
- erreur de confiance : prediction trop confiante ou pas assez.

Lien avec descente de gradient :

- la fonction de perte mesure les erreurs ;
- l'optimiseur ajuste les poids du reseau ;
- le but est de reduire la perte.

Phrase utile :

> Le principe general est le meme que dans la descente de gradient : on mesure une erreur, puis on ajuste les parametres du modele pour la reduire. La difference est que YOLO a des millions de parametres.

---

## 11. Fonction de perte : version intuitive

La loss combine plusieurs objectifs :

1. Bien classer les objets.
2. Bien placer les boites.
3. Bien estimer l'orientation.
4. Eviter les fausses detections.
5. Donner un bon score de confiance.

Phrase simple :

> Si le modele predit un avion mais le place au mauvais endroit, il est penalise. S'il place bien la boite mais predit la mauvaise classe, il est aussi penalise.

Important :

- le modele ne "comprend" pas comme un humain ;
- il optimise une fonction mathematique ;
- la qualite depend des donnees, du format des labels et de l'evaluation.

---

## 12. NMS / filtrage des predictions

YOLO peut proposer plusieurs boites proches pour le meme objet.

NMS = Non-Maximum Suppression.

Role :

- garder la meilleure detection ;
- supprimer les detections trop similaires ;
- eviter de compter le meme objet plusieurs fois.

Avec OBB :

- le raisonnement est similaire ;
- mais il faut comparer des boites orientees, pas seulement horizontales.

Phrase utile :

> Apres la prediction, il faut filtrer les detections. Sinon, le modele peut detecter plusieurs fois le meme objet.

---

## 13. Evaluation : IoU et mAP

### IoU

IoU = Intersection over Union.

Idee :

- mesure le chevauchement entre la boite predite et la boite reelle ;
- plus l'IoU est proche de 1, meilleure est la localisation.

Phrase simple :

> L'IoU repond a la question : est-ce que ma boite predite recouvre bien l'objet annote ?

### mAP

mAP = mean Average Precision.

Idee :

- mesure la qualite globale de detection ;
- tient compte des classes, des scores de confiance et de la localisation.

Phrase simple :

> La mAP resume la capacite du modele a detecter les bons objets, avec les bonnes classes et des boites assez precises.

---

## 14. Pourquoi ne pas utiliser seulement un classifieur

Un classifieur global serait inadapte.

Raison :

- une image DOTA peut contenir plusieurs objets ;
- plusieurs classes peuvent apparaitre dans la meme image ;
- il faut localiser les objets, pas seulement dire ce qu'il y a dans l'image.

Phrase utile :

> Un classifieur pourrait dire qu'une image contient des avions, mais il ne dirait pas combien il y en a ni ou ils se trouvent.

---

## 15. Pourquoi ne pas convertir directement en boites horizontales

Possible, mais moins ideal.

Avantages :

- plus simple ;
- compatible avec plus de modeles ;
- utile pour baseline.

Inconvenients :

- perte de precision geometrique ;
- plus de fond dans les boites ;
- moins adapte aux objets tournes.

Phrase equilibree :

> Je garde les boites horizontales comme option de baseline, mais mon choix principal reste YOLO-OBB parce qu'il respecte mieux les annotations DOTA.

---

## 16. Controle qualite du pretraitement

Ce que j'ai verifie :

- tailles d'images presentes ;
- classes encodees ;
- coordonnees normalisees ;
- boites avec largeur/hauteur positives ;
- coordonnees hors image.

Resultat :

- 127 843 objets au total ;
- 171 objets exclus provisoirement ;
- 127 672 objets utilisables.

Pourquoi exclure ?

- pour eviter des labels invalides ;
- pour ne pas casser l'entrainement ;
- pour garder les donnees brutes intactes.

Phrase utile :

> Je ne supprime pas les donnees originales. Je cree seulement un sous-ensemble propre pour un premier entrainement.

---

## 17. Export YOLO-OBB

Ce qui a ete cree :

- labels train ;
- labels validation ;
- fichier YAML ;
- structure compatible Ultralytics.

Format d'une ligne :

```text
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Exemple d'idee :

- `class_index` : nombre correspondant a la classe ;
- les 8 valeurs suivantes : 4 coins de la boite orientee ;
- toutes les coordonnees sont normalisees entre 0 et 1.

Phrase utile :

> Cette etape fait le pont entre l'exploration des donnees et l'entrainement du modele.

---

## 18. Prochaines etapes

Prochaines actions techniques :

- verifier l'environnement GPU ;
- installer/configurer Ultralytics si necessaire ;
- preparer les images dans la structure YOLO ;
- probablement faire du tuilage, car les images DOTA sont grandes ;
- entrainer un premier modele YOLO-OBB ;
- evaluer avec IoU/mAP ;
- analyser les erreurs ;
- relier les erreurs a une question causale.

Question causale future possible :

> Est-ce que les petits objets ou les objets fortement orientes augmentent la probabilite d'erreur du detecteur ?

---

## 19. Usage de l'IA : comment etre transparent

Je peux etre ouvert sur l'utilisation de l'IA, mais je dois le formuler correctement.

Bonne formulation :

> J'utilise l'IA comme assistant pedagogique et technique. Elle m'aide a structurer le notebook, a expliquer les concepts et a proposer du code. Mais je verifie les resultats, j'execute les cellules, je consulte la documentation officielle et je m'assure de comprendre les choix algorithmiques.

Important a montrer :

- je ne presente pas seulement du code genere ;
- je comprends le format DOTA ;
- je comprends pourquoi YOLO-OBB est choisi ;
- je comprends ce que fait le pretraitement ;
- je comprends les limites actuelles ;
- je sais ce qu'il reste a faire.

Phrase forte :

> L'IA accelere mon apprentissage, mais elle ne remplace pas ma responsabilite de comprendre, verifier et justifier les choix techniques.

Si le professeur insiste :

> Justement, l'objectif de cette presentation est de montrer que je ne me limite pas a executer du code. Je peux expliquer le probleme, les donnees, le format des annotations, l'algorithme choisi et les controles que j'ai faits.

---

## 20. Questions probables et reponses

### Pourquoi YOLO-OBB ?

Parce que DOTA fournit des boites orientees et que les objets aeriennes peuvent etre tournes dans toutes les directions. YOLO-OBB respecte mieux la geometrie que des boites horizontales.

### Pourquoi pas YOLO classique ?

YOLO classique est possible avec des boites horizontales, mais on perd l'orientation. Pour DOTA, cette perte est importante parce que beaucoup d'objets sont diagonaux ou allonges.

### Est-ce que le modele est deja entraine ?

Pas encore. J'ai d'abord construit une base propre : exploration, parsing des annotations, controle qualite, normalisation et export YOLO-OBB. L'entrainement est la prochaine etape.

### Pourquoi est-ce acceptable de ne pas avoir encore entraine ?

Parce qu'un entrainement sans comprendre les donnees peut produire des resultats trompeurs. J'ai d'abord verifie que le format, les classes, les coordonnees et les annotations sont coherents.

### Que representent les 171 objets exclus ?

Ce sont des annotations dont au moins une coordonnee normalisee sort de l'intervalle `[0, 1]`. Pour un premier export propre, je les exclus provisoirement, mais je garde leur trace pour analyse.

### Que veut dire normaliser les coordonnees ?

Transformer les pixels en valeurs entre 0 et 1. Par exemple, si l'image fait 2000 pixels de large et que `x = 500`, alors `x_norm = 0.25`.

### Qu'est-ce que le YAML ?

C'est un fichier de configuration qui indique a Ultralytics ou trouver les images, les labels et les noms des classes.

### Quelle est la difference entre detection et classification ?

La classification donne une classe pour toute l'image. La detection donne plusieurs objets, avec leur classe et leur position.

### Comment YOLO apprend ?

Il compare ses predictions aux annotations, calcule une perte, puis ajuste ses poids par descente de gradient pour reduire les erreurs.

### Comment evaluer le modele ?

Avec des metriques comme IoU et mAP. L'IoU mesure le chevauchement entre boite predite et boite reelle. La mAP resume la qualite globale des detections.

### Quelle est la difficulte principale du projet ?

Les images sont grandes, les objets sont parfois petits, les classes sont desequilibrees, et les objets sont souvent orientes.

---

## 21. Phrases courtes a garder en tete

- "Ce projet est un probleme de detection, pas de classification."
- "DOTA annote les objets avec des boites orientees."
- "YOLO-OBB est choisi parce qu'il respecte cette orientation."
- "Le pretraitement transforme les annotations brutes en labels utilisables par le modele."
- "Je garde les donnees brutes intactes et je cree un sous-ensemble propre pour l'entrainement."
- "L'IA m'aide a apprendre et a verifier, mais je dois comprendre l'algorithme."
- "La prochaine etape est l'entrainement et l'evaluation avec IoU/mAP."

---

## 22. Mini-plan oral flexible

Utiliser comme guide, pas comme texte.

### Debut

- Presenter le sujet.
- Dire que le dataset est DOTA-v1.0.
- Expliquer que c'est de la detection d'objets dans des images aeriennes.

### Avancement

- Parsing des annotations.
- Exploration des classes et tailles.
- Visualisation des boites.
- Analyse des orientations.
- Pretraitement et export YOLO-OBB.

### Algorithme

- YOLO = detecteur rapide en une passe.
- Backbone / neck / head.
- YOLO-OBB ajoute les boites orientees.
- Pertinent pour DOTA.

### Conclusion

- Base de donnees preparee.
- Choix algorithmique justifie.
- Prochaine etape : entrainement + evaluation + analyse causale des erreurs.

---

## 23. Si je perds le fil

Revenir a cette sequence :

1. DOTA = images aeriennes + objets orientes.
2. J'ai explore les donnees pour comprendre le probleme.
3. Les orientations justifient YOLO-OBB.
4. J'ai prepare les labels au bon format.
5. La prochaine etape est l'entrainement.

Phrase de secours :

> Le point central de mon avancement est que le choix du modele vient directement de la structure des donnees : comme DOTA fournit des boites orientees, je prepare le projet autour de YOLO-OBB.
