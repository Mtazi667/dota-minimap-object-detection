# Concepts oraux a maitriser - ML, detection, causalite

Objectif du fichier : identifier les concepts a etudier pour etre capable de defendre le projet a l'oral.

Ce fichier n'est pas encore un cours complet. Il sert de carte de travail :

- quoi apprendre ;
- pourquoi le professeur peut le demander ;
- lien avec le projet DOTA / YOLO-OBB ;
- niveau de priorite ;
- statut de maitrise.

Derniere mise a jour : 2026-07-29

---

## 1. Diagnostic global

Le notebook avance correctement, mais la presentation du 2026-07-09 a revele un risque important : certaines questions conceptuelles simples peuvent bloquer si elles ne sont pas preparees.

Questions deja posees par le professeur :

- schema causal avec `smoking`, `cancer`, `cancer gene` ;
- explication de l'IoU dans le contexte de YOLO.

Conclusion :

- il ne suffit pas d'avoir du code ;
- il faut pouvoir expliquer les concepts sans support immediat ;
- les concepts ML et causalite doivent etre etudies en parallele du notebook.

---

## 2. Niveaux de priorite

### Priorite 0 - A maitriser en premier

Concepts qui peuvent etre demandes directement et qui sont centraux.

### Priorite 1 - A maitriser pour defendre le pipeline

Concepts importants pour expliquer les choix techniques du projet.

### Priorite 2 - A maitriser pour la suite du rapport

Concepts qui deviendront centraux dans les parties causalite et interpretation.

---

## 3. Statuts possibles

- `A faire` : pas encore etudie serieusement.
- `Fragile` : deja vu, mais pas encore defendable oralement.
- `Correct` : explicable simplement, mais a renforcer.
- `Solide` : explicable avec exemple, limites et lien au projet.

---

## 4. Carte principale des concepts

| Priorite | Bloc | Concept | Statut actuel | Pourquoi c'est important |
|---|---|---|---|---|
| 0 | Causalite | Causalite vs correlation | Fragile | Base de toutes les questions causales |
| 0 | Causalite | Traitement / outcome | Fragile | Necessaire pour formuler une question causale |
| 0 | Causalite | Confounder | Premier passage couvert | Question probable du professeur |
| 0 | Causalite | Mediator | Premier passage couvert | Piege classique dans les DAG |
| 0 | Causalite | Collider | Premier passage couvert | Piege classique : ne pas controler par erreur |
| 0 | Causalite | DAG | Premier passage couvert | Lire et expliquer un schema causal |
| 0 | Causalite | Backdoor path | Premier passage couvert | Comprendre l'ajustement causal |
| 0 | Detection | IoU | Fragile | Metrique de base en detection d'objets |
| 0 | Detection | Precision / recall | A faire | Base de l'evaluation du modele |
| 0 | Detection | mAP | A faire | Metrique principale pour detecteurs |
| 0 | YOLO | Prediction de classe et boite | Correct | Expliquer ce que le modele produit |
| 0 | YOLO | Confidence score | Fragile | Comprendre les detections retenues |
| 0 | YOLO | NMS | A faire | Comprendre le filtrage des detections |
| 0 | YOLO-OBB | Boite orientee vs horizontale | Correct | Justifie le choix YOLO-OBB |
| 1 | ML general | Train / validation / test | Correct | Eviter fuite et evaluation trompeuse |
| 1 | ML general | Overfitting / underfitting | Fragile | Expliquer generalisation |
| 1 | ML general | Fonction de perte | Fragile | Comprendre apprentissage modele |
| 1 | ML general | Descente de gradient | Correct | Relier au mini-exemple du notebook |
| 1 | Deep learning | Reseau de neurones | Fragile | Base de YOLO |
| 1 | Deep learning | CNN / convolution | A faire | Base de vision par ordinateur |
| 1 | YOLO | Backbone | Correct | Architecture YOLO |
| 1 | YOLO | Neck | Fragile | Detection multi-echelle |
| 1 | YOLO | Head | Fragile | Sorties du modele |
| 1 | Donnees | Desequilibre des classes | Correct | DOTA est desequilibre |
| 1 | Donnees | Objets difficiles | Correct | Analyse DOTA et erreurs futures |
| 1 | Donnees | Tuilage | Fragile | Necessaire pour grandes images |
| 1 | Methodologie | Data leakage | Correct | Risque majeur avec tuiles |
| 2 | Causalite appliquee | Question causale | A faire | Partie 4 du projet |
| 2 | Causalite appliquee | Variables d'ajustement | A faire | Partie 4/5 du projet |
| 2 | Causalite appliquee | Effet causal moyen | A faire | Estimation causale |
| 2 | Causalite appliquee | Effet heterogene | A faire | Lien avec arbres causaux |
| 2 | Causalite appliquee | Arbres causaux | A faire | Partie 5 du projet |
| 2 | Causalite appliquee | Forets causales | A faire | Partie 5 du projet |
| 2 | Analyse erreurs | Faux positif | A faire | Evaluer detections incorrectes |
| 2 | Analyse erreurs | Faux negatif | A faire | Evaluer objets manques |
| 2 | Analyse erreurs | Erreur de localisation | A faire | Lien direct avec IoU |
| 2 | Analyse erreurs | Erreur de classification | A faire | Classe predite incorrecte |

---

## 5. Concepts deja presents dans le notebook, mais a rendre oraux

Ces concepts sont deja dans le notebook ou proches du notebook, mais il faut etre capable de les expliquer sans lire le code.

| Concept | Section notebook | Risque oral |
|---|---|---|
| Format DOTA | Exploration | Expliquer les 8 coordonnees |
| Boites orientees | Exploration + YOLO-OBB | Justifier pourquoi OBB |
| Boites horizontales | Pretraitement | Expliquer perte d'information |
| Normalisation des coordonnees | Pretraitement | Expliquer pourquoi entre 0 et 1 |
| Encodage des classes | Pretraitement | Expliquer pourquoi noms -> entiers |
| Objets difficiles | Exploration | Expliquer impact sur performance |
| Desequilibre de classes | Exploration | Expliquer pourquoi performance globale peut tromper |
| Tuilage | Mentionne, pas encore implemente | Expliquer risque de fuite |
| YOLO-OBB | Section modele | Expliquer architecture et sortie |
| Loss / gradient | Mini-exemple SGD | Relier a entrainement YOLO |

---

## 6. Concepts absents ou trop peu vus

Ces concepts doivent etre ajoutes dans l'apprentissage oral avant d'aller trop loin.

### Detection / evaluation

- IoU
- seuil IoU
- true positive
- false positive
- false negative
- precision
- recall
- courbe precision-recall
- AP
- mAP
- NMS
- confidence threshold

### Causalite

- causalite vs association
- DAG
- traitement
- outcome
- confounder
- mediator
- collider
- backdoor path
- ajustement
- effet causal
- effet causal moyen
- effet heterogene
- variable observee vs non observee

### Machine learning general

- apprentissage supervise
- features
- labels
- train / validation / test
- generalisation
- overfitting
- underfitting
- hyperparametres
- batch size
- epoch
- learning rate
- loss

### Deep learning / vision

- reseau de neurones
- poids
- convolution
- feature map
- backbone
- neck
- head
- transfert learning
- fine-tuning
- GPU / CUDA

### Methodologie projet

- data leakage
- split par image
- split par tuile
- stratification
- baseline
- comparaison de modeles
- reproductibilite
- limites experimentales

---

## 7. Ordre d'etude recommande

### Phase A - Questions que le professeur vient deja de poser

1. IoU
2. Precision / recall
3. mAP
4. DAG simple
5. Confounder / mediator / collider
6. Exemple `smoking`, `cancer`, `cancer gene`

Objectif : ne plus etre surpris par une question courte.

### Phase B - Defense du choix YOLO-OBB

1. Detection vs classification
2. YOLO en une passe
3. Backbone / neck / head
4. Confidence score
5. NMS
6. Boites orientees vs horizontales
7. Loss YOLO : classe + boite + confiance + orientation

Objectif : expliquer l'algorithme sans code.

### Phase C - Methodologie d'entrainement

1. Train / validation / test
2. Overfitting / underfitting
3. Epoch / batch size / learning rate
4. GPU / CUDA
5. Tuilage
6. Data leakage
7. Baseline vs modele moderne

Objectif : defendre les choix experimentaux.

### Phase D - Causalite appliquee au projet

1. Question causale dans DOTA
2. Traitement possible : petit objet, forte orientation, densite, classe rare
3. Outcome possible : erreur de detection, faux negatif, IoU faible
4. Variables d'ajustement
5. Arbres causaux
6. Forets causales
7. Interpretation : prediction vs causalite

Objectif : preparer les parties 4 et 5 du projet.

---

## 8. Questions orales probables

### Detection / YOLO

- Qu'est-ce que l'IoU ?
- Pourquoi l'IoU ne suffit pas seule ?
- Quelle est la difference entre precision et recall ?
- C'est quoi la mAP ?
- Pourquoi YOLO plutot qu'un classifieur ?
- Pourquoi YOLO-OBB plutot que YOLO classique ?
- Que fait la NMS ?
- Que represente le confidence score ?
- Qu'est-ce que la loss du modele penalise ?
- Pourquoi les petits objets sont difficiles ?

### Donnees / methodologie

- Pourquoi separer train et validation ?
- Qu'est-ce qu'une fuite de donnees ?
- Pourquoi le tuilage peut creer une fuite ?
- Pourquoi les classes desequilibrees posent probleme ?
- Pourquoi exclure les 171 annotations hors image ?
- Pourquoi normaliser les coordonnees ?
- Pourquoi garder les boites orientees ?

### Causalite

- Difference entre correlation et causalite ?
- C'est quoi un confounder ?
- C'est quoi un mediator ?
- C'est quoi un collider ?
- Dans le schema smoking/cancer/gene, quel est le probleme causal ?
- Qu'est-ce qu'un traitement ?
- Qu'est-ce qu'un outcome ?
- Quelles variables faut-il ajuster ?
- Pourquoi prediction et causalite ne sont pas la meme chose ?

### Projet DOTA

- Quelle est ta question causale future ?
- Quel serait ton traitement ?
- Quel serait ton outcome ?
- Quelles variables d'ajustement envisages-tu ?
- Comment relier les erreurs de YOLO a une analyse causale ?

---

## 9. Liens directs avec le projet DOTA

| Concept | Lien avec le projet |
|---|---|
| IoU | Mesurer si une boite predite recouvre bien une boite reelle |
| mAP | Evaluer globalement le detecteur |
| Faux negatif | Objet present mais non detecte |
| Faux positif | Detection d'un objet qui n'existe pas |
| Orientation | DOTA contient des objets tournes |
| Petit objet | Vehicules et bateaux peuvent etre difficiles |
| Classe rare | Certaines classes auront moins d'exemples |
| Tuilage | Necessaire pour grandes images |
| Data leakage | Tuiles proches ne doivent pas etre dans train et validation |
| Confounder | Variable qui influence a la fois le traitement et l'erreur |
| Traitement causal | Exemple : objet petit vs grand |
| Outcome causal | Exemple : erreur du detecteur |
| Arbres causaux | Trouver pour quels objets l'effet est plus fort |

---

## 10. Definition du socle minimal a atteindre

Avant de reprendre un gros entrainement, l'objectif est d'atteindre ce niveau :

### Pour chaque concept priorite 0

Je dois pouvoir dire :

1. definition simple ;
2. exemple hors projet ;
3. exemple dans DOTA ;
4. piege courant ;
5. pourquoi ca compte pour mon projet.

### Exemple de niveau attendu

Pour `IoU`, je dois savoir :

- definir intersection et union ;
- expliquer pourquoi `IoU = 1` est parfait ;
- expliquer pourquoi `IoU = 0` est mauvais ;
- dire que des seuils IoU servent a decider si une detection est correcte ;
- relier ca aux boites YOLO-OBB.

Pour `confounder`, je dois savoir :

- dire que c'est une variable qui influence le traitement et l'outcome ;
- donner l'exemple smoking / cancer / gene si pertinent ;
- expliquer pourquoi il faut ajuster ;
- expliquer qu'une mauvaise variable controlee peut biaiser l'analyse.

---

## 11. Plan d'action concret

### Etape 1

Etudier et expliquer :

- IoU
- true positive / false positive / false negative
- precision / recall
- mAP

### Etape 2

Etudier et expliquer :

- DAG
- traitement
- outcome
- confounder
- mediator
- collider

### Etape 3

Etudier et expliquer :

- YOLO
- YOLO-OBB
- NMS
- confidence score
- loss

### Etape 4

Etudier et expliquer :

- overfitting
- validation
- data leakage
- tuilage
- baseline

### Etape 5

Formuler pour le projet :

- question causale provisoire ;
- traitement ;
- outcome ;
- variables d'ajustement ;
- variables a ne pas ajuster ;
- methode causale envisagee.

---

## 12. Regle de travail pour la suite

Pour chaque concept etudie, produire une mini-fiche :

```text
Concept :
Definition simple :
Intuition :
Exemple general :
Lien avec DOTA :
Ou ca apparait dans notre notebook/pipeline :
Piege frequent :
Phrase orale :
Question probable :
Reponse courte :
```

Ces mini-fiches pourront ensuite etre reprises dans :

- le notebook ;
- le rapport ;
- une future presentation ;
- une preparation orale rapide.

---

## 13. Suivi rapide

Objectif : garder une vision claire de l'avancement sans passer trop de temps sur la preparation orale.

### Blocs a couvrir

| Bloc | Concepts | Statut |
|---|---|---|
| Evaluation detection | IoU, TP/FP/FN, precision, recall, mAP | Couvert, a pratiquer |
| YOLO pratique | confidence, NMS, loss, OBB | Premier passage couvert |
| Causalite de base | DAG, traitement, outcome, confounder, mediator, collider | Premier passage couvert |
| Methodologie ML | overfitting, validation, leakage, baseline | A faire |
| Causalite appliquee DOTA | traitement, outcome, ajustements, erreurs du detecteur | A faire |

Regle : couvrir 2 ou 3 cartes, puis revenir au projet si le bloc est suffisamment clair.

---

## 14. Cartes orales - Evaluation en detection

### Carte 1 - IoU

Question probable du prof :

- Comment sais-tu si une boite predite par YOLO est correcte ?

Reponse en 20 secondes :

- On utilise l'IoU, pour Intersection over Union. Ca mesure le chevauchement entre la boite predite et la boite reelle. Si les deux boites se superposent parfaitement, l'IoU vaut 1. Si elles ne se touchent pas, l'IoU vaut 0. En detection, on utilise souvent un seuil d'IoU pour decider si une prediction compte comme correcte.

Intuition :

- Intersection = partie commune entre les deux boites.
- Union = surface totale couverte par les deux boites.
- IoU = intersection / union.

Mini-exemple :

- Boite predite presque au bon endroit : IoU elevee.
- Bonne classe mais boite loin de l'objet : IoU faible.

Lien avec DOTA :

- Dans DOTA, on veut savoir si YOLO-OBB localise bien les avions, bateaux, vehicules, etc.
- Comme les objets peuvent etre tournes, l'IoU doit idealement comparer des boites orientees.

Ou ca apparait dans notre notebook/pipeline :

- On n'a pas encore calcule l'IoU dans le notebook, parce qu'il faut d'abord entrainer un modele et obtenir des predictions.
- Les labels YOLO-OBB exportes serviront de boites reelles.
- Les futures predictions YOLO-OBB seront comparees a ces boites reelles avec une metrique de type IoU.

Piege :

- Predire la bonne classe ne suffit pas. Si la boite est mauvaise, la detection est mauvaise.

Phrase orale utile :

- "L'IoU me permet de mesurer la qualite geometrique de la detection, pas seulement la classe predite."

Statut : Correct, a pratiquer oralement.

---

### Carte 2 - True positive, false positive, false negative

Question probable du prof :

- Quand est-ce qu'une detection est consideree bonne ou mauvaise ?

Reponse en 20 secondes :

- Une detection est un true positive si le modele predit la bonne classe et que la boite recouvre assez bien un objet reel, selon un seuil d'IoU. Un false positive est une detection annoncee par le modele mais qui ne correspond pas a un vrai objet. Un false negative est un objet reel que le modele n'a pas detecte.

Intuition :

- True positive = objet trouve correctement.
- False positive = fausse alerte.
- False negative = objet manque.

Mini-exemple :

- YOLO detecte un avion au bon endroit : true positive.
- YOLO detecte un avion dans une zone vide : false positive.
- Un avion existe dans l'image mais YOLO ne le detecte pas : false negative.

Lien avec DOTA :

- Les petits objets peuvent augmenter les false negatives.
- Les zones denses peuvent augmenter les false positives.
- Les objets orientes peuvent causer des erreurs de localisation.

Ou ca apparait dans notre notebook/pipeline :

- L'exploration a montre des objets petits, des classes desequilibrees et des boites orientees.
- Apres entrainement, on pourra transformer les predictions en categories d'erreurs : objet trouve, objet manque, fausse detection, mauvaise localisation.
- Ces erreurs seront aussi utiles plus tard pour la partie causale du projet.

Piege :

- Une prediction peut etre mauvaise meme si elle "semble proche" : il faut une regle, souvent basee sur l'IoU.

Phrase orale utile :

- "Les erreurs du detecteur ne sont pas toutes du meme type : il peut inventer un objet, manquer un objet, ou mal localiser un objet."

Statut : A renforcer.

---

---

### Carte 3 - Precision et recall

Question probable du prof :

- Quelle est la difference entre precision et recall ?

Reponse en 20 secondes :

- La precision repond a la question : parmi les detections du modele, combien sont correctes ? Le recall repond a la question : parmi les vrais objets presents, combien le modele a-t-il retrouves ? Donc la precision mesure les fausses alertes, tandis que le recall mesure les objets manques.

Intuition :

- Precision elevee = peu de fausses detections.
- Recall eleve = peu d'objets oublies.

Mini-exemple :

- Modele tres prudent : peu de detections, precision haute, recall bas.
- Modele trop agressif : beaucoup de detections, recall haut, precision basse.

Lien avec DOTA :

- Si on veut detecter tous les petits vehicules, il faut un bon recall.
- Si on veut eviter de fausses detections dans des zones complexes, il faut une bonne precision.

Ou ca apparait dans notre notebook/pipeline :

- Le notebook a deja montre que certaines classes sont beaucoup plus frequentes que d'autres.
- Une precision globale peut cacher de mauvaises performances sur des classes rares.
- Quand on evaluera YOLO-OBB, il faudra regarder precision et recall, idealement par classe.

Piege :

- Precision et recall sont souvent en tension. Ameliorer l'un peut degrader l'autre.

Phrase orale utile :

- "La precision me dit si les detections sont fiables; le recall me dit si le modele oublie beaucoup d'objets."

Statut : A renforcer.

---

### Carte 4 - mAP

Question probable du prof :

- C'est quoi la mAP et pourquoi on l'utilise en detection ?

Reponse en 20 secondes :

- La mAP veut dire mean Average Precision. Elle resume la performance d'un detecteur en combinant precision et recall, puis en moyennant le resultat sur les classes. C'est utile parce qu'un detecteur ne produit pas seulement une classe : il produit plusieurs boites avec des scores de confiance.

Intuition :

- Pour une classe, on regarde comment la precision et le recall changent quand on varie le seuil de confiance.
- Cette relation donne une courbe precision-recall.
- L'AP resume cette courbe pour une classe.
- La mAP moyenne les AP sur plusieurs classes.

Mini-exemple :

- Si le modele detecte tres bien les `plane`, AP plane sera elevee.
- S'il detecte mal les `bridge`, AP bridge sera faible.
- La mAP resume les performances sur toutes les classes.

Lien avec DOTA :

- DOTA contient 15 classes.
- Certaines classes sont tres frequentes, comme `ship` ou `small-vehicle`.
- D'autres sont plus rares.
- La mAP aide a evaluer le detecteur sans regarder seulement une classe dominante.

Ou ca apparait dans notre notebook/pipeline :

- Le notebook a deja une section `Evaluation prevue du detecteur` qui introduit la mAP.
- La vraie mAP sera calculee apres l'entrainement YOLO-OBB, quand on aura des predictions sur la validation.
- Il faudra idealement regarder la mAP globale et les resultats par classe.

Piege :

- Une bonne mAP globale peut cacher une mauvaise performance sur une classe rare ou difficile.
- Il ne faut pas conclure que le modele est bon partout seulement avec une moyenne.

Phrase orale utile :

- "La mAP donne une evaluation globale du detecteur, mais dans DOTA je devrai aussi regarder les performances par classe, taille et orientation."

Statut : A renforcer.

---

### Carte 5 - Confidence score

Question probable du prof :

- Que represente le score de confiance dans YOLO ?

Reponse en 20 secondes :

- Le score de confiance indique a quel point le modele est confiant dans une detection. Il sert a classer et filtrer les predictions. Une detection avec une confiance tres faible peut etre ignoree, tandis qu'une detection tres confiante a plus de chances d'etre gardee.

Intuition :

- Le modele ne dit pas seulement "il y a un avion".
- Il dit plutot "je pense qu'il y a un avion ici, avec une certaine confiance".

Mini-exemple :

- Detection `plane` avec confiance 0.92 : prediction assez forte.
- Detection `plane` avec confiance 0.18 : prediction faible, probablement filtree.

Lien avec DOTA :

- Les images aeriennes peuvent contenir des zones complexes : batiments, routes, ports, petits objets.
- Le modele peut produire des detections incertaines dans ces zones.
- Le score de confiance aide a decider quelles detections garder.

Ou ca apparait dans notre notebook/pipeline :

- Le notebook n'a pas encore de predictions YOLO, donc pas encore de scores de confiance reels.
- Apres entrainement, chaque prediction YOLO-OBB aura un score.
- Ces scores serviront a calculer precision, recall et mAP.

Piege :

- Une confiance elevee ne garantit pas que la boite est bien placee.
- Il faut combiner confiance, classe predite et IoU.

Phrase orale utile :

- "Le score de confiance sert a filtrer les detections, mais il ne remplace pas l'evaluation geometrique avec l'IoU."

Statut : A faire pratiquer.

---

### Carte 6 - NMS

Question probable du prof :

- Pourquoi YOLO a besoin de NMS ?

Reponse en 20 secondes :

- YOLO peut predire plusieurs boites pour le meme objet. La NMS, Non-Maximum Suppression, garde la detection la plus confiante et supprime les detections trop proches qui representent probablement le meme objet.

Intuition :

- Un seul avion peut generer plusieurs detections proches.
- Sans filtrage, on compterait plusieurs fois le meme objet.
- La NMS nettoie les predictions.

Mini-exemple :

- Trois boites detectent le meme avion.
- Elles ont des confiances 0.91, 0.84 et 0.70.
- La NMS garde souvent celle a 0.91 et supprime les autres si elles se chevauchent trop.

Lien avec DOTA :

- DOTA peut contenir des zones denses avec beaucoup de vehicules ou bateaux proches.
- La NMS est utile, mais elle doit etre reglee prudemment.
- Si elle est trop agressive, elle peut supprimer des objets reels proches les uns des autres.

Ou ca apparait dans notre notebook/pipeline :

- La NMS n'apparaitra vraiment qu'apres les predictions YOLO-OBB.
- Elle fait partie du post-traitement du modele.
- Elle influencera les faux positifs, faux negatifs, precision et recall.

Piege :

- NMS n'est pas l'entrainement du modele; c'est une etape apres prediction.
- Elle ne corrige pas un modele mauvais, elle filtre seulement les detections candidates.

Phrase orale utile :

- "La NMS evite de compter plusieurs fois le meme objet, mais dans les zones denses elle peut aussi supprimer des objets proches si elle est trop stricte."

Statut : A faire pratiquer.

---

### Carte 7 - Loss YOLO

Question probable du prof :

- Qu'est-ce que YOLO apprend pendant l'entrainement ?

Reponse en 20 secondes :

- YOLO apprend en minimisant une loss, c'est-a-dire une mesure d'erreur entre ses predictions et les annotations. Cette erreur penalise surtout trois choses : une mauvaise localisation de la boite, une mauvaise classe et une mauvaise confiance. En YOLO-OBB, la localisation doit aussi respecter l'orientation de l'objet.

Intuition :

- La loss est le signal qui dit au modele dans quelle direction corriger ses poids.
- Plus la prediction est loin de l'annotation, plus la penalite est forte.
- Pendant l'entrainement, le modele fait beaucoup de petites corrections.

Mini-exemple :

- Si un avion est annote a gauche de l'image mais que YOLO le predit trop a droite, la partie localisation de la loss augmente.
- Si la boite est bonne mais la classe predite est `ship` au lieu de `plane`, la partie classification augmente.
- Si le modele est tres confiant dans une mauvaise detection, l'erreur est plus grave.

Lien avec DOTA :

- Dans DOTA, les objets sont souvent petits, denses et orientes.
- La loss doit donc aider le modele a apprendre des boites precises, pas seulement des classes.
- Pour YOLO-OBB, une boite mal orientee peut etre penalisee meme si le centre est proche de l'objet.

Ou ca apparait dans notre notebook/pipeline :

- Le notebook prepare les labels YOLO-OBB qui serviront de verite terrain pendant l'entrainement.
- La cellule d'entrainement est preparee mais desactivee par defaut avec `RUN_YOLO_OBB_TRAINING = False`.
- Quand l'entrainement sera lance, Ultralytics affichera des pertes comme la loss de boite et la loss de classe.

Piege :

- La loss n'est pas la meme chose que la mAP.
- La loss sert a entrainer le modele; la mAP sert a evaluer sa qualite sur validation.
- Une loss qui baisse est bon signe, mais il faut verifier les metriques de validation.

Phrase orale utile :

- "La loss est le signal d'apprentissage : elle penalise les erreurs de boite, de classe et de confiance, puis le modele ajuste ses poids pour reduire cette erreur."

Statut : A renforcer.

---

### Carte 8 - YOLO classique vs YOLO-OBB

Question probable du prof :

- Pourquoi utiliser YOLO-OBB au lieu d'un YOLO classique ?

Reponse en 20 secondes :

- Un YOLO classique predit des boites horizontales, alignees avec les axes de l'image. YOLO-OBB predit des boites orientees, donc des boites qui peuvent tourner avec l'objet. Dans DOTA, c'est important parce que les avions, bateaux, ponts et vehicules peuvent etre dans n'importe quelle orientation.

Intuition :

- Boite horizontale = rectangle droit, meme si l'objet est diagonal.
- Boite orientee = rectangle qui suit mieux la direction de l'objet.
- Moins de fond inutile dans la boite signifie une localisation plus precise.

Mini-exemple :

- Un avion diagonal dans une image satellite.
- Avec une boite horizontale, la boite englobe l'avion plus beaucoup de fond.
- Avec une boite orientee, la boite suit l'avion et represente mieux sa vraie forme.

Lien avec DOTA :

- DOTA fournit directement des annotations avec 4 coins, donc le dataset est naturellement compatible avec les boites orientees.
- Notre exploration a montre que l'orientation mediane est loin de 0 degre.
- Cela justifie de garder les OBB comme representation principale.

Ou ca apparait dans notre notebook/pipeline :

- Le notebook exporte les labels YOLO-OBB au format `class_index x1 y1 x2 y2 x3 y3 x4 y4`.
- On garde aussi des boites horizontales derivees pour une baseline simple ou pour analyser certaines erreurs.
- La section YOLO explique deja pourquoi YOLO-OBB est plus coherent avec DOTA.

Piege :

- YOLO-OBB n'est pas automatiquement meilleur dans tous les cas.
- Il est plus adapte a DOTA, mais il demande des annotations correctes, un format exact et une evaluation compatible avec les boites orientees.

Phrase orale utile :

- "YOLO classique localise avec des rectangles horizontaux; YOLO-OBB localise avec des rectangles orientes, ce qui correspond beaucoup mieux aux objets de DOTA."

Statut : A renforcer.

---

## 15. Cartes orales - Causalite de base et application DOTA

### Carte 9 - DAG

Question probable du prof :

- A quoi sert un DAG en causalite ?

Reponse en 20 secondes :

- Un DAG est un graphe oriente sans cycle qui represente mes hypotheses sur les relations causales. Les fleches indiquent la direction supposee des causes. Il m'aide a identifier les chemins de confusion, choisir les variables d'ajustement et eviter de controler sur un mediator ou un collider.

Mini-exemple :

- `classe -> tres petit`
- `classe -> detection`
- `tres petit -> detection`

Lien DOTA :

- La classe peut influencer la taille typique et la facilite de detection. Le DAG montre pourquoi la classe doit etre consideree avant d'interpreter la difference petits/grands.

Piege :

- Un DAG n'est pas appris automatiquement par les correlations. Il formalise des hypotheses qui doivent etre defendues.

Phrase orale utile :

- "Le DAG rend mes hypotheses visibles et me dit quels chemins non causaux je dois bloquer."

Statut : Premier passage couvert, a redessiner sans support.

---

### Carte 10 - Traitement, outcome et estimand

Question probable du prof :

- Quelle est exactement ta question causale ?

Reponse en 20 secondes :

- Mon traitement vaut 1 si l'objet appartient au premier quartile de taille relative calcule sur train. Mon outcome vaut 1 si le modele produit une detection de meme classe, de confiance au moins 0.25 et d'IoU orientee au moins 0.50. J'estime l'effet moyen sur les objets uniques de validation.

Lien DOTA :

- Le seuil est defini avec train pour ne pas le choisir apres avoir vu les outcomes validation.
- Une seule instance est gardee par objet original pour eviter de surponderer les objets dupliques par la grille.

Piege :

- "Petit" est une condition observationnelle. Plusieurs interventions differentes pourraient rendre un objet petit.

Phrase orale utile :

- "Je definis D, Y et la population avant l'estimation, sinon le mot effet reste ambigu."

Statut : Premier passage couvert.

---

### Carte 11 - Confounder

Question probable du prof :

- Qu'est-ce qu'un confounder et quel est ton exemple dans DOTA ?

Reponse en 20 secondes :

- Un confounder est une cause commune du traitement et de l'outcome. Dans DOTA, la classe est un exemple : certaines classes sont typiquement plus petites et la classe influence aussi la facilite de detection. Sans ajuster sur elle, la comparaison petits/grands melange taille et composition des classes.

Schema :

- `tres petit <- classe -> detection correcte`

Piege :

- Une variable associee aux deux n'est pas automatiquement un confounder. Elle doit preceder causalement les deux dans le DAG.

Phrase orale utile :

- "J'ajuste sur la classe pour bloquer un chemin arriere, pas parce qu'elle est seulement correlee."

Statut : Premier passage couvert.

---

### Carte 12 - Mediator

Question probable du prof :

- Pourquoi ne faut-il pas toujours ajuster sur toutes les variables disponibles ?

Reponse en 20 secondes :

- Un mediator est cause par le traitement et transmet une partie de son effet vers l'outcome. Si je veux l'effet total, l'ajuster bloquerait precisement une partie de l'effet recherche. Par exemple, la petite taille peut reduire la qualite des features, qui reduit ensuite la detection.

Schema :

- `tres petit -> qualite des features -> detection`

Lien DOTA :

- Le score de confiance et les features internes sont produits apres le traitement, donc ne sont pas des ajustements principaux.

Piege :

- Ajuster sur un mediator n'est pas toujours faux, mais cela change la question vers un effet direct.

Phrase orale utile :

- "Plus de variables n'est pas toujours mieux : une variable post-traitement peut retirer l'effet que je cherche."

Statut : Premier passage couvert.

---

### Carte 13 - Collider

Question probable du prof :

- Qu'est-ce qu'un collider et pourquoi est-ce un piege ?

Reponse en 20 secondes :

- Un collider est une consequence commune de deux variables. Sans conditionnement, le chemin est ferme. Si je conditionne sur le collider, je peux creer une association artificielle entre ses causes. Par exemple, petite taille et scene complexe peuvent toutes deux influencer le fait qu'une tuile soit retenue.

Schema :

- `tres petit -> tuile selectionnee <- scene complexe`

Lien DOTA :

- Le rejet des fragments ambigus peut selectionner certaines positions ou tailles. Cette limite doit etre declaree.

Piege :

- Un collider est l'inverse logique d'un confounder pour l'ajustement : le controler peut ouvrir un biais.

Phrase orale utile :

- "Je ne controle pas une variable seulement parce qu'elle est disponible ; je regarde sa place dans le DAG."

Statut : Premier passage couvert.

---

### Carte 14 - Backdoor path et ajustement

Question probable du prof :

- Comment choisis-tu les variables d'ajustement ?

Reponse en 20 secondes :

- Je cherche les chemins entre traitement et outcome qui commencent par une fleche entrant dans le traitement. Je choisis un ensemble de variables pre-traitement qui bloque ces backdoors, sans bloquer les mediateurs ni ouvrir les colliders.

Ensemble principal DOTA :

- classe ;
- orientation ;
- ratio de forme ;
- densite ;
- source et GSD ;
- position du centre dans la tuile.

Variables evitees :

- aire exacte, car elle definit D ;
- fraction conservee, car elle releve de la selection par tuilage et peut etre
  une variable post-traitement ;
- confiance et IoU, car elles sont post-traitement ou outcome.

Phrase orale utile :

- "Mon ensemble d'ajustement vient du DAG et de la chronologie causale, pas d'une selection automatique de toutes les colonnes."

Statut : Premier passage couvert.

---

### Carte 15 - AIPW et cross-fitting

Question probable du prof :

- Pourquoi utiliser AIPW plutot qu'une simple difference de moyennes ?

Reponse en 20 secondes :

- La difference brute peut etre confondue. AIPW combine un modele de l'outcome et un score de propension, puis corrige les predictions avec des residus ponderes. Le cross-fitting produit ces predictions sur des folds non utilises pour entrainer les modeles nuisances, groupes par image source.

Intuition :

- g-computation predit les deux mondes ;
- propension mesure le mecanisme de traitement observe ;
- la correction residuelle rapproche l'estimation des outcomes observes.

Piege :

- Doublement robuste ne signifie pas robuste aux confounders non observes, a la mauvaise positivite ou a un traitement mal defini.

Phrase orale utile :

- "AIPW combine deux modeles et une correction hors fold, mais mes hypotheses causales restent indispensables."

Statut : Premier passage couvert, formule a revoir.

---

### Carte 16 - Arbre causal et effet heterogene

Question probable du prof :

- Pourquoi un arbre causal si tu as deja un effet moyen ?

Reponse en 20 secondes :

- L'effet moyen peut cacher des sous-groupes. Un arbre causal cherche des divisions ou l'effet estime differe, par exemple selon la classe, l'orientation ou la densite. Dans mon pipeline, une moitie des images choisit les divisions et l'autre estime les effets des feuilles pour limiter le biais.

Lien DOTA :

- Une penalite de petite taille peut etre plus forte dans une scene dense ou pour certaines classes.

Piege :

- Une feuille extreme avec peu d'images peut etre du bruit. L'arbre est exploratoire et demande des intervalles et une validation externe.

Phrase orale utile :

- "L'arbre ne remplace pas l'ATE ; il explore ou l'effet peut etre different et avec quelle incertitude."

Statut : Premier passage couvert.

---
