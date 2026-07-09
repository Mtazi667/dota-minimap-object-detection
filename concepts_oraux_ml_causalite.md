# Concepts oraux a maitriser - ML, detection, causalite

Objectif du fichier : identifier les concepts a etudier pour etre capable de defendre le projet a l'oral.

Ce fichier n'est pas encore un cours complet. Il sert de carte de travail :

- quoi apprendre ;
- pourquoi le professeur peut le demander ;
- lien avec le projet DOTA / YOLO-OBB ;
- niveau de priorite ;
- statut de maitrise.

Derniere mise a jour : 2026-07-09

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
| 0 | Causalite | Confounder | A faire | Question probable du professeur |
| 0 | Causalite | Mediator | A faire | Piege classique dans les DAG |
| 0 | Causalite | Collider | A faire | Piege classique : ne pas controler par erreur |
| 0 | Causalite | DAG | Fragile | Lire et expliquer un schema causal |
| 0 | Causalite | Backdoor path | A faire | Comprendre l'ajustement causal |
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
| Evaluation detection | IoU, TP/FP/FN, precision, recall, mAP | En cours |
| YOLO pratique | confidence, NMS, loss, OBB | A faire |
| Causalite de base | DAG, traitement, outcome, confounder, mediator, collider | A faire |
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
