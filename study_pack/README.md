# Pack d'etude hors ligne - projet DOTA

Ce dossier est concu pour un vol de 8 heures sans Internet. Il ne faut pas
tout lire passivement. Le but est de pouvoir expliquer le projet, refaire les
raisonnements essentiels et reconnaitre les limites methodologiques.

## Avant de couper Internet

1. Ouvrir ce fichier et verifier que les liens fonctionnent.
2. Ouvrir `projet_dota.ipynb` avec le noyau `Python (DOTA GPU)`.
3. Executer :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\verify_offline_ready.py
```

4. Garder le dossier du projet sur le disque local.
5. Brancher le chargeur si un entrainement doit etre relance.

Le modele de base `yolo26n.pt`, l'environnement `.venv_gpu`, les donnees
preparees et les poids entraines sont deja locaux. Ils sont volontairement
ignores par Git parce qu'ils sont volumineux.

## Methode de travail pour chaque bloc

Utiliser toujours le meme cycle :

1. Fermer le document et repondre de memoire aux questions de depart.
2. Lire seulement ce qui manque.
3. Expliquer le concept a voix basse en 20 secondes.
4. Faire un mini-exercice sans regarder la correction.
5. Noter `0`, `1` ou `2` :
   - `0` : je bloque ;
   - `1` : je comprends avec de l'aide ;
   - `2` : je peux l'expliquer avec un exemple et une limite.

Une notion n'est pas maitrisee parce qu'elle semble familiere. Elle est
maitrisee quand la reponse sort sans support.

## Programme principal de 8 heures

Chaque bloc dure 50 minutes, suivi de 10 minutes de pause. Les pauses font
partie du programme : elles evitent de confondre fatigue et incomprehension.

### Heure 1 - Carte du projet et du bareme

- Lire [01_projet_de_bout_en_bout.md](01_projet_de_bout_en_bout.md).
- Dessiner de memoire le pipeline : images brutes -> tuiles -> labels ->
  entrainement -> predictions -> metriques -> table causale -> effets.
- Associer chaque etape aux six questions du sujet.
- Expliquer pourquoi le split est fait par image source et non par tuile.

Livrable personnel : une page manuscrite contenant le pipeline et les six
questions.

### Heure 2 - Detection, IoU et metriques

- Etudier les cartes 1 a 6 de `concepts_oraux_ml_causalite.md`.
- Lire les sections IoU, matching, precision, recall, AP et mAP dans
  [02_detection_yolo_obb.md](02_detection_yolo_obb.md).
- Faire les exercices D1 a D6 de
  [05_exercices_sans_corrige.md](05_exercices_sans_corrige.md).

Livrable personnel : expliquer la difference entre une prediction confiante et
une prediction correcte.

### Heure 3 - YOLO, OBB et entrainement

- Etudier les cartes 7 et 8 de `concepts_oraux_ml_causalite.md`.
- Revoir backbone, neck, head, loss, transfert learning et NMS.
- Comparer HBB et OBB sans dire qu'OBB est automatiquement meilleur.
- Lire dans le notebook les cellules de la question 3.

Livrable personnel : reponse de 90 secondes a "Pourquoi YOLO-OBB pour DOTA ?".

### Heure 4 - Tuilage, donnees et reproductibilite

- Relire la partie pipeline de
  [01_projet_de_bout_en_bout.md](01_projet_de_bout_en_bout.md).
- Examiner `preparation_summary.json` et les manifestes CSV.
- Repondre aux questions sur les fragments de bord, les tuiles negatives, les
  classes rares et la stratification.
- Faire les exercices P1 a P6.

Livrable personnel : defendre les choix 1024 pixels, stride 824 en train et
stride 1024 en validation.

### Heure 5 - DAG et vocabulaire causal

- Lire [03_causalite_appliquee.md](03_causalite_appliquee.md) jusqu'a la
  section sur les hypotheses d'identification.
- Refaire les DAG `smoking -> cancer` et DOTA sans regarder.
- Classer chaque variable comme traitement, outcome, confounder, mediator,
  collider ou variable de precision.
- Faire les exercices C1 a C7.

Livrable personnel : expliquer pourquoi correlation n'est pas causalite avec
un exemple DOTA.

### Heure 6 - Estimation causale

- Etudier regression ajustee, score de propension, IPW, g-computation et AIPW.
- Comprendre le role du cross-fitting et du bootstrap groupe par image.
- Etudier l'arbre causal et la foret sur pseudo-outcomes.
- Faire les exercices C8 a C14.

Livrable personnel : comparer AIPW et difference brute en moins d'une minute.

### Heure 7 - Resultats reels et limites

- Lire [04_resultats_et_interpretation.md](04_resultats_et_interpretation.md).
- Ouvrir les figures dans `outputs/analysis/`.
- Trouver trois constats predictifs, deux constats causaux et cinq limites.
- Verifier que chaque conclusion est soutenue par une mesure.

Livrable personnel : une conclusion de deux minutes sans confondre mAP et
effet causal.

### Heure 8 - Simulation orale et examen

- Utiliser [06_questions_orales.md](06_questions_orales.md).
- Tirer 20 questions au hasard.
- Repondre d'abord sans support, puis corriger.
- Faire le mini-examen final de
  [05_exercices_sans_corrige.md](05_exercices_sans_corrige.md).
- Consulter [07_corrige.md](07_corrige.md) seulement a la fin.

Livrable personnel : liste des cinq notions qui restent fragiles.

## Parcours reduit de 3 heures

Si la fatigue est forte :

1. Heure 1 : carte du projet, tuilage, HBB vs OBB.
2. Heure 2 : IoU, precision, recall, mAP, DAG, confounder, collider.
3. Heure 3 : resultats reels, limites et 15 questions orales.

Ne pas sacrifier les limites. Une reponse prudente et rigoureuse vaut mieux
qu'une affirmation forte non defendable.

## Index des ressources

- [01_projet_de_bout_en_bout.md](01_projet_de_bout_en_bout.md) : fil complet
  du projet et correspondance avec le bareme.
- [02_detection_yolo_obb.md](02_detection_yolo_obb.md) : detection, YOLO,
  metriques, entrainement et erreurs.
- [03_causalite_appliquee.md](03_causalite_appliquee.md) : DAG, identification,
  estimateurs et heterogeneite.
- [04_resultats_et_interpretation.md](04_resultats_et_interpretation.md) :
  chiffres et figures produits par l'experience.
- [05_exercices_sans_corrige.md](05_exercices_sans_corrige.md) : exercices.
- [06_questions_orales.md](06_questions_orales.md) : questions courtes.
- [07_corrige.md](07_corrige.md) : solutions separees.
- `flashcards.csv` : cartes importables dans Anki ou lisibles dans un tableur.
- `concepts_oraux_ml_causalite.md` : cartes orales detaillees deja commencees.

## Grille finale d'auto-evaluation

| Bloc | Score avant | Score apres | Preuve de maitrise |
|---|---:|---:|---|
| Pipeline complet |  |  | Je le dessine sans support |
| Tuilage et fuite |  |  | Je justifie le split par image |
| IoU, AP, mAP |  |  | Je donne formules et limites |
| YOLO et OBB |  |  | Je relie architecture et DOTA |
| DAG |  |  | Je distingue les types de noeuds |
| AIPW |  |  | Je compare aux methodes simples |
| Arbre causal |  |  | J'explique heterogeneite et prudence |
| Resultats reels |  |  | Je cite des chiffres exacts |
| Limites |  |  | J'en explique au moins cinq |
