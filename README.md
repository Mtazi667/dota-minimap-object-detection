# Detection DOTA-v1.0 et inference causale

Projet individuel de detection d'objets dans des images satellites et
aeriennes. Le pipeline compare YOLO26n avec boites horizontales et YOLO26n-OBB,
puis estime l'effet de la condition "objet tres petit" sur la probabilite de
detection correcte.

## Lecture sur un autre ordinateur

Pour lire le projet apres un `git pull`, sans donnees DOTA et sans environnement
Python, ouvrir directement :

- [`output/notebook/projet_dota.html`](output/notebook/projet_dota.html) :
  notebook pre-execute, autonome et lisible hors ligne ;
- [`output/pdf/guide_etude_vol_dota.pdf`](output/pdf/guide_etude_vol_dota.pdf) :
  parcours d'etude de 8 heures ;
- [`output/pdf/rapport_final_dota.pdf`](output/pdf/rapport_final_dota.pdf) :
  rapport final.

Les tableaux, figures et resultats sont deja inclus. Il ne faut pas reexecuter
le notebook sur l'ordinateur de lecture. Voir
[`PORTABLE_READING.md`](PORTABLE_READING.md) pour la checklist complete.

## Livrables

- `projet_dota.ipynb` : notebook principal en francais, organise selon les six
  questions du sujet.
- `output/pdf/rapport_final_dota.pdf` : rapport final.
- `study_pack/` : parcours d'etude actif pour 8 heures hors ligne.
- `output/pdf/guide_etude_vol_dota.pdf` : version PDF du pack d'etude.
- `output/study_pack/guide_etude_dota.html` : version HTML autonome.
- `outputs/analysis/` : metriques, tables causales et figures reproductibles.

## Pipeline

1. Charger et controler DOTA-v1.0.
2. Selectionner 180 images train et 60 validation avec stratification par
   classe et type de scene.
3. Tuiler en 1024 pixels sans partager d'image source entre splits.
4. Produire les memes tuiles avec labels HBB et OBB.
5. Entrainer une baseline HBB et deux configurations OBB.
6. Evaluer globalement, par classe, taille et orientation.
7. Associer predictions et objets de validation de facon un-a-un.
8. Estimer effets brut, g-computation, IPW et AIPW.
9. Explorer l'heterogeneite avec arbre honnete et foret sur pseudo-outcomes.

## Resultats finaux

- YOLO26n-OBB-1024 : precision 0.574, recall 0.290, F1 0.385,
  mAP50 0.274 et mAP50-95 0.186 ;
- HBB-640 et OBB-640 : mAP50-95 respectives 0.145 et 0.146 ;
- table causale : 1 646 objets uniques de 53 images ;
- AIPW principal : -0.368, IC 95 % bootstrap image
  [-0.530 ; -0.166] ;
- sensibilite : estimations ponctuelles negatives, avec un intervalle
  traversant zero lorsque la propension est clippee a 0.02.

## Demarrage local

Cette section concerne la reproduction complete sur l'ordinateur qui contient
les donnees, les poids et l'environnement GPU. Elle n'est pas necessaire pour
la lecture portable.

```powershell
.\.venv_gpu\Scripts\python.exe scripts\verify_offline_ready.py
.\.venv_gpu\Scripts\python.exe -m jupyter lab
```

Choisir le noyau `Python (DOTA GPU)`.

## Tests

```powershell
.\.venv_gpu\Scripts\python.exe -m pytest -q
```

## Reproduction des etapes longues

```powershell
.\.venv_gpu\Scripts\python.exe scripts\prepare_experiment.py
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task hbb --epochs 20 --imgsz 640 --batch 16 --workers 0 --name hbb_run
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task obb --epochs 20 --imgsz 640 --batch 16 --workers 0 --name obb_run
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task obb --epochs 20 --imgsz 1024 --batch 4 --workers 0 --name obb_1024_run
.\.venv_gpu\Scripts\python.exe scripts\evaluate_experiment.py --extra-obb YOLO26n-OBB-640:runs/dota_experiment_v1/main_obb_yolo26n_e20_img640/weights/best.pt:640
```

Les donnees, environnements, poids et runs sont volumineux et restent locaux.
Ils sont ignores par Git. Les metriques, figures, notebook, rapport et
documents d'etude sont versionnes.

Voir [OFFLINE_READY.md](OFFLINE_READY.md) pour les commandes sans Internet.
