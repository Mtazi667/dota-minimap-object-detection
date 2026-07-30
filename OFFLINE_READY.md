# Utilisation hors ligne

## Demarrage

Depuis PowerShell, a la racine du projet :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\verify_offline_ready.py
.\.venv_gpu\Scripts\python.exe -m jupyter lab
```

Dans Jupyter, choisir le noyau `Python (DOTA GPU)`.

## Ressources deja locales

- donnees DOTA train et validation ;
- dataset experimental HBB/OBB sous `prepared_data/dota_experiment_v1` ;
- environnement Python GPU sous `.venv_gpu` ;
- poids de depart `yolo26n.pt` ;
- poids des runs sous `runs/dota_experiment_v1` ;
- notebook, analyses, figures et pack d'etude.

Les dossiers `.venv_gpu`, `prepared_data`, `runs` et les fichiers de poids sont
ignores par Git, mais restent disponibles sur cet ordinateur.

## Commandes reproductibles

Recreer le sous-ensemble dans un nouveau dossier :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\prepare_experiment.py --output prepared_data\dota_experiment_rebuild
```

Entrainer une nouvelle baseline HBB :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task hbb --epochs 20 --imgsz 640 --batch 16 --workers 0 --name hbb_rebuild
```

Entrainer un nouveau modele OBB :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task obb --epochs 20 --imgsz 640 --batch 16 --workers 0 --name obb_rebuild
```

Entrainer la variante haute resolution :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task obb --epochs 20 --imgsz 1024 --batch 4 --workers 0 --name obb_1024_rebuild
```

Reprendre un run interrompu :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\train_models.py --task obb --workers 0 --resume-from runs\dota_experiment_v1\RUN\weights\last.pt
```

Relancer l'evaluation finale :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\evaluate_experiment.py --extra-obb YOLO26n-OBB-640:runs/dota_experiment_v1/main_obb_yolo26n_e20_img640/weights/best.pt:640
```

Cette commande utilise OBB-1024 comme modele principal par defaut et ajoute
OBB-640 a la comparaison avec la baseline HBB.

## Lecture pendant le vol

- `output/pdf/guide_etude_vol_dota.pdf` : guide de 66 pages ;
- `output/study_pack/guide_etude_dota.html` : version navigable hors ligne ;
- `output/notebook/projet_dota.html` : notebook execute, lisible sans Jupyter ;
- `output/pdf/rapport_final_dota.pdf` : rapport final de 14 pages.

## En cas de probleme

- CUDA absent : verifier que la commande utilise bien
  `.\.venv_gpu\Scripts\python.exe`.
- Dossier de run deja present : choisir un nouveau `--name`. Le script refuse
  l'ecrasement volontairement.
- Memoire GPU insuffisante : diminuer `--batch`, pas la validation.
- Blocage des workers Windows : garder `--workers 0`.
- Analyse manquante : verifier que les poids `best.pt` existent avant
  `evaluate_experiment.py`.
