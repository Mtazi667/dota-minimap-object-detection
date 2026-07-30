# Utilisation hors ligne et lecture sur un autre ordinateur

## Mode lecture portable

Apres un `git pull`, aucun environnement Python, dataset ou poids n'est
necessaire pour consulter les resultats deja calcules.

Ouvrir dans cet ordre :

1. `PORTABLE_READING.md` ;
2. `output/notebook/projet_dota.html` ;
3. `output/pdf/guide_etude_vol_dota.pdf` ;
4. `output/pdf/rapport_final_dota.pdf`.

Le fichier HTML du notebook contient localement le code, les tableaux, les
figures et toutes les sorties. Il ne charge aucune bibliotheque distante.

Le fichier `projet_dota.ipynb` peut aussi etre ouvert dans VS Code ou Jupyter.
Ses sorties sont sauvegardees. Ne pas utiliser `Restart and Run All` sur
l'ordinateur de lecture.

Controle facultatif avec n'importe quel Python 3 :

```powershell
python scripts\verify_portable_reader.py
```

## Reproduction complete sur l'ordinateur d'origine

Depuis PowerShell, a la racine du projet :

```powershell
.\.venv_gpu\Scripts\python.exe scripts\verify_offline_ready.py
.\.venv_gpu\Scripts\python.exe -m jupyter lab
```

Dans Jupyter, choisir le noyau `Python (DOTA GPU)`.

## Ressources non versionnees

Les donnees DOTA, le dataset experimental, `.venv_gpu`, les poids et les runs
sont ignores par Git. Ils sont presents sur l'ordinateur d'origine, mais ne
sont pas recuperes par un `git pull`.

Ils sont necessaires seulement pour reexecuter le pipeline. Le notebook
pre-execute, les analyses, figures, PDF et documents d'etude sont versionnes et
suffisent pour la lecture.

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

- `PORTABLE_READING.md` : point de depart sur le second ordinateur ;
- `output/pdf/guide_etude_vol_dota.pdf` : guide de 66 pages ;
- `output/study_pack/guide_etude_dota.html` : version navigable hors ligne ;
- `output/notebook/projet_dota.html` : notebook pre-execute, autonome et
  lisible sans Jupyter ;
- `output/pdf/rapport_final_dota.pdf` : rapport final de 14 pages.

## En cas de probleme

- Sorties absentes dans le `.ipynb` : utiliser le lecteur HTML versionne, qui
  ne depend pas du noyau Jupyter.
- Message demandant un noyau : ignorer ce message si le but est seulement de
  lire les sorties, ou utiliser le lecteur HTML.
- CUDA absent : verifier que la commande utilise bien
  `.\.venv_gpu\Scripts\python.exe`.
- Dossier de run deja present : choisir un nouveau `--name`. Le script refuse
  l'ecrasement volontairement.
- Memoire GPU insuffisante : diminuer `--batch`, pas la validation.
- Blocage des workers Windows : garder `--workers 0`.
- Analyse manquante : verifier que les poids `best.pt` existent avant
  `evaluate_experiment.py`.
