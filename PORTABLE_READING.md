# Lecture du projet sur un autre ordinateur

Ce mode est concu pour lire le projet apres un `git pull`, sans reexecuter le
notebook et sans copier DOTA, les poids YOLO, les runs ou l'environnement
Python de l'ordinateur d'origine.

## Chemin recommande

1. Recuperer la branche `main`.
2. Ouvrir directement
   [`output/notebook/projet_dota.html`](output/notebook/projet_dota.html)
   dans un navigateur.
3. Utiliser ensuite :
   - [`output/pdf/guide_etude_vol_dota.pdf`](output/pdf/guide_etude_vol_dota.pdf)
     pour le parcours d'etude de 8 heures ;
   - [`output/pdf/rapport_final_dota.pdf`](output/pdf/rapport_final_dota.pdf)
     pour le rapport final ;
   - [`output/study_pack/guide_etude_dota.html`](output/study_pack/guide_etude_dota.html)
     pour la version navigable du pack d'etude.

Le lecteur HTML du notebook contient le code, les tableaux, les graphiques et
les sorties deja calculees. Il ne depend ni de Python, ni de Jupyter, ni
d'Internet.

## Ouvrir le fichier `.ipynb`

Le fichier [`projet_dota.ipynb`](projet_dota.ipynb) est lui aussi
pre-execute. VS Code ou Jupyter peut afficher ses sorties sans disposer des
donnees ou des poids.

Ne pas utiliser `Restart and Run All` sur l'ordinateur de lecture : une
reexecution complete necessite les donnees DOTA locales et les poids ignores
par Git. La consultation des sorties sauvegardees ne les necessite pas.

## Verification facultative

Si Python 3 est deja installe sur l'ordinateur de lecture :

```powershell
python scripts\verify_portable_reader.py
```

Ce controle utilise uniquement la bibliotheque standard de Python. Il verifie
que le notebook est completement execute, sans erreur sauvegardee, que les
fichiers de lecture sont presents et que le lecteur HTML n'utilise aucune
ressource distante.

## Checklist avant de partir

- effectuer le `git pull` ;
- ouvrir une fois le lecteur HTML du notebook ;
- couper temporairement le Wi-Fi et verifier que les figures restent visibles ;
- ouvrir les deux PDF ;
- garder tout le dossier du depot sur le disque local.
