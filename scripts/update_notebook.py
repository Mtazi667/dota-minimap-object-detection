"""Remplace les sections provisoires 3-5 par les questions 3-6 finales."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "projet_dota.ipynb"
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"


def _without_future_import(source: str) -> str:
    return source.replace("from __future__ import annotations\n\n", "")


def _format_percent(value: float) -> str:
    return f"{100 * value:.1f} %"


def _markdown_cell(source: str) -> nbformat.NotebookNode:
    return new_markdown_cell(
        source=source.strip() + "\n",
        metadata={"tags": ["flight-completion-v1"]},
    )


def _code_cell(
    source: str,
    hide_input: bool = False,
) -> nbformat.NotebookNode:
    tags = ["flight-completion-v1"]
    if hide_input:
        tags.append("hide-input")
    return new_code_cell(
        source=source.strip() + "\n",
        metadata={"tags": tags},
    )


def main() -> None:
    required = [
        ANALYSIS_DIR / "analysis_summary.json",
        ANALYSIS_DIR / "model_comparison.csv",
        ANALYSIS_DIR / "per_class_metrics.csv",
        ANALYSIS_DIR / "causal_effect_estimates.csv",
        ANALYSIS_DIR / "causal_sensitivity.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Executer scripts/evaluate_experiment.py avant la mise a jour : "
            + ", ".join(str(path) for path in missing)
        )

    summary = json.loads(
        (ANALYSIS_DIR / "analysis_summary.json").read_text(encoding="utf-8")
    )
    models = pd.read_csv(ANALYSIS_DIR / "model_comparison.csv")
    classes = pd.read_csv(ANALYSIS_DIR / "per_class_metrics.csv")
    effects = pd.read_csv(ANALYSIS_DIR / "causal_effect_estimates.csv")
    sensitivity = pd.read_csv(ANALYSIS_DIR / "causal_sensitivity.csv")
    primary_name = summary["primary_model"]["model"]
    primary_model = models.loc[models["model"].eq(primary_name)].iloc[0]
    baseline = models.loc[models["task"].eq("hbb")].iloc[0]
    best_model = models.sort_values("map50_95", ascending=False).iloc[0]
    primary_classes = classes.loc[classes["model"].eq(primary_name)]
    best_classes = primary_classes.nlargest(3, "ap50_95")[
        ["class_name", "ap50_95"]
    ]
    weak_classes = primary_classes.nsmallest(3, "ap50_95")[
        ["class_name", "ap50_95"]
    ]
    aipw = effects.loc[
        effects["method"].eq("AIPW doublement robuste")
    ].iloc[0]

    pipeline_source = _without_future_import(
        (PROJECT_DIR / "src" / "dota_pipeline.py").read_text(encoding="utf-8")
    )
    analysis_source = _without_future_import(
        (PROJECT_DIR / "src" / "experiment_analysis.py").read_text(
            encoding="utf-8"
        )
    )

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "Cette date peut etre corrigee si necessaire.",
        "Cette date correspond a l'organisation locale verifiee des fichiers.",
    )
    portability_replacements = {
        '| {path}")': '| {path.relative_to(PROJECT_DIR)}")',
        '"dossier_labels": str(label_dir),': (
            '"dossier_labels": label_dir.relative_to(PROJECT_DIR).as_posix(),'
        ),
        'f"path: {YOLO_OBB_DIR.as_posix()}",': (
            'f"path: {YOLO_OBB_DIR.relative_to(PROJECT_DIR).as_posix()}",'
        ),
        "print(YOLO_OBB_YAML_PATH)": (
            "print(YOLO_OBB_YAML_PATH.relative_to(PROJECT_DIR))"
        ),
    }
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old_text, new_text in portability_replacements.items():
            cell.source = cell.source.replace(old_text, new_text)

    cut_index = None
    for index, cell in enumerate(notebook.cells):
        if (
            cell.cell_type == "markdown"
            and cell.source.lstrip().startswith("## 3.")
        ):
            cut_index = index
            break
    if cut_index is None:
        raise RuntimeError("La section 3 provisoire du notebook est introuvable.")
    notebook.cells = notebook.cells[:cut_index]

    model_interpretation = (
        f"La baseline HBB atteint une mAP50 de {baseline['map50']:.3f} et une "
        f"mAP50-95 de {baseline['map50_95']:.3f}. Le modele causal principal "
        f"`{primary_name}` atteint respectivement {primary_model['map50']:.3f} "
        f"et {primary_model['map50_95']:.3f}, avec un F1 global de "
        f"{primary_model['f1']:.3f}. Le meilleur score mAP50-95 du "
        f"protocole est obtenu par `{best_model['model']}` avec "
        f"{best_model['map50_95']:.3f}. Ces scores restent modestes : le "
        "sous-ensemble est limite, les objets sont souvent minuscules et "
        "l'entrainement est court. La comparaison mesure ce protocole precis, "
        "pas une superiorite universelle d'une representation."
    )
    best_class_text = ", ".join(
        f"`{row.class_name}` ({row.ap50_95:.3f})"
        for row in best_classes.itertuples()
    )
    weak_class_text = ", ".join(
        f"`{row.class_name}` ({row.ap50_95:.3f})"
        for row in weak_classes.itertuples()
    )
    effect_interpretation = (
        f"L'estimation AIPW principale vaut {aipw['effect']:.3f}, avec un "
        f"intervalle bootstrap groupe a 95 % "
        f"[{aipw['ci_lower']:.3f}, {aipw['ci_upper']:.3f}]. Cela correspond a "
        f"{_format_percent(aipw['effect'])} de difference absolue de "
        "probabilite sous les hypotheses d'identification. Le taux observe "
        f"est {_format_percent(aipw['outcome_treated'])} chez les objets tres "
        f"petits et {_format_percent(aipw['outcome_control'])} chez les "
        "controles. Cette estimation ne transforme pas l'etude en experience "
        "randomisee : elle reste sensible aux confounders non observes, a la "
        "selection et a la definition operationnelle de la taille."
    )
    sensitivity_signs = ", ".join(
        f"{row.specification}: {row.effect:.3f}"
        for row in sensitivity.itertuples()
    )

    cells = [
        _markdown_cell(
            """
## 3. Modeles predictifs de detection et apprentissage profond

### Tache et protocole

La tache predictive consiste a localiser chaque objet dans une tuile et a
predire sa classe parmi les 15 classes DOTA. Deux detecteurs YOLO26n sont
compares sur exactement les memes images sources et les memes tuiles :

- baseline HBB : boites horizontales derivees des quatre coins ;
- modele principal OBB : quatre coins et orientation conserves.

Les deux architectures partent d'un backbone pre-entraine sur COCO. Aucun poids
OBB deja ajuste sur DOTA n'est utilise, afin de ne pas avantager le modele
oriente. La selection contient 180 images train et 60 images validation. Le
split est fait avant le tuilage, donc aucune image source ne traverse les deux
ensembles.
"""
        ),
        _markdown_cell(
            """
### Fonctions autonomes de preparation

Le sujet impose que les fonctions auxiliaires soient presentes dans le
notebook. La cellule suivante contient donc le code utilise pour lire DOTA,
selectionner les images, tuiler, exporter HBB/OBB et valider le dataset. La
construction volumineuse est desactivee par defaut apres sa premiere execution,
mais elle peut etre relancee en changeant un seul drapeau.
"""
        ),
        _code_cell(pipeline_source, hide_input=True),
        _code_cell(
            """
EXPERIMENT_DIR = PROJECT_DIR / "prepared_data" / "dota_experiment_v1"
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"
BUILD_EXPERIMENT_DATASET = False

if BUILD_EXPERIMENT_DATASET:
    preparation_summary = build_experiment_dataset(
        project_dir=PROJECT_DIR,
        output_root=EXPERIMENT_DIR,
        train_image_count=180,
        val_image_count=60,
        tile_size=1024,
        train_stride=824,
        val_stride=1024,
        seed=42,
    )
else:
    summary_path = EXPERIMENT_DIR / "preparation_summary.json"
    if not summary_path.exists():
        summary_path = ANALYSIS_DIR / "preparation_summary_snapshot.json"
    preparation_summary = json.loads(summary_path.read_text(encoding="utf-8"))

compact_preparation_report(preparation_summary)
"""
        ),
        _markdown_cell(
            """
Le dataset final contient 912 tuiles train et 245 tuiles validation. Les 15
classes sont presentes dans les deux splits et le controle trouve zero image
source commune. Les 8 487 lignes du manifeste sont des instances tuilees ;
elles correspondent a 6 292 objets originaux uniques. Cette distinction est
importante lorsque des tuiles train se chevauchent.

Les fragments conservant au moins 70 % de l'objet sont gardes. Une tuile est
rejetee si elle contient un fragment ambigu entre 20 % et 70 %. Un echantillon
de tuiles negatives est garde pour apprendre le fond.
"""
        ),
        _markdown_cell(
            """
### Code d'entrainement reproductible

Les runs longs ne sont pas relances lors d'un simple `Run All`. Le code reste
present et executable en activant `RUN_TRAINING`. Les dossiers existants ne
sont jamais ecrases.
"""
        ),
        _code_cell(
            """
import os
import sys
import torch
from ultralytics import YOLO

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics_config"))

def train_detector(task, name, epochs=20, imgsz=640, batch=16):
    if task not in {"hbb", "obb"}:
        raise ValueError("task doit etre hbb ou obb")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA est requis pour ce run.")
    run_dir = PROJECT_DIR / "runs" / "dota_experiment_v1" / name
    if run_dir.exists():
        raise FileExistsError(f"Run deja present: {run_dir}")
    model_config = "yolo26n.yaml" if task == "hbb" else "yolo26n-obb.yaml"
    model = YOLO(model_config).load(str(PROJECT_DIR / "yolo26n.pt"))
    return model.train(
        data=str(EXPERIMENT_DIR / task / f"dota_{task}.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=0,
        project=str(PROJECT_DIR / "runs" / "dota_experiment_v1"),
        name=name,
        seed=42,
        deterministic=True,
        patience=7,
        amp=True,
        plots=True,
        save=True,
    )

RUN_TRAINING = False
if RUN_TRAINING:
    hbb_run = train_detector("hbb", "notebook_hbb_e20", batch=16)
    obb_run = train_detector("obb", "notebook_obb_e20", batch=16)

pd.DataFrame(
    {
        "python": [sys.version.split()[0]],
        "torch": [torch.__version__],
        "cuda_available": [torch.cuda.is_available()],
        "gpu": [torch.cuda.get_device_name(0) if torch.cuda.is_available() else None],
    }
)
"""
        ),
        _markdown_cell(
            """
### Evaluation globale

La validation finale utilise jusqu'a 1 000 detections par tuile pour ne pas
tronquer trop vite les scenes denses. La mAP50 mesure une localisation
tolerante a IoU 0.50 ; la mAP50-95 exige aussi des boites tres precises.
"""
        ),
        _code_cell(
            """
model_comparison = pd.read_csv(ANALYSIS_DIR / "model_comparison.csv")
model_comparison.round(4)
"""
        ),
        _code_cell(
            """
from IPython.display import Image as IPythonImage, display
display(IPythonImage(filename=str(ANALYSIS_DIR / "model_comparison.png")))
"""
        ),
        _markdown_cell(model_interpretation),
        _markdown_cell(
            f"""
### Resultats par classe

Pour le modele principal, les trois AP50-95 les plus hautes sont
{best_class_text}. Les trois plus faibles sont {weak_class_text}. Les classes
rares doivent etre interpretees avec leurs effectifs : une AP basse peut
refleter a la fois la difficulte visuelle et le peu d'exemples.
"""
        ),
        _code_cell(
            """
per_class_metrics = pd.read_csv(ANALYSIS_DIR / "per_class_metrics.csv")
display(IPythonImage(filename=str(ANALYSIS_DIR / "per_class_metrics.png")))
per_class_metrics.sort_values(["model", "ap50_95"], ascending=[True, False]).round(4)
"""
        ),
        _markdown_cell(
            """
### Exemples visuels et analyse d'erreurs

Les planches suivantes utilisent des tuiles validation. Elles doivent etre
lues avec les metriques : une prediction visuellement plausible peut echouer a
cause de la classe, de la confiance ou d'une IoU insuffisante.
"""
        ),
        _code_cell(
            f"""
prediction_sheets = [
    ANALYSIS_DIR / "prediction_examples_yolo26n-hbb-640.jpg",
    ANALYSIS_DIR / "prediction_examples_{primary_name.lower()}.jpg",
]
for sheet in prediction_sheets:
    if sheet.exists():
        display(IPythonImage(filename=str(sheet)))
"""
        ),
        _code_cell(
            """
display(IPythonImage(filename=str(ANALYSIS_DIR / "detection_by_size_orientation.png")))
"""
        ),
        _markdown_cell(
            """
Les limites predictives principales sont les objets minuscules, les scenes
denses, les orientations, les classes rares, le redimensionnement et la courte
duree d'entrainement. Ces observations motivent la question causale, mais elles
ne constituent pas encore une estimation causale.
"""
        ),
        _markdown_cell(
            """
## 4. Formulation de la question causale

### Traitement, outcome et estimand

- `D = 1` : objet tres petit, defini par le premier quartile de l'aire relative
  de tuile parmi les objets train uniques.
- `Y = 1` : prediction OBB de meme classe, confiance au moins 0.25 et IoU
  orientee au moins 0.50.
- population : objets originaux uniques du sous-ensemble validation.
- estimand : effet moyen sur la probabilite de detection correcte.

Question : quel est l'effet d'etre un objet tres petit sur la probabilite d'une
detection correcte, apres ajustement sur la classe, l'orientation, le ratio de
forme, la densite, la source, le GSD et les conditions de bord de tuile ?

La petite taille est une condition observationnelle, pas une intervention
randomisee. Le mot effet est donc conditionnel a des hypotheses fortes.
"""
        ),
        _code_cell(
            """
import networkx as nx

dag = nx.DiGraph()
dag.add_edges_from(
    [
        ("Classe", "Tres petit"),
        ("Classe", "Detection"),
        ("GSD", "Tres petit"),
        ("GSD", "Detection"),
        ("Source", "GSD"),
        ("Source", "Detection"),
        ("Orientation", "Detection"),
        ("Densite", "Detection"),
        ("Ratio", "Detection"),
        ("Bord tuile", "Detection"),
        ("Tres petit", "Detection"),
    ]
)
positions = {
    "Source": (0, 2),
    "GSD": (1, 2),
    "Classe": (1, 0),
    "Orientation": (2, -1),
    "Densite": (3, -1),
    "Ratio": (4, -1),
    "Bord tuile": (5, -1),
    "Tres petit": (3, 1.5),
    "Detection": (5, 1.5),
}
plt.figure(figsize=(12, 5))
nx.draw_networkx(
    dag,
    positions,
    node_color=["#E69F00" if node == "Tres petit" else "#56B4E9" for node in dag],
    node_size=2300,
    font_size=9,
    arrowsize=18,
)
plt.title("DAG simplifie de la question causale")
plt.axis("off")
plt.show()
"""
        ),
        _markdown_cell(
            """
### Ensemble d'ajustement et hypotheses

La classe, le GSD et la source peuvent ouvrir des chemins arriere vers
l'outcome. L'orientation, la densite, le ratio et la position au bord ameliorent
la comparabilite et la precision. L'aire exacte n'est pas incluse car elle
definit le traitement. La confiance et l'IoU sont post-traitement ou definissent
l'outcome.

Hypotheses :

1. exchangeabilite conditionnelle apres les variables observees ;
2. positivite : petits et non petits comparables pour les profils analyses ;
3. coherence d'une definition operationnelle de la petite taille ;
4. absence de fuite entre images source ;
5. interference limitee entre objets, malgre NMS et densite ;
6. mesures et annotations suffisamment fiables.

Les confounders non observes, la selection des tuiles et le sous-ensemble
restent des limites.
"""
        ),
        _markdown_cell(
            """
## 5. Estimation causale et heterogeneite

Les fonctions ci-dessous calculent l'IoU polygonale, imposent un matching
un-a-un, choisissent une instance par objet et estiment plusieurs contrastes.
L'AIPW combine un modele de propension et deux modeles d'outcome avec
cross-fitting groupe par image. Les intervalles reechantillonnent les images,
pas les objets independamment.
"""
        ),
        _code_cell(analysis_source, hide_input=True),
        _code_cell(
            """
causal_objects = pd.read_csv(ANALYSIS_DIR / "causal_object_table.csv")
causal_effects = pd.read_csv(ANALYSIS_DIR / "causal_effect_estimates.csv")
causal_sensitivity = pd.read_csv(ANALYSIS_DIR / "causal_sensitivity.csv")
causal_subgroups = pd.read_csv(ANALYSIS_DIR / "causal_tree_subgroups.csv")

causal_effects.round(4)
"""
        ),
        _markdown_cell(
            """
### Methodes comparees

- difference brute : descriptive, sans ajustement ;
- g-computation : moyenne des outcomes predits sous les deux traitements ;
- IPW : pseudo-population ponderee par la propension ;
- AIPW : combinaison doublement robuste des deux modeles.

Le cross-fitting limite l'overfitting des modeles nuisances. Il ne corrige pas
un confounder non observe ni une violation de positivite.
"""
        ),
        _code_cell(
            """
display(IPythonImage(filename=str(ANALYSIS_DIR / "causal_effects.png")))
display(IPythonImage(filename=str(ANALYSIS_DIR / "propensity_overlap.png")))
"""
        ),
        _markdown_cell(effect_interpretation),
        _markdown_cell(
            f"""
### Sensibilite

Effets AIPW observes pour les specifications testees :

{sensitivity_signs}.

La sensibilite sert a verifier si le signe ou l'amplitude depend fortement d'un
seuil. Elle ne teste pas les confounders non observes.
"""
        ),
        _code_cell(
            """
causal_sensitivity.round(4)
"""
        ),
        _markdown_cell(
            """
### Effets heterogenes

L'arbre causal utilise une moitie des images pour les divisions et l'autre pour
les effets des feuilles. Il est volontairement limite a une division et a de
grandes feuilles afin de reduire les sous-groupes artificiels. La foret predit
des pseudo-outcomes dans des folds groupes par image. Cette foret est une
approximation pedagogique d'une foret causale specialisee ; les sous-groupes
exploratoires ne sont pas des lois generales.
"""
        ),
        _code_cell(
            """
display(IPythonImage(filename=str(ANALYSIS_DIR / "causal_tree.png")))
display(IPythonImage(filename=str(ANALYSIS_DIR / "cate_distribution.png")))
causal_subgroups.round(4)
"""
        ),
        _markdown_cell(
            f"""
## 6. Interpretation, conclusion et limites

### Conclusion predictive

Trois modeles comparables ont ete entraines et evalues sur des images source
validation separees. `{best_model['model']}` obtient la meilleure mAP50-95 du
protocole ({best_model['map50_95']:.3f}). Les resultats par classe, taille et
orientation montrent que la moyenne globale ne suffit pas. Un score modeste
est coherent avec les objets minuscules, les classes rares, la petite taille du
sous-ensemble et seulement 20 epochs.

### Conclusion causale

{effect_interpretation}

### Limites

1. sous-ensemble stratifie plutot que DOTA complet ;
2. entrainements courts et une seule petite architecture ;
3. classes rares et support causal parfois limite ;
4. traitement non randomise et intervention de taille ambigue ;
5. flou, contraste, occultation et qualite d'annotation non observes ;
6. selection produite par le tuilage et les fragments ;
7. interference possible via NMS et scenes denses ;
8. outcome depend d'un detecteur et de seuils choisis ;
9. foret sur pseudo-outcomes plutot qu'une implementation GRF complete ;
10. arbre volontairement peu profond et heterogeneite exploratoire ;
11. bootstrap groupe sans re-ajustement integral des nuisances.

La performance predictive et l'effet causal repondent a deux questions
complementaires. La premiere mesure ce que le detecteur accomplit ; la seconde
essaie d'expliquer une difference d'erreur sous des hypotheses explicites.
"""
        ),
        _code_cell(
            """
reproducibility = pd.DataFrame(
    {
        "element": [
            "graine",
            "images train/validation",
            "taille de tuile",
            "stride train/validation",
            "classes",
            "overlap image source",
            "python",
            "torch",
            "cuda",
        ],
        "valeur": [
            42,
            "180 / 60",
            1024,
            "824 / 1024",
            15,
            preparation_summary["validation"]["original_image_overlap_count"],
            sys.version.split()[0],
            torch.__version__,
            torch.version.cuda,
        ],
    }
)
reproducibility
"""
        ),
        _markdown_cell(
            """
### Prochaines ameliorations

- entrainer plus longtemps sur un sous-ensemble plus grand ;
- tester une architecture plus grande si la memoire le permet ;
- mesurer calibration, erreurs par source et courbes d'apprentissage ;
- valider les sous-groupes causaux sur un autre echantillon ;
- enrichir les covariables de contraste, flou et contexte ;
- utiliser une implementation specialisee de foret causale compatible ;
- rediger le rapport final en gardant les conclusions proportionnees aux
  donnees.
"""
        ),
    ]
    notebook.cells.extend(cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {
        **notebook.metadata.get("language_info", {}),
        "name": "python",
        "version": "3.14.3",
    }

    backup_dir = PROJECT_DIR / "tmp" / "notebook_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK_PATH, backup_dir / "projet_dota_before_final.ipynb")
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook mis a jour: {NOTEBOOK_PATH} ({len(notebook.cells)} cellules)")


if __name__ == "__main__":
    main()
