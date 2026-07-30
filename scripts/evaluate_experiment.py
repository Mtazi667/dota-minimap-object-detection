"""Evalue HBB/OBB, construit la table causale et genere les figures finales."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics_config"))

from src.dota_pipeline import DOTA_V1_CLASSES
from src.experiment_analysis import (
    add_instance_geometry,
    choose_unique_object_instances,
    estimate_doubly_robust_effect,
    match_predictions_to_ground_truth,
    run_causal_analysis,
)
from ultralytics import YOLO


DEFAULT_DATASET = PROJECT_DIR / "prepared_data" / "dota_experiment_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "analysis"
DEFAULT_HBB = (
    PROJECT_DIR
    / "runs"
    / "dota_experiment_v1"
    / "baseline_hbb_yolo26n_e20"
    / "weights"
    / "best.pt"
)
DEFAULT_OBB = (
    PROJECT_DIR
    / "runs"
    / "dota_experiment_v1"
    / "tuned_obb_yolo26n_e20_img1024"
    / "weights"
    / "best.pt"
)


def portable_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hbb-weights", type=Path, default=DEFAULT_HBB)
    parser.add_argument("--obb-weights", type=Path, default=DEFAULT_OBB)
    parser.add_argument("--obb-name", type=str, default="YOLO26n-OBB-1024")
    parser.add_argument("--obb-imgsz", type=int, default=1024)
    parser.add_argument("--hbb-imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-obb",
        action="append",
        default=[],
        metavar="NAME:PATH:IMGSZ",
        help="Ajoute une variante OBB, par exemple obb_1024:C:/run/best.pt:1024.",
    )
    return parser.parse_args()


def parse_extra_models(values: list[str]) -> list[dict]:
    models = []
    for value in values:
        try:
            name, raw_path, raw_imgsz = value.rsplit(":", 2)
        except ValueError as error:
            raise ValueError(
                f"Format --extra-obb invalide: {value!r}. "
                "Utiliser NAME:PATH:IMGSZ."
            ) from error
        models.append(
            {
                "name": name,
                "task": "obb",
                "weights": Path(raw_path),
                "imgsz": int(raw_imgsz),
            }
        )
    return models


def validate_model(
    specification: dict,
    dataset_root: Path,
    output_root: Path,
    batch: int,
    workers: int,
    max_det: int,
) -> tuple[dict, pd.DataFrame]:
    weights = Path(specification["weights"])
    if not weights.exists():
        raise FileNotFoundError(weights)
    task = specification["task"]
    data_yaml = dataset_root / task / f"dota_{task}.yaml"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=specification["imgsz"],
        batch=batch,
        device=0,
        workers=workers,
        conf=0.001,
        iou=0.7,
        max_det=max_det,
        plots=True,
        project=str(output_root / "validation_runs"),
        name=specification["name"],
        exist_ok=True,
        verbose=False,
    )
    metric_box = metrics.box
    global_row = {
        "model": specification["name"],
        "task": task,
        "imgsz": specification["imgsz"],
        "weights": portable_project_path(weights),
        "precision": float(metric_box.mp),
        "recall": float(metric_box.mr),
        "f1": float(
            2 * metric_box.mp * metric_box.mr
            / max(metric_box.mp + metric_box.mr, 1e-12)
        ),
        "map50": float(metric_box.map50),
        "map50_95": float(metric_box.map),
        "fitness": float(metrics.fitness),
        "inference_ms_per_image": float(metrics.speed.get("inference", np.nan)),
    }

    class_rows = []
    class_indices = np.asarray(metric_box.ap_class_index, dtype=int)
    ap50 = np.asarray(metric_box.ap50, dtype=float)
    ap50_95 = np.asarray(metric_box.ap, dtype=float)
    for metric_position, class_id in enumerate(class_indices):
        class_rows.append(
            {
                "model": specification["name"],
                "task": task,
                "class_id": int(class_id),
                "class_name": DOTA_V1_CLASSES[int(class_id)],
                "ap50": float(ap50[metric_position]),
                "ap50_95": float(ap50_95[metric_position]),
            }
        )
    return global_row, pd.DataFrame(class_rows)


def _prediction_arrays(result, task: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if task == "obb":
        predictions = result.obb
        if predictions is None or len(predictions) == 0:
            return [], np.empty(0, dtype=int), np.empty(0, dtype=float)
        polygons = [
            polygon.astype(np.float32)
            for polygon in predictions.xyxyxyxy.cpu().numpy()
        ]
    else:
        predictions = result.boxes
        if predictions is None or len(predictions) == 0:
            return [], np.empty(0, dtype=int), np.empty(0, dtype=float)
        polygons = []
        for x_min, y_min, x_max, y_max in predictions.xyxy.cpu().numpy():
            polygons.append(
                np.asarray(
                    [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    dtype=np.float32,
                )
            )
    return (
        polygons,
        predictions.cls.cpu().numpy().astype(int),
        predictions.conf.cpu().numpy().astype(float),
    )


def build_matched_object_table(
    weights: Path,
    task: str,
    dataset_root: Path,
    imgsz: int,
    confidence: float,
    max_det: int,
    workers: int,
) -> pd.DataFrame:
    manifest = pd.read_csv(dataset_root / "manifests" / "object_manifest.csv")
    tile_manifest = pd.read_csv(dataset_root / "manifests" / "tile_manifest.csv")
    selected_val = choose_unique_object_instances(manifest, "val")
    selected_ids = set(selected_val["object_id"])
    ground_truth_by_tile = {
        tile_id: group.reset_index(drop=True)
        for tile_id, group in manifest.loc[manifest["split"].eq("val")].groupby(
            "tile_id"
        )
    }

    model = YOLO(str(weights))
    image_directory = dataset_root / task / "images" / "val"
    matched_parts = []
    prediction_stream = model.predict(
        source=str(image_directory),
        stream=True,
        imgsz=imgsz,
        conf=confidence,
        iou=0.7,
        max_det=max_det,
        device=0,
        workers=workers,
        verbose=False,
    )
    for result in prediction_stream:
        tile_id = Path(result.path).stem
        ground_truth = ground_truth_by_tile.get(tile_id)
        if ground_truth is None or ground_truth.empty:
            continue
        polygons, classes, confidences = _prediction_arrays(result, task)
        matched_parts.append(
            match_predictions_to_ground_truth(
                ground_truth,
                polygons,
                classes,
                confidences,
            )
        )

    matched_all = pd.concat(matched_parts, ignore_index=True)
    table = matched_all.loc[matched_all["object_id"].isin(selected_ids)].copy()
    if table["object_id"].duplicated().any():
        selected_lookup = selected_val.set_index("object_id")["tile_id"]
        table = table.loc[
            table["tile_id"].eq(table["object_id"].map(selected_lookup))
        ].copy()
    if table["object_id"].duplicated().any():
        raise AssertionError("Des objets sont dupliques apres l'appariement.")
    if len(table) != len(selected_val):
        missing = set(selected_val["object_id"]) - set(table["object_id"])
        raise AssertionError(
            f"{len(missing)} objets selectionnes n'ont pas ete evalues."
        )
    table = add_instance_geometry(table)

    tile_columns = tile_manifest[
        ["split", "tile_id", "object_count", "is_negative_tile"]
    ].rename(columns={"object_count": "tile_object_count"})
    table = table.merge(tile_columns, on=["split", "tile_id"], how="left")

    train_unique = choose_unique_object_instances(manifest, "train")
    small_area_threshold = float(train_unique["relative_area_tile"].quantile(0.25))
    train_quantiles = train_unique["relative_area_tile"].quantile(
        [0.25, 0.5, 0.75]
    )
    table["very_small"] = (
        table["relative_area_tile"] <= small_area_threshold
    ).astype(int)
    table["detected_iou40"] = (table["matched_iou"] >= 0.40).astype(int)
    table["detected_iou50"] = (table["matched_iou"] >= 0.50).astype(int)
    table["detected_iou60"] = (table["matched_iou"] >= 0.60).astype(int)
    table["size_group"] = pd.cut(
        table["relative_area_tile"],
        bins=[
            -np.inf,
            float(train_quantiles.loc[0.25]),
            float(train_quantiles.loc[0.50]),
            float(train_quantiles.loc[0.75]),
            np.inf,
        ],
        labels=["tres petit", "petit", "moyen", "grand"],
        include_lowest=True,
    ).astype(str)
    table["orientation_group"] = pd.cut(
        table["orientation_abs_deg"],
        bins=[-0.01, 15, 30, 45, 60, 75, 90.01],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"],
        include_lowest=True,
    ).astype(str)
    table["small_area_threshold_train_q25"] = small_area_threshold
    return table.sort_values(["image_id", "object_id"]).reset_index(drop=True)


def create_prediction_sheet(
    specification: dict,
    dataset_root: Path,
    output_path: Path,
    confidence: float,
    max_det: int,
    seed: int,
    count: int = 9,
) -> None:
    image_directory = (
        dataset_root / specification["task"] / "images" / "val"
    )
    label_directory = (
        dataset_root / specification["task"] / "labels" / "val"
    )
    image_paths = sorted(
        path
        for path in image_directory.glob("*.jpg")
        if (label_directory / f"{path.stem}.txt").stat().st_size > 0
    )
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(
        rng.choice(len(image_paths), size=min(count, len(image_paths)), replace=False)
    )
    selected_paths = [image_paths[index] for index in selected_indices]
    model = YOLO(str(specification["weights"]))
    results = model.predict(
        source=[str(path) for path in selected_paths],
        imgsz=specification["imgsz"],
        conf=confidence,
        iou=0.7,
        max_det=max_det,
        device=0,
        workers=0,
        verbose=False,
    )

    panels = []
    for result in results:
        plotted = result.plot(labels=True, conf=True, boxes=True)
        image = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        image.thumbnail((480, 480))
        panel = Image.new("RGB", (500, 525), "white")
        panel.paste(image, ((500 - image.width) // 2, 5))
        ImageDraw.Draw(panel).text(
            (10, 503),
            Path(result.path).stem,
            fill="black",
        )
        panels.append(panel)

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    sheet = Image.new("RGB", (columns * 500, rows * 525), "#e8e8e8")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 500, (index // columns) * 525))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def run_sensitivity_analysis(
    table: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows = []
    specifications = [
        ("IoU >= 0.40", "detected_iou40", 0.05, seed),
        ("IoU >= 0.50", "detected_iou50", 0.05, seed),
        ("IoU >= 0.60", "detected_iou60", 0.05, seed),
        ("Propension clip 0.02", "detected_iou50", 0.02, seed),
        ("Propension clip 0.10", "detected_iou50", 0.10, seed),
        ("Graine nuisances 7", "detected_iou50", 0.05, 7),
        ("Graine nuisances 99", "detected_iou50", 0.05, 99),
    ]
    for index, (label, outcome, clip, model_seed) in enumerate(specifications):
        sensitivity_table = table.copy()
        _, estimates = estimate_doubly_robust_effect(
            sensitivity_table,
            outcome_column=outcome,
            folds=5,
            propensity_clip=clip,
            bootstrap_repetitions=350,
            seed=model_seed,
        )
        aipw = estimates.loc[
            estimates["method"].eq("AIPW doublement robuste")
        ].iloc[0]
        rows.append(
            {
                "specification": label,
                "effect": aipw["effect"],
                "ci_lower": aipw["ci_lower"],
                "ci_upper": aipw["ci_upper"],
                "treatment_prevalence": aipw["treatment_prevalence"],
                "outcome_treated": aipw["outcome_treated"],
                "outcome_control": aipw["outcome_control"],
            }
        )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
        }
    )


def plot_model_comparison(model_table: pd.DataFrame, output_path: Path) -> None:
    melted = model_table.melt(
        id_vars=["model"],
        value_vars=["precision", "recall", "f1", "map50", "map50_95"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(11, 6))
    axis = sns.barplot(
        data=melted,
        x="metric",
        y="score",
        hue="model",
        palette="colorblind",
    )
    axis.set_ylim(0, max(0.35, melted["score"].max() * 1.18))
    axis.set_title("Comparaison des detecteurs sur la validation")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.legend(title="Modele", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_per_class(
    class_table: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered_classes = (
        class_table.groupby("class_name")["ap50_95"]
        .mean()
        .sort_values()
        .index
    )
    plt.figure(figsize=(12, 8.8))
    axis = sns.barplot(
        data=class_table,
        y="class_name",
        x="ap50_95",
        hue="model",
        order=ordered_classes,
        palette="colorblind",
    )
    axis.set_title("mAP50-95 par classe")
    axis.set_xlabel("AP50-95")
    axis.set_ylabel("Classe")
    axis.legend(
        title="Modele",
        loc="upper right",
        ncol=1,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_detection_strata(table: pd.DataFrame, output_path: Path) -> None:
    size_order = ["tres petit", "petit", "moyen", "grand"]
    orientation_order = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]
    rates = (
        table.groupby(["size_group", "orientation_group"], observed=True)[
            "detected_iou50"
        ]
        .mean()
        .unstack()
        .reindex(index=size_order, columns=orientation_order)
    )
    counts = (
        table.groupby(["size_group", "orientation_group"], observed=True)[
            "detected_iou50"
        ]
        .size()
        .unstack()
        .reindex(index=size_order, columns=orientation_order)
        .fillna(0)
        .astype(int)
    )
    annotations = np.asarray(
        [
            [
                f"{rates.loc[size, orientation]:.2f}\n(n={counts.loc[size, orientation]})"
                if counts.loc[size, orientation] > 0
                else "n=0"
                for orientation in orientation_order
            ]
            for size in size_order
        ]
    )
    plt.figure(figsize=(11, 5.8))
    axis = sns.heatmap(
        rates,
        annot=annotations,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=max(0.5, float(rates.max().max())),
    )
    axis.set_title("Taux de detection correcte par taille et orientation")
    axis.set_xlabel("Orientation absolue (degres)")
    axis.set_ylabel("Groupe de taille")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_causal_effects(estimates: pd.DataFrame, output_path: Path) -> None:
    ordered = estimates.sort_values("effect").reset_index(drop=True)
    positions = np.arange(len(ordered))
    plt.figure(figsize=(10, 5.8))
    plt.errorbar(
        ordered["effect"],
        positions,
        xerr=[
            ordered["effect"] - ordered["ci_lower"],
            ordered["ci_upper"] - ordered["effect"],
        ],
        fmt="o",
        capsize=5,
        color="#0072B2",
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(positions, ordered["method"])
    plt.xlabel("Effet sur la probabilite de detection correcte")
    plt.title("Effet estime d'un objet tres petit")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_propensity_overlap(table: pd.DataFrame, output_path: Path) -> None:
    plot_table = table.copy()
    plot_table["Objet tres petit"] = plot_table["very_small"].map(
        {0: "Non", 1: "Oui"}
    )
    plt.figure(figsize=(10, 5.8))
    axis = sns.histplot(
        data=plot_table,
        x="propensity_score",
        hue="Objet tres petit",
        bins=25,
        stat="density",
        common_norm=False,
        element="step",
        palette=["#009E73", "#D55E00"],
    )
    axis.set_title("Recouvrement des scores de propension")
    axis.set_xlabel("P(objet tres petit | variables d'ajustement)")
    axis.set_ylabel("Densite")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_causal_tree_figure(
    tree_model,
    preprocessor,
    subgroups: pd.DataFrame,
    output_path: Path,
) -> None:
    feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    tree = tree_model.tree_
    root_feature = feature_names[int(tree.feature[0])]
    readable_features = {
        "tile_object_count": "Nombre d'objets dans la tuile",
        "orientation_abs_deg": "Orientation absolue",
        "log_aspect_ratio": "Log du ratio de forme",
        "gsd": "Resolution au sol (GSD)",
        "center_margin_normalized": "Position du centre dans la tuile",
    }
    root_label = readable_features.get(root_feature, root_feature)
    threshold = float(tree.threshold[0])
    left_leaf = int(tree.children_left[0])
    right_leaf = int(tree.children_right[0])
    subgroup_lookup = subgroups.set_index("leaf_id")

    figure, axis = plt.subplots(figsize=(13, 6.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.5,
        0.78,
        (
            "Division choisie sur la moitie structure\n"
            f"{root_label} <= {threshold:.1f}"
        ),
        ha="center",
        va="center",
        fontsize=13,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F2B179",
            "edgecolor": "#A85D20",
        },
    )

    for x_position, leaf_id, condition in (
        (0.25, left_leaf, f"<= {threshold:.1f}"),
        (0.75, right_leaf, f"> {threshold:.1f}"),
    ):
        row = subgroup_lookup.loc[leaf_id]
        axis.annotate(
            "",
            xy=(x_position, 0.38),
            xytext=(0.5, 0.70),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#333333"},
        )
        axis.text(
            x_position,
            0.27,
            (
                f"{condition}\n"
                f"Effet holdout = {row.effect:.3f}\n"
                f"IC 95 % [{row.ci_lower:.3f}; {row.ci_upper:.3f}]\n"
                f"{int(row.n_objects)} objets | {int(row.n_images)} images"
            ),
            ha="center",
            va="center",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#F6D7BC",
                "edgecolor": "#A85D20",
            },
        )
    axis.set_title(
        "Arbre causal honnete conservateur sur pseudo-outcomes AIPW",
        fontsize=17,
        pad=18,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cate_distribution(table: pd.DataFrame, output_path: Path) -> None:
    plot_table = table.copy()
    plot_table["Objet tres petit"] = plot_table["very_small"].map(
        {0: "Non", 1: "Oui"}
    )
    plt.figure(figsize=(11, 5.8))
    axis = sns.histplot(
        data=plot_table,
        x="cate_effect_forest",
        hue="Objet tres petit",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette=["#009E73", "#D55E00"],
    )
    axis.axvline(
        table["cate_effect_forest"].mean(),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Moyenne",
    )
    axis.set_title("Distribution des effets heterogenes estimes")
    axis.set_xlabel("CATE estimee")
    axis.set_ylabel("Densite")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_causal_dag(output_path: Path) -> None:
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
    colors = [
        "#E69F00"
        if node == "Tres petit"
        else "#009E73"
        if node == "Detection"
        else "#56B4E9"
        for node in dag
    ]
    plt.figure(figsize=(12, 5.5))
    nx.draw_networkx(
        dag,
        positions,
        node_color=colors,
        node_size=2600,
        font_size=9,
        arrowsize=20,
        edge_color="#4d4d4d",
    )
    plt.title("DAG simplifie de l'effet de la petite taille")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    preparation_summary = (
        args.dataset / "preparation_summary.json"
    ).read_text(encoding="utf-8")
    (args.output / "preparation_summary_snapshot.json").write_text(
        preparation_summary,
        encoding="utf-8",
    )
    pd.read_csv(
        args.dataset / "manifests" / "class_coverage.csv"
    ).to_csv(
        args.output / "class_coverage_snapshot.csv",
        index=False,
    )

    model_specs = [
        {
            "name": "YOLO26n-HBB-640",
            "task": "hbb",
            "weights": args.hbb_weights,
            "imgsz": args.hbb_imgsz,
        },
        {
            "name": args.obb_name,
            "task": "obb",
            "weights": args.obb_weights,
            "imgsz": args.obb_imgsz,
        },
        *parse_extra_models(args.extra_obb),
    ]
    for specification in model_specs:
        specification["weights"] = Path(specification["weights"]).resolve()

    model_rows = []
    class_parts = []
    for specification in model_specs:
        global_row, class_table = validate_model(
            specification,
            dataset_root=args.dataset,
            output_root=args.output,
            batch=args.batch,
            workers=args.workers,
            max_det=args.max_det,
        )
        model_rows.append(global_row)
        class_parts.append(class_table)
        create_prediction_sheet(
            specification,
            dataset_root=args.dataset,
            output_path=(
                args.output
                / f"prediction_examples_{specification['name'].lower()}.jpg"
            ),
            confidence=args.confidence,
            max_det=args.max_det,
            seed=args.seed,
        )

    model_table = pd.DataFrame(model_rows)
    class_table = pd.concat(class_parts, ignore_index=True)
    model_table.to_csv(args.output / "model_comparison.csv", index=False)
    class_table.to_csv(args.output / "per_class_metrics.csv", index=False)

    causal_table = build_matched_object_table(
        weights=args.obb_weights,
        task="obb",
        dataset_root=args.dataset,
        imgsz=args.obb_imgsz,
        confidence=args.confidence,
        max_det=args.max_det,
        workers=args.workers,
    )
    causal_result = run_causal_analysis(
        causal_table,
        seed=args.seed,
        bootstrap_repetitions=1000,
    )
    sensitivity = run_sensitivity_analysis(
        causal_result.object_table,
        seed=args.seed,
    )

    causal_result.object_table.to_csv(
        args.output / "causal_object_table.csv",
        index=False,
    )
    causal_result.estimates.to_csv(
        args.output / "causal_effect_estimates.csv",
        index=False,
    )
    causal_result.tree_subgroups.to_csv(
        args.output / "causal_tree_subgroups.csv",
        index=False,
    )
    sensitivity.to_csv(
        args.output / "causal_sensitivity.csv",
        index=False,
    )
    (args.output / "causal_tree_rules.txt").write_text(
        causal_result.tree_rules,
        encoding="utf-8",
    )

    plot_model_comparison(
        model_table,
        args.output / "model_comparison.png",
    )
    plot_per_class(
        class_table,
        args.output / "per_class_metrics.png",
    )
    plot_detection_strata(
        causal_result.object_table,
        args.output / "detection_by_size_orientation.png",
    )
    plot_causal_effects(
        causal_result.estimates,
        args.output / "causal_effects.png",
    )
    plot_propensity_overlap(
        causal_result.object_table,
        args.output / "propensity_overlap.png",
    )
    plot_causal_tree_figure(
        causal_result.tree_model,
        causal_result.tree_preprocessor,
        causal_result.tree_subgroups,
        args.output / "causal_tree.png",
    )
    plot_cate_distribution(
        causal_result.object_table,
        args.output / "cate_distribution.png",
    )
    plot_causal_dag(args.output / "causal_dag.png")

    primary_effect = causal_result.estimates.loc[
        causal_result.estimates["method"].eq("AIPW doublement robuste")
    ].iloc[0]
    obb_row = model_table.loc[model_table["model"].eq(args.obb_name)].iloc[0]
    summary = {
        "causal_question": (
            "Quel est l'effet d'etre un objet tres petit sur la probabilite "
            "d'une detection correcte (IoU >= 0.50), apres ajustement sur la "
            "classe, l'orientation, la densite, la source, le GSD, le ratio "
            "de forme et la position du centre dans la tuile ?"
        ),
        "treatment_definition": {
            "column": "very_small",
            "threshold_relative_area_tile": float(
                causal_result.object_table[
                    "small_area_threshold_train_q25"
                ].iloc[0]
            ),
            "reference": "Premier quartile des objets train uniques.",
        },
        "outcome_definition": (
            f"Meme classe, confiance >= {args.confidence:.2f} et IoU OBB >= 0.50."
        ),
        "primary_model": _json_ready(obb_row.to_dict()),
        "primary_aipw_effect": _json_ready(primary_effect.to_dict()),
        "n_causal_objects": len(causal_result.object_table),
        "n_causal_images": int(
            causal_result.object_table["image_id"].nunique()
        ),
        "models": _json_ready(model_table.to_dict(orient="records")),
        "sensitivity": _json_ready(sensitivity.to_dict(orient="records")),
        "limitations": [
            "Traitement non randomise et variables non observees possibles.",
            "Sous-ensemble stratifie plutot que DOTA complet.",
            "Forêt sur pseudo-outcomes AIPW, approximation pedagogique d'une foret causale.",
            "Arbre honnete volontairement limite a une division et grandes feuilles.",
            "Intervalles bootstrap par image sans re-ajustement complet des modeles nuisances.",
        ],
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
