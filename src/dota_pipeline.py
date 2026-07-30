"""Pipeline de preparation DOTA-v1.0 pour une experience YOLO HBB/OBB.

Le module reste volontairement sans dependance specialisee autre que celles
deja necessaires au notebook (NumPy, pandas, Pillow et OpenCV). Les fonctions
importantes seront aussi reprises dans le notebook final afin que celui-ci
reste executable independamment.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


DOTA_V1_CLASSES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]

CLASS_TO_ID = {name: index for index, name in enumerate(DOTA_V1_CLASSES)}


def polygon_area(points: np.ndarray) -> float:
    """Retourne l'aire absolue d'un polygone ordonne."""
    points = np.asarray(points, dtype=np.float64)
    x_values = points[:, 0]
    y_values = points[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_values, np.roll(y_values, 1))
            - np.dot(y_values, np.roll(x_values, 1))
        )
    )


def principal_orientation_degrees(points: np.ndarray) -> float:
    """Estime l'orientation absolue de l'axe le plus long, entre 0 et 90 degres."""
    points = np.asarray(points, dtype=np.float64)
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[1]
    edge = first_edge if np.linalg.norm(first_edge) >= np.linalg.norm(second_edge) else second_edge
    angle = np.degrees(np.arctan2(edge[1], edge[0])) % 180
    return float(min(angle, 180 - angle))


def parse_dota_annotation_file(label_path: Path, split: str) -> tuple[dict, list[dict]]:
    """Lit un fichier DOTA et retourne ses metadonnees et ses objets."""
    metadata = {"image_source": "unknown", "gsd": np.nan}
    objects: list[dict] = []

    with label_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith("imagesource:"):
                metadata["image_source"] = line.split(":", 1)[1].strip() or "unknown"
                continue
            if line.startswith("gsd:"):
                raw_value = line.split(":", 1)[1].strip()
                metadata["gsd"] = np.nan if raw_value == "null" else float(raw_value)
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            class_name = parts[8]
            if class_name not in CLASS_TO_ID:
                raise ValueError(f"Classe DOTA inconnue dans {label_path}: {class_name}")

            coordinates = np.asarray([float(value) for value in parts[:8]], dtype=np.float32)
            polygon = coordinates.reshape(4, 2)
            object_index = len(objects)
            edge_lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)

            row = {
                "split": split,
                "image_id": label_path.stem,
                "object_index": object_index,
                "object_id": f"{split}:{label_path.stem}:{object_index}",
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],
                "difficult": int(parts[9]),
                "polygon_area_px": polygon_area(polygon),
                "bbox_width_px": float(polygon[:, 0].max() - polygon[:, 0].min()),
                "bbox_height_px": float(polygon[:, 1].max() - polygon[:, 1].min()),
                "oriented_long_side_px": float(edge_lengths.max()),
                "oriented_short_side_px": float(edge_lengths.min()),
                "orientation_abs_deg": principal_orientation_degrees(polygon),
            }
            for point_index, (x_value, y_value) in enumerate(polygon, start=1):
                row[f"x{point_index}"] = float(x_value)
                row[f"y{point_index}"] = float(y_value)
            objects.append(row)

    return metadata, objects


def load_dota_split(
    image_dir: Path,
    label_dir: Path,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les images et annotations d'un split en deux tables."""
    image_rows: list[dict] = []
    object_rows: list[dict] = []

    label_paths = sorted(label_dir.glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"Aucune annotation trouvee dans {label_dir}")

    for label_path in label_paths:
        image_path = image_dir / f"{label_path.stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image manquante pour {label_path.name}: {image_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        metadata, objects = parse_dota_annotation_file(label_path, split)
        image_rows.append(
            {
                "split": split,
                "image_id": label_path.stem,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "image_width": image_width,
                "image_height": image_height,
                **metadata,
            }
        )

        image_area = image_width * image_height
        for row in objects:
            row["image_width"] = image_width
            row["image_height"] = image_height
            row["image_source"] = metadata["image_source"]
            row["gsd"] = metadata["gsd"]
            row["relative_area_image"] = (
                row["polygon_area_px"] / image_area if image_area > 0 else np.nan
            )
            row["aspect_ratio"] = (
                row["oriented_long_side_px"] / max(row["oriented_short_side_px"], 1e-6)
            )
            object_rows.append(row)

    images = pd.DataFrame(image_rows)
    objects = pd.DataFrame(object_rows)
    return images, objects


def build_image_summary(images: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    """Cree les strates categorie dominante x densite pour l'echantillonnage."""
    object_counts = objects.groupby("image_id").size().rename("object_count")
    class_counts = (
        objects.groupby(["image_id", "class_name"])
        .size()
        .rename("class_count")
        .reset_index()
    )
    dominant_rows = class_counts.sort_values(
        ["image_id", "class_count", "class_name"],
        ascending=[True, False, True],
    ).drop_duplicates("image_id")
    dominant_rows = dominant_rows[["image_id", "class_name"]].rename(
        columns={"class_name": "dominant_class"}
    )

    summary = images.merge(object_counts, on="image_id", how="left")
    summary = summary.merge(dominant_rows, on="image_id", how="left")
    summary["object_count"] = summary["object_count"].fillna(0).astype(int)
    summary["dominant_class"] = summary["dominant_class"].fillna("empty")

    percentile_rank = summary["object_count"].rank(method="average", pct=True)
    summary["density_group"] = pd.cut(
        percentile_rank,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["sparse", "medium", "dense"],
        include_lowest=True,
    ).astype(str)
    summary["scene_stratum"] = (
        summary["dominant_class"].astype(str) + "|" + summary["density_group"].astype(str)
    )

    presence = pd.crosstab(objects["image_id"], objects["class_name"]).gt(0)
    for class_name in DOTA_V1_CLASSES:
        values = presence[class_name] if class_name in presence else False
        summary[f"has_{class_name}"] = summary["image_id"].map(values).fillna(False).astype(bool)

    return summary


def select_stratified_images(
    image_summary: pd.DataFrame,
    number_to_select: int,
    seed: int = 42,
    minimum_images_per_class: int = 4,
) -> pd.DataFrame:
    """Selectionne des images avec couverture des classes et des types de scene.

    La categorie est representee par la presence de chacune des 15 classes.
    Le type de scene est une strate operationnelle combinant la classe dominante
    et la densite d'objets (sparse, medium, dense).
    """
    if number_to_select <= 0:
        raise ValueError("number_to_select doit etre strictement positif")
    if number_to_select >= len(image_summary):
        result = image_summary.copy()
        result["selection_order"] = np.arange(len(result))
        return result

    rng = random.Random(seed)
    selected: list[str] = []
    selected_set: set[str] = set()

    class_image_counts = {
        class_name: int(image_summary[f"has_{class_name}"].sum())
        for class_name in DOTA_V1_CLASSES
    }
    rarity_weight = {
        class_name: 1.0 / max(class_image_counts[class_name], 1)
        for class_name in DOTA_V1_CLASSES
    }

    # Premier passage: garantir plusieurs images de chaque categorie, en
    # commencant par les categories les plus rares.
    ordered_classes = sorted(DOTA_V1_CLASSES, key=lambda name: class_image_counts[name])
    for class_name in ordered_classes:
        target = min(minimum_images_per_class, class_image_counts[class_name])
        while (
            sum(
                image_id in selected_set
                for image_id in image_summary.loc[
                    image_summary[f"has_{class_name}"], "image_id"
                ]
            )
            < target
            and len(selected) < number_to_select
        ):
            candidates = image_summary[
                image_summary[f"has_{class_name}"]
                & ~image_summary["image_id"].isin(selected_set)
            ].copy()
            if candidates.empty:
                break

            candidates["coverage_score"] = 0.0
            for candidate_class in DOTA_V1_CLASSES:
                candidates["coverage_score"] += (
                    candidates[f"has_{candidate_class}"].astype(float)
                    * rarity_weight[candidate_class]
                )
            candidates["random_tie_break"] = [rng.random() for _ in range(len(candidates))]
            chosen = candidates.sort_values(
                ["coverage_score", "random_tie_break"],
                ascending=[False, True],
            ).iloc[0]["image_id"]
            selected.append(chosen)
            selected_set.add(chosen)

    # Deuxieme passage: rapprocher la selection de la distribution des strates.
    stratum_proportions = image_summary["scene_stratum"].value_counts(normalize=True)
    while len(selected) < number_to_select:
        remaining = image_summary[~image_summary["image_id"].isin(selected_set)].copy()
        if remaining.empty:
            break

        current_counts = (
            image_summary[image_summary["image_id"].isin(selected_set)]["scene_stratum"]
            .value_counts()
            .to_dict()
        )
        target_counts = (stratum_proportions * number_to_select).to_dict()
        remaining_strata = set(remaining["scene_stratum"])
        chosen_stratum = max(
            remaining_strata,
            key=lambda stratum: target_counts.get(stratum, 0)
            - current_counts.get(stratum, 0),
        )
        candidates = remaining[remaining["scene_stratum"] == chosen_stratum].copy()
        candidates["random_tie_break"] = [rng.random() for _ in range(len(candidates))]
        chosen = candidates.sort_values("random_tie_break").iloc[0]["image_id"]
        selected.append(chosen)
        selected_set.add(chosen)

    result = image_summary[image_summary["image_id"].isin(selected_set)].copy()
    order_map = {image_id: index for index, image_id in enumerate(selected)}
    result["selection_order"] = result["image_id"].map(order_map)
    return result.sort_values("selection_order").reset_index(drop=True)


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Retourne des positions couvrant toute la dimension, bord final inclus."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(length - tile_size + 1, 1), stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def _object_polygon(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [[row[f"x{index}"], row[f"y{index}"]] for index in range(1, 5)],
        dtype=np.float32,
    )


def clip_object_to_tile(
    polygon: np.ndarray,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> tuple[float, np.ndarray | None]:
    """Intersecte une boite orientee avec une tuile et retourne un OBB clippe."""
    polygon = cv2.convexHull(np.asarray(polygon, dtype=np.float32))
    tile_rectangle = np.asarray(
        [
            [tile_x, tile_y],
            [tile_x + tile_size, tile_y],
            [tile_x + tile_size, tile_y + tile_size],
            [tile_x, tile_y + tile_size],
        ],
        dtype=np.float32,
    )
    tile_rectangle = cv2.convexHull(tile_rectangle)
    original_area = abs(float(cv2.contourArea(polygon)))
    if original_area <= 0:
        return 0.0, None

    intersection_area, intersection_polygon = cv2.intersectConvexConvex(
        polygon,
        tile_rectangle,
    )
    if intersection_polygon is None or intersection_area <= 0:
        return 0.0, None

    retained_fraction = float(intersection_area / original_area)
    intersection_points = intersection_polygon.reshape(-1, 2)
    if retained_fraction >= 0.999:
        clipped_box = polygon.reshape(-1, 2)
    else:
        clipped_box = cv2.boxPoints(cv2.minAreaRect(intersection_points)).astype(np.float32)
    clipped_box[:, 0] -= tile_x
    clipped_box[:, 1] -= tile_y
    clipped_box = np.clip(clipped_box, 0, tile_size)
    return min(retained_fraction, 1.0), clipped_box


def _format_obb_label(class_id: int, points: np.ndarray, tile_size: int) -> str:
    normalized = np.clip(np.asarray(points, dtype=float) / tile_size, 0.0, 1.0)
    coordinate_text = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
    return f"{class_id} {coordinate_text}"


def _format_hbb_label(class_id: int, points: np.ndarray, tile_size: int) -> str:
    points = np.asarray(points, dtype=float)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_center = (x_min + x_max) / 2 / tile_size
    y_center = (y_min + y_max) / 2 / tile_size
    width = (x_max - x_min) / tile_size
    height = (y_max - y_min) / tile_size
    values = np.clip([x_center, y_center, width, height], 0.0, 1.0)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def _safe_hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_dataset_yaml(dataset_root: Path, task_name: str) -> Path:
    yaml_path = dataset_root / f"dota_{task_name}.yaml"
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {class_name}"
        for class_id, class_name in enumerate(DOTA_V1_CLASSES)
    )
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def prepare_tiled_split(
    selected_images: pd.DataFrame,
    objects: pd.DataFrame,
    split: str,
    output_root: Path,
    tile_size: int,
    stride: int,
    seed: int,
    negative_tile_fraction: float = 0.15,
    minimum_retained_fraction: float = 0.70,
    ambiguous_fragment_fraction: float = 0.20,
    jpeg_quality: int = 92,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decoupe un split et produit simultanement les labels OBB et HBB."""
    rng = random.Random(seed)
    obb_root = output_root / "obb"
    hbb_root = output_root / "hbb"
    for task_root in (obb_root, hbb_root):
        (task_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (task_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    object_manifest_rows: list[dict] = []
    tile_manifest_rows: list[dict] = []

    for _, image_row in selected_images.iterrows():
        image_id = image_row["image_id"]
        image_objects = objects[objects["image_id"] == image_id].copy()
        image_path = Path(image_row["image_path"])

        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
            width, height = source_image.size

            x_positions = tile_positions(width, tile_size, stride)
            y_positions = tile_positions(height, tile_size, stride)

            for tile_y in y_positions:
                for tile_x in x_positions:
                    kept_objects: list[tuple[pd.Series, float, np.ndarray]] = []
                    ambiguous_fragment = False

                    for _, object_row in image_objects.iterrows():
                        retained_fraction, clipped_box = clip_object_to_tile(
                            _object_polygon(object_row),
                            tile_x,
                            tile_y,
                            tile_size,
                        )
                        if (
                            ambiguous_fragment_fraction
                            <= retained_fraction
                            < minimum_retained_fraction
                        ):
                            ambiguous_fragment = True
                            break
                        if retained_fraction >= minimum_retained_fraction and clipped_box is not None:
                            side_lengths = np.linalg.norm(
                                np.roll(clipped_box, -1, axis=0) - clipped_box,
                                axis=1,
                            )
                            if side_lengths.min() >= 2:
                                kept_objects.append(
                                    (object_row, retained_fraction, clipped_box)
                                )

                    if ambiguous_fragment:
                        continue
                    if not kept_objects and rng.random() > negative_tile_fraction:
                        continue

                    tile_id = f"{image_id}__x{tile_x}_y{tile_y}"
                    obb_image_path = obb_root / "images" / split / f"{tile_id}.jpg"
                    hbb_image_path = hbb_root / "images" / split / f"{tile_id}.jpg"
                    obb_label_path = obb_root / "labels" / split / f"{tile_id}.txt"
                    hbb_label_path = hbb_root / "labels" / split / f"{tile_id}.txt"

                    crop = source_image.crop(
                        (
                            tile_x,
                            tile_y,
                            min(tile_x + tile_size, width),
                            min(tile_y + tile_size, height),
                        )
                    )
                    canvas = Image.new("RGB", (tile_size, tile_size), (114, 114, 114))
                    canvas.paste(crop, (0, 0))
                    canvas.save(obb_image_path, format="JPEG", quality=jpeg_quality)
                    link_method = _safe_hardlink_or_copy(obb_image_path, hbb_image_path)

                    obb_lines: list[str] = []
                    hbb_lines: list[str] = []
                    for label_index, (object_row, retained_fraction, clipped_box) in enumerate(
                        kept_objects
                    ):
                        class_id = int(object_row["class_id"])
                        obb_lines.append(_format_obb_label(class_id, clipped_box, tile_size))
                        hbb_lines.append(_format_hbb_label(class_id, clipped_box, tile_size))

                        row = object_row.to_dict()
                        row.update(
                            {
                                "tile_id": tile_id,
                                "label_index": label_index,
                                "tile_x": tile_x,
                                "tile_y": tile_y,
                                "tile_size": tile_size,
                                "retained_fraction": retained_fraction,
                                "clipped_polygon_area_px": polygon_area(clipped_box),
                            }
                        )
                        for point_index, (x_value, y_value) in enumerate(
                            clipped_box,
                            start=1,
                        ):
                            row[f"tile_x{point_index}"] = float(x_value)
                            row[f"tile_y{point_index}"] = float(y_value)
                        object_manifest_rows.append(row)

                    obb_label_path.write_text(
                        "\n".join(obb_lines) + ("\n" if obb_lines else ""),
                        encoding="utf-8",
                    )
                    hbb_label_path.write_text(
                        "\n".join(hbb_lines) + ("\n" if hbb_lines else ""),
                        encoding="utf-8",
                    )
                    tile_manifest_rows.append(
                        {
                            "split": split,
                            "tile_id": tile_id,
                            "image_id": image_id,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "tile_size": tile_size,
                            "object_count": len(kept_objects),
                            "is_negative_tile": len(kept_objects) == 0,
                            "hbb_image_link_method": link_method,
                        }
                    )

    object_manifest = pd.DataFrame(object_manifest_rows)
    tile_manifest = pd.DataFrame(tile_manifest_rows)
    if not object_manifest.empty:
        duplicate_counts = object_manifest.groupby("object_id").size()
        object_manifest["tile_instances_for_object"] = object_manifest["object_id"].map(
            duplicate_counts
        )
    return object_manifest, tile_manifest


def validate_prepared_dataset(
    output_root: Path,
    selections: dict[str, pd.DataFrame],
    object_manifest: pd.DataFrame,
) -> dict:
    """Verifie structure, labels, couverture et separation des images sources."""
    train_ids = set(selections["train"]["image_id"])
    val_ids = set(selections["val"]["image_id"])
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise AssertionError(f"Fuite train/validation: {overlap[:5]}")

    validation: dict[str, object] = {
        "original_image_overlap_count": len(overlap),
        "tasks": {},
    }
    for task in ("obb", "hbb"):
        task_summary: dict[str, dict] = {}
        expected_columns = 9 if task == "obb" else 5
        for split in ("train", "val"):
            image_paths = sorted((output_root / task / "images" / split).glob("*.jpg"))
            label_paths = sorted((output_root / task / "labels" / split).glob("*.txt"))
            image_stems = {path.stem for path in image_paths}
            label_stems = {path.stem for path in label_paths}
            if image_stems != label_stems:
                raise AssertionError(f"Images/labels non alignes pour {task}/{split}")

            invalid_lines = 0
            coordinate_violations = 0
            object_count = 0
            classes_present: set[int] = set()
            for label_path in label_paths:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    values = line.split()
                    if len(values) != expected_columns:
                        invalid_lines += 1
                        continue
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                    classes_present.add(class_id)
                    object_count += 1
                    if any(value < 0 or value > 1 for value in coordinates):
                        coordinate_violations += 1

            if invalid_lines or coordinate_violations:
                raise AssertionError(
                    f"Labels invalides pour {task}/{split}: "
                    f"{invalid_lines=}, {coordinate_violations=}"
                )
            task_summary[split] = {
                "image_count": len(image_paths),
                "negative_image_count": sum(
                    path.stat().st_size == 0 for path in label_paths
                ),
                "object_count": object_count,
                "class_count": len(classes_present),
                "classes_present": sorted(classes_present),
            }
        validation["tasks"][task] = task_summary

    validation["manifest_row_count"] = int(len(object_manifest))
    validation["unique_original_object_count"] = int(
        object_manifest["object_id"].nunique()
    )
    return validation


def build_experiment_dataset(
    project_dir: Path,
    output_root: Path,
    train_image_count: int = 180,
    val_image_count: int = 60,
    tile_size: int = 1024,
    train_stride: int = 824,
    val_stride: int = 1024,
    seed: int = 42,
) -> dict:
    """Construit l'experience complete HBB/OBB et retourne son resume."""
    project_dir = Path(project_dir).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"{output_root} existe deja et n'est pas vide. "
            "Choisir un nouveau dossier pour proteger les resultats existants."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_objects = load_dota_split(
        project_dir / "Training Data" / "Images",
        project_dir / "Training Data" / "LabelTxt",
        "train",
    )
    val_images, val_objects = load_dota_split(
        project_dir / "Validation Data" / "Images",
        project_dir / "Validation Data" / "LabelTxt",
        "val",
    )

    train_summary = build_image_summary(train_images, train_objects)
    val_summary = build_image_summary(val_images, val_objects)
    selected_train = select_stratified_images(
        train_summary,
        train_image_count,
        seed=seed,
    )
    selected_val = select_stratified_images(
        val_summary,
        val_image_count,
        seed=seed + 1,
    )

    selected_train.to_csv(manifests_dir / "selected_train_images.csv", index=False)
    selected_val.to_csv(manifests_dir / "selected_val_images.csv", index=False)

    train_object_manifest, train_tile_manifest = prepare_tiled_split(
        selected_train,
        train_objects,
        "train",
        output_root,
        tile_size,
        train_stride,
        seed=seed,
    )
    val_object_manifest, val_tile_manifest = prepare_tiled_split(
        selected_val,
        val_objects,
        "val",
        output_root,
        tile_size,
        val_stride,
        seed=seed + 1,
    )

    object_manifest = pd.concat(
        [train_object_manifest, val_object_manifest],
        ignore_index=True,
    )
    tile_manifest = pd.concat(
        [train_tile_manifest, val_tile_manifest],
        ignore_index=True,
    )
    object_manifest.to_csv(manifests_dir / "object_manifest.csv", index=False)
    tile_manifest.to_csv(manifests_dir / "tile_manifest.csv", index=False)

    obb_yaml = _write_dataset_yaml(output_root / "obb", "obb")
    hbb_yaml = _write_dataset_yaml(output_root / "hbb", "hbb")
    validation = validate_prepared_dataset(
        output_root,
        {"train": selected_train, "val": selected_val},
        object_manifest,
    )

    class_coverage = (
        object_manifest.groupby(["split", "class_name"])["object_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=DOTA_V1_CLASSES, fill_value=0)
    )
    class_coverage.to_csv(manifests_dir / "class_coverage.csv")

    summary = {
        "project_dir": str(project_dir),
        "output_root": str(output_root),
        "configuration": {
            "train_image_count": train_image_count,
            "val_image_count": val_image_count,
            "tile_size": tile_size,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "seed": seed,
        },
        "yaml": {"obb": str(obb_yaml), "hbb": str(hbb_yaml)},
        "validation": validation,
        "class_coverage": class_coverage.to_dict(),
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def compact_preparation_report(summary: dict) -> pd.DataFrame:
    """Transforme le resume JSON en petit tableau lisible."""
    rows = []
    for task, split_values in summary["validation"]["tasks"].items():
        for split, values in split_values.items():
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "tiles": values["image_count"],
                    "negative_tiles": values["negative_image_count"],
                    "objects": values["object_count"],
                    "classes": values["class_count"],
                }
            )
    return pd.DataFrame(rows)


def iter_label_coordinates(label_paths: Iterable[Path]) -> Iterable[list[float]]:
    """Petit utilitaire de test pour parcourir les coordonnees exportees."""
    for label_path in label_paths:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            yield [float(value) for value in line.split()[1:]]
