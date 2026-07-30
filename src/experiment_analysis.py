"""Evaluation par objet et estimation causale pour l'experience DOTA.

Les fonctions restent independantes d'Ultralytics afin de pouvoir etre testees
sur de petits exemples et recopiees dans le notebook final.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text


NUMERIC_ADJUSTMENT_COLUMNS = [
    "orientation_abs_deg",
    "log_aspect_ratio",
    "tile_object_count",
    "gsd",
    "center_margin_normalized",
]

CATEGORICAL_ADJUSTMENT_COLUMNS = ["class_name", "image_source"]


@dataclass
class CausalAnalysisResult:
    """Conteneur compact pour les sorties de l'analyse causale."""

    object_table: pd.DataFrame
    estimates: pd.DataFrame
    tree_subgroups: pd.DataFrame
    tree_rules: str
    tree_model: DecisionTreeRegressor
    tree_preprocessor: ColumnTransformer


def convex_polygon_iou(polygon_a: np.ndarray, polygon_b: np.ndarray) -> float:
    """Calcule l'IoU de deux polygones convexes."""
    polygon_a = cv2.convexHull(np.asarray(polygon_a, dtype=np.float32))
    polygon_b = cv2.convexHull(np.asarray(polygon_b, dtype=np.float32))
    area_a = abs(float(cv2.contourArea(polygon_a)))
    area_b = abs(float(cv2.contourArea(polygon_b)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    union_area = area_a + area_b - float(intersection_area)
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def hbb_to_polygon(box: Iterable[float]) -> np.ndarray:
    """Convertit ``xmin, ymin, xmax, ymax`` en quadrilatere."""
    x_min, y_min, x_max, y_max = [float(value) for value in box]
    return np.asarray(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def manifest_polygon(row: pd.Series, prefix: str = "tile_") -> np.ndarray:
    """Reconstruit le quadrilatere d'une ligne du manifeste."""
    return np.asarray(
        [
            [row[f"{prefix}x{index}"], row[f"{prefix}y{index}"]]
            for index in range(1, 5)
        ],
        dtype=np.float32,
    )


def add_instance_geometry(manifest: pd.DataFrame) -> pd.DataFrame:
    """Ajoute aire relative a la tuile et marge minimale aux bords."""
    result = manifest.copy()
    x_columns = [f"tile_x{index}" for index in range(1, 5)]
    y_columns = [f"tile_y{index}" for index in range(1, 5)]
    minimum_x = result[x_columns].min(axis=1)
    maximum_x = result[x_columns].max(axis=1)
    minimum_y = result[y_columns].min(axis=1)
    maximum_y = result[y_columns].max(axis=1)
    result["edge_margin_px"] = pd.concat(
        [
            minimum_x,
            minimum_y,
            result["tile_size"] - maximum_x,
            result["tile_size"] - maximum_y,
        ],
        axis=1,
    ).min(axis=1)
    result["edge_margin_normalized"] = (
        result["edge_margin_px"] / result["tile_size"]
    ).clip(lower=0)
    center_x = result[x_columns].mean(axis=1)
    center_y = result[y_columns].mean(axis=1)
    result["center_margin_normalized"] = (
        pd.concat(
            [
                center_x,
                center_y,
                result["tile_size"] - center_x,
                result["tile_size"] - center_y,
            ],
            axis=1,
        ).min(axis=1)
        / result["tile_size"]
    ).clip(lower=0)
    result["relative_area_tile"] = (
        result["clipped_polygon_area_px"] / result["tile_size"].pow(2)
    )
    result["log_aspect_ratio"] = np.log1p(result["aspect_ratio"].clip(lower=0))
    return result


def choose_unique_object_instances(
    manifest: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Conserve une seule tuile par objet original.

    Certaines positions finales de la grille se chevauchent. On choisit
    d'abord l'instance qui conserve la plus grande fraction de l'objet, puis
    celle qui se trouve le plus loin du bord de sa tuile.
    """
    candidates = add_instance_geometry(
        manifest.loc[manifest["split"].eq(split)].copy()
    )
    ordered = candidates.sort_values(
        ["object_id", "retained_fraction", "edge_margin_px", "tile_id"],
        ascending=[True, False, False, True],
    )
    selected = ordered.drop_duplicates("object_id", keep="first").copy()
    if selected["object_id"].duplicated().any():
        raise AssertionError("La selection contient encore des objets dupliques.")
    return selected.reset_index(drop=True)


def match_predictions_to_ground_truth(
    ground_truth: pd.DataFrame,
    prediction_polygons: list[np.ndarray],
    prediction_classes: np.ndarray,
    prediction_confidences: np.ndarray,
) -> pd.DataFrame:
    """Associe predictions et verites terrain, classe par classe.

    L'algorithme hongrois maximise la somme des IoU tout en imposant une
    association un-a-un. Les objets sans prediction gardent une IoU et une
    confiance nulles.
    """
    matched = ground_truth.copy().reset_index(drop=True)
    matched["matched_iou"] = 0.0
    matched["matched_confidence"] = 0.0
    matched["matched_prediction_index"] = -1

    prediction_classes = np.asarray(prediction_classes, dtype=int)
    prediction_confidences = np.asarray(prediction_confidences, dtype=float)
    if len(prediction_polygons) != len(prediction_classes):
        raise ValueError("Le nombre de polygones et de classes predites differe.")

    for class_id in sorted(matched["class_id"].astype(int).unique()):
        ground_indices = matched.index[matched["class_id"].astype(int).eq(class_id)]
        prediction_indices = np.flatnonzero(prediction_classes == class_id)
        if len(ground_indices) == 0 or len(prediction_indices) == 0:
            continue

        iou_matrix = np.zeros(
            (len(ground_indices), len(prediction_indices)),
            dtype=np.float32,
        )
        for ground_position, ground_index in enumerate(ground_indices):
            ground_polygon = manifest_polygon(matched.loc[ground_index])
            for prediction_position, prediction_index in enumerate(prediction_indices):
                iou_matrix[ground_position, prediction_position] = convex_polygon_iou(
                    ground_polygon,
                    prediction_polygons[prediction_index],
                )

        assigned_ground, assigned_prediction = linear_sum_assignment(-iou_matrix)
        for ground_position, prediction_position in zip(
            assigned_ground,
            assigned_prediction,
            strict=True,
        ):
            ground_index = ground_indices[ground_position]
            prediction_index = int(prediction_indices[prediction_position])
            matched.loc[ground_index, "matched_iou"] = float(
                iou_matrix[ground_position, prediction_position]
            )
            matched.loc[ground_index, "matched_confidence"] = float(
                prediction_confidences[prediction_index]
            )
            matched.loc[ground_index, "matched_prediction_index"] = prediction_index

    return matched


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_ADJUSTMENT_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_ADJUSTMENT_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _build_tree_preprocessor() -> ColumnTransformer:
    """Prepare les covariables d'arbre sans standardiser les seuils numeriques."""
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_ADJUSTMENT_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_ADJUSTMENT_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.full(len(features), float(classes[0] == 1))
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def _cluster_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    seed: int,
    repetitions: int = 1000,
) -> tuple[float, float]:
    """Intervalle percentile en reechantillonnant les images sources."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return np.nan, np.nan

    positions = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_positions = np.concatenate(
            [positions[group] for group in sampled_groups]
        )
        bootstrap_means[repetition] = values[sampled_positions].mean()
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def estimate_doubly_robust_effect(
    object_table: pd.DataFrame,
    treatment_column: str = "very_small",
    outcome_column: str = "detected_iou50",
    group_column: str = "image_id",
    folds: int = 5,
    propensity_clip: float = 0.05,
    bootstrap_repetitions: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estime des effets naif, g-computation, IPW et AIPW avec cross-fitting."""
    table = object_table.copy().reset_index(drop=True)
    treatment = table[treatment_column].astype(int).to_numpy()
    outcome = table[outcome_column].astype(int).to_numpy()
    groups = table[group_column].astype(str).to_numpy()
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("Le traitement doit contenir les deux groupes 0 et 1.")
    if set(np.unique(outcome)) - {0, 1}:
        raise ValueError("L'outcome doit etre binaire.")

    adjustment_columns = (
        NUMERIC_ADJUSTMENT_COLUMNS + CATEGORICAL_ADJUSTMENT_COLUMNS
    )
    features = table[adjustment_columns]
    propensity = np.zeros(len(table), dtype=float)
    mu_zero = np.zeros(len(table), dtype=float)
    mu_one = np.zeros(len(table), dtype=float)

    number_of_splits = min(folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=number_of_splits)
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(features, treatment, groups)
    ):
        propensity_model = Pipeline(
            [
                ("preprocess", _build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed + fold_index,
                    ),
                ),
            ]
        )
        propensity_model.fit(
            features.iloc[train_indices],
            treatment[train_indices],
        )
        propensity[test_indices] = _positive_probability(
            propensity_model,
            features.iloc[test_indices],
        )

        for level, destination in ((0, mu_zero), (1, mu_one)):
            level_indices = train_indices[treatment[train_indices] == level]
            outcome_model = Pipeline(
                [
                    ("preprocess", _build_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=350,
                            min_samples_leaf=15,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=seed + 10 * fold_index + level,
                        ),
                    ),
                ]
            )
            outcome_model.fit(
                features.iloc[level_indices],
                outcome[level_indices],
            )
            destination[test_indices] = _positive_probability(
                outcome_model,
                features.iloc[test_indices],
            )

    propensity = np.clip(propensity, propensity_clip, 1 - propensity_clip)
    mu_zero = np.clip(mu_zero, 0, 1)
    mu_one = np.clip(mu_one, 0, 1)
    aipw_score = (
        mu_one
        - mu_zero
        + treatment * (outcome - mu_one) / propensity
        - (1 - treatment) * (outcome - mu_zero) / (1 - propensity)
    )
    ipw_score = (
        treatment * outcome / propensity
        - (1 - treatment) * outcome / (1 - propensity)
    )
    naive_score = np.where(
        treatment == 1,
        outcome / max(treatment.mean(), 1e-9),
        -outcome / max((1 - treatment).mean(), 1e-9),
    )
    g_computation_score = mu_one - mu_zero

    table["propensity_score"] = propensity
    table["predicted_outcome_control"] = mu_zero
    table["predicted_outcome_treated"] = mu_one
    table["aipw_score"] = aipw_score
    table["cate_g_computation"] = g_computation_score

    score_map = {
        "Difference brute": naive_score,
        "G-computation": g_computation_score,
        "IPW": ipw_score,
        "AIPW doublement robuste": aipw_score,
    }
    estimate_rows = []
    for method_index, (method, scores) in enumerate(score_map.items()):
        lower, upper = _cluster_bootstrap_interval(
            scores,
            groups,
            seed=seed + method_index,
            repetitions=bootstrap_repetitions,
        )
        estimate_rows.append(
            {
                "method": method,
                "effect": float(np.mean(scores)),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_objects": len(table),
                "n_images": int(pd.Series(groups).nunique()),
                "treatment_prevalence": float(treatment.mean()),
                "outcome_treated": float(outcome[treatment == 1].mean()),
                "outcome_control": float(outcome[treatment == 0].mean()),
                "propensity_overlap_0_1_0_9": float(
                    ((propensity >= 0.1) & (propensity <= 0.9)).mean()
                ),
            }
        )
    return table, pd.DataFrame(estimate_rows)


def fit_honest_causal_tree(
    object_table: pd.DataFrame,
    score_column: str = "aipw_score",
    group_column: str = "image_id",
    max_depth: int = 1,
    minimum_leaf: int = 200,
    bootstrap_repetitions: int = 800,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    str,
    DecisionTreeRegressor,
    ColumnTransformer,
]:
    """Construit un arbre de sous-groupes sur des pseudo-outcomes AIPW."""
    table = object_table.copy().reset_index(drop=True)
    adjustment_columns = (
        NUMERIC_ADJUSTMENT_COLUMNS + CATEGORICAL_ADJUSTMENT_COLUMNS
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    structure_indices, estimation_indices = next(
        splitter.split(table, groups=table[group_column])
    )

    preprocessor = _build_tree_preprocessor()
    structure_features = preprocessor.fit_transform(
        table.loc[structure_indices, adjustment_columns]
    )
    estimation_features = preprocessor.transform(
        table.loc[estimation_indices, adjustment_columns]
    )
    feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=minimum_leaf,
        random_state=seed,
    )
    tree.fit(
        structure_features,
        table.loc[structure_indices, score_column],
    )
    estimation_table = table.loc[estimation_indices].copy()
    estimation_table["causal_tree_leaf"] = tree.apply(estimation_features)

    subgroup_rows = []
    for leaf_id, subgroup in estimation_table.groupby("causal_tree_leaf"):
        lower, upper = _cluster_bootstrap_interval(
            subgroup[score_column].to_numpy(),
            subgroup[group_column].astype(str).to_numpy(),
            seed=seed + int(leaf_id),
            repetitions=bootstrap_repetitions,
        )
        subgroup_rows.append(
            {
                "leaf_id": int(leaf_id),
                "n_objects": len(subgroup),
                "n_images": int(subgroup[group_column].nunique()),
                "treated_objects": int(subgroup["very_small"].sum()),
                "control_objects": int((1 - subgroup["very_small"]).sum()),
                "effect": float(subgroup[score_column].mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "dominant_class": subgroup["class_name"].mode().iat[0],
                "median_orientation": float(
                    subgroup["orientation_abs_deg"].median()
                ),
                "median_density": float(subgroup["tile_object_count"].median()),
            }
        )

    rules = export_text(tree, feature_names=feature_names, decimals=3)
    return (
        pd.DataFrame(subgroup_rows).sort_values("effect").reset_index(drop=True),
        rules,
        tree,
        preprocessor,
    )


def add_cross_fitted_effect_forest(
    object_table: pd.DataFrame,
    score_column: str = "aipw_score",
    group_column: str = "image_id",
    seed: int = 42,
) -> pd.DataFrame:
    """Ajoute des CATE issues d'une foret sur pseudo-outcomes, avec split honnete."""
    table = object_table.copy().reset_index(drop=True)
    adjustment_columns = (
        NUMERIC_ADJUSTMENT_COLUMNS + CATEGORICAL_ADJUSTMENT_COLUMNS
    )
    splitter = GroupKFold(n_splits=2)
    cate = np.zeros(len(table), dtype=float)

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(table, groups=table[group_column])
    ):
        preprocessor = _build_preprocessor()
        train_features = preprocessor.fit_transform(
            table.loc[train_indices, adjustment_columns]
        )
        test_features = preprocessor.transform(
            table.loc[test_indices, adjustment_columns]
        )
        forest = RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=25,
            max_depth=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed + fold_index,
        )
        forest.fit(train_features, table.loc[train_indices, score_column])
        cate[test_indices] = forest.predict(test_features)

    # Un pseudo-outcome AIPW individuel peut sortir de [-1, 1], alors que
    # l'effet conditionnel d'un outcome binaire reste dans ces bornes.
    table["cate_effect_forest"] = np.clip(cate, -1.0, 1.0)
    table["cate_quartile"] = pd.qcut(
        table["cate_effect_forest"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    return table


def run_causal_analysis(
    object_table: pd.DataFrame,
    seed: int = 42,
    bootstrap_repetitions: int = 1000,
) -> CausalAnalysisResult:
    """Execute la chaine causale complete sur une table deja appariee."""
    scored_table, estimates = estimate_doubly_robust_effect(
        object_table,
        seed=seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    scored_table = add_cross_fitted_effect_forest(scored_table, seed=seed)
    (
        tree_subgroups,
        tree_rules,
        tree_model,
        tree_preprocessor,
    ) = fit_honest_causal_tree(
        scored_table,
        seed=seed,
        bootstrap_repetitions=max(400, bootstrap_repetitions // 2),
    )
    return CausalAnalysisResult(
        object_table=scored_table,
        estimates=estimates,
        tree_subgroups=tree_subgroups,
        tree_rules=tree_rules,
        tree_model=tree_model,
        tree_preprocessor=tree_preprocessor,
    )
