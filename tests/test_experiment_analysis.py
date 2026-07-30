import numpy as np
import pandas as pd

from src.experiment_analysis import (
    choose_unique_object_instances,
    convex_polygon_iou,
    match_predictions_to_ground_truth,
)


def _manifest_row(object_id: str, tile_id: str, retained: float, margin: float) -> dict:
    return {
        "split": "val",
        "object_id": object_id,
        "tile_id": tile_id,
        "class_id": 0,
        "tile_size": 100,
        "clipped_polygon_area_px": 400,
        "aspect_ratio": 1.0,
        "retained_fraction": retained,
        "tile_x1": margin,
        "tile_y1": margin,
        "tile_x2": margin + 20,
        "tile_y2": margin,
        "tile_x3": margin + 20,
        "tile_y3": margin + 20,
        "tile_x4": margin,
        "tile_y4": margin + 20,
    }


def test_convex_polygon_iou() -> None:
    first = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    second = np.asarray([[5, 0], [15, 0], [15, 10], [5, 10]], dtype=float)
    assert np.isclose(convex_polygon_iou(first, first), 1.0)
    assert np.isclose(convex_polygon_iou(first, second), 1 / 3)


def test_choose_unique_instance_prefers_retention_then_margin() -> None:
    rows = [
        _manifest_row("a", "tile_1", 0.8, 20),
        _manifest_row("a", "tile_2", 1.0, 2),
        _manifest_row("b", "tile_3", 1.0, 1),
        _manifest_row("b", "tile_4", 1.0, 10),
    ]
    selected = choose_unique_object_instances(pd.DataFrame(rows), "val")
    choices = dict(zip(selected["object_id"], selected["tile_id"], strict=True))
    assert choices == {"a": "tile_2", "b": "tile_4"}


def test_matching_is_one_to_one_within_class() -> None:
    ground_truth = pd.DataFrame(
        [
            _manifest_row("a", "tile", 1.0, 0),
            {
                **_manifest_row("b", "tile", 1.0, 0),
                "tile_x1": 30,
                "tile_x2": 50,
                "tile_x3": 50,
                "tile_x4": 30,
            },
        ]
    )
    predictions = [
        np.asarray([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float),
        np.asarray([[30, 0], [50, 0], [50, 20], [30, 20]], dtype=float),
    ]
    matched = match_predictions_to_ground_truth(
        ground_truth,
        predictions,
        np.asarray([0, 0]),
        np.asarray([0.8, 0.9]),
    )
    assert np.allclose(matched["matched_iou"], 1.0)
    assert set(matched["matched_prediction_index"]) == {0, 1}
