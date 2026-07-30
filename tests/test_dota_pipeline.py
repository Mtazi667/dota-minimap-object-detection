from pathlib import Path

import numpy as np

from src.dota_pipeline import (
    clip_object_to_tile,
    polygon_area,
    principal_orientation_degrees,
    tile_positions,
)


def test_polygon_area_and_orientation() -> None:
    rectangle = np.asarray([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=np.float32)
    assert polygon_area(rectangle) == 40
    assert principal_orientation_degrees(rectangle) == 0


def test_tile_positions_cover_final_border() -> None:
    assert tile_positions(500, 1024, 824) == [0]
    assert tile_positions(2500, 1024, 824) == [0, 824, 1476]


def test_clip_object_to_tile() -> None:
    polygon = np.asarray(
        [[900, 100], [1100, 100], [1100, 300], [900, 300]],
        dtype=np.float32,
    )
    retained, clipped = clip_object_to_tile(polygon, 0, 0, 1024)
    assert 0.61 < retained < 0.63
    assert clipped is not None
    assert clipped[:, 0].min() >= 0
    assert clipped[:, 0].max() <= 1024


def test_project_root_exists() -> None:
    assert (Path(__file__).resolve().parents[1] / "Subject.pdf").exists()
