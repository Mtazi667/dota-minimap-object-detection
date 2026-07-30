"""Cree une planche de controle visuel pour un dataset YOLO HBB ou OBB."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--task", choices=["obb", "hbb"], default="obb")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def draw_labels(image_path: Path, label_path: Path, task: str) -> Image.Image:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        class_id = int(values[0])
        coordinates = np.asarray([float(value) for value in values[1:]], dtype=float)
        if task == "obb":
            points = coordinates.reshape(4, 2)
            points[:, 0] *= width
            points[:, 1] *= height
        else:
            x_center, y_center, box_width, box_height = coordinates
            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            x_max = (x_center + box_width / 2) * width
            y_max = (y_center + box_height / 2) * height
            points = np.asarray(
                [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            )

        points_int = np.round(points).astype(np.int32)
        color = tuple(int(value) for value in np.random.default_rng(class_id).integers(60, 255, 3))
        cv2.polylines(image, [points_int], isClosed=True, color=color, thickness=2)
        anchor = tuple(points_int[0])
        cv2.putText(
            image,
            CLASS_NAMES[class_id],
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main() -> None:
    args = parse_args()
    image_dir = args.dataset_root / args.task / "images" / args.split
    label_dir = args.dataset_root / args.task / "labels" / args.split
    image_paths = sorted(image_dir.glob("*.jpg"))
    labelled_paths = [
        path
        for path in image_paths
        if (label_dir / f"{path.stem}.txt").stat().st_size > 0
    ]
    selected = random.Random(args.seed).sample(
        labelled_paths,
        min(args.count, len(labelled_paths)),
    )

    panels: list[Image.Image] = []
    for image_path in selected:
        panel = draw_labels(
            image_path,
            label_dir / f"{image_path.stem}.txt",
            args.task,
        )
        panel.thumbnail((480, 480))
        canvas = Image.new("RGB", (500, 530), "white")
        canvas.paste(panel, ((500 - panel.width) // 2, 10))
        ImageDraw.Draw(canvas).text((12, 505), image_path.stem, fill="black")
        panels.append(canvas)

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    contact_sheet = Image.new("RGB", (columns * 500, rows * 530), "#eeeeee")
    for index, panel in enumerate(panels):
        contact_sheet.paste(panel, ((index % columns) * 500, (index // columns) * 530))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
