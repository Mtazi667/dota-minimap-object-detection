"""Construit le sous-ensemble stratifie et les tuiles HBB/OBB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.dota_pipeline import build_experiment_dataset, compact_preparation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "prepared_data" / "dota_experiment_v1",
    )
    parser.add_argument("--train-images", type=int, default=180)
    parser.add_argument("--val-images", type=int, default=60)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--train-stride", type=int, default=824)
    parser.add_argument("--val-stride", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_experiment_dataset(
        project_dir=PROJECT_DIR,
        output_root=args.output,
        train_image_count=args.train_images,
        val_image_count=args.val_images,
        tile_size=args.tile_size,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        seed=args.seed,
    )
    print(compact_preparation_report(summary).to_string(index=False))
    print(json.dumps(summary["configuration"], indent=2))


if __name__ == "__main__":
    main()
