"""Entraine la baseline HBB ou le modele principal OBB sur les memes tuiles."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics_config"))

import torch
from ultralytics import YOLO


DEFAULT_DATASET = PROJECT_DIR / "prepared_data" / "dota_experiment_v1"
RUNS_DIR = PROJECT_DIR / "runs" / "dota_experiment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["hbb", "obb"], required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--name", type=str)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Checkpoint last.pt a reprendre sans ecraser le run existant.",
    )
    return parser.parse_args()


def build_model(task: str) -> YOLO:
    """Charge un backbone COCO commun, sans poids OBB deja ajustes sur DOTA."""
    base_weights = PROJECT_DIR / "yolo26n.pt"
    if not base_weights.exists():
        raise FileNotFoundError(
            "yolo26n.pt est absent. Le telecharger avant le vol pour garantir "
            "une execution hors ligne."
        )
    model_config = "yolo26n.yaml" if task == "hbb" else "yolo26n-obb.yaml"
    return YOLO(model_config).load(str(base_weights))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA est requis pour cette experience.")

    data_yaml = args.dataset / args.task / f"dota_{args.task}.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    resume_checkpoint = args.resume_from.resolve() if args.resume_from else None
    if resume_checkpoint:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(resume_checkpoint)
        if resume_checkpoint.name != "last.pt":
            raise ValueError("--resume-from doit pointer vers un checkpoint last.pt.")
        run_dir = resume_checkpoint.parents[1]
        run_name = run_dir.name
        model = YOLO(str(resume_checkpoint))
    else:
        run_name = args.name or (
            f"{args.task}_yolo26n_coco_e{args.epochs}_img{args.imgsz}_seed{args.seed}"
        )
        run_dir = RUNS_DIR / run_name
        if run_dir.exists():
            raise FileExistsError(
                f"Le dossier de run existe deja: {run_dir}. "
                "Choisir un autre --name pour proteger les resultats."
            )
        model = build_model(args.task)

    started_at = time.time()
    if resume_checkpoint:
        train_results = model.train(
            resume=str(resume_checkpoint),
            device=0,
            workers=args.workers,
            patience=args.patience,
            plots=True,
        )
    else:
        train_results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            workers=args.workers,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=False,
            seed=args.seed,
            deterministic=True,
            patience=args.patience,
            optimizer="auto",
            amp=True,
            close_mosaic=max(2, min(5, args.epochs // 4)),
            cache=False,
            plots=True,
            save=True,
            verbose=True,
        )
    elapsed_seconds = time.time() - started_at

    run_dir = Path(train_results.save_dir)
    results_csv = run_dir / "results.csv"
    completed_epochs = (
        max(0, len(results_csv.read_text(encoding="utf-8").splitlines()) - 1)
        if results_csv.exists()
        else None
    )
    metadata = {
        "task": args.task,
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "elapsed_seconds": elapsed_seconds,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "transfer_source": str(base_weights.resolve())
        if (base_weights := PROJECT_DIR / "yolo26n.pt").exists()
        else None,
        "pretraining_note": (
            "Backbone YOLO26n preentraine sur COCO; aucun poids OBB "
            "preentraine sur DOTA n'a ete utilise."
        ),
    }
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
