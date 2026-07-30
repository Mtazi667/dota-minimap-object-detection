"""Controle local, sans acces reseau, de l'environnement de travail DOTA."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics_config"))


REQUIRED_PATHS = [
    PROJECT_DIR / "Subject.pdf",
    PROJECT_DIR / "projet_dota.ipynb",
    PROJECT_DIR / "Training Data" / "Images",
    PROJECT_DIR / "Training Data" / "LabelTxt",
    PROJECT_DIR / "Validation Data" / "Images",
    PROJECT_DIR / "Validation Data" / "LabelTxt",
    PROJECT_DIR / "prepared_data" / "dota_experiment_v1" / "preparation_summary.json",
    PROJECT_DIR / "yolo26n.pt",
    PROJECT_DIR
    / "runs"
    / "dota_experiment_v1"
    / "baseline_hbb_yolo26n_e20"
    / "weights"
    / "best.pt",
    PROJECT_DIR
    / "runs"
    / "dota_experiment_v1"
    / "main_obb_yolo26n_e20_img640"
    / "weights"
    / "best.pt",
    PROJECT_DIR
    / "runs"
    / "dota_experiment_v1"
    / "tuned_obb_yolo26n_e20_img1024"
    / "weights"
    / "best.pt",
    PROJECT_DIR / "outputs" / "analysis" / "analysis_summary.json",
    PROJECT_DIR / "outputs" / "analysis" / "model_comparison.csv",
    PROJECT_DIR / "outputs" / "analysis" / "causal_object_table.csv",
    PROJECT_DIR / "output" / "pdf" / "rapport_final_dota.pdf",
    PROJECT_DIR / "output" / "pdf" / "guide_etude_vol_dota.pdf",
    PROJECT_DIR / "output" / "study_pack" / "guide_etude_dota.html",
    PROJECT_DIR / "output" / "notebook" / "projet_dota.html",
    PROJECT_DIR / "study_pack" / "README.md",
    PROJECT_DIR / "study_pack" / "01_projet_de_bout_en_bout.md",
    PROJECT_DIR / "study_pack" / "02_detection_yolo_obb.md",
    PROJECT_DIR / "study_pack" / "03_causalite_appliquee.md",
    PROJECT_DIR / "study_pack" / "04_resultats_et_interpretation.md",
]

REQUIRED_MODULES = [
    "cv2",
    "fitz",
    "jupyter",
    "matplotlib",
    "mistune",
    "nbclient",
    "nbformat",
    "networkx",
    "numpy",
    "pandas",
    "PIL",
    "pypdf",
    "reportlab",
    "scipy",
    "seaborn",
    "sklearn",
    "statsmodels",
    "torch",
    "torchvision",
    "ultralytics",
]


def main() -> None:
    checks: list[dict] = []
    for path in REQUIRED_PATHS:
        checks.append(
            {
                "check": f"path:{path.relative_to(PROJECT_DIR)}",
                "ok": path.exists(),
                "detail": str(path),
            }
        )

    versions = {}
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "installed")
            checks.append(
                {
                    "check": f"module:{module_name}",
                    "ok": True,
                    "detail": str(versions[module_name]),
                }
            )
        except Exception as error:
            checks.append(
                {
                    "check": f"module:{module_name}",
                    "ok": False,
                    "detail": repr(error),
                }
            )

    cuda_detail = "torch indisponible"
    cuda_ok = False
    try:
        import torch

        if torch.cuda.is_available():
            first = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
            second = first @ first.T
            torch.cuda.synchronize()
            cuda_ok = bool(torch.isfinite(second).all().item())
            cuda_detail = (
                f"{torch.cuda.get_device_name(0)} | torch={torch.__version__} "
                f"| cuda={torch.version.cuda}"
            )
        else:
            cuda_detail = "torch.cuda.is_available() == False"
    except Exception as error:
        cuda_detail = repr(error)
    checks.append({"check": "cuda_calculation", "ok": cuda_ok, "detail": cuda_detail})

    summary_path = (
        PROJECT_DIR
        / "prepared_data"
        / "dota_experiment_v1"
        / "preparation_summary.json"
    )
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = summary["validation"]
        data_ok = (
            validation["original_image_overlap_count"] == 0
            and validation["tasks"]["obb"]["train"]["class_count"] == 15
            and validation["tasks"]["obb"]["val"]["class_count"] == 15
            and validation["tasks"]["hbb"]["train"]["class_count"] == 15
            and validation["tasks"]["hbb"]["val"]["class_count"] == 15
        )
        checks.append(
            {
                "check": "prepared_dataset_validation",
                "ok": data_ok,
                "detail": (
                    f"overlap={validation['original_image_overlap_count']}, "
                    f"manifest_rows={validation['manifest_row_count']}"
                ),
            }
        )

    report = {
        "project": str(PROJECT_DIR),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
