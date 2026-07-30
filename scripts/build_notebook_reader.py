"""Build a dependency-free HTML reader from the executed notebook."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_DIR / "projet_dota.ipynb"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "notebook" / "projet_dota.html"

ABSOLUTE_USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+", re.IGNORECASE)
REMOTE_RESOURCE_PATTERNS = (
    re.compile(r"<script\b[^>]*\bsrc=[\"']https?://", re.IGNORECASE),
    re.compile(r"<link\b[^>]*\bhref=[\"']https?://", re.IGNORECASE),
    re.compile(r"<img\b[^>]*\bsrc=[\"']https?://", re.IGNORECASE),
    re.compile(r"\bimport\s*\(\s*[\"']https?://", re.IGNORECASE),
    re.compile(r"url\(\s*[\"']?https?://", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_executed_notebook(notebook: nbformat.NotebookNode) -> dict[str, int]:
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    missing_counts = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and cell.execution_count is None
    ]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    serialized = nbformat.writes(notebook)

    if missing_counts:
        raise RuntimeError(f"Cellules non executees: {missing_counts}")
    if errors:
        raise RuntimeError(f"Sorties d'erreur sauvegardees: {len(errors)}")
    if ABSOLUTE_USER_PATH.search(serialized):
        raise RuntimeError("Le notebook contient encore un chemin utilisateur absolu.")

    return {
        "cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "stored_outputs": sum(len(cell.get("outputs", [])) for cell in code_cells),
    }


def remove_unused_online_bootstrap(document: str) -> str:
    document = re.sub(
        r"\s*<script\s+src=[\"']https://cdnjs\.cloudflare\.com/ajax/libs/"
        r"require\.js/[^\"']+[\"']></script>",
        "",
        document,
        flags=re.IGNORECASE,
    )
    document = re.sub(
        r"\s*<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->",
        "",
        document,
        flags=re.DOTALL | re.IGNORECASE,
    )
    document = re.sub(
        r"\s*<script type=[\"']module[\"']>\s*document\.addEventListener"
        r".*?<!-- End of mermaid configuration -->",
        "",
        document,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return document


def add_reader_banner(document: str) -> str:
    style = """
<style id="portable-reader-style">
.portable-reader {
  box-sizing: border-box;
  margin: 0 auto 1.25rem;
  max-width: 1120px;
  padding: 0.9rem 1.1rem;
  border: 1px solid #9cc7d1;
  border-radius: 10px;
  background: #eef8fa;
  color: #17324d;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.portable-reader strong { color: #075f6b; }
.portable-reader a { color: #075f6b; font-weight: 600; }
</style>
"""
    banner = """
<aside class="portable-reader" aria-label="Mode lecture hors ligne">
  <strong>Version pre-executee pour lecture hors ligne.</strong>
  Tous les tableaux, graphiques et resultats sont deja inclus. Aucune donnee
  DOTA, aucun poids YOLO, aucun noyau Jupyter et aucune connexion Internet ne
  sont necessaires pour lire cette page.
  <a href="../pdf/guide_etude_vol_dota.pdf">Guide d'etude</a> ·
  <a href="../pdf/rapport_final_dota.pdf">Rapport final</a> ·
  <a href="../../PORTABLE_READING.md">Instructions</a>
</aside>
"""
    document = document.replace("</head>", f"{style}</head>", 1)
    document = re.sub(r"(<body\b[^>]*>)", rf"\1{banner}", document, count=1)
    document = re.sub(
        r"<title>.*?</title>",
        "<title>Projet DOTA - notebook pre-execute</title>",
        document,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return document


def assert_offline_resources(document: str) -> None:
    failures = [
        pattern.pattern
        for pattern in REMOTE_RESOURCE_PATTERNS
        if pattern.search(document)
    ]
    if failures:
        raise RuntimeError(
            "Le lecteur HTML contient encore des ressources distantes actives: "
            + ", ".join(failures)
        )
    if ABSOLUTE_USER_PATH.search(document):
        raise RuntimeError("Le lecteur HTML contient un chemin utilisateur absolu.")


def build_reader(notebook_path: Path, output_path: Path) -> dict[str, object]:
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    summary = validate_executed_notebook(notebook)

    exporter = HTMLExporter(template_name="lab")
    document, _ = exporter.from_notebook_node(notebook)
    document = remove_unused_online_bootstrap(document)
    document = add_reader_banner(document)
    assert_offline_resources(document)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8", newline="\n")
    return {
        "notebook": str(notebook_path.resolve()),
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        **summary,
    }


def main() -> None:
    args = parse_args()
    summary = build_reader(args.notebook, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
