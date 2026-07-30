"""Verify the committed read-only bundle using only Python's standard library."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "projet_dota.ipynb"
NOTEBOOK_HTML = PROJECT_DIR / "output" / "notebook" / "projet_dota.html"
REQUIRED_FILES = (
    NOTEBOOK_PATH,
    NOTEBOOK_HTML,
    PROJECT_DIR / "PORTABLE_READING.md",
    PROJECT_DIR / "output" / "pdf" / "guide_etude_vol_dota.pdf",
    PROJECT_DIR / "output" / "pdf" / "rapport_final_dota.pdf",
    PROJECT_DIR / "output" / "study_pack" / "guide_etude_dota.html",
    PROJECT_DIR / "outputs" / "analysis" / "analysis_summary.json",
    PROJECT_DIR / "outputs" / "analysis" / "model_comparison.csv",
    PROJECT_DIR / "outputs" / "analysis" / "causal_object_table.csv",
)
ABSOLUTE_USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+", re.IGNORECASE)
ACCENTED_VOWELS = set("àâäáãåæéèêëíìîïóòôöõœúùûüýÿÀÂÄÁÃÅÆÉÈÊËÍÌÎÏÓÒÔÖÕŒÚÙÛÜÝ")
REMOTE_RESOURCE_PATTERNS = (
    re.compile(r"<script\b[^>]*\bsrc=[\"']https?://", re.IGNORECASE),
    re.compile(r"<link\b[^>]*\bhref=[\"']https?://", re.IGNORECASE),
    re.compile(r"<img\b[^>]*\bsrc=[\"']https?://", re.IGNORECASE),
    re.compile(r"\bimport\s*\(\s*[\"']https?://", re.IGNORECASE),
    re.compile(r"url\(\s*[\"']?https?://", re.IGNORECASE),
)


def check(condition: bool, name: str, detail: str, results: list[dict[str, object]]) -> None:
    results.append({"check": name, "ok": bool(condition), "detail": detail})


def verify_notebook(results: list[dict[str, object]]) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    missing_counts = [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and cell.get("execution_count") is None
    ]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
    )
    serialized = json.dumps(notebook, ensure_ascii=False)
    headings = [sources.find(f"## {number}.") for number in range(1, 7)]
    kernel_name = notebook.get("metadata", {}).get("kernelspec", {}).get("name")

    check(notebook.get("nbformat") == 4, "notebook_format", str(notebook.get("nbformat")), results)
    check(len(cells) == 95, "notebook_cells", str(len(cells)), results)
    check(len(code_cells) == 48, "notebook_code_cells", str(len(code_cells)), results)
    check(not missing_counts, "notebook_all_executed", str(missing_counts), results)
    check(not errors, "notebook_no_saved_errors", str(len(errors)), results)
    check(
        all(position >= 0 for position in headings) and headings == sorted(headings),
        "notebook_six_sections_in_order",
        str(headings),
        results,
    )
    check(
        not any(character in ACCENTED_VOWELS for character in sources),
        "notebook_no_accented_vowels",
        "source cells",
        results,
    )
    check(
        not ABSOLUTE_USER_PATH.search(serialized),
        "notebook_no_absolute_user_path",
        "serialized notebook",
        results,
    )
    check(kernel_name == "python3", "notebook_portable_kernel", str(kernel_name), results)


def verify_html(results: list[dict[str, object]]) -> None:
    document = NOTEBOOK_HTML.read_text(encoding="utf-8")
    remote_patterns = [
        pattern.pattern
        for pattern in REMOTE_RESOURCE_PATTERNS
        if pattern.search(document)
    ]
    check(not remote_patterns, "html_no_remote_resources", str(remote_patterns), results)
    check(
        not ABSOLUTE_USER_PATH.search(document),
        "html_no_absolute_user_path",
        "portable reader",
        results,
    )
    check("portable-reader" in document, "html_reader_banner", "portable-reader", results)
    check("data:image/" in document, "html_embedded_images", "data URI images", results)


def verify_files(results: list[dict[str, object]]) -> None:
    for path in REQUIRED_FILES:
        present = path.is_file() and path.stat().st_size > 0
        check(present, f"file:{path.relative_to(PROJECT_DIR)}", str(path), results)
    for path in (
        PROJECT_DIR / "output" / "pdf" / "guide_etude_vol_dota.pdf",
        PROJECT_DIR / "output" / "pdf" / "rapport_final_dota.pdf",
    ):
        signature = path.read_bytes()[:5] if path.is_file() else b""
        check(signature == b"%PDF-", f"pdf_signature:{path.name}", repr(signature), results)


def main() -> None:
    results: list[dict[str, object]] = []
    verify_files(results)
    if NOTEBOOK_PATH.is_file():
        verify_notebook(results)
    if NOTEBOOK_HTML.is_file():
        verify_html(results)
    ok = all(result["ok"] for result in results)
    print(json.dumps({"project": str(PROJECT_DIR), "checks": results, "ok": ok}, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
