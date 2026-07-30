"""Genere un guide d'etude hors ligne en HTML et PDF."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

import mistune
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
STUDY_DIR = PROJECT_DIR / "study_pack"
DEFAULT_HTML = PROJECT_DIR / "output" / "study_pack" / "guide_etude_dota.html"
DEFAULT_PDF = PROJECT_DIR / "output" / "pdf" / "guide_etude_vol_dota.pdf"
DOCUMENTS = [
    STUDY_DIR / "README.md",
    STUDY_DIR / "01_projet_de_bout_en_bout.md",
    STUDY_DIR / "02_detection_yolo_obb.md",
    STUDY_DIR / "03_causalite_appliquee.md",
    STUDY_DIR / "04_resultats_et_interpretation.md",
    STUDY_DIR / "05_exercices_sans_corrige.md",
    STUDY_DIR / "06_questions_orales.md",
    STUDY_DIR / "07_corrige.md",
    PROJECT_DIR / "concepts_oraux_ml_causalite.md",
]

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#087F8C")
ORANGE = colors.HexColor("#D97706")
LIGHT = colors.HexColor("#F4F7FA")
GRAY = colors.HexColor("#4B5563")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser.parse_args()


def slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def rewrite_local_links(markdown_text: str) -> str:
    mapping = {path.name: f"#doc-{slug(path)}" for path in DOCUMENTS}
    for filename, anchor in mapping.items():
        markdown_text = markdown_text.replace(f"({filename})", f"({anchor})")
    return markdown_text


def build_html(output_path: Path) -> None:
    markdown = mistune.create_markdown(plugins=["table", "strikethrough"])
    sections = []
    navigation = []
    for path in DOCUMENTS:
        if not path.exists():
            raise FileNotFoundError(path)
        anchor = f"doc-{slug(path)}"
        title = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        navigation.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
        source = rewrite_local_links(path.read_text(encoding="utf-8"))
        sections.append(
            f'<section class="document" id="{anchor}">'
            f'<div class="source-label">{html.escape(path.name)}</div>'
            f"{markdown(source)}"
            "</section>"
        )

    css = """
    :root { --navy:#17324D; --teal:#087F8C; --orange:#D97706; --paper:#fff; --ink:#17202A; }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin:0; color:var(--ink); font-family:Arial, Helvetica, sans-serif; background:#EEF2F6; line-height:1.55; }
    header { background:linear-gradient(120deg,var(--navy),#28597D); color:white; padding:2.2rem max(5vw,1rem); }
    header h1 { margin:0 0 .4rem; font-size:clamp(1.8rem,4vw,3rem); }
    header p { margin:0; max-width:62rem; opacity:.92; }
    nav { position:sticky; top:0; z-index:5; display:flex; gap:.65rem; overflow:auto; padding:.65rem max(3vw,.75rem); background:#fff; box-shadow:0 2px 10px #0002; }
    nav a { white-space:nowrap; text-decoration:none; color:var(--navy); border:1px solid #CBD5E1; border-radius:999px; padding:.35rem .7rem; font-size:.82rem; }
    nav a:hover { color:white; background:var(--teal); border-color:var(--teal); }
    main { max-width:980px; margin:1.5rem auto 5rem; padding:0 1rem; }
    .document { background:var(--paper); padding:clamp(1.1rem,4vw,3rem); margin:0 0 1.5rem; border-radius:14px; box-shadow:0 8px 30px #17324D14; }
    .source-label { float:right; color:#64748B; font-size:.72rem; background:#F1F5F9; padding:.25rem .5rem; border-radius:6px; }
    h1,h2,h3,h4 { color:var(--navy); line-height:1.2; scroll-margin-top:4.5rem; }
    h2 { border-bottom:2px solid #DDE8F0; padding-bottom:.3rem; margin-top:2rem; }
    h3 { color:var(--teal); margin-top:1.5rem; }
    code { background:#EFF6FF; color:#0F3D5E; padding:.08rem .28rem; border-radius:4px; }
    pre { overflow:auto; background:#102A43; color:#E6EDF3; padding:1rem; border-radius:8px; line-height:1.4; }
    blockquote { margin:1rem 0; padding:.8rem 1rem; border-left:5px solid var(--orange); background:#FFF7ED; }
    table { width:100%; border-collapse:collapse; display:block; overflow:auto; margin:1rem 0; font-size:.9rem; }
    th { background:var(--navy); color:#fff; }
    th,td { border:1px solid #CBD5E1; padding:.45rem .55rem; text-align:left; }
    tr:nth-child(even) td { background:#F8FAFC; }
    a { color:#0369A1; }
    .toolbar { position:fixed; right:1rem; bottom:1rem; z-index:10; }
    .toolbar button { border:0; border-radius:999px; background:var(--orange); color:white; padding:.7rem 1rem; box-shadow:0 4px 15px #0003; cursor:pointer; }
    @media print {
      body { background:white; }
      header, nav, .toolbar { display:none; }
      main { max-width:none; margin:0; padding:0; }
      .document { box-shadow:none; border-radius:0; page-break-before:always; margin:0; }
      a { color:inherit; text-decoration:none; }
      pre, table, blockquote { break-inside:avoid; }
    }
    """
    script = """
    function toggleCorrections(){
      const target=document.getElementById('doc-07-corrige');
      if(!target) return;
      const hidden=target.style.display==='none';
      target.style.display=hidden?'block':'none';
      if(hidden) target.scrollIntoView({behavior:'smooth'});
    }
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Guide d'etude DOTA hors ligne</title>"
        f"<style>{css}</style></head><body>"
        "<header><h1>Guide d'etude DOTA hors ligne</h1>"
        "<p>Parcours de 8 heures, detection, YOLO-OBB, causalite, resultats, "
        "questions orales et exercices corriges.</p></header>"
        f"<nav>{''.join(navigation)}</nav><main>{''.join(sections)}</main>"
        "<div class=\"toolbar\"><button onclick=\"toggleCorrections()\">"
        "Afficher/masquer le corrige</button></div>"
        f"<script>{script}</script></body></html>",
        encoding="utf-8",
    )
    shutil.copy2(STUDY_DIR / "flashcards.csv", output_path.parent / "flashcards.csv")


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("StudyArial", str(regular)))
        pdfmetrics.registerFont(TTFont("StudyArialBold", str(bold)))
        return "StudyArial", "StudyArialBold"
    return "Helvetica", "Helvetica-Bold"


def inline_markup(text: str) -> str:
    protected = escape(text)
    protected = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', protected)
    protected = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", protected)
    protected = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", protected)
    protected = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", protected)
    return protected


def table_from_markdown(
    lines: list[str],
    body_style: ParagraphStyle,
    font_regular: str,
    font_bold: str,
) -> Table:
    parsed = []
    for line_index, line in enumerate(lines):
        if line_index == 1 and re.fullmatch(r"\|?[\s:|-]+\|?", line.strip()):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        parsed.append(
            [Paragraph(inline_markup(cell), body_style) for cell in cells]
        )
    columns = max(len(row) for row in parsed)
    width = 17.0 * cm / columns
    table = Table(parsed, colWidths=[width] * columns, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_regular),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def markdown_to_flowables(
    source: str,
    styles: dict,
    font_regular: str,
    font_bold: str,
) -> list:
    lines = source.splitlines()
    flowables = []
    index = 0
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            flowables.append(Paragraph(inline_markup(text), styles["body"]))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            label = f"[{language}]\n" if language else ""
            flowables.append(
                Preformatted(
                    label + "\n".join(code_lines),
                    styles["code"],
                )
            )
        elif stripped.startswith("|") and index + 1 < len(lines) and re.fullmatch(
            r"\|?[\s:|-]+\|?",
            lines[index + 1].strip(),
        ):
            flush_paragraph()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            flowables.append(
                table_from_markdown(
                    table_lines,
                    styles["table"],
                    font_regular,
                    font_bold,
                )
            )
            flowables.append(Spacer(1, 0.15 * cm))
            continue
        elif stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[level:].strip()
            style_name = "h1" if level == 1 else "h2" if level == 2 else "h3"
            flowables.append(Paragraph(inline_markup(heading), styles[style_name]))
        elif stripped.startswith(">"):
            flush_paragraph()
            quote_lines = [stripped.lstrip("> ").strip()]
            index += 1
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip("> ").strip())
                index += 1
            flowables.append(
                Paragraph(
                    inline_markup(" ".join(quote_lines)),
                    styles["quote"],
                )
            )
            continue
        elif re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            item = re.sub(r"^[-*]\s+", "", stripped)
            flowables.append(
                Paragraph(
                    inline_markup(item),
                    styles["bullet"],
                    bulletText="•",
                )
            )
        elif re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            match = re.match(r"^(\d+)\.\s+(.*)", stripped)
            flowables.append(
                Paragraph(
                    inline_markup(match.group(2)),
                    styles["bullet"],
                    bulletText=f"{match.group(1)}.",
                )
            )
        elif stripped == "---":
            flush_paragraph()
            flowables.append(Spacer(1, 0.25 * cm))
        elif not stripped:
            flush_paragraph()
        else:
            paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    return flowables


def page_decorator(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.5 * cm, height - 1.15 * cm, width - 1.5 * cm, height - 1.15 * cm)
    font = "StudyArial" if "StudyArial" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    canvas.setFont(font, 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.5 * cm, height - 0.9 * cm, "Guide d'etude DOTA - hors ligne")
    canvas.drawRightString(width - 1.5 * cm, 0.8 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf(output_path: Path) -> None:
    font_regular, font_bold = register_fonts()
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "StudyTitle",
            parent=base["Title"],
            fontName=font_bold,
            fontSize=25,
            leading=31,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "StudyH1",
            parent=base["Heading1"],
            fontName=font_bold,
            fontSize=18,
            leading=23,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "StudyH2",
            parent=base["Heading2"],
            fontName=font_bold,
            fontSize=13.5,
            leading=18,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "StudyH3",
            parent=base["Heading3"],
            fontName=font_bold,
            fontSize=11,
            leading=15,
            textColor=ORANGE,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "StudyBody",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=9.2,
            leading=13.5,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "StudyBullet",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=9,
            leading=13,
            leftIndent=15,
            firstLineIndent=0,
            bulletIndent=3,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "StudyQuote",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=9.2,
            leading=13.5,
            leftIndent=12,
            rightIndent=8,
            borderColor=ORANGE,
            borderWidth=1,
            borderPadding=7,
            backColor=colors.HexColor("#FFF7ED"),
            spaceBefore=5,
            spaceAfter=7,
        ),
        "code": ParagraphStyle(
            "StudyCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=9.2,
            leftIndent=7,
            rightIndent=7,
            backColor=colors.HexColor("#EEF2F6"),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "StudyTable",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=6.7,
            leading=8.5,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.4 * cm,
        title="Guide d'etude DOTA hors ligne",
        author="Projet individuel",
    )
    story = [
        Spacer(1, 3.0 * cm),
        Paragraph("Guide d'etude DOTA hors ligne", styles["title"]),
        Paragraph(
            "Parcours de 8 heures - detection, YOLO-OBB, causalite, resultats, exercices et questions orales",
            ParagraphStyle(
                "StudySubtitle",
                parent=styles["body"],
                fontSize=13,
                leading=19,
                alignment=TA_CENTER,
                textColor=GRAY,
            ),
        ),
        Spacer(1, 1.0 * cm),
        Paragraph(
            "Conseil : repondre de memoire avant de lire. Les exercices et leur corrige sont separes.",
            styles["quote"],
        ),
        PageBreak(),
    ]
    for document_index, path in enumerate(DOCUMENTS):
        if not path.exists():
            raise FileNotFoundError(path)
        if document_index:
            story.append(PageBreak())
        story.extend(
            markdown_to_flowables(
                path.read_text(encoding="utf-8"),
                styles,
                font_regular,
                font_bold,
            )
        )
    document.build(story, onFirstPage=page_decorator, onLaterPages=page_decorator)
    reader = PdfReader(str(output_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(reader.pages) < 20:
        raise RuntimeError("Le PDF d'etude est anormalement court.")
    if "Programme principal de 8 heures" not in text or "Corrige" not in text:
        raise RuntimeError("Le controle textuel du guide d'etude a echoue.")


def main() -> None:
    args = parse_args()
    build_html(args.html)
    build_pdf(args.pdf)
    report = {
        "html": str(args.html.resolve()),
        "html_bytes": args.html.stat().st_size,
        "pdf": str(args.pdf.resolve()),
        "pdf_bytes": args.pdf.stat().st_size,
        "pdf_pages": len(PdfReader(str(args.pdf)).pages),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
