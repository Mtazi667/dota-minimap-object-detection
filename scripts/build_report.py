"""Construit le rapport PDF final a partir des sorties verifiees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "pdf" / "rapport_final_dota.pdf"

NAVY = colors.HexColor("#15324B")
TEAL = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#D97706")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_TEAL = colors.HexColor("#E7F5F2")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
DARK_GRAY = colors.HexColor("#374151")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ReportArial", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportArialBold", str(bold)))
        return "ReportArial", "ReportArialBold"
    return "Helvetica", "Helvetica-Bold"


def build_styles(font_regular: str, font_bold: str) -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=font_bold,
            fontSize=25,
            leading=31,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName=font_regular,
            fontSize=13,
            leading=19,
            textColor=DARK_GRAY,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontName=font_bold,
            fontSize=18,
            leading=23,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontName=font_bold,
            fontSize=13,
            leading=17,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1F2937"),
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=7.6,
            leading=10.5,
            textColor=DARK_GRAY,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=7.6,
            leading=10,
            textColor=DARK_GRAY,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=9,
        ),
        "callout": ParagraphStyle(
            "ReportCallout",
            parent=base["BodyText"],
            fontName=font_bold,
            fontSize=10,
            leading=15,
            textColor=NAVY,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=7,
            spaceAfter=7,
            borderColor=TEAL,
            borderWidth=1,
            borderPadding=8,
            backColor=LIGHT_TEAL,
        ),
        "toc": ParagraphStyle(
            "ReportToc",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=10,
            leading=17,
            textColor=NAVY,
            leftIndent=12,
            spaceAfter=3,
        ),
    }


def page_decorator(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.setLineWidth(0.5)
    canvas.line(1.6 * cm, height - 1.25 * cm, width - 1.6 * cm, height - 1.25 * cm)
    canvas.setFillColor(DARK_GRAY)
    canvas.setFont("ReportArial" if "ReportArial" in pdfmetrics.getRegisteredFontNames() else "Helvetica", 7.5)
    canvas.drawString(1.6 * cm, height - 0.95 * cm, "Projet DOTA-v1.0 - detection et inference causale")
    canvas.drawRightString(width - 1.6 * cm, 0.85 * cm, f"Page {document.page}")
    canvas.restoreState()


def first_page_decorator(canvas, document) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 1.2 * cm, A4[0], 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, A4[0], 0.55 * cm, fill=1, stroke=0)
    canvas.restoreState()


def report_image(
    path: Path,
    max_width: float = 17.0 * cm,
    max_height: float = 11.2 * cm,
) -> Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def paragraph(text: str, styles: dict) -> Paragraph:
    return Paragraph(text, styles["body"])


def bullet(text: str, styles: dict) -> Paragraph:
    return Paragraph(f"• {text}", styles["body"])


def make_table(
    rows: list[list],
    column_widths: list[float],
    font_regular: str,
    font_bold: str,
    repeat_rows: int = 1,
) -> Table:
    table = Table(rows, colWidths=column_widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_regular),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def pct(value: float) -> str:
    return f"{100 * value:.1f} %"


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(
        (args.analysis / "analysis_summary.json").read_text(encoding="utf-8")
    )
    preparation = json.loads(
        (args.analysis / "preparation_summary_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    models = pd.read_csv(args.analysis / "model_comparison.csv")
    classes = pd.read_csv(args.analysis / "per_class_metrics.csv")
    effects = pd.read_csv(args.analysis / "causal_effect_estimates.csv")
    sensitivity = pd.read_csv(args.analysis / "causal_sensitivity.csv")
    subgroups = pd.read_csv(args.analysis / "causal_tree_subgroups.csv")
    primary_name = summary["primary_model"]["model"]
    primary_model = models.loc[models["model"].eq(primary_name)].iloc[0]
    best_model = models.sort_values("map50_95", ascending=False).iloc[0]
    primary_effect = effects.loc[
        effects["method"].eq("AIPW doublement robuste")
    ].iloc[0]
    primary_classes = classes.loc[classes["model"].eq(primary_name)].copy()
    top_classes = primary_classes.nlargest(5, "ap50_95")
    weak_classes = primary_classes.nsmallest(5, "ap50_95")

    font_regular, font_bold = register_fonts()
    styles = build_styles(font_regular, font_bold)
    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.5 * cm,
        title="Detection d'objets DOTA et inference causale",
        author="Projet individuel",
        subject="DOTA-v1.0, YOLO HBB/OBB et effet de la petite taille",
    )
    story = []

    story.extend(
        [
            Spacer(1, 2.2 * cm),
            Paragraph(
                "Détection d'objets dans des images satellites et aériennes",
                styles["title"],
            ),
            Paragraph(
                "DOTA-v1.0 - YOLO HBB, YOLO-OBB et inférence causale sur les erreurs de détection",
                styles["subtitle"],
            ),
            Spacer(1, 0.5 * cm),
            Table(
                [
                    ["Projet", "Projet individuel 11"],
                    ["Dataset", "DOTA-v1.0"],
                    ["Environnement", "Python 3.14, PyTorch CUDA 13.0, RTX 4060"],
                    ["Protocole", "180 images train, 60 images validation, tuiles 1024"],
                    ["Date", "29 juillet 2026"],
                ],
                colWidths=[4.0 * cm, 10.5 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), NAVY),
                        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                        ("FONTNAME", (0, 0), (0, -1), font_bold),
                        ("FONTNAME", (1, 0), (1, -1), font_regular),
                        ("BACKGROUND", (1, 0), (1, -1), LIGHT_BLUE),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.white),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            Spacer(1, 1.2 * cm),
            Paragraph(
                "Ce rapport sépare explicitement la performance prédictive du détecteur et l'estimation causale d'un effet sur ses erreurs.",
                styles["callout"],
            ),
            PageBreak(),
        ]
    )

    story.append(Paragraph("Table des matières", styles["h1"]))
    toc_items = [
        "Résumé exécutif",
        "1. Exploration du jeu de données et choix des algorithmes",
        "2. Prétraitement, tuilage et variables dérivées",
        "3. Modèles prédictifs et apprentissage profond",
        "4. Formulation de la question causale",
        "5. Estimation causale et effets hétérogènes",
        "6. Interprétation, limites et conclusion",
        "Reproductibilité et références",
    ]
    for index, item in enumerate(toc_items):
        story.append(Paragraph(f"{index + 1}. {item}", styles["toc"]))
    story.append(PageBreak())

    story.append(Paragraph("Résumé exécutif", styles["h1"]))
    story.append(
        paragraph(
            "Ce projet étudie la détection de 15 catégories d'objets dans des "
            "images aériennes DOTA-v1.0. Les images sources sont séparées avant "
            "le tuilage afin d'éviter toute fuite. Un sous-ensemble stratifié "
            "de 180 images d'entraînement et 60 images de validation est "
            "converti simultanément en boîtes horizontales et orientées.",
            styles,
        )
    )
    story.append(
        paragraph(
            f"Trois configurations comparables sont évaluées. Le meilleur "
            f"score mAP50-95 est obtenu par <b>{best_model['model']}</b> "
            f"({best_model['map50_95']:.3f}), avec une mAP50 de "
            f"{best_model['map50']:.3f}. Les résultats modestes sont cohérents "
            "avec la brièveté de l'entraînement, la petite taille du "
            "sous-ensemble et la difficulté des objets minuscules.",
            styles,
        )
    )
    story.append(
        paragraph(
            f"L'analyse causale définit le traitement comme l'appartenance au "
            f"premier quartile de taille calculé sur train, et l'outcome comme "
            f"une détection de même classe, confiance au moins 0,25 et IoU "
            f"orientée au moins 0,50. L'effet AIPW vaut "
            f"<b>{primary_effect['effect']:.3f}</b>, IC bootstrap groupe 95 % "
            f"[{primary_effect['ci_lower']:.3f}, "
            f"{primary_effect['ci_upper']:.3f}]. Cette valeur est interprétée "
            "comme une différence absolue de probabilité sous des hypothèses "
            "d'identification explicites, et non comme le résultat d'une "
            "expérience randomisée.",
            styles,
        )
    )
    story.append(report_image(args.analysis / "model_comparison.png", max_height=9.0 * cm))
    story.append(
        Paragraph(
            "Figure 1 - Comparaison globale des détecteurs sur la validation.",
            styles["caption"],
        )
    )
    story.append(PageBreak())

    story.append(
        Paragraph(
            "1. Exploration du jeu de données et choix des algorithmes",
            styles["h1"],
        )
    )
    story.append(Paragraph("1.1 Données et difficultés", styles["h2"]))
    for text in [
        "DOTA-v1.0 contient de grandes images satellites et aériennes. Chaque objet est décrit par quatre coins, une classe et un indicateur de difficulté.",
        "L'exploration complète du notebook recense 98 990 objets train et 28 853 objets validation. Les orientations médianes proches de 47° en train et 52° en validation justifient une représentation orientée.",
        "Les objets varient fortement en aire et les classes sont déséquilibrées. Les petits véhicules sont nombreux, tandis que les hélicoptères et certaines infrastructures sont rares.",
    ]:
        story.append(paragraph(text, styles))
    story.append(Paragraph("1.2 Algorithmes considérés", styles["h2"]))
    algorithm_rows = [
        ["Algorithme", "Pertinence", "Décision"],
        ["YOLO-OBB", "Détection rapide avec boîtes orientées", "Modèle principal"],
        ["YOLO HBB", "Détecteur simple sur boîtes converties", "Baseline"],
        ["Faster R-CNN / RetinaNet", "Alternatives HBB plus lourdes", "Non retenues pour ce protocole"],
        ["DETR", "Approche moderne mais coûteuse en données/calcul", "Piste future"],
        ["Classifieur global", "Ignore localisation et multiplicité", "Inadapté à l'état brut"],
    ]
    story.append(
        make_table(
            algorithm_rows,
            [4.0 * cm, 7.3 * cm, 5.1 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        paragraph(
            "Les risques identifiés avant l'entraînement sont la fuite entre "
            "tuiles d'une même source, la perte géométrique des HBB, le "
            "déséquilibre des classes et la confusion entre précision "
            "prédictive et causalité.",
            styles,
        )
    )

    story.append(Paragraph("2. Prétraitement, tuilage et variables dérivées", styles["h1"]))
    story.append(Paragraph("2.1 Sélection stratifiée", styles["h2"]))
    story.append(
        paragraph(
            "Les images sont sélectionnées avant le tuilage. La stratégie "
            "garantit plusieurs images par classe quand les données le "
            "permettent, puis équilibre des strates combinant classe dominante "
            "et densité d'objets. Elle améliore la couverture sans prétendre "
            "supprimer le déséquilibre naturel.",
            styles,
        )
    )
    validation = preparation["validation"]
    preparation_rows = [["Tâche", "Split", "Tuiles", "Négatives", "Objets", "Classes"]]
    for task in ("hbb", "obb"):
        for split in ("train", "val"):
            values = validation["tasks"][task][split]
            preparation_rows.append(
                [
                    task.upper(),
                    split,
                    str(values["image_count"]),
                    str(values["negative_image_count"]),
                    str(values["object_count"]),
                    str(values["class_count"]),
                ]
            )
    story.append(
        make_table(
            preparation_rows,
            [2.5 * cm, 2.5 * cm, 2.5 * cm, 2.7 * cm, 3.0 * cm, 2.5 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            f"Contrôle critique : <b>{validation['original_image_overlap_count']} image source partagée</b> entre train et validation.",
            styles["callout"],
        )
    )
    story.append(Paragraph("2.2 Tuilage et fragments", styles["h2"]))
    for text in [
        "Les tuiles mesurent 1024 x 1024 pixels. Le stride train de 824 crée un chevauchement de 200 pixels pour mieux conserver les objets proches des bords. Le stride validation de 1024 limite la redondance.",
        "Un objet est gardé si au moins 70 % de sa surface reste dans la tuile. Une tuile contenant un fragment entre 20 % et 70 % est rejetée comme ambiguë. Les fragments inférieurs à 20 % sont ignorés.",
        "Un échantillon de tuiles négatives est conservé pour apprendre le fond. Les HBB et OBB partagent exactement les mêmes fichiers image.",
    ]:
        story.append(paragraph(text, styles))
    story.append(Paragraph("2.3 Variables", styles["h2"]))
    variable_rows = [
        ["Niveau", "Variables principales"],
        ["Image", "source, GSD, largeur, hauteur, classe dominante, densité"],
        ["Objet", "classe, difficult, aire, orientation, ratio de forme"],
        ["Tuile", "position x/y, taille, nombre d'objets, tuile négative"],
        ["Fragment", "fraction conservée, aire clippée, marge au bord"],
        ["Prédiction", "classe, confiance, IoU associée, détection correcte"],
    ]
    story.append(
        make_table(
            variable_rows,
            [3.2 * cm, 13.2 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("3. Modèles prédictifs et apprentissage profond", styles["h1"]))
    story.append(Paragraph("3.1 Protocole d'entraînement", styles["h2"]))
    for text in [
        "Les modèles utilisent YOLO26n, un backbone initialisé par transfert depuis COCO, 20 epochs, la graine 42, l'AMP et l'early stopping. Les workers sont fixés à zéro pour éviter les blocages observés sous Windows.",
        "La baseline HBB et le modèle OBB à 640 utilisent les mêmes tuiles, la même graine et le même point de départ général. Une variante OBB à 1024 teste l'hypothèse qu'une résolution d'entrée plus élevée aide les petits objets.",
        "Les poids best.pt sont sélectionnés selon la validation plutôt que de supposer que la dernière époque est la meilleure.",
    ]:
        story.append(paragraph(text, styles))
    model_rows = [["Modèle", "Tâche", "imgsz", "Précision", "Recall", "F1", "mAP50", "mAP50-95"]]
    for row in models.itertuples():
        model_rows.append(
            [
                row.model,
                row.task.upper(),
                str(row.imgsz),
                f"{row.precision:.3f}",
                f"{row.recall:.3f}",
                f"{row.f1:.3f}",
                f"{row.map50:.3f}",
                f"{row.map50_95:.3f}",
            ]
        )
    story.append(
        make_table(
            model_rows,
            [3.7 * cm, 1.3 * cm, 1.3 * cm, 1.8 * cm, 1.7 * cm, 1.5 * cm, 1.8 * cm, 2.1 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        paragraph(
            f"Le modèle retenu pour construire l'outcome causal est "
            f"<b>{primary_name}</b> : précision {primary_model['precision']:.3f}, "
            f"recall {primary_model['recall']:.3f}, F1 {primary_model['f1']:.3f}, mAP50 "
            f"{primary_model['map50']:.3f} et mAP50-95 "
            f"{primary_model['map50_95']:.3f}.",
            styles,
        )
    )
    story.append(report_image(args.analysis / "per_class_metrics.png", max_height=12.5 * cm))
    story.append(
        Paragraph(
            "Figure 2 - AP50-95 par classe et par modèle.",
            styles["caption"],
        )
    )
    story.append(Paragraph("3.2 Résultats par classe", styles["h2"]))
    class_rows = [["Groupe", "Classe", "AP50-95"]]
    for label, frame in (("Meilleure", top_classes), ("Faible", weak_classes)):
        for row in frame.itertuples():
            class_rows.append([label, row.class_name, f"{row.ap50_95:.3f}"])
    story.append(
        make_table(
            class_rows,
            [3.2 * cm, 8.0 * cm, 3.0 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(
        paragraph(
            "Ces AP doivent être lues avec les effectifs : une classe rare peut "
            "avoir une estimation très variable. Inversement, une classe "
            "fréquente peut rester difficile si ses objets sont minuscules ou "
            "très denses.",
            styles,
        )
    )
    prediction_path = (
        args.analysis / f"prediction_examples_{primary_name.lower()}.jpg"
    )
    story.append(report_image(prediction_path, max_height=12.5 * cm))
    story.append(
        Paragraph(
            f"Figure 3 - Exemples de prédictions de {primary_name} sur validation.",
            styles["caption"],
        )
    )
    story.append(report_image(args.analysis / "detection_by_size_orientation.png", max_height=10.0 * cm))
    story.append(
        Paragraph(
            "Figure 4 - Taux de détection correcte selon taille et orientation.",
            styles["caption"],
        )
    )
    story.append(
        paragraph(
            "L'analyse d'erreurs confirme que les scores globaux cachent des "
            "écarts de taille, orientation et classe. Quelques pixels de "
            "décalage réduisent beaucoup plus l'IoU d'un petit objet que celle "
            "d'un grand objet.",
            styles,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("4. Formulation de la question causale", styles["h1"]))
    story.append(Paragraph("4.1 Traitement, outcome et estimand", styles["h2"]))
    causal_rows = [
        ["Élément", "Définition"],
        ["Traitement D", "1 si aire relative de tuile <= premier quartile train, 0 sinon"],
        ["Outcome Y", "1 si même classe, confiance >= 0,25 et IoU OBB >= 0,50"],
        ["Population", f"{summary['n_causal_objects']} objets uniques de {summary['n_causal_images']} images validation"],
        ["Estimand", "ATE sur la probabilité de détection correcte"],
    ]
    story.append(
        make_table(
            causal_rows,
            [3.5 * cm, 12.9 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(summary["causal_question"], styles["callout"])
    )
    story.append(report_image(args.analysis / "causal_dag.png", max_height=9.0 * cm))
    story.append(
        Paragraph(
            "Figure 5 - DAG simplifié guidant l'ajustement.",
            styles["caption"],
        )
    )
    story.append(Paragraph("4.2 Ajustement et hypothèses", styles["h2"]))
    for text in [
        "L'ajustement inclut classe, orientation, log du ratio de forme, densité de tuile, source, GSD et position du centre dans la tuile. L'aire exacte est exclue parce qu'elle définit le traitement. La fraction conservée, la confiance et l'IoU sont exclues car elles relèvent de la sélection, du post-traitement ou de l'outcome.",
        "L'échangeabilité conditionnelle suppose l'absence de confondeurs non observés après ajustement. Cette hypothèse est menacée par le flou, le contraste, l'occultation et la qualité d'annotation.",
        "La positivité exige des petits et non petits comparables dans les profils analysés. Les classes rares et typiquement minuscules peuvent avoir un support limité.",
        "La cohérence est fragile parce que plusieurs interventions pourraient rendre un objet petit. L'absence d'interférence est aussi imparfaite dans les scènes denses à cause de la NMS et de max_det.",
    ]:
        story.append(paragraph(text, styles))

    story.append(Paragraph("5. Estimation causale et effets hétérogènes", styles["h1"]))
    story.append(Paragraph("5.1 Méthodes comparées", styles["h2"]))
    method_rows = [["Méthode", "Effet", "IC bas", "IC haut"]]
    for row in effects.itertuples():
        method_rows.append(
            [
                row.method,
                f"{row.effect:.3f}",
                f"{row.ci_lower:.3f}",
                f"{row.ci_upper:.3f}",
            ]
        )
    story.append(
        make_table(
            method_rows,
            [7.3 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    for text in [
        "La différence brute compare les taux sans ajustement. La g-computation prédit les deux outcomes. L'IPW pondère par la probabilité du traitement observé. L'AIPW combine les deux modèles et corrige avec les résidus.",
        "Les modèles nuisances sont prédits hors fold avec GroupKFold par image. Les intervalles bootstrap rééchantillonnent les images entières afin de respecter la dépendance entre objets d'une même scène.",
    ]:
        story.append(paragraph(text, styles))
    story.append(report_image(args.analysis / "causal_effects.png", max_height=8.5 * cm))
    story.append(
        Paragraph(
            "Figure 6 - Effets moyens et intervalles bootstrap groupe.",
            styles["caption"],
        )
    )
    story.append(
        Paragraph(
            f"Résultat principal : AIPW = <b>{primary_effect['effect']:.3f}</b>, "
            f"IC 95 % [{primary_effect['ci_lower']:.3f}, "
            f"{primary_effect['ci_upper']:.3f}]. Le taux observé est "
            f"{pct(primary_effect['outcome_treated'])} chez les traités et "
            f"{pct(primary_effect['outcome_control'])} chez les contrôles.",
            styles["callout"],
        )
    )
    story.append(report_image(args.analysis / "propensity_overlap.png", max_height=8.5 * cm))
    story.append(
        Paragraph(
            "Figure 7 - Recouvrement des scores de propension.",
            styles["caption"],
        )
    )
    story.append(
        paragraph(
            f"La fraction des objets dont le score est entre 0,1 et 0,9 vaut "
            f"{pct(primary_effect['propensity_overlap_0_1_0_9'])}. Un bon "
            "recouvrement réduit l'extrapolation, sans démontrer l'absence de "
            "confondeurs non observés.",
            styles,
        )
    )
    story.append(Paragraph("5.2 Sensibilité", styles["h2"]))
    sensitivity_rows = [["Spécification", "Effet", "IC bas", "IC haut"]]
    for row in sensitivity.itertuples():
        sensitivity_rows.append(
            [
                row.specification,
                f"{row.effect:.3f}",
                f"{row.ci_lower:.3f}",
                f"{row.ci_upper:.3f}",
            ]
        )
    story.append(
        make_table(
            sensitivity_rows,
            [7.6 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(
        paragraph(
            "La variation des seuils IoU et du clipping montre quelles parties "
            "de la conclusion sont stables et lesquelles dépendent du choix de "
            "spécification. Elle ne remplace pas une analyse de confondeur non "
            "observé.",
            styles,
        )
    )
    story.append(Paragraph("5.3 Hétérogénéité", styles["h2"]))
    story.append(report_image(args.analysis / "causal_tree.png", max_height=11.8 * cm))
    story.append(
        Paragraph(
            "Figure 8 - Arbre causal honnête sur pseudo-outcomes AIPW.",
            styles["caption"],
        )
    )
    subgroup_rows = [["Feuille", "Objets", "Images", "Effet", "IC 95 %", "Classe dominante"]]
    for row in subgroups.sort_values("effect").itertuples():
        subgroup_rows.append(
            [
                str(row.leaf_id),
                str(row.n_objects),
                str(row.n_images),
                f"{row.effect:.3f}",
                f"[{row.ci_lower:.3f}; {row.ci_upper:.3f}]",
                row.dominant_class,
            ]
        )
    story.append(
        make_table(
            subgroup_rows,
            [1.7 * cm, 2.1 * cm, 2.0 * cm, 2.2 * cm, 4.0 * cm, 4.4 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(report_image(args.analysis / "cate_distribution.png", max_height=8.5 * cm))
    story.append(
        Paragraph(
            "Figure 9 - Distribution des effets conditionnels estimés par forêt sur pseudo-outcomes.",
            styles["caption"],
        )
    )
    story.append(
        paragraph(
            "La structure de l'arbre est choisie sur une moitié des images et "
            "les effets des feuilles sur l'autre. La forêt utilise des "
            "prédictions croisées par image. L'arbre est volontairement limité "
            "à une division et à de grandes feuilles pour réduire les sous-groupes "
            "artificiels. Ces résultats sont exploratoires : "
            "la méthode est une approximation pédagogique d'une forêt causale "
            "spécialisée.",
            styles,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("6. Interprétation, limites et conclusion", styles["h1"]))
    story.append(Paragraph("6.1 Interprétation prédictive", styles["h2"]))
    story.append(
        paragraph(
            f"Le meilleur modèle du protocole est {best_model['model']} avec "
            f"mAP50-95 {best_model['map50_95']:.3f}. Ce résultat décrit la "
            "capacité de localisation et classification sur les images "
            "validation sélectionnées. Il ne doit pas être transformé en "
            "affirmation causale sur la taille ou la résolution.",
            styles,
        )
    )
    story.append(Paragraph("6.2 Interprétation causale", styles["h2"]))
    story.append(
        paragraph(
            f"Sous les hypothèses d'identification, l'effet AIPW de la "
            f"condition très petite est estimé à {primary_effect['effect']:.3f} "
            f"point de probabilité, avec IC 95 % "
            f"[{primary_effect['ci_lower']:.3f}, "
            f"{primary_effect['ci_upper']:.3f}]. Il s'agit d'une différence "
            "absolue moyenne dans la population analysée, pas d'une baisse "
            "relative universelle.",
            styles,
        )
    )
    story.append(Paragraph("6.3 Limites", styles["h2"]))
    limitations = [
        "Sous-ensemble stratifié plutôt que DOTA complet.",
        "Vingt epochs et une petite famille de modèles.",
        "Classes rares et positivité locale limitée.",
        "Traitement non randomisé et intervention de taille ambiguë.",
        "Flou, contraste, occultation et qualité d'annotation non observés.",
        "Sélection induite par le tuilage et le rejet des fragments ambigus.",
        "Interférence possible entre objets via NMS, densité et max_det.",
        "Outcome dépendant d'un détecteur et de seuils choisis.",
        "Forêt sur pseudo-outcomes, non implémentation complète d'une GRF.",
        "Arbre honnête volontairement peu profond pour limiter les sous-groupes instables.",
        "Bootstrap groupe sans réajustement complet des modèles nuisances.",
    ]
    for item in limitations:
        story.append(bullet(item, styles))
    story.append(Paragraph("6.4 Conclusion", styles["h2"]))
    story.append(
        Paragraph(
            "Le projet fournit une chaîne complète et reproductible allant des "
            "annotations DOTA aux tuiles, aux modèles HBB/OBB, à l'évaluation "
            "par classe et à une table causale par objet. Le résultat prédictif "
            "mesure ce que le détecteur accomplit. L'analyse causale estime "
            "ensuite un contraste sur ses erreurs, avec des hypothèses, des "
            "intervalles et une sensibilité explicites. Cette séparation est la "
            "conclusion méthodologique centrale.",
            styles["callout"],
        )
    )

    story.append(Paragraph("Reproductibilité et références", styles["h1"]))
    reproducibility_rows = [
        ["Élément", "Valeur"],
        ["Graine", "42"],
        ["Images source", "180 train / 60 validation"],
        ["Tuiles", "1024 pixels"],
        ["Stride", "824 train / 1024 validation"],
        ["Chevauchement source", "0 image"],
        ["Architecture", "YOLO26n HBB et OBB"],
        ["Environnement", "Python 3.14.3, torch 2.11.0+cu130"],
        ["GPU", "NVIDIA GeForce RTX 4060"],
        ["Artefacts", "CSV, JSON, figures, poids best/last et notebook exécuté"],
    ]
    story.append(
        make_table(
            reproducibility_rows,
            [5.0 * cm, 11.4 * cm],
            font_regular,
            font_bold,
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    references = [
        "DOTA-v1.0 : https://captain-whu.github.io/DOTA/",
        "Ultralytics YOLO OBB : https://docs.ultralytics.com/tasks/obb/",
        "Sujet officiel du projet, fichier Subject.pdf fourni.",
        "Code, notebook et sorties : dépôt dota-minimap-object-detection.",
    ]
    for reference in references:
        story.append(bullet(reference, styles))

    document.build(
        story,
        onFirstPage=first_page_decorator,
        onLaterPages=page_decorator,
    )
    reader = PdfReader(str(args.output))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(reader.pages) < 8:
        raise RuntimeError("Le rapport genere est anormalement court.")
    if "Résumé exécutif" not in extracted or "Conclusion" not in extracted:
        raise RuntimeError("Le controle textuel du PDF a echoue.")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pages": len(reader.pages),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
