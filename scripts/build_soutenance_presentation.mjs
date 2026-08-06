import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  layers,
  shape,
  text,
} from "@oai/artifact-tool";

const finalPptx = path.resolve(process.argv[2] ?? "outputs/soutenance_yolo_obb.pptx");
const projectDir = path.resolve(process.argv[3] ?? process.cwd());
const qaDir = path.resolve(process.argv[4] ?? path.join(path.dirname(finalPptx), "presentation_qa"));

const COLORS = {
  canvas: "#FFFFFF",
  ink: "#000000",
  muted: "#5D6470",
  panel: "#F2F2F2",
  rule: "#B8BCC4",
  accent: "#6DCBF4",
  accentStrong: "#3D8DFF",
  baseline: "#A7ADB7",
};

const FONT = "Arial";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function imageBuffer(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function tx(value, position, style = {}) {
  return text([value], {
    position,
    width: position.width,
    height: position.height,
    style: {
      fontSize: style.fontSize ?? "20px",
      typeface: FONT,
      color: style.color ?? COLORS.ink,
      bold: style.bold ?? false,
      alignment: style.alignment ?? "left",
      verticalAlignment: style.verticalAlignment ?? "top",
      autoFit: style.autoFit ?? "shrinkText",
      wrap: "square",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  });
}

function buildSlide1(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;

  const checklist = [
    ["01", "TUILE 1024", "Préserver les petits détails"],
    ["02", "BACKBONE CNN", "Extraire les caractéristiques"],
    ["03", "NECK MULTI-ÉCHELLE", "Fusionner plusieurs résolutions"],
    ["04", "TÊTE OBB", "Classe, confiance, position et angle"],
    ["05", "FILTRAGE", "Confiance puis NMS orientée"],
  ];

  const nodes = [
    tx("YOLO-OBB suit l’orientation des objets DOTA", { left: 42, top: 34, width: 1196, height: 64 }, { fontSize: "42px" }),
    tx("Pourquoi ce choix ?", { left: 42, top: 126, width: 570, height: 40 }, { fontSize: "27px", bold: true }),
    tx("Petits objets  •  orientations libres  •  scènes denses", { left: 42, top: 166, width: 570, height: 32 }, { fontSize: "19px", color: COLORS.muted }),
    shape({ geometry: "rect", fill: COLORS.rule, position: { left: 640, top: 126 }, width: 1.5, height: 498 }),
    tx("Le pipeline en cinq étapes", { left: 687, top: 126, width: 520, height: 40 }, { fontSize: "27px", bold: true }),
    tx("OBB : moins d’arrière-plan  •  1024 : plus de détail  •  nano : modèle compact", { left: 42, top: 575, width: 570, height: 48 }, { fontSize: "17px", color: COLORS.muted }),
    tx("DOTA-v1.0 — modèle principal : YOLO26n-OBB-1024", { left: 42, top: 662, width: 600, height: 22 }, { fontSize: "13px", color: COLORS.muted }),
    tx("1", { left: 1184, top: 660, width: 54, height: 22 }, { fontSize: "13px", alignment: "right" }),
  ];

  checklist.forEach(([number, label, detail], index) => {
    const top = 188 + index * 86;
    nodes.push(
      tx(number, { left: 687, top, width: 58, height: 48 }, { fontSize: "30px", bold: true, color: COLORS.accentStrong }),
      tx(label, { left: 762, top, width: 445, height: 28 }, { fontSize: "21px", bold: true }),
      tx(detail, { left: 762, top: top + 31, width: 445, height: 28 }, { fontSize: "17px", color: COLORS.muted }),
      shape({ geometry: "rect", fill: index === 4 ? COLORS.accentStrong : COLORS.panel, position: { left: 762, top: top + 65 }, width: 445, height: 1.5 }),
    );
  });

  slide.compose(
    layers({ name: "codex-grid-adapted-slide-10", width: "fill", height: "fill" }, nodes),
    { frame: { left: 0, top: 0, width: 1280, height: 720 }, baseUnit: 1 },
  );

  return slide;
}

async function addSlide1Evidence(slide) {
  const demoPath = path.join(projectDir, "outputs", "presentation_assets", "demo_p0249_yolo_obb.jpg");
  slide.images.add({
    blob: await imageBuffer(demoPath),
    contentType: "image/jpeg",
    alt: "Tuile aérienne de validation avec seize prédictions d'avions encadrées par des boîtes orientées bleues",
    fit: "cover",
    position: { left: 42, top: 210, width: 570, height: 345 },
    geometry: "rect",
  });
}

function buildSlide2(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;

  const nodes = [
    tx("L’IoU relie la géométrie aux résultats", { left: 42, top: 34, width: 1196, height: 64 }, { fontSize: "42px" }),
    tx("Même formule pour HBB et OBB ; seule l’intersection polygonale devient plus complexe.", { left: 42, top: 108, width: 1196, height: 34 }, { fontSize: "20px", color: COLORS.muted }),
    shape({ geometry: "rect", fill: COLORS.rule, position: { left: 640, top: 162 }, width: 1.5, height: 466 }),
    tx("Intersection", { left: 64, top: 176, width: 210, height: 30 }, { fontSize: "19px", bold: true, color: COLORS.accentStrong }),
    tx("IoU(A,B) = Aire(A ∩ B)", { left: 333, top: 222, width: 285, height: 44 }, { fontSize: "25px", bold: true, alignment: "center" }),
    shape({ geometry: "rect", fill: COLORS.ink, position: { left: 365, top: 276 }, width: 220, height: 2 }),
    tx("Aire(A) + Aire(B) − Aire(A ∩ B)", { left: 329, top: 288, width: 293, height: 72 }, { fontSize: "21px", alignment: "center" }),
    tx("0 = aucune superposition\n1 = superposition parfaite", { left: 338, top: 388, width: 270, height: 74 }, { fontSize: "19px", color: COLORS.muted, alignment: "center" }),
    tx("Exemple : 60 / (100 + 100 − 60) = 0,43", { left: 58, top: 515, width: 555, height: 40 }, { fontSize: "21px", bold: true, alignment: "center" }),
    tx("L’IoU sert à évaluer la localisation et à repérer les prédictions redondantes.", { left: 58, top: 565, width: 555, height: 55 }, { fontSize: "17px", color: COLORS.muted, alignment: "center" }),
    tx("Validation préliminaire", { left: 680, top: 166, width: 520, height: 38 }, { fontSize: "27px", bold: true }),
    tx("7,9 ms/image", { left: 680, top: 482, width: 240, height: 42 }, { fontSize: "30px", bold: true, color: COLORS.accentStrong }),
    tx("contre 17,1 ms pour Faster R-CNN", { left: 916, top: 491, width: 290, height: 30 }, { fontSize: "17px", color: COLORS.muted }),
    tx("YOLO-OBB est ≈ 2,2× plus rapide et obtient les meilleurs scores dans cette expérience.", { left: 680, top: 548, width: 520, height: 54 }, { fontSize: "20px", bold: true }),
    tx("Limite : rappel = 0,291 — une partie importante des objets reste manquée.", { left: 680, top: 607, width: 520, height: 38 }, { fontSize: "17px", color: COLORS.muted }),
    tx("Source : validation locale exécutée dans projet_dota_soutenance.ipynb", { left: 42, top: 662, width: 720, height: 22 }, { fontSize: "13px", color: COLORS.muted }),
    tx("2", { left: 1184, top: 660, width: 54, height: 22 }, { fontSize: "13px", alignment: "right" }),
  ];

  slide.compose(
    layers({ name: "codex-grid-adapted-slide-11", width: "fill", height: "fill" }, nodes),
    { frame: { left: 0, top: 0, width: 1280, height: 720 }, baseUnit: 1 },
  );

  return slide;
}

async function addSlide2Evidence(slide) {
  const iouPath = path.join(projectDir, "outputs", "presentation_assets", "iou_obb_diagram.png");
  slide.images.add({
    blob: await imageBuffer(iouPath),
    contentType: "image/png",
    alt: "Deux boîtes orientées translucides dont la zone d'intersection est colorée en bleu foncé",
    fit: "contain",
    position: { left: 48, top: 204, width: 286, height: 286 },
    geometry: "rect",
  });

  slide.charts.add("bar", {
    position: { left: 680, top: 218, width: 520, height: 242 },
    categories: ["F1", "mAP50-95"],
    series: [
      { name: "Faster R-CNN HBB", values: [0.213, 0.060], fill: COLORS.baseline, valuesFormatCode: "0.000" },
      { name: "YOLO26n-OBB-1024", values: [0.386, 0.185], fill: COLORS.accentStrong, valuesFormatCode: "0.000" },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 42 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fill: COLORS.muted, fontSize: 13 } },
    xAxis: { textStyle: { fill: COLORS.ink, fontSize: 15, bold: true }, line: { style: "solid", fill: COLORS.rule, width: 1 } },
    yAxis: {
      min: 0,
      max: 0.45,
      majorUnit: 0.10,
      numberFormatCode: "0.0",
      textStyle: { fill: COLORS.muted, fontSize: 12 },
      majorGridlines: { style: "solid", fill: "#E4E6EA", width: 1 },
      line: { style: "solid", fill: "none", width: 0 },
    },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: COLORS.ink, fontSize: 13, bold: true } },
    chartFill: "#FFFFFF",
    chartLine: { style: "solid", fill: "none", width: 0 },
    plotAreaFill: "#FFFFFF",
    plotAreaLine: { style: "solid", fill: "none", width: 0 },
  });
}

async function main() {
  await fs.mkdir(path.dirname(finalPptx), { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slide1 = buildSlide1(presentation);
  await addSlide1Evidence(slide1);
  const slide2 = buildSlide2(presentation);
  await addSlide2Evidence(slide2);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(qaDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(qaDir, `${stem}.layout.json`), await layout.text());
  }

  await writeBlob(
    path.join(qaDir, "deck-montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );
  const inspect = await presentation.inspect({ kind: "slide,textbox,shape,image,chart", maxChars: 30000 });
  await fs.writeFile(path.join(qaDir, "deck-inspect.ndjson"), inspect.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  console.log(`PPTX=${finalPptx}`);
  console.log(`QA=${qaDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
