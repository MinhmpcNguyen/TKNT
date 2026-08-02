import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "/Users/Yuki/TKNT/work/roomflow_slide13_14_link/tmp";
const inputPptx = `${workspace}/template-starter.pptx`;
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide13_14_linked.pptx";
const previewDir = `${workspace}/preview`;
const finalLayoutDir = `${workspace}/layout/final`;

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function shapeElement(layout, id, label) {
  const element = layout.elements.find((item) => item.kind === "shape" && item.id === id);
  if (!element) {
    throw new Error(`Missing ${label} (${id})`);
  }
  return element;
}

function resolveShape(presentation, layout, id, label) {
  return presentation.resolve(shapeElement(layout, id, label).aid);
}

function setText(shape, text, style = {}) {
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize ?? 16,
    bold: style.bold ?? false,
    color: style.color ?? "#1F2937",
    alignment: style.alignment ?? "left",
    typeface: style.bold ? "Arial Bold" : "Arial",
  };
}

function setTitle(shape, text) {
  shape.text = text;
  shape.text.style = {
    fontSize: 24,
    bold: true,
    color: "#FFFFFF",
    alignment: "left",
    typeface: "Arial Bold",
  };
}

function styleCard(shape, { left, top, width, height, fill, line }) {
  shape.position = { left, top, width, height };
  shape.fill = fill;
  shape.line = { style: "solid", fill: line, width: 1.2 };
  shape.borderRadius = 12;
}

function setStageTitle(shape, text) {
  setText(shape, text, {
    fontSize: 20,
    bold: false,
    color: "#003A70",
    alignment: "left",
  });
}

function setStageBody(shape, text) {
  setText(shape, text, {
    fontSize: 14.5,
    color: "#263445",
    alignment: "center",
  });
}

function setListLine(shape, text, top) {
  shape.position = { left: 94, top, width: 424, height: 30 };
  setText(shape, text, {
    fontSize: 15.7,
    color: "#1F2937",
    alignment: "left",
  });
}

function setUseDot(shape, top, fill) {
  shape.position = { left: 572, top, width: 22, height: 22 };
  shape.fill = fill;
  shape.line = { style: "solid", fill: "none", width: 0 };
  shape.text.style = {
    fontSize: 12,
    bold: true,
    color: "#FFFFFF",
    alignment: "center",
    typeface: "Arial Bold",
  };
}

function setUseLine(shape, text, top) {
  shape.position = { left: 606, top: top - 2, width: 320, height: 28 };
  setText(shape, text, {
    fontSize: 14.2,
    color: "#263445",
    alignment: "left",
  });
}

async function editSlide13(presentation) {
  const slide = presentation.slides.items[12];
  const layout = JSON.parse(await (await slide.export({ format: "layout" })).text());

  setTitle(resolveShape(presentation, layout, "2", "slide 13 title"), "AI Converts the Brief into a Placement-Ready Plan");

  setStageTitle(resolveShape(presentation, layout, "7", "stage 1 title"), "User brief");
  setStageBody(
    resolveShape(presentation, layout, "8", "stage 1 body"),
    "Living room with sofa,\nTV console, coffee table,\nrug, bookshelf, and\n1-2 armchairs.",
  );

  setStageTitle(resolveShape(presentation, layout, "11", "stage 2 title"), "Parse needs");
  setStageBody(
    resolveShape(presentation, layout, "12", "stage 2 body"),
    "Required items\nOptional items\nRelations and preferences",
  );

  setStageTitle(resolveShape(presentation, layout, "15", "stage 3 title"), "Validate plan");
  setStageBody(
    resolveShape(presentation, layout, "16", "stage 3 body"),
    "Schema checks and retries\nproduce a complete plan\nbefore placement.",
  );

  styleCard(resolveShape(presentation, layout, "4", "placement plan card"), {
    left: 64,
    top: 328,
    width: 464,
    height: 232,
    fill: "#F3FAFF",
    line: "#0877B9",
  });
  const planTitle = resolveShape(presentation, layout, "37", "placement plan title");
  planTitle.position = { left: 84, top: 344, width: 424, height: 34 };
  setText(planTitle, "Placement-ready plan", {
    fontSize: 22,
    bold: false,
    color: "#003A70",
    alignment: "center",
  });
  setListLine(resolveShape(presentation, layout, "39", "required line"), "Required: sofa, TV console, coffee table, rug, bookshelf", 390);
  setListLine(resolveShape(presentation, layout, "40", "optional line"), "Optional: 1-2 armchairs", 430);
  setListLine(resolveShape(presentation, layout, "41", "relation line"), "Relation: sofa faces TV/media group", 470);
  setListLine(resolveShape(presentation, layout, "42", "preference line"), "Preference: keep the center open", 510);

  styleCard(resolveShape(presentation, layout, "25", "placement use card"), {
    left: 552,
    top: 328,
    width: 408,
    height: 232,
    fill: "#F7FCFA",
    line: "#0B8F6B",
  });
  const useTitle = resolveShape(presentation, layout, "26", "placement use title");
  useTitle.position = { left: 572, top: 344, width: 368, height: 34 };
  setText(useTitle, "Used by placement", {
    fontSize: 20,
    bold: false,
    color: "#003A70",
    alignment: "center",
  });

  const uses = [
    ["27", "28", 392, "#0B78A8", "place required items first"],
    ["29", "30", 430, "#F59E0B", "add optional items later"],
    ["31", "32", 468, "#0B8F6B", "use relations as constraints"],
    ["33", "34", 506, "#003A70", "keep preferences for scoring"],
  ];
  for (const [dotId, textId, top, fill, text] of uses) {
    setUseDot(resolveShape(presentation, layout, dotId, `use dot ${dotId}`), top, fill);
    setUseLine(resolveShape(presentation, layout, textId, `use line ${textId}`), text, top);
  }

  setText(
    resolveShape(presentation, layout, "36", "slide 13 handoff"),
    "This checked plan is the input to placement search on the next slide.",
    {
      fontSize: 17.5,
      bold: false,
      color: "#003A70",
      alignment: "center",
    },
  );

  return slide;
}

function setAlgorithmText(shape, text, { fontSize = 14.2, bold = false, alignment = "center" } = {}) {
  setText(shape, text, {
    fontSize,
    bold,
    color: "#003A70",
    alignment,
  });
}

async function editSlide14(presentation) {
  const slide = presentation.slides.items[13];
  const layout = JSON.parse(await (await slide.export({ format: "layout" })).text());

  setTitle(resolveShape(presentation, layout, "2", "slide 14 title"), "The Checked Plan Drives Placement Search");

  setAlgorithmText(resolveShape(presentation, layout, "7", "step 1 title"), "Use plan to find\ncandidate spots", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "15", "step 1 caption"), "room zones, required items,\nand relations", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "18", "step 2 title"), "Try allowed\ndirections", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "24", "step 2 caption"), "each spot x 4 directions", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "27", "step 3 title"), "Place required\nitems", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "35", "step 3 caption"), "reject severe conflicts;\nflag minor ones", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "38", "step 4 title"), "Add related\nfurniture", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "45", "step 4 caption"), "follow relation fields", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "48", "step 5 title"), "Repair small\nconflicts", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "56", "step 5 caption"), "small shifts inside\nallowed zones", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "59", "step 6 title"), "Add optional\nitems", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "68", "step 6 caption"), "only after required\nlayout fits", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "71", "step 7 title"), "Score valid\nlayouts", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "81", "step 7 caption"), "function, spacing,\nand preferences", {
    fontSize: 11.4,
  });

  setAlgorithmText(resolveShape(presentation, layout, "84", "step 8 title"), "Rank best\noptions", {
    fontSize: 14.8,
    alignment: "left",
  });
  setAlgorithmText(resolveShape(presentation, layout, "91", "step 8 caption"), "ranked layouts with\nnotes", {
    fontSize: 11.4,
  });

  setText(
    resolveShape(presentation, layout, "100", "slide 14 summary"),
    "Checked plan in \u2192 Generate candidates \u2192 Repair/complete \u2192 Score and rank",
    {
      fontSize: 16.8,
      bold: false,
      color: "#003A70",
      alignment: "center",
    },
  );

  return slide;
}

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(finalLayoutDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPptx));
const editedSlides = [
  { number: 13, slide: await editSlide13(presentation) },
  { number: 14, slide: await editSlide14(presentation) },
];

for (const { number, slide } of editedSlides) {
  const padded = String(number).padStart(2, "0");
  await writeBlob(`${previewDir}/slide-${padded}.png`, await presentation.export({ slide, format: "png", scale: 1 }));
  await fs.writeFile(`${finalLayoutDir}/slide-${padded}.layout.json`, await (await slide.export({ format: "layout" })).text());
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);

console.log(JSON.stringify({ outputPptx, previewDir, finalLayoutDir }, null, 2));
