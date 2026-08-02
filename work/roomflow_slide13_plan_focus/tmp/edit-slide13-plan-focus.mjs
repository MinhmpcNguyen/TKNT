import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "/Users/Yuki/TKNT/work/roomflow_slide13_plan_focus/tmp";
const inputPptx = `${workspace}/template-starter.pptx`;
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide13_plan_focus.pptx";
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
    fontSize: style.fontSize ?? 15,
    bold: style.bold ?? false,
    color: style.color ?? "#263445",
    alignment: style.alignment ?? "left",
    typeface: style.bold ? "Arial Bold" : "Arial",
  };
}

function styleCard(shape, { left, top, width, height, fill, line }) {
  shape.position = { left, top, width, height };
  shape.fill = fill;
  shape.line = { style: "solid", fill: line, width: 1.2 };
  shape.borderRadius = 12;
}

function setPlanLine(shape, text, top) {
  shape.position = { left: 94, top, width: 426, height: 34 };
  setText(shape, text, {
    fontSize: 14.6,
    color: "#1F2937",
    alignment: "left",
  });
}

function setCheckDot(shape, top, fill) {
  shape.position = { left: 574, top, width: 22, height: 22 };
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

function setCheckLine(shape, text, top) {
  shape.position = { left: 608, top: top - 3, width: 324, height: 32 };
  setText(shape, text, {
    fontSize: 13.4,
    color: "#263445",
    alignment: "left",
  });
}

async function editSlide13(presentation) {
  const slide = presentation.slides.items[12];
  const layout = JSON.parse(await (await slide.export({ format: "layout" })).text());

  styleCard(resolveShape(presentation, layout, "4", "plan field card"), {
    left: 64,
    top: 328,
    width: 464,
    height: 232,
    fill: "#F3FAFF",
    line: "#0877B9",
  });
  const planTitle = resolveShape(presentation, layout, "37", "plan title");
  planTitle.position = { left: 84, top: 344, width: 424, height: 34 };
  setText(planTitle, "Structured plan fields", {
    fontSize: 21,
    color: "#003A70",
    alignment: "center",
  });
  setPlanLine(resolveShape(presentation, layout, "39", "required items"), "Required items: sofa, TV console, coffee table, rug, bookshelf", 390);
  setPlanLine(resolveShape(presentation, layout, "40", "optional items"), "Optional items: 1-2 armchairs, lower priority", 430);
  setPlanLine(resolveShape(presentation, layout, "41", "relations"), "Relation: seating faces TV/media group", 470);
  setPlanLine(resolveShape(presentation, layout, "42", "preference"), "Preference: keep the center open", 510);

  styleCard(resolveShape(presentation, layout, "25", "plan checks card"), {
    left: 552,
    top: 328,
    width: 408,
    height: 232,
    fill: "#F7FCFA",
    line: "#0B8F6B",
  });
  const checksTitle = resolveShape(presentation, layout, "26", "checks title");
  checksTitle.position = { left: 572, top: 344, width: 368, height: 34 };
  setText(checksTitle, "Checks before handoff", {
    fontSize: 21,
    color: "#003A70",
    alignment: "center",
  });

  const checkRows = [
    ["27", "28", 392, "#0B78A8", "Schema complete: all required fields are present"],
    ["29", "30", 430, "#F59E0B", "Catalog matched: furniture names are valid"],
    ["31", "32", 468, "#0B8F6B", "Relations explicit: no vague instruction left"],
    ["33", "34", 506, "#003A70", "Retry if invalid: output must be clean"],
  ];

  for (const [dotId, textId, top, fill, text] of checkRows) {
    setCheckDot(resolveShape(presentation, layout, dotId, `check dot ${dotId}`), top, fill);
    setCheckLine(resolveShape(presentation, layout, textId, `check line ${textId}`), text, top);
  }

  setText(
    resolveShape(presentation, layout, "36", "slide 13 takeaway"),
    "The output is a checked structured plan, not free-form text.",
    {
      fontSize: 17.5,
      color: "#003A70",
      alignment: "center",
    },
  );

  return slide;
}

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(finalLayoutDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPptx));
const slide = await editSlide13(presentation);

await writeBlob(`${previewDir}/slide-13.png`, await presentation.export({ slide, format: "png", scale: 1 }));
await fs.writeFile(`${finalLayoutDir}/slide-13.layout.json`, await (await slide.export({ format: "layout" })).text());

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);

console.log(JSON.stringify({ outputPptx, previewDir, finalLayoutDir }, null, 2));
