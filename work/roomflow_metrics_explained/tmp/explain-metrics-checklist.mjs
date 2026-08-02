import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const inputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_metrics_clean_text.pptx";
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_metrics_explained.pptx";
const previewDir = "/Users/Yuki/TKNT/work/roomflow_metrics_explained/tmp/previews";

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

function setChecklistText(shape, text, { color = "#003A70" } = {}) {
  shape.text = text;
  shape.text.style = {
    fontSize: 17,
    bold: true,
    color,
    alignment: "center",
    typeface: "Arial Bold",
  };
}

async function editSlide(presentation, slideIndex) {
  const slide = presentation.slides.items[slideIndex];
  const layout = JSON.parse(await (await slide.export({ format: "layout" })).text());

  const rows = [
    ["68", "Add\ninside target box", 185, "#003A70"],
    ["71", "Recolor\ncolor changes; shape stays", 258, "#003A70"],
    ["74", "Remove\ntarget gone; background fixed", 331, "#003A70"],
    ["77", "Replace\nright object; same footprint", 404, "#003A70"],
    ["80", "Common checks\ncamera, room, objects stay", 477, "#B51F2A"],
  ];

  for (const [id, text, top, color] of rows) {
    const shape = resolveShape(presentation, layout, id, text);
    shape.position = { left: 682, top: top + 7, width: 276, height: 48 };
    setChecklistText(shape, text, { color });
  }

  return slide;
}

await fs.mkdir(previewDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPptx));
const slides = [];
for (const slideIndex of [23, 31]) {
  slides.push(await editSlide(presentation, slideIndex));
}

for (const [index, slide] of slides.entries()) {
  const slideNumber = index === 0 ? 24 : 32;
  await writeBlob(
    `${previewDir}/slide-${slideNumber}.png`,
    await presentation.export({ slide, format: "png", scale: 1 }),
  );
  await fs.writeFile(
    `${previewDir}/slide-${slideNumber}.layout.json`,
    await (await slide.export({ format: "layout" })).text(),
  );
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);

console.log(JSON.stringify({ outputPptx, previewDir }, null, 2));
