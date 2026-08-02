import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const inputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_metrics_revised.pptx";
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_metrics_clean_text.pptx";
const previewDir = "/Users/Yuki/TKNT/work/roomflow_metrics_clean_text/tmp/previews";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function shapeById(layout, id, label) {
  const element = layout.elements.find((item) => item.kind === "shape" && item.id === id);
  if (!element) {
    throw new Error(`Missing ${label} (${id})`);
  }
  return element;
}

function resolveShape(presentation, layout, id, label) {
  return presentation.resolve(shapeById(layout, id, label).aid);
}

function deleteShape(shape) {
  if (shape && typeof shape.delete === "function") {
    shape.delete();
    return;
  }
  if (shape && typeof shape.remove === "function") {
    shape.remove();
    return;
  }
  shape.text = "";
}

function setLargeText(shape, text, { fontSize, color = "#003A70", bold = true, alignment = "center" }) {
  shape.text = text;
  shape.text.style = {
    fontSize,
    bold,
    color,
    alignment,
    typeface: bold ? "Arial Bold" : "Arial",
  };
}

async function editSlide(presentation, slideIndex) {
  const slide = presentation.slides.items[slideIndex];
  const layout = JSON.parse(await (await slide.export({ format: "layout" })).text());

  const smallCaptionIds = ["64", "69", "72", "75", "78", "81", "82"];
  for (const id of smallCaptionIds) {
    const element = layout.elements.find((item) => item.kind === "shape" && item.id === id);
    if (element) {
      deleteShape(presentation.resolve(element.aid));
    }
  }

  const chartTitle = resolveShape(presentation, layout, "63", "chart title");
  chartTitle.position = { left: 72, top: 142, width: 540, height: 38 };
  setLargeText(chartTitle, "Edit score ratio by edit type", {
    fontSize: 24,
    color: "#003A70",
    bold: true,
    alignment: "left",
  });

  const rightTitle = resolveShape(presentation, layout, "66", "checklist title");
  rightTitle.position = { left: 660, top: 142, width: 330, height: 42 };
  setLargeText(rightTitle, "Scoring checklist (5 pts)", {
    fontSize: 25,
    color: "#003A70",
    bold: true,
    alignment: "center",
  });

  const cardRows = [
    ["68", "Add", 185, "#003A70"],
    ["71", "Recolor", 258, "#003A70"],
    ["74", "Remove", 331, "#003A70"],
    ["77", "Replace", 404, "#003A70"],
    ["80", "Common checks", 477, "#B51F2A"],
  ];

  for (const [id, text, top, color] of cardRows) {
    const label = resolveShape(presentation, layout, id, `${text} label`);
    label.position = { left: 674, top: top + 12, width: 292, height: 40 };
    setLargeText(label, text, {
      fontSize: text === "Common checks" ? 23 : 25,
      color,
      bold: true,
      alignment: "center",
    });
  }

  const takeaway = resolveShape(presentation, layout, "84", "bottom takeaway");
  takeaway.position = { left: 110, top: 630, width: 804, height: 32 };
  setLargeText(
    takeaway,
    "Selected target boxes improve edit score; add/replace still need tighter target control.",
    {
      fontSize: 18,
      color: "#003A70",
      bold: true,
      alignment: "center",
    },
  );

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
