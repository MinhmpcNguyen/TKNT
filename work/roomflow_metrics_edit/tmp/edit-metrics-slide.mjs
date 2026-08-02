import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const inputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_revised.pptx";
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_metrics_revised.pptx";
const previewDir = "/Users/Yuki/TKNT/work/roomflow_metrics_edit/tmp/previews";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function findElement(layout, id, label) {
  const element = layout.elements.find((item) => item.kind === "shape" && item.id === id);
  if (!element) {
    throw new Error(`Missing ${label} (${id})`);
  }
  return element;
}

function resolveShape(presentation, layout, id, label) {
  return presentation.resolve(findElement(layout, id, label).aid);
}

function setText(shape, text, style = {}) {
  shape.text = text;
  shape.text.style = style;
}

function applyMetricSlideEdits(presentation, slideIndex) {
  const slide = presentation.slides.items[slideIndex];
  return slide.export({ format: "layout" }).then(async (layoutBlob) => {
    const layout = JSON.parse(await layoutBlob.text());

    setText(resolveShape(presentation, layout, "63", "chart title"), "Edit score ratio by edit type", {
      fontSize: 20,
      color: "#003A70",
      typeface: "Arial",
    });
    setText(resolveShape(presentation, layout, "64", "chart subtitle"), "Higher is better; score normalized to 100.", {
      fontSize: 12,
      color: "#3D5063",
      typeface: "Arial",
    });

    setText(resolveShape(presentation, layout, "66", "right heading"), "Scoring checklist (5 pts)", {
      fontSize: 20,
      color: "#003A70",
      typeface: "Arial",
    });

    const rows = [
      ["68", "69", "Add", "Inside bbox; identity and scale are correct"],
      ["71", "72", "Recolor", "Requested color/material; target shape is kept"],
      ["74", "75", "Remove", "Target removed; background is repaired"],
      ["77", "78", "Replace", "Correct object; footprint and scale are kept"],
      ["80", "81", "Common checks", "Camera/room, other objects, and artifacts"],
    ];

    for (const [titleId, detailId, title, detail] of rows) {
      setText(resolveShape(presentation, layout, titleId, `${title} title`), title, {
        fontSize: 18,
        bold: true,
        color: title === "Common checks" ? "#B51F2A" : "#003A70",
        typeface: "Arial",
      });
      setText(resolveShape(presentation, layout, detailId, `${title} detail`), detail, {
        fontSize: 11.2,
        color: "#20364E",
        typeface: "Arial",
      });
    }

    setText(
      resolveShape(presentation, layout, "82", "source note"),
      "Source: edit_manual_v1 checklist, n=64 per variant.",
      {
        fontSize: 10.5,
        color: "#5B6B7F",
        typeface: "Arial",
      },
    );
    setText(
      resolveShape(presentation, layout, "84", "takeaway"),
      "Selected target boxes improve edit score; add/replace still need tighter target control.",
      {
        fontSize: 16,
        color: "#003A70",
        typeface: "Arial",
      },
    );

    return slide;
  });
}

await fs.mkdir(previewDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPptx));
const editedSlides = [];
for (const slideIndex of [23, 31]) {
  editedSlides.push(await applyMetricSlideEdits(presentation, slideIndex));
}

for (const [index, slide] of editedSlides.entries()) {
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
