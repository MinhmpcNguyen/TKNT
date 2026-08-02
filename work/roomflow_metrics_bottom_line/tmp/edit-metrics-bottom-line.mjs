import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const inputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide13_plan_focus.pptx";
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_metrics_bottom_line.pptx";
const previewDir = "/Users/Yuki/TKNT/work/roomflow_metrics_bottom_line/tmp/preview";
const layoutDir = "/Users/Yuki/TKNT/work/roomflow_metrics_bottom_line/tmp/layout/final";
const replacement = "Selected target boxes improve scores across most edit types.";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function resolveShapeById(presentation, layout, id, label) {
  const element = layout.elements.find((item) => item.kind === "shape" && item.id === id);
  if (!element) {
    throw new Error(`Missing ${label} (${id})`);
  }
  return presentation.resolve(element.aid);
}

async function editSlide(presentation, slideIndex, slideNumber) {
  const slide = presentation.slides.items[slideIndex];
  const layout = JSON.parse(await (await slide.export({ format: "layout" })).text());
  const target = resolveShapeById(presentation, layout, "84", `bottom takeaway on slide ${slideNumber}`);

  target.text = replacement;
  target.text.style = {
    fontSize: 16,
    color: "#003A70",
    alignment: "center",
    typeface: "Arial",
  };

  await writeBlob(`${previewDir}/slide-${slideNumber}.png`, await presentation.export({ slide, format: "png", scale: 1 }));
  await fs.writeFile(`${layoutDir}/slide-${slideNumber}.layout.json`, await (await slide.export({ format: "layout" })).text());
}

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPptx));
await editSlide(presentation, 23, 24);
await editSlide(presentation, 31, 32);

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);

console.log(JSON.stringify({ outputPptx, previewDir, layoutDir, replacement }, null, 2));
