import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "/Users/Yuki/TKNT/work/roomflow_slide16_edit/tmp";
const inputPptx = `${workspace}/template-starter.pptx`;
const outputPptx = "/Users/Yuki/TKNT/outputs/RoomFlow_Thesis_Defense_4x3_slide16_revised.pptx";
const outputPreview = `${workspace}/final-slide-16.png`;
const outputLayout = `${workspace}/final-slide-16.layout.json`;
const generatedImagePath = `${workspace}/template-inspect/assets/ppt/media/image21.png`;

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function findOne(layout, predicate, label) {
  const match = layout.elements.find(predicate);
  if (!match) {
    throw new Error(`Missing slide 16 element: ${label}`);
  }
  return match;
}

function resolved(presentation, layout, predicate, label) {
  const element = findOne(layout, predicate, label);
  return presentation.resolve(element.aid);
}

function resolvedAll(presentation, layout, predicate) {
  return layout.elements.filter(predicate).map((element) => presentation.resolve(element.aid));
}

function deleteIfPossible(target) {
  if (target && typeof target.delete === "function") {
    target.delete();
    return;
  }
  if (target && typeof target.remove === "function") {
    target.remove();
  }
}

function addTextbox(slide, { name, left, top, width, height, text, fontSize, bold = false, color = "#1F2937", alignment = "center" }) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize,
    bold,
    color,
    alignment,
    typeface: bold ? "Arial Bold" : "Arial",
  };
  return shape;
}

function addArrow(slide, { left, top, width, height }) {
  return slide.shapes.add({
    geometry: "rightArrow",
    name: "pipeline-arrow",
    position: { left, top, width, height },
    fill: "#003A70",
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addStage(slide, { index, left, top, title, body }) {
  const card = slide.shapes.add({
    geometry: "roundRect",
    name: `pipeline-stage-${index}`,
    position: { left, top, width: 148, height: 108 },
    fill: "#F8FBFE",
    line: { style: "solid", fill: "#0877B9", width: 1.2 },
    borderRadius: 9,
  });

  const badge = slide.shapes.add({
    geometry: "ellipse",
    name: `pipeline-stage-${index}-badge`,
    position: { left: left + 12, top: top + 12, width: 26, height: 26 },
    fill: "#003A70",
    line: { style: "solid", fill: "none", width: 0 },
  });
  badge.text = String(index);
  badge.text.style = {
    fontSize: 12,
    bold: true,
    color: "#FFFFFF",
    alignment: "center",
    typeface: "Arial Bold",
  };

  addTextbox(slide, {
    name: `pipeline-stage-${index}-title`,
    left: left + 44,
    top: top + 10,
    width: 92,
    height: 32,
    text: title,
    fontSize: 13,
    bold: true,
    color: "#003A70",
    alignment: "left",
  });

  addTextbox(slide, {
    name: `pipeline-stage-${index}-body`,
    left: left + 12,
    top: top + 50,
    width: 124,
    height: 44,
    text: body,
    fontSize: 11.5,
    color: "#1F2937",
  });

  return card;
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPptx));
const slide = presentation.slides.items[15];
const initialLayout = JSON.parse(await (await slide.export({ format: "layout" })).text());

const title = resolved(
  presentation,
  initialLayout,
  (element) => element.name === "slide-title",
  "slide title",
);
title.text.replace(
  "The Generated Image Follows the 3D Furniture Arrangement",
  "Layout Constraints Keep the Generated Image Aligned",
);

for (const target of [
  resolved(
    presentation,
    initialLayout,
    (element) => element.kind === "image" && element.name === "Picture 11",
    "central technical diagram",
  ),
  ...resolvedAll(
    presentation,
    initialLayout,
    (element) =>
      element.kind === "shape" &&
      (["10", "15", "20"].includes(element.id) || String(element.name || "").startsWith("Straight Arrow Connector")),
  ),
]) {
  deleteIfPossible(target);
}

const panel = resolved(
  presentation,
  initialLayout,
  (element) => element.kind === "shape" && element.id === "7",
  "central panel",
);
panel.position = { left: 234, top: 220, width: 585, height: 324 };
panel.fill = "none";
panel.line = { style: "solid", fill: "#003A70", width: 1.2 };
panel.bringToFront();

addTextbox(slide, {
  name: "pipeline-heading",
  left: 275,
  top: 244,
  width: 505,
  height: 32,
  text: "Constraint-guided image generation",
  fontSize: 17,
  bold: true,
  color: "#003A70",
});

addTextbox(slide, {
  name: "pipeline-subhead",
  left: 286,
  top: 274,
  width: 485,
  height: 28,
  text: "The image model receives visual structure before it applies the requested style.",
  fontSize: 12.5,
  color: "#1F2937",
});

addStage(slide, {
  index: 1,
  left: 270,
  top: 318,
  title: "Visual constraints",
  body: "Room geometry,\nfurniture positions,\ncamera view",
});
addArrow(slide, { left: 425, top: 357, width: 28, height: 18 });
addStage(slide, {
  index: 2,
  left: 462,
  top: 318,
  title: "Guidance package",
  body: "Visual guide,\nobject constraints,\nstyle prompt",
});
addArrow(slide, { left: 617, top: 357, width: 28, height: 18 });
addStage(slide, {
  index: 3,
  left: 654,
  top: 318,
  title: "Image model",
  body: "Generates a\nperspective image\nfrom constraints",
});

const takeaway = slide.shapes.add({
  geometry: "roundRect",
  name: "pipeline-takeaway",
  position: { left: 284, top: 456, width: 485, height: 46 },
  fill: "#EAF4FC",
  line: { style: "solid", fill: "#0877B9", width: 1 },
  borderRadius: 7,
});
takeaway.text = "Geometry and furniture positions stay fixed while style changes.";
takeaway.text.style = {
  fontSize: 13.5,
  bold: true,
  color: "#003A70",
  alignment: "center",
  typeface: "Arial Bold",
};

addArrow(slide, { left: 202, top: 354, width: 34, height: 20 });
addArrow(slide, { left: 818, top: 354, width: 30, height: 20 });

const outputImage = resolved(
  presentation,
  initialLayout,
  (element) => element.kind === "image" && element.name === "Picture 50",
  "output image placeholder",
);
const generatedImage = await fs.readFile(generatedImagePath);
outputImage.replace({
  blob: generatedImage,
  contentType: "image/png",
  alt: "Generated realistic room image",
  fit: "cover",
});
outputImage.frame = { left: 852, top: 250, width: 128, height: 118 };
outputImage.fit = "cover";
outputImage.geometry = "roundRect";
outputImage.borderRadius = 8;

const outputLabel = resolved(
  presentation,
  initialLayout,
  (element) => element.kind === "shape" && element.id === "25",
  "output label",
);
outputLabel.text = "Realistic image\nwith fixed layout";
outputLabel.position = { left: 852, top: 391, width: 128, height: 48 };
outputLabel.text.style = {
  fontSize: 14,
  bold: true,
  color: "#1F2937",
  alignment: "center",
  typeface: "Arial Bold",
};
outputLabel.bringToFront();

await fs.writeFile(outputLayout, await (await slide.export({ format: "layout" })).text());
await writeBlob(outputPreview, await presentation.export({ slide, format: "png", scale: 1 }));

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);

console.log(JSON.stringify({ outputPptx, outputPreview, outputLayout }, null, 2));
