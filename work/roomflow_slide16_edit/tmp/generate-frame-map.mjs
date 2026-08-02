import fs from "node:fs/promises";

const workspace = "/Users/Yuki/TKNT/work/roomflow_slide16_edit/tmp";
const slideCount = 33;

const outputSlides = Array.from({ length: slideCount }, (_, index) => {
  const slide = index + 1;
  if (slide !== 16) {
    return {
      outputSlide: slide,
      sourceSlide: slide,
      narrativeRole: `unchanged source slide ${slide}`,
      reuseMode: "duplicate-slide",
      editTargets: [],
    };
  }

  return {
    outputSlide: 16,
    sourceSlide: 16,
    narrativeRole: "constraint-guided image generation pipeline revision",
    reuseMode: "duplicate-slide",
    editTargets: [
      {
        action: "rewrite",
        sourceElementId: "sh/8z2h8bq1",
        reason: "Convert the title from a topic label into a defense-ready takeaway.",
      },
      {
        action: "delete",
        sourceElementId: "im/wza1czap",
        reason: "Remove the dense internal technical diagram from the central panel.",
      },
      {
        action: "rewrite-and-reposition",
        sourceElementId: "sh/lwbyxwra",
        reason: "Reuse the inherited central panel as the simplified pipeline area.",
      },
      {
        action: "replace",
        sourceElementId: "im/x0ji54ra",
        reason: "Replace the generic output icon with generated-image evidence from slide 17.",
      },
      {
        action: "rewrite",
        sourceElementId: "sh/zm50veps",
        reason: "Clarify that the output preserves layout constraints.",
      },
      {
        action: "add",
        newPrimitiveAllowed: true,
        mustNotOverlapInherited: true,
        zone: { left: 250, top: 245, width: 540, height: 230 },
        reason: "Add editable stage cards inside the freed central panel after deleting the technical screenshot.",
      },
    ],
  };
});

await fs.writeFile(
  `${workspace}/template-frame-map.json`,
  `${JSON.stringify({ outputSlides, omittedSourceSlides: [] }, null, 2)}\n`,
);

await fs.writeFile(
  `${workspace}/template-audit.txt`,
  [
    "Source deck: RoomFlow thesis defense, 4:3, 33 slides.",
    "Visual contract: preserve the HUST footer, title bar, page number, palette, and margins.",
    "Slide 16 issue: the center diagram is too dense for a live thesis defense and uses internal implementation terms.",
    "Slide 16 edit plan: keep the input/output structure, replace the central technical diagram with a simplified constraint-guided pipeline, and use the real generated image as output evidence.",
  ].join("\n") + "\n",
);

await fs.writeFile(
  `${workspace}/deviation-log.txt`,
  [
    "Slide 16: title rewritten as a takeaway.",
    "Slide 16: dense central diagram removed and replaced with editable step cards in the same visual region.",
    "Slide 16: generic output icon replaced with real generated image extracted from slide 17.",
    "All other slides: no intended content or layout changes.",
  ].join("\n") + "\n",
);
