import fs from "node:fs/promises";

const workspace = "/Users/Yuki/TKNT/work/roomflow_slide13_plan_focus/tmp";
const slideCount = 33;

const slide13Targets = [
  { action: "rewrite-and-reposition", shapeId: "4", reason: "Keep the lower-left card as the plan field block." },
  { action: "rewrite-and-reposition", shapeId: "37", reason: "Clarify the lower-left card as structured plan fields." },
  { action: "rewrite-and-reposition", shapeId: "39", reason: "Explain the required field." },
  { action: "rewrite-and-reposition", shapeId: "40", reason: "Explain the optional field." },
  { action: "rewrite-and-reposition", shapeId: "41", reason: "Explain the relation field." },
  { action: "rewrite-and-reposition", shapeId: "42", reason: "Explain the preference field." },
  { action: "rewrite-and-reposition", shapeId: "25", reason: "Reuse the right card for plan validation instead of placement behavior." },
  { action: "rewrite-and-reposition", shapeId: "26", reason: "Replace the placement heading with validation content." },
  { action: "rewrite-and-reposition", shapeId: "27", reason: "Keep an inherited bullet marker for validation checks." },
  { action: "rewrite-and-reposition", shapeId: "28", reason: "Replace placement content with a validation check." },
  { action: "rewrite-and-reposition", shapeId: "29", reason: "Keep an inherited bullet marker for validation checks." },
  { action: "rewrite-and-reposition", shapeId: "30", reason: "Replace placement content with a validation check." },
  { action: "rewrite-and-reposition", shapeId: "31", reason: "Keep an inherited bullet marker for validation checks." },
  { action: "rewrite-and-reposition", shapeId: "32", reason: "Replace placement content with a validation check." },
  { action: "rewrite-and-reposition", shapeId: "33", reason: "Keep an inherited bullet marker for validation checks." },
  { action: "rewrite-and-reposition", shapeId: "34", reason: "Replace placement content with a validation check." },
  { action: "rewrite", shapeId: "36", reason: "Make the slide takeaway focus on checked structured output." }
];

const outputSlides = Array.from({ length: slideCount }, (_, index) => {
  const slide = index + 1;
  return {
    outputSlide: slide,
    sourceSlide: slide,
    narrativeRole: slide === 13 ? "content bridge focused on checked plan output" : "preserve-only chrome source slide",
    reuseMode: "duplicate-slide",
    editTargets: slide === 13 ? slide13Targets : []
  };
});

await fs.writeFile(
  `${workspace}/template-frame-map.json`,
  `${JSON.stringify({ outputSlides, omittedSourceSlides: [] }, null, 2)}\n`,
  "utf8"
);

await fs.writeFile(
  `${workspace}/template-audit.txt`,
  [
    "Source deck: RoomFlow thesis defense deck, edited copy with slides 13-14 linked.",
    "Task: revise slide 13 based on comments.",
    "Requested changes: remove the 'Used by placement' content because slide 14 covers it; make the plan block clearer and more prominent.",
    "Approach: preserve template chrome and the two-card lower layout, but rewrite the right card as plan validation rather than placement behavior.",
    "No new primitives are needed; existing inherited elements are rewritten/repositioned only.",
    ""
  ].join("\n"),
  "utf8"
);

await fs.writeFile(
  `${workspace}/deviation-log.txt`,
  [
    "Slide 13: removed the 'Used by placement' wording and all placement-behavior bullets.",
    "Slide 13: rewrote the lower-left plan block with clearer field descriptions.",
    "Slide 13: repurposed the lower-right card as 'Plan checks' to explain schema/catalog/retry validation.",
    "Slide 13: changed the bottom takeaway to emphasize structured checked output rather than placement details.",
    ""
  ].join("\n"),
  "utf8"
);

console.log(`${workspace}/template-frame-map.json`);
