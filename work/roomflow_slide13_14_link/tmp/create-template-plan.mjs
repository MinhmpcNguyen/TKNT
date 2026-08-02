import fs from "node:fs/promises";

const workspace = "/Users/Yuki/TKNT/work/roomflow_slide13_14_link/tmp";
const slideCount = 33;

const editedTargets = {
  13: [
    { action: "rewrite", shapeId: "2", reason: "Retitle the slide around the handoff to placement." },
    { action: "rewrite", shapeId: "7", reason: "Clarify the first stage in the spoken flow." },
    { action: "rewrite", shapeId: "8", reason: "Shorten the user brief so the input is easier to scan." },
    { action: "rewrite", shapeId: "11", reason: "Clarify that the AI extracts requirements." },
    { action: "rewrite", shapeId: "12", reason: "Show the parsed fields that matter downstream." },
    { action: "rewrite", shapeId: "15", reason: "Name the output as placement-ready." },
    { action: "rewrite", shapeId: "16", reason: "Explain the validation before placement." },
    { action: "rewrite-and-reposition", shapeId: "4", reason: "Make the output plan the main visual focus." },
    { action: "rewrite-and-reposition", shapeId: "37", reason: "Rename the plan card and keep hierarchy readable." },
    { action: "rewrite-and-reposition", shapeId: "39", reason: "List required fields for placement." },
    { action: "rewrite-and-reposition", shapeId: "40", reason: "List optional fields for placement." },
    { action: "rewrite-and-reposition", shapeId: "41", reason: "List relation fields for placement." },
    { action: "rewrite-and-reposition", shapeId: "42", reason: "List preference fields for placement." },
    { action: "rewrite-and-reposition", shapeId: "25", reason: "Use the second card as a downstream-use card." },
    { action: "rewrite-and-reposition", shapeId: "26", reason: "Rename the downstream-use card." },
    { action: "rewrite-and-reposition", shapeId: "27", reason: "Keep checklist marker aligned with new content." },
    { action: "rewrite-and-reposition", shapeId: "28", reason: "Explain what placement does with the plan." },
    { action: "rewrite-and-reposition", shapeId: "29", reason: "Keep checklist marker aligned with new content." },
    { action: "rewrite-and-reposition", shapeId: "30", reason: "Explain what placement does with the plan." },
    { action: "rewrite-and-reposition", shapeId: "31", reason: "Keep checklist marker aligned with new content." },
    { action: "rewrite-and-reposition", shapeId: "32", reason: "Explain what placement does with the plan." },
    { action: "rewrite-and-reposition", shapeId: "33", reason: "Keep checklist marker aligned with new content." },
    { action: "rewrite-and-reposition", shapeId: "34", reason: "Explain what placement does with the plan." },
    { action: "rewrite", shapeId: "36", reason: "Make the slide-to-slide handoff explicit." }
  ],
  14: [
    { action: "rewrite", shapeId: "2", reason: "Retitle the slide around the checked plan input." },
    { action: "rewrite", shapeId: "7", reason: "Connect the first placement step to the checked plan." },
    { action: "rewrite", shapeId: "15", reason: "Clarify candidate spots are derived from plan fields." },
    { action: "rewrite", shapeId: "18", reason: "Tie orientation search to allowed directions." },
    { action: "rewrite", shapeId: "24", reason: "Clarify each candidate is tested in several directions." },
    { action: "rewrite", shapeId: "27", reason: "Use the same required/optional wording as slide 13." },
    { action: "rewrite", shapeId: "35", reason: "Clarify validation of conflicts." },
    { action: "rewrite", shapeId: "38", reason: "Connect related furniture to the relation fields." },
    { action: "rewrite", shapeId: "45", reason: "Clarify relation-based placement." },
    { action: "rewrite", shapeId: "48", reason: "Clarify local repair step." },
    { action: "rewrite", shapeId: "56", reason: "Clarify small shifts happen inside constraints." },
    { action: "rewrite", shapeId: "59", reason: "Clarify optional objects come after required layout." },
    { action: "rewrite", shapeId: "68", reason: "Use required/optional wording from the checked plan." },
    { action: "rewrite", shapeId: "71", reason: "Clarify scoring uses preferences." },
    { action: "rewrite", shapeId: "81", reason: "Clarify score dimensions." },
    { action: "rewrite", shapeId: "84", reason: "Clarify final output." },
    { action: "rewrite", shapeId: "91", reason: "Clarify final result contains ranked layouts." },
    { action: "rewrite", shapeId: "100", reason: "Make the bottom summary start from the checked plan." }
  ]
};

const outputSlides = Array.from({ length: slideCount }, (_, index) => {
  const slide = index + 1;
  return {
    outputSlide: slide,
    sourceSlide: slide,
    narrativeRole:
      slide === 13
        ? "content bridge from AI planning to placement"
        : slide === 14
          ? "content continuation showing placement search"
          : "preserve-only chrome source slide",
    reuseMode: "duplicate-slide",
    editTargets: editedTargets[slide] ?? []
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
    "Source deck: RoomFlow thesis defense, 4:3 deck with HUST footer/chrome.",
    "Visual route: template-following edit of the latest revised PPTX.",
    "Typography and chrome: preserve the source title bar, footer, page markers, rounded cards, and existing palette.",
    "Slide 13 issue: the previous slide had many equal-weight boxes, so the downstream handoff was implicit.",
    "Slide 13 edit: reuse existing cards to show brief -> parsed needs -> validated plan, then emphasize a placement-ready plan and what placement consumes.",
    "Slide 14 edit: preserve the eight-step placement-search layout and rewrite text so step 1 clearly consumes the checked plan.",
    "No new visual primitives are required; all visible changes rewrite or reposition inherited objects.",
    ""
  ].join("\n"),
  "utf8"
);

await fs.writeFile(
  `${workspace}/deviation-log.txt`,
  [
    "Slide 13: moved the two lower cards into a stronger main-output + downstream-use hierarchy while keeping source card style.",
    "Slide 13: changed the takeaway to explicitly link to placement search.",
    "Slide 14: changed title, first step, captions, and bottom summary to make the checked-plan input explicit.",
    ""
  ].join("\n"),
  "utf8"
);

console.log(`${workspace}/template-frame-map.json`);
