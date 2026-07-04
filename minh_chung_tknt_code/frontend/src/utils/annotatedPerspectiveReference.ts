import {
  type SnapshotOpeningRecord,
  type ThreeSnapshotArtifact,
  type VisibleSnapshotObject,
} from './threeSnapshot';

const GUIDE_COLORS = [
  '#f97316',
  '#2563eb',
  '#16a34a',
  '#dc2626',
  '#9333ea',
  '#0891b2',
  '#ca8a04',
  '#db2777',
];
const TV_CONSOLE_RENDER_IDENTITY_LABEL = 'TV console with integrated flat-screen TV';

type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type LabelPlacement = {
  color: string;
  label: string;
  objectBox: Rect;
  labelBox: Rect;
  anchor: {
    x: number;
    y: number;
  };
};

export async function buildStructuralPerspectiveReferenceImageDataUrl(params: {
  sourceImageDataUrl: string;
  snapshot: ThreeSnapshotArtifact;
}): Promise<string | null> {
  if (typeof document === 'undefined') {
    return null;
  }

  const sourceImage = await loadImage(params.sourceImageDataUrl);
  const sourceWidth = Math.max(1, sourceImage.naturalWidth || sourceImage.width);
  const sourceHeight = Math.max(1, sourceImage.naturalHeight || sourceImage.height);
  const canvas = document.createElement('canvas');
  canvas.width = sourceWidth;
  canvas.height = sourceHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    return null;
  }

  context.save();
  context.filter = 'contrast(1.22) saturate(1.08) brightness(1.03)';
  context.drawImage(sourceImage, 0, 0, sourceWidth, sourceHeight);
  context.restore();
  const sourcePixels = context.getImageData(0, 0, sourceWidth, sourceHeight);

  context.save();
  context.fillStyle = 'rgba(255, 255, 255, 0.14)';
  context.fillRect(0, 0, sourceWidth, sourceHeight);
  context.restore();

  drawEdgeHighlights(context, sourcePixels, sourceWidth, sourceHeight);

  const scaleX = sourceWidth / Math.max(1, params.snapshot.canvas.widthPx);
  const scaleY = sourceHeight / Math.max(1, params.snapshot.canvas.heightPx);
  params.snapshot.visibleObjects.forEach((object, index) => {
    drawObjectStructuralOutline(context, {
      object,
      index,
      scaleX,
      scaleY,
      canvasWidth: sourceWidth,
      canvasHeight: sourceHeight,
    });
  });
  const openingItems = snapshotOpenings(params.snapshot);
  openingItems.forEach((opening, index) => {
    drawOpeningStructuralOutline(context, {
      opening,
      index: params.snapshot.visibleObjects.length + index,
      scaleX,
      scaleY,
      canvasWidth: sourceWidth,
      canvasHeight: sourceHeight,
    });
  });

  drawObjectIdentityCallouts(context, {
    objects: params.snapshot.visibleObjects,
    openings: openingItems,
    scaleX,
    scaleY,
    canvasWidth: sourceWidth,
    canvasHeight: sourceHeight,
  });

  return canvas.toDataURL('image/png');
}

export async function buildEdgeMapReferenceImageDataUrl(params: {
  sourceImageDataUrl: string;
}): Promise<string | null> {
  if (typeof document === 'undefined') {
    return null;
  }

  const sourceImage = await loadImage(params.sourceImageDataUrl);
  const width = Math.max(1, sourceImage.naturalWidth || sourceImage.width);
  const height = Math.max(1, sourceImage.naturalHeight || sourceImage.height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    return null;
  }

  context.drawImage(sourceImage, 0, 0, width, height);
  const sourcePixels = context.getImageData(0, 0, width, height);
  const edgePixels = context.createImageData(width, height);
  const source = sourcePixels.data;
  const target = edgePixels.data;
  const threshold = 18;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const topLeft = luminanceAt(source, width, x - 1, y - 1);
      const top = luminanceAt(source, width, x, y - 1);
      const topRight = luminanceAt(source, width, x + 1, y - 1);
      const left = luminanceAt(source, width, x - 1, y);
      const right = luminanceAt(source, width, x + 1, y);
      const bottomLeft = luminanceAt(source, width, x - 1, y + 1);
      const bottom = luminanceAt(source, width, x, y + 1);
      const bottomRight = luminanceAt(source, width, x + 1, y + 1);

      const gx =
        -topLeft -
        2 * left -
        bottomLeft +
        topRight +
        2 * right +
        bottomRight;
      const gy =
        -topLeft -
        2 * top -
        topRight +
        bottomLeft +
        2 * bottom +
        bottomRight;
      const magnitude = Math.hypot(gx, gy);
      const offset = (y * width + x) * 4;
      const value = magnitude >= threshold ? 255 : 0;
      target[offset] = value;
      target[offset + 1] = value;
      target[offset + 2] = value;
      target[offset + 3] = 255;
    }
  }

  context.fillStyle = '#000000';
  context.fillRect(0, 0, width, height);
  context.putImageData(edgePixels, 0, 0);
  return canvas.toDataURL('image/png');
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load perspective guide image.'));
    image.src = dataUrl;
  });
}

function drawEdgeHighlights(
  context: CanvasRenderingContext2D,
  sourcePixels: ImageData,
  width: number,
  height: number
): void {
  if (width < 3 || height < 3) {
    return;
  }

  const source = sourcePixels.data;
  const edgePixels = context.createImageData(width, height);
  const target = edgePixels.data;
  const threshold = 16;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const topLeft = luminanceAt(source, width, x - 1, y - 1);
      const top = luminanceAt(source, width, x, y - 1);
      const topRight = luminanceAt(source, width, x + 1, y - 1);
      const left = luminanceAt(source, width, x - 1, y);
      const right = luminanceAt(source, width, x + 1, y);
      const bottomLeft = luminanceAt(source, width, x - 1, y + 1);
      const bottom = luminanceAt(source, width, x, y + 1);
      const bottomRight = luminanceAt(source, width, x + 1, y + 1);

      const gx =
        -topLeft -
        2 * left -
        bottomLeft +
        topRight +
        2 * right +
        bottomRight;
      const gy =
        -topLeft -
        2 * top -
        topRight +
        bottomLeft +
        2 * bottom +
        bottomRight;
      const magnitude = Math.hypot(gx, gy);
      if (magnitude < threshold) {
        continue;
      }

      const offset = (y * width + x) * 4;
      const alpha = clamp((magnitude - threshold) * 2.4, 0, 230);
      target[offset] = 12;
      target[offset + 1] = 18;
      target[offset + 2] = 28;
      target[offset + 3] = alpha;
    }
  }

  const edgeCanvas = document.createElement('canvas');
  edgeCanvas.width = width;
  edgeCanvas.height = height;
  const edgeContext = edgeCanvas.getContext('2d');
  if (!edgeContext) {
    return;
  }
  edgeContext.putImageData(edgePixels, 0, 0);
  context.drawImage(edgeCanvas, 0, 0);
  context.save();
  context.globalAlpha = 0.38;
  context.drawImage(edgeCanvas, 1, 0);
  context.drawImage(edgeCanvas, 0, 1);
  context.restore();
}

function drawObjectStructuralOutline(
  context: CanvasRenderingContext2D,
  params: {
    object: VisibleSnapshotObject;
    index: number;
    scaleX: number;
    scaleY: number;
    canvasWidth: number;
    canvasHeight: number;
  }
): void {
  const { object, index, scaleX, scaleY, canvasWidth, canvasHeight } = params;
  const box = object.screenBboxPx;
  if (!box) {
    return;
  }

  const color = GUIDE_COLORS[index % GUIDE_COLORS.length];
  const x = clamp(box.minX * scaleX, 0, canvasWidth);
  const y = clamp(box.minY * scaleY, 0, canvasHeight);
  const width = clamp(box.width * scaleX, 8, canvasWidth - x);
  const height = clamp(box.height * scaleY, 8, canvasHeight - y);
  const lineWidth = Math.max(2, Math.min(canvasWidth, canvasHeight) * 0.003);

  context.save();
  context.fillStyle = withAlpha(color, 0.1);
  context.fillRect(x, y, width, height);
  context.strokeStyle = 'rgba(15, 23, 42, 0.78)';
  context.lineWidth = Math.max(2, lineWidth * 1.45);
  context.strokeRect(x, y, width, height);
  context.strokeStyle = withAlpha(color, 0.88);
  context.lineWidth = lineWidth;
  context.setLineDash([lineWidth * 4, lineWidth * 2]);
  context.strokeRect(x, y, width, height);
  context.setLineDash([]);
  context.strokeStyle = 'rgba(255, 255, 255, 0.72)';
  context.lineWidth = Math.max(1, lineWidth * 0.45);
  context.strokeRect(x + lineWidth, y + lineWidth, width - lineWidth * 2, height - lineWidth * 2);
  context.restore();
}

function drawOpeningStructuralOutline(
  context: CanvasRenderingContext2D,
  params: {
    opening: SnapshotOpeningRecord;
    index: number;
    scaleX: number;
    scaleY: number;
    canvasWidth: number;
    canvasHeight: number;
  }
): void {
  const { opening, index, scaleX, scaleY, canvasWidth, canvasHeight } = params;
  const box = scaledOpeningRect(opening, scaleX, scaleY, canvasWidth, canvasHeight);
  if (!box) {
    return;
  }

  const color = GUIDE_COLORS[index % GUIDE_COLORS.length];
  const lineWidth = Math.max(2, Math.min(canvasWidth, canvasHeight) * 0.0035);

  context.save();
  context.fillStyle = withAlpha(color, 0.12);
  context.fillRect(box.x, box.y, box.width, box.height);
  context.strokeStyle = 'rgba(2, 6, 23, 0.86)';
  context.lineWidth = Math.max(2, lineWidth * 1.6);
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.strokeStyle = withAlpha(color, 0.92);
  context.lineWidth = lineWidth;
  context.setLineDash([lineWidth * 3, lineWidth * 1.5]);
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.setLineDash([]);
  context.restore();
}

function drawObjectIdentityCallouts(
  context: CanvasRenderingContext2D,
  params: {
    objects: VisibleSnapshotObject[];
    openings?: SnapshotOpeningRecord[];
    scaleX: number;
    scaleY: number;
    canvasWidth: number;
    canvasHeight: number;
  }
): void {
  const { objects, openings = [], scaleX, scaleY, canvasWidth, canvasHeight } = params;
  if (objects.length === 0 && openings.length === 0) {
    return;
  }

  context.save();
  const objectBoxes = [
    ...objects.map((object) =>
      scaledObjectRect(object, scaleX, scaleY, canvasWidth, canvasHeight)
    ),
    ...openings.map((opening) =>
      scaledOpeningRect(opening, scaleX, scaleY, canvasWidth, canvasHeight)
    ),
  ]
    .filter((box): box is Rect => box !== null);
  const placements: LabelPlacement[] = [];

  const lockItems = [
    ...objects.map((object, index) => ({
      box: scaledObjectRect(object, scaleX, scaleY, canvasWidth, canvasHeight),
      color: GUIDE_COLORS[index % GUIDE_COLORS.length],
      label: `${String(index + 1).padStart(2, '0')} ${formatObjectLabel(object)}`,
    })),
    ...openings.map((opening, index) => ({
      box: scaledOpeningRect(opening, scaleX, scaleY, canvasWidth, canvasHeight),
      color: GUIDE_COLORS[(objects.length + index) % GUIDE_COLORS.length],
      label: `${String(objects.length + index + 1).padStart(2, '0')} ${formatOpeningLabel(opening)}`,
    })),
  ];

  lockItems.forEach((item) => {
    const objectBox = item.box;
    if (!objectBox) {
      return;
    }
    const color = item.color;
    const label = item.label;
    const labelBox = measureCalloutLabel(context, {
      label,
      canvasWidth,
      canvasHeight,
    });
    const placement = placeCalloutLabel({
      color,
      label,
      objectBox,
      labelBox,
      objectBoxes,
      placedLabels: placements.map((item) => item.labelBox),
      canvasWidth,
      canvasHeight,
    });
    placements.push(placement);
  });

  placements.forEach((placement) => {
    drawCalloutLeader(context, placement);
  });
  placements.forEach((placement) => {
    drawCalloutLabel(context, placement);
  });
  context.restore();
}

function snapshotOpenings(snapshot: ThreeSnapshotArtifact): SnapshotOpeningRecord[] {
  return [
    ...snapshot.room.openings.doors,
    ...snapshot.room.openings.windows,
  ].filter((opening) => Boolean(opening.screenBboxPx));
}

function scaledObjectRect(
  object: VisibleSnapshotObject,
  scaleX: number,
  scaleY: number,
  canvasWidth: number,
  canvasHeight: number
): Rect | null {
  const box = object.screenBboxPx;
  if (!box) {
    return null;
  }
  const x = clamp(box.minX * scaleX, 0, canvasWidth);
  const y = clamp(box.minY * scaleY, 0, canvasHeight);
  const width = clamp(box.width * scaleX, 8, canvasWidth - x);
  const height = clamp(box.height * scaleY, 8, canvasHeight - y);
  return { x, y, width, height };
}

function scaledOpeningRect(
  opening: SnapshotOpeningRecord,
  scaleX: number,
  scaleY: number,
  canvasWidth: number,
  canvasHeight: number
): Rect | null {
  const box = opening.screenBboxPx;
  if (!box) {
    return null;
  }
  const x = clamp(box.minX * scaleX, 0, canvasWidth);
  const y = clamp(box.minY * scaleY, 0, canvasHeight);
  const width = clamp(box.width * scaleX, 8, canvasWidth - x);
  const height = clamp(box.height * scaleY, 8, canvasHeight - y);
  return { x, y, width, height };
}

function measureCalloutLabel(
  context: CanvasRenderingContext2D,
  params: {
    label: string;
    canvasWidth: number;
    canvasHeight: number;
  }
): Rect {
  const { label, canvasWidth, canvasHeight } = params;
  const fontSize = resolveCalloutFontSize(canvasWidth, canvasHeight);
  const paddingX = Math.round(fontSize * 0.55);
  const paddingY = Math.round(fontSize * 0.32);
  const maxWidth = Math.max(132, Math.min(360, canvasWidth * 0.22));

  context.font = `800 ${fontSize}px sans-serif`;
  const measuredWidth = Math.ceil(context.measureText(label).width + paddingX * 2);
  return {
    x: 0,
    y: 0,
    width: Math.min(maxWidth, measuredWidth),
    height: fontSize + paddingY * 2,
  };
}

function placeCalloutLabel(params: {
  color: string;
  label: string;
  objectBox: Rect;
  labelBox: Rect;
  objectBoxes: Rect[];
  placedLabels: Rect[];
  canvasWidth: number;
  canvasHeight: number;
}): LabelPlacement {
  const {
    color,
    label,
    objectBox,
    labelBox,
    objectBoxes,
    placedLabels,
    canvasWidth,
    canvasHeight,
  } = params;
  const margin = Math.max(10, Math.min(canvasWidth, canvasHeight) * 0.018);
  const candidates = buildLabelCandidates({
    objectBox,
    labelBox,
    margin,
    canvasWidth,
    canvasHeight,
  });
  const best = candidates
    .map((box) => ({
      box,
      score: scoreLabelCandidate({
        candidate: box,
        objectBox,
        objectBoxes,
        placedLabels,
      }),
    }))
    .sort((left, right) => left.score - right.score)[0]?.box ?? candidates[0];

  return {
    color,
    label,
    objectBox,
    labelBox: best,
    anchor: nearestPointOnRect(best, rectCenter(objectBox)),
  };
}

function buildLabelCandidates(params: {
  objectBox: Rect;
  labelBox: Rect;
  margin: number;
  canvasWidth: number;
  canvasHeight: number;
}): Rect[] {
  const { objectBox, labelBox, margin, canvasWidth, canvasHeight } = params;
  const center = rectCenter(objectBox);
  const positions = [
    {
      x: objectBox.x + objectBox.width + margin,
      y: center.y - labelBox.height / 2,
    },
    {
      x: objectBox.x - labelBox.width - margin,
      y: center.y - labelBox.height / 2,
    },
    {
      x: center.x - labelBox.width / 2,
      y: objectBox.y - labelBox.height - margin,
    },
    {
      x: center.x - labelBox.width / 2,
      y: objectBox.y + objectBox.height + margin,
    },
    {
      x: objectBox.x + objectBox.width + margin,
      y: objectBox.y - labelBox.height - margin,
    },
    {
      x: objectBox.x - labelBox.width - margin,
      y: objectBox.y - labelBox.height - margin,
    },
    {
      x: objectBox.x + objectBox.width + margin,
      y: objectBox.y + objectBox.height + margin,
    },
    {
      x: objectBox.x - labelBox.width - margin,
      y: objectBox.y + objectBox.height + margin,
    },
  ];
  return positions.map((position) => ({
    x: clamp(
      position.x,
      margin,
      Math.max(margin, canvasWidth - labelBox.width - margin)
    ),
    y: clamp(
      position.y,
      margin,
      Math.max(margin, canvasHeight - labelBox.height - margin)
    ),
    width: labelBox.width,
    height: labelBox.height,
  }));
}

function scoreLabelCandidate(params: {
  candidate: Rect;
  objectBox: Rect;
  objectBoxes: Rect[];
  placedLabels: Rect[];
}): number {
  const { candidate, objectBox, objectBoxes, placedLabels } = params;
  const center = rectCenter(objectBox);
  const labelCenter = rectCenter(candidate);
  const distance = Math.hypot(labelCenter.x - center.x, labelCenter.y - center.y);
  const objectOverlap = objectBoxes.reduce(
    (sum, box) => sum + rectOverlapArea(candidate, expandRect(box, 3)),
    0
  );
  const labelOverlap = placedLabels.reduce(
    (sum, box) => sum + rectOverlapArea(candidate, expandRect(box, 5)),
    0
  );
  const selfOverlap = rectOverlapArea(candidate, expandRect(objectBox, 4));
  return distance + objectOverlap * 120 + labelOverlap * 180 + selfOverlap * 220;
}

function drawCalloutLeader(
  context: CanvasRenderingContext2D,
  placement: LabelPlacement
): void {
  const objectCenter = rectCenter(placement.objectBox);
  const objectEdge = nearestPointOnRect(placement.objectBox, placement.anchor);
  context.save();
  context.strokeStyle = placement.color;
  context.lineWidth = 3;
  context.shadowColor = 'rgba(255, 255, 255, 0.9)';
  context.shadowBlur = 4;
  context.beginPath();
  context.moveTo(objectEdge.x, objectEdge.y);
  context.lineTo(placement.anchor.x, placement.anchor.y);
  context.stroke();
  context.fillStyle = placement.color;
  context.beginPath();
  context.arc(objectCenter.x, objectCenter.y, 4, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawCalloutLabel(
  context: CanvasRenderingContext2D,
  placement: LabelPlacement
): void {
  const fontSize = resolveCalloutFontSize(
    context.canvas.width,
    context.canvas.height
  );
  const paddingX = Math.round(fontSize * 0.55);
  const paddingY = Math.round(fontSize * 0.32);
  const swatchSize = Math.max(10, Math.round(fontSize * 0.62));
  const textX = placement.labelBox.x + paddingX + swatchSize + 7;
  const maxTextWidth = Math.max(
    12,
    placement.labelBox.width - paddingX * 2 - swatchSize - 7
  );

  context.save();
  context.shadowColor = 'rgba(15, 23, 42, 0.28)';
  context.shadowBlur = 8;
  context.fillStyle = 'rgba(255, 255, 255, 0.97)';
  roundRect(context, placement.labelBox, 7);
  context.fill();
  context.shadowBlur = 0;
  context.strokeStyle = placement.color;
  context.lineWidth = 3;
  roundRect(context, placement.labelBox, 7);
  context.stroke();

  context.fillStyle = placement.color;
  context.fillRect(
    placement.labelBox.x + paddingX,
    placement.labelBox.y + (placement.labelBox.height - swatchSize) / 2,
    swatchSize,
    swatchSize
  );

  context.fillStyle = '#0f172a';
  context.font = `800 ${fontSize}px sans-serif`;
  drawEllipsizedText(
    context,
    placement.label,
    textX,
    placement.labelBox.y + paddingY + fontSize - 1,
    maxTextWidth
  );
  context.restore();
}

function resolveCalloutFontSize(canvasWidth: number, canvasHeight: number): number {
  return clamp(Math.round(Math.min(canvasWidth, canvasHeight) * 0.034), 18, 38);
}

function rectCenter(rect: Rect): { x: number; y: number } {
  return {
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
  };
}

function nearestPointOnRect(
  rect: Rect,
  point: {
    x: number;
    y: number;
  }
): { x: number; y: number } {
  return {
    x: clamp(point.x, rect.x, rect.x + rect.width),
    y: clamp(point.y, rect.y, rect.y + rect.height),
  };
}

function rectOverlapArea(left: Rect, right: Rect): number {
  const xOverlap = Math.max(
    0,
    Math.min(left.x + left.width, right.x + right.width) - Math.max(left.x, right.x)
  );
  const yOverlap = Math.max(
    0,
    Math.min(left.y + left.height, right.y + right.height) - Math.max(left.y, right.y)
  );
  return xOverlap * yOverlap;
}

function expandRect(rect: Rect, amount: number): Rect {
  return {
    x: rect.x - amount,
    y: rect.y - amount,
    width: rect.width + amount * 2,
    height: rect.height + amount * 2,
  };
}

function roundRect(
  context: CanvasRenderingContext2D,
  rect: Rect,
  radius: number
): void {
  const resolvedRadius = Math.min(radius, rect.width / 2, rect.height / 2);
  context.beginPath();
  context.moveTo(rect.x + resolvedRadius, rect.y);
  context.lineTo(rect.x + rect.width - resolvedRadius, rect.y);
  context.quadraticCurveTo(
    rect.x + rect.width,
    rect.y,
    rect.x + rect.width,
    rect.y + resolvedRadius
  );
  context.lineTo(rect.x + rect.width, rect.y + rect.height - resolvedRadius);
  context.quadraticCurveTo(
    rect.x + rect.width,
    rect.y + rect.height,
    rect.x + rect.width - resolvedRadius,
    rect.y + rect.height
  );
  context.lineTo(rect.x + resolvedRadius, rect.y + rect.height);
  context.quadraticCurveTo(
    rect.x,
    rect.y + rect.height,
    rect.x,
    rect.y + rect.height - resolvedRadius
  );
  context.lineTo(rect.x, rect.y + resolvedRadius);
  context.quadraticCurveTo(rect.x, rect.y, rect.x + resolvedRadius, rect.y);
  context.closePath();
}

function drawEllipsizedText(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number
): void {
  if (context.measureText(text).width <= maxWidth) {
    context.fillText(text, x, y);
    return;
  }

  let nextText = text;
  while (
    nextText.length > 3 &&
    context.measureText(`${nextText}...`).width > maxWidth
  ) {
    nextText = nextText.slice(0, -1);
  }
  context.fillText(`${nextText}...`, x, y);
}

function luminanceAt(
  data: Uint8ClampedArray,
  width: number,
  x: number,
  y: number
): number {
  const offset = (y * width + x) * 4;
  return data[offset] * 0.2126 + data[offset + 1] * 0.7152 + data[offset + 2] * 0.0722;
}

function formatObjectLabel(object: VisibleSnapshotObject): string {
  if (object.renderIdentityLabel?.trim()) {
    return object.renderIdentityLabel.trim();
  }
  if (object.type.trim().toLowerCase() === 'tv_console') {
    return TV_CONSOLE_RENDER_IDENTITY_LABEL;
  }
  const rawLabel =
    object.assetId || object.rawAssetId || object.type || 'object';
  return rawLabel.replace(/[_-]+/g, ' ').trim() || 'object';
}

function formatOpeningLabel(opening: SnapshotOpeningRecord): string {
  const normalizedId = opening.id.replace(/[_-]+/g, ' ').trim();
  if (/window/i.test(opening.id)) {
    return normalizedId || 'window';
  }
  if (/door/i.test(opening.id)) {
    return normalizedId || 'door';
  }
  return normalizedId || 'opening';
}

function withAlpha(hexColor: string, alpha: number): string {
  const hex = hexColor.replace('#', '');
  if (hex.length !== 6) {
    return `rgba(249, 115, 22, ${alpha})`;
  }
  const red = Number.parseInt(hex.slice(0, 2), 16);
  const green = Number.parseInt(hex.slice(2, 4), 16);
  const blue = Number.parseInt(hex.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
