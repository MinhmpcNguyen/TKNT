import {
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

export async function buildAnnotatedPerspectiveReferenceImageDataUrl(params: {
  sourceImageDataUrl: string;
  snapshot: ThreeSnapshotArtifact;
}): Promise<string | null> {
  if (typeof document === 'undefined') {
    return null;
  }

  const sourceImage = await loadImage(params.sourceImageDataUrl);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, sourceImage.naturalWidth || sourceImage.width);
  canvas.height = Math.max(1, sourceImage.naturalHeight || sourceImage.height);
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  context.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
  context.fillStyle = 'rgba(0, 0, 0, 0.16)';
  context.fillRect(0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / Math.max(1, params.snapshot.canvas.widthPx);
  const scaleY = canvas.height / Math.max(1, params.snapshot.canvas.heightPx);
  params.snapshot.visibleObjects.forEach((object, index) => {
    drawObjectAnnotation(context, {
      object,
      index,
      scaleX,
      scaleY,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
    });
  });

  drawGuideHeader(context, params.snapshot.visibleObjects.length);
  return canvas.toDataURL('image/png');
}

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

  drawObjectIdentityCallouts(context, {
    objects: params.snapshot.visibleObjects,
    scaleX,
    scaleY,
    canvasWidth: sourceWidth,
    canvasHeight: sourceHeight,
  });

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

function drawObjectAnnotation(
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
  const color = GUIDE_COLORS[index % GUIDE_COLORS.length];
  const label = `${String(index + 1).padStart(2, '0')} ${formatObjectLabel(object)}`;
  const box = object.screenBboxPx;
  const center = object.screenCenterPx;

  const centerX = clamp(
    (center?.x ?? (box ? box.minX + box.width / 2 : canvasWidth / 2)) * scaleX,
    0,
    canvasWidth
  );
  const centerY = clamp(
    (center?.y ?? (box ? box.minY + box.height / 2 : canvasHeight / 2)) * scaleY,
    0,
    canvasHeight
  );

  context.save();
  context.lineWidth = Math.max(3, Math.min(canvasWidth, canvasHeight) * 0.004);
  context.strokeStyle = color;
  context.fillStyle = withAlpha(color, 0.14);

  if (box) {
    const x = clamp(box.minX * scaleX, 0, canvasWidth);
    const y = clamp(box.minY * scaleY, 0, canvasHeight);
    const width = clamp(box.width * scaleX, 8, canvasWidth - x);
    const height = clamp(box.height * scaleY, 8, canvasHeight - y);
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
    drawLabel(context, label, color, x, Math.max(4, y - 28), canvasWidth);
  } else {
    context.beginPath();
    context.arc(centerX, centerY, 12, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    drawLabel(context, label, color, centerX + 14, centerY - 14, canvasWidth);
  }

  context.restore();
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

function drawObjectIdentityCallouts(
  context: CanvasRenderingContext2D,
  params: {
    objects: VisibleSnapshotObject[];
    scaleX: number;
    scaleY: number;
    canvasWidth: number;
    canvasHeight: number;
  }
): void {
  const { objects, scaleX, scaleY, canvasWidth, canvasHeight } = params;
  if (objects.length === 0) {
    return;
  }

  context.save();
  const objectBoxes = objects
    .map((object) => scaledObjectRect(object, scaleX, scaleY, canvasWidth, canvasHeight))
    .filter((box): box is Rect => box !== null);
  const placements: LabelPlacement[] = [];

  objects.forEach((object, index) => {
    const objectBox = scaledObjectRect(object, scaleX, scaleY, canvasWidth, canvasHeight);
    if (!objectBox) {
      return;
    }
    const color = GUIDE_COLORS[index % GUIDE_COLORS.length];
    const label = `${String(index + 1).padStart(2, '0')} ${formatObjectLabel(object)}`;
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

function drawLabel(
  context: CanvasRenderingContext2D,
  label: string,
  color: string,
  x: number,
  y: number,
  canvasWidth: number
): void {
  const fontSize = 22;
  const paddingX = 8;
  const paddingY = 5;
  context.font = `700 ${fontSize}px sans-serif`;
  const width = Math.ceil(context.measureText(label).width + paddingX * 2);
  const height = fontSize + paddingY * 2;
  const labelX = clamp(x, 4, Math.max(4, canvasWidth - width - 4));
  const labelY = Math.max(4, y);

  context.fillStyle = 'rgba(255, 255, 255, 0.94)';
  context.fillRect(labelX, labelY, width, height);
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.strokeRect(labelX, labelY, width, height);
  context.fillStyle = '#111827';
  context.fillText(label, labelX + paddingX, labelY + fontSize + paddingY - 2);
}

function drawGuideHeader(
  context: CanvasRenderingContext2D,
  objectCount: number
): void {
  const label = `Same-camera object guide - ${objectCount} visible objects`;
  context.save();
  context.font = '700 22px sans-serif';
  const width = Math.ceil(context.measureText(label).width + 24);
  context.fillStyle = 'rgba(17, 24, 39, 0.82)';
  context.fillRect(12, 12, width, 38);
  context.fillStyle = '#ffffff';
  context.fillText(label, 24, 38);
  context.restore();
}

function formatObjectLabel(object: VisibleSnapshotObject): string {
  const rawLabel = object.assetId || object.rawAssetId || object.type || 'object';
  return rawLabel.replace(/[_-]+/g, ' ').trim() || 'object';
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
