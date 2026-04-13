import { DesignPlanV1, FloorplanGeometry, Placement, Point } from '../types';
import { polygonBounds } from './geometry';

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 900;
const PADDING = 56;

type CanvasPoint = {
  x: number;
  y: number;
};

export function buildLayoutReferenceImageDataUrl(params: {
  floorplan: FloorplanGeometry | null;
  design: DesignPlanV1 | null;
}): string | null {
  const { floorplan, design } = params;
  if (typeof document === 'undefined' || !floorplan) {
    return null;
  }

  const roomPoints = floorplan.room?.polygon_mm ?? [];
  if (roomPoints.length < 3) {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  const bounds = polygonBounds(roomPoints);
  const roomWidth = Math.max(1, bounds.maxX - bounds.minX);
  const roomHeight = Math.max(1, bounds.maxY - bounds.minY);
  const scale = Math.min(
    (CANVAS_WIDTH - PADDING * 2) / roomWidth,
    (CANVAS_HEIGHT - PADDING * 2) / roomHeight
  );
  const offsetX =
    (CANVAS_WIDTH - roomWidth * scale) / 2 - bounds.minX * scale;
  const offsetY =
    (CANVAS_HEIGHT - roomHeight * scale) / 2 + bounds.maxY * scale;

  const toCanvasPoint = (point: Point): CanvasPoint => ({
    x: point.x * scale + offsetX,
    y: offsetY - point.y * scale,
  });

  context.fillStyle = '#f7f4ed';
  context.fillRect(0, 0, canvas.width, canvas.height);

  const roomPath = new Path2D();
  roomPoints.forEach((point, index) => {
    const canvasPoint = toCanvasPoint(point);
    if (index === 0) {
      roomPath.moveTo(canvasPoint.x, canvasPoint.y);
      return;
    }
    roomPath.lineTo(canvasPoint.x, canvasPoint.y);
  });
  roomPath.closePath();

  context.fillStyle = '#d6d8dd';
  context.fill(roomPath);
  context.lineWidth = 18;
  context.strokeStyle = '#eef1f5';
  context.lineJoin = 'round';
  context.stroke(roomPath);

  drawOpenings(context, floorplan, toCanvasPoint);

  const placements = design?.placements ?? [];
  const sortedPlacements = [...placements].sort(
    (left, right) =>
      right.footprint_mm.w * right.footprint_mm.d -
      left.footprint_mm.w * left.footprint_mm.d
  );
  sortedPlacements.forEach((placement) => {
    drawPlacement(context, placement, toCanvasPoint, scale);
  });

  return canvas.toDataURL('image/png');
}

function drawOpenings(
  context: CanvasRenderingContext2D,
  floorplan: FloorplanGeometry,
  toCanvasPoint: (point: Point) => CanvasPoint
): void {
  context.save();
  context.lineCap = 'round';

  floorplan.openings.windows.forEach((windowRow) => {
    const start = toCanvasPoint(windowRow.start_mm);
    const end = toCanvasPoint(windowRow.end_mm);
    context.strokeStyle = '#ffffff';
    context.lineWidth = 10;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  });

  floorplan.openings.doors.forEach((door) => {
    const start = toCanvasPoint(door.start_mm);
    const end = toCanvasPoint(door.end_mm);
    context.strokeStyle = '#c16363';
    context.lineWidth = 8;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  });

  context.restore();
}

function drawPlacement(
  context: CanvasRenderingContext2D,
  placement: Placement,
  toCanvasPoint: (point: Point) => CanvasPoint,
  scale: number
): void {
  const center = toCanvasPoint(placement.position_mm);
  const width = Math.max(18, placement.footprint_mm.w * scale);
  const depth = Math.max(18, placement.footprint_mm.d * scale);
  const rotationRad = (-placement.rotation_deg * Math.PI) / 180;
  const label = formatPlacementLabel(placement);
  const fillColor = resolvePlacementFillColor(placement);

  context.save();
  context.translate(center.x, center.y);
  context.rotate(rotationRad);

  context.fillStyle = fillColor;
  context.strokeStyle = '#49515d';
  context.lineWidth = Math.max(1.5, Math.min(width, depth) * 0.04);
  drawRoundedRect(context, -width / 2, -depth / 2, width, depth, 12);
  context.fill();
  context.stroke();

  context.strokeStyle = 'rgba(255, 255, 255, 0.95)';
  context.lineWidth = Math.max(2, Math.min(width, depth) * 0.08);
  context.beginPath();
  context.moveTo(-width * 0.18, depth * 0.28);
  context.lineTo(width * 0.18, depth * 0.28);
  context.stroke();

  if (width >= 58 && depth >= 24) {
    const fontSize = clampNumber(Math.min(width, depth) * 0.22, 11, 20);
    context.font = `600 ${fontSize}px sans-serif`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = '#1f232a';
    context.fillText(label, 0, 0, width - 18);
  }

  context.restore();
}

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  const clampedRadius = Math.max(0, Math.min(radius, width / 2, height / 2));
  context.beginPath();
  context.moveTo(x + clampedRadius, y);
  context.arcTo(x + width, y, x + width, y + height, clampedRadius);
  context.arcTo(x + width, y + height, x, y + height, clampedRadius);
  context.arcTo(x, y + height, x, y, clampedRadius);
  context.arcTo(x, y, x + width, y, clampedRadius);
  context.closePath();
}

function formatPlacementLabel(placement: Placement): string {
  const raw = placement.category || placement.asset_id || 'item';
  return raw.replace(/[_-]+/g, ' ').trim();
}

function resolvePlacementFillColor(placement: Placement): string {
  if (isHexColor(placement.color_hex)) {
    return placement.color_hex;
  }
  return '#8b95a4';
}

function isHexColor(value: string | null | undefined): value is string {
  return typeof value === 'string' && /^#(?:[0-9a-fA-F]{3}){1,2}$/.test(value);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
