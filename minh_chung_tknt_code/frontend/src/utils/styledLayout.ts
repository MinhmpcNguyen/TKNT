import { Footprint, Placement, Point } from '../types';
import { normalizePlanRotationDeg } from './planOrientation';

export type NumericBbox = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

function usesSwappedFootprint(rotationDeg: number): boolean {
  const normalized = normalizePlanRotationDeg(rotationDeg);
  return Math.abs(normalized - 90) < 0.5 || Math.abs(normalized - 270) < 0.5;
}

function almostEqual(left: number, right: number, tolerance = 0.5): boolean {
  return Math.abs(left - right) <= tolerance;
}

export function getNumericBbox(value: unknown): NumericBbox {
  const bbox = (value as { bbox?: Record<string, unknown> } | null)?.bbox ?? {};
  return {
    minX: Number((bbox as Record<string, unknown>).min_x ?? (bbox as Record<string, unknown>).minX ?? 0),
    minY: Number((bbox as Record<string, unknown>).min_y ?? (bbox as Record<string, unknown>).minY ?? 0),
    maxX: Number((bbox as Record<string, unknown>).max_x ?? (bbox as Record<string, unknown>).maxX ?? 0),
    maxY: Number((bbox as Record<string, unknown>).max_y ?? (bbox as Record<string, unknown>).maxY ?? 0),
  };
}

export function getBboxCenter(bbox: NumericBbox): Point {
  return {
    x: (bbox.minX + bbox.maxX) / 2,
    y: (bbox.minY + bbox.maxY) / 2,
  };
}

export function getPlacementPolygon(
  placement: Pick<Placement, 'position_mm' | 'footprint_mm' | 'rotation_deg'>
): Point[] {
  const centerX = Number(placement.position_mm?.x ?? 0);
  const centerY = Number(placement.position_mm?.y ?? 0);
  const halfWidth = Math.max(0, Number(placement.footprint_mm?.w ?? 0)) / 2;
  const halfDepth = Math.max(0, Number(placement.footprint_mm?.d ?? 0)) / 2;
  const radians = (normalizePlanRotationDeg(Number(placement.rotation_deg ?? 0)) * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return [
    { x: -halfWidth, y: -halfDepth },
    { x: halfWidth, y: -halfDepth },
    { x: halfWidth, y: halfDepth },
    { x: -halfWidth, y: halfDepth },
  ].map((point) => ({
    x: centerX + point.x * cos - point.y * sin,
    y: centerY + point.x * sin + point.y * cos,
  }));
}

function getBboxFromPoints(points: Point[]): NumericBbox {
  if (points.length === 0) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, Number(point.x ?? 0));
    minY = Math.min(minY, Number(point.y ?? 0));
    maxX = Math.max(maxX, Number(point.x ?? 0));
    maxY = Math.max(maxY, Number(point.y ?? 0));
  }
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  }
  return { minX, minY, maxX, maxY };
}

export function getLocalFootprintFromWorldBbox(
  bbox: NumericBbox,
  rotationDeg: number
): Footprint {
  const worldWidth = Math.max(0, bbox.maxX - bbox.minX);
  const worldDepth = Math.max(0, bbox.maxY - bbox.minY);
  if (usesSwappedFootprint(rotationDeg)) {
    return { w: worldDepth, d: worldWidth };
  }
  return { w: worldWidth, d: worldDepth };
}

export function getWorldBboxFromPlacement(
  placement: Pick<Placement, 'position_mm' | 'footprint_mm' | 'rotation_deg'>
): NumericBbox {
  return getBboxFromPoints(getPlacementPolygon(placement));
}

export function placementMatchesStyledObject(
  placement: Pick<Placement, 'position_mm' | 'footprint_mm' | 'rotation_deg'>,
  styledBBox: NumericBbox,
  styledRotationDeg: number
): boolean {
  const styledCenter = getBboxCenter(styledBBox);
  const placementBbox = getWorldBboxFromPlacement(placement);
  const placementRotation = normalizePlanRotationDeg(Number(placement.rotation_deg ?? 0));
  const styledRotation = normalizePlanRotationDeg(styledRotationDeg);
  return (
    almostEqual(Number(placement.position_mm?.x ?? 0), styledCenter.x) &&
    almostEqual(Number(placement.position_mm?.y ?? 0), styledCenter.y) &&
    almostEqual(placementBbox.minX, styledBBox.minX) &&
    almostEqual(placementBbox.minY, styledBBox.minY) &&
    almostEqual(placementBbox.maxX, styledBBox.maxX) &&
    almostEqual(placementBbox.maxY, styledBBox.maxY) &&
    almostEqual(placementRotation, styledRotation)
  );
}
