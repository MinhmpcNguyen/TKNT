import { Footprint, Placement, Point } from '../types';

export type NumericBbox = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

function normalizeRotationDeg(value: number): number {
  const normalized = Number.isFinite(value) ? value % 360 : 0;
  return (normalized + 360) % 360;
}

function usesSwappedFootprint(rotationDeg: number): boolean {
  const normalized = normalizeRotationDeg(rotationDeg);
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
  const centerX = Number(placement.position_mm?.x ?? 0);
  const centerY = Number(placement.position_mm?.y ?? 0);
  const localWidth = Math.max(0, Number(placement.footprint_mm?.w ?? 0));
  const localDepth = Math.max(0, Number(placement.footprint_mm?.d ?? 0));
  const worldWidth = usesSwappedFootprint(Number(placement.rotation_deg ?? 0)) ? localDepth : localWidth;
  const worldDepth = usesSwappedFootprint(Number(placement.rotation_deg ?? 0)) ? localWidth : localDepth;
  return {
    minX: centerX - worldWidth / 2,
    minY: centerY - worldDepth / 2,
    maxX: centerX + worldWidth / 2,
    maxY: centerY + worldDepth / 2,
  };
}

export function placementMatchesStyledObject(
  placement: Pick<Placement, 'position_mm' | 'footprint_mm' | 'rotation_deg'>,
  styledBBox: NumericBbox,
  styledRotationDeg: number
): boolean {
  const styledCenter = getBboxCenter(styledBBox);
  const styledFootprint = getLocalFootprintFromWorldBbox(styledBBox, styledRotationDeg);
  const placementRotation = normalizeRotationDeg(Number(placement.rotation_deg ?? 0));
  const styledRotation = normalizeRotationDeg(styledRotationDeg);
  return (
    almostEqual(Number(placement.position_mm?.x ?? 0), styledCenter.x) &&
    almostEqual(Number(placement.position_mm?.y ?? 0), styledCenter.y) &&
    almostEqual(Number(placement.footprint_mm?.w ?? 0), styledFootprint.w) &&
    almostEqual(Number(placement.footprint_mm?.d ?? 0), styledFootprint.d) &&
    almostEqual(placementRotation, styledRotation)
  );
}
