import { Point } from '../types';

export function toCanvas(
  point: Point,
  scale: number,
  offset: Point,
  bounds?: { minY: number; maxY: number }
): Point {
  const x = point.x * scale + offset.x;
  if (!bounds) {
    return { x, y: point.y * scale + offset.y };
  }
  const y = (bounds.maxY - point.y) * scale + offset.y;
  return { x, y };
}

export function fromCanvas(
  point: Point,
  scale: number,
  offset: Point,
  bounds?: { minY: number; maxY: number }
): Point {
  const x = (point.x - offset.x) / scale;
  if (!bounds) {
    return { x, y: (point.y - offset.y) / scale };
  }
  const y = bounds.maxY - (point.y - offset.y) / scale;
  return { x, y };
}

export function polygonBounds(points: Point[]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  points.forEach((p) => {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  });
  return { minX, minY, maxX, maxY };
}

export function normalizePoints(points: Point[]): { points: Point[]; dx: number; dy: number } {
  if (points.length === 0) return { points, dx: 0, dy: 0 };
  const { minX, minY } = polygonBounds(points);
  if (!isFinite(minX) || !isFinite(minY)) return { points, dx: 0, dy: 0 };
  if (Math.abs(minX) < 1e-6 && Math.abs(minY) < 1e-6) {
    return { points, dx: 0, dy: 0 };
  }
  return {
    points: points.map((p) => ({ x: p.x - minX, y: p.y - minY })),
    dx: minX,
    dy: minY,
  };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
