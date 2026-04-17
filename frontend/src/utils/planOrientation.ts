export function normalizePlanRotationDeg(rotationDeg: number): number {
  return ((rotationDeg % 360) + 360) % 360;
}

export type CardinalSide = 'top' | 'bottom' | 'left' | 'right';
export type DirectionVector = { dx: number; dy: number };

export function normalizeCardinalSide(value: unknown): CardinalSide | null {
  const text = String(value ?? '').trim().toLowerCase();
  if (text === 'top' || text === 'bottom' || text === 'left' || text === 'right') {
    return text;
  }
  return null;
}

export function sideToVec(side: CardinalSide | null | undefined): DirectionVector | null {
  switch (side) {
    case 'top':
      return { dx: 0, dy: 1 };
    case 'bottom':
      return { dx: 0, dy: -1 };
    case 'left':
      return { dx: -1, dy: 0 };
    case 'right':
      return { dx: 1, dy: 0 };
    default:
      return null;
  }
}

export function vecToSide(
  vec: Partial<DirectionVector> | null | undefined,
): CardinalSide | null {
  if (!vec) return null;
  const dx = Number(vec.dx ?? NaN);
  const dy = Number(vec.dy ?? NaN);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return null;
  }
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? 'right' : 'left';
  }
  return dy >= 0 ? 'top' : 'bottom';
}

export function rotateVecCcw(
  vec: Partial<DirectionVector> | null | undefined,
  rotationDeg: number,
): DirectionVector | null {
  if (!vec) return null;
  const dx = Number(vec.dx ?? NaN);
  const dy = Number(vec.dy ?? NaN);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return null;
  }
  switch (normalizePlanRotationDeg(rotationDeg)) {
    case 0:
      return { dx, dy };
    case 90:
      return { dx: -dy, dy: dx };
    case 180:
      return { dx: -dx, dy: -dy };
    case 270:
      return { dx: dy, dy: -dx };
    default:
      return null;
  }
}

export function frontWorldToLocalSide(
  frontWorld: Partial<DirectionVector> | null | undefined,
  rotationDeg: number,
): CardinalSide | null {
  const inverseRotation = (360 - normalizePlanRotationDeg(rotationDeg)) % 360;
  return vecToSide(rotateVecCcw(frontWorld, inverseRotation));
}

export function planRotationToYawRad(rotationDeg: number): number {
  return (normalizePlanRotationDeg(rotationDeg) * Math.PI) / 180;
}

export function yawRadToPlanRotationDeg(yawRad: number): number {
  return normalizePlanRotationDeg((yawRad * 180) / Math.PI);
}

export function isCardinalPlanRotation(rotationDeg: number): boolean {
  const normalized = normalizePlanRotationDeg(rotationDeg);
  return normalized % 90 === 0;
}
