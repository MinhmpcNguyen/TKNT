export function normalizePlanRotationDeg(rotationDeg: number): number {
  if (!Number.isFinite(rotationDeg)) {
    return 0;
  }
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

export function normalizeDirectionVector(value: unknown): DirectionVector | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const record = value as Partial<DirectionVector>;
  const dx = Number(record.dx ?? NaN);
  const dy = Number(record.dy ?? NaN);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return null;
  }
  if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) {
    return null;
  }
  return { dx, dy };
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
  const normalized = normalizePlanRotationDeg(rotationDeg);
  switch (normalized) {
    case 0:
      return { dx, dy };
    case 90:
      return { dx: -dy, dy: dx };
    case 180:
      return { dx: -dx, dy: -dy };
    case 270:
      return { dx: dy, dy: -dx };
    default:
      break;
  }

  const radians = (normalized * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return {
    dx: dx * cos - dy * sin,
    dy: dx * sin + dy * cos,
  };
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

export function planRotationToCanvasDeg(rotationDeg: number): number {
  return -normalizePlanRotationDeg(rotationDeg);
}

export function planFrontWorldToYawRad(
  frontWorld: Partial<DirectionVector> | null | undefined,
): number | null {
  const front = normalizeDirectionVector(frontWorld);
  if (!front) {
    return null;
  }
  return Math.atan2(-front.dx, front.dy);
}

export function resolvePlanObjectYawRad(params: {
  rotationDeg: number;
  frontWorld?: unknown;
  frontSideWorld?: unknown;
  frontSideLocal?: unknown;
}): number {
  const worldFront =
    normalizeDirectionVector(params.frontWorld) ??
    sideToVec(normalizeCardinalSide(params.frontSideWorld));
  const fromWorld = planFrontWorldToYawRad(worldFront);
  if (fromWorld !== null) {
    return fromWorld;
  }

  const localFront = sideToVec(normalizeCardinalSide(params.frontSideLocal));
  const rotatedLocalFront = rotateVecCcw(localFront, params.rotationDeg);
  return planFrontWorldToYawRad(rotatedLocalFront) ?? planRotationToYawRad(params.rotationDeg);
}

export function yawRadToPlanRotationDeg(yawRad: number): number {
  return normalizePlanRotationDeg((yawRad * 180) / Math.PI);
}
