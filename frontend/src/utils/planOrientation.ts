export function normalizePlanRotationDeg(rotationDeg: number): number {
  return ((rotationDeg % 360) + 360) % 360;
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
