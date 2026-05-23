import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal, flushSync } from 'react-dom';
import {
  fetchRenderPresets,
  getGeneratedRenderDetail,
  renderSnapshotImage,
  updateGeneratedRenderMeta,
  type SnapshotImageEditOperation,
  type SnapshotImageLockedGeometryAtlas,
  type SnapshotImagePresetOption,
  type SnapshotImagePresetSelection,
  type SnapshotImagePresets,
  type SnapshotImageRenderApiResponse,
  type SnapshotImageRenderResult,
} from '../api';
import {
  buildPaymentReference,
  formatVnd,
  getMonthlyImagesRemaining,
  type BillingChargeKind,
} from '../billing';
import { BillingPaymentModal } from './BillingPaymentModal';
import { useEditorStore } from '../store';
import {
  DesignPlanV1,
  FloorplanGeometry,
  GeneratedRenderRecord,
  SavedLayoutRecord,
} from '../types';
import { polygonBounds } from '../utils/geometry';
import { getNumericBbox, getWorldBboxFromPlacement, placementMatchesStyledObject } from '../utils/styledLayout';
import {
  buildThreeSnapshotArtifact,
  type SnapshotOpeningRecord,
  type SnapshotObjectRecord,
  type ThreeSnapshotArtifact,
  type ThreeSnapshotRenderState,
  type VisibleSnapshotObject,
} from '../utils/threeSnapshot';
import {
  buildEdgeMapReferenceImageDataUrl,
  buildStructuralPerspectiveReferenceImageDataUrl,
} from '../utils/annotatedPerspectiveReference';
import {
  resolvePlanObjectYawRad,
  yawRadToPlanRotationDeg,
} from '../utils/planOrientation';
import {
  applyInventoryModelPreviewOverride,
  buildGlbPreviewModel,
  disposeThreeObject,
  type GlbFitStrategy,
  type InventoryModelDescriptor,
} from '../three/models';
import {
  collectReintroducedObjects,
  getInventoryDimensionsMm,
  isReintroducedIdentifier,
  isShelfLikePreviewType,
  isUserGeneratedPreviewItem,
  resolveInventoryItemForPreview,
  resolveWallHungPreviewDepthMm,
  type InventoryPreviewResolution,
} from '../three/inventoryDiagnostics';
import { buildObjectProxy } from '../three/proxies';

type RenderObject = {
  id: string;
  assetId?: string | null;
  rawAssetId?: string | null;
  type: string;
  colorHex: string;
  material?: string | null;
  rotationCcw?: number;
  frontWorld?: unknown;
  frontSideWorld?: unknown;
  frontSideLocal?: unknown;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  placementFootprintMm?: { w: number; d: number } | null;
  placeOn?: { method: string; targetId?: string | null } | null;
  draggable?: boolean;
  reintroduced?: boolean;
};

type ResolvedRenderObject = RenderObject & {
  inventoryResolution: InventoryPreviewResolution;
};

type ObjectVisualDiagnostic = {
  objectId: string;
  label: string;
  rawAssetId: string | null;
  resolvedAssetId: string | null;
  reintroduced: boolean;
  modelUrl?: string | null;
  previewRotationOffsetDeg?: number | null;
  finalDescriptorRotationDeg?: number | null;
  status:
    | 'glb_loading'
    | 'glb_loaded'
    | 'proxy_no_inventory_match'
    | 'proxy_no_model_descriptor'
    | 'proxy_low_confidence_match'
    | 'proxy_aspect_mismatch'
    | 'proxy_no_loader'
    | 'proxy_model_load_error';
  reason: string;
};

type RenderPresetGroupName = 'styles' | 'lights' | 'sceneries';
type CameraPresetGroupName = 'mid' | 'ceiling';
type ImageRenderAction = 'generate' | 'edit' | 'camera_rerender';
type RoomPointMm = {
  x: number;
  y: number;
};
type CameraPreset = {
  id: string;
  label: string;
  group: CameraPresetGroupName;
  description: string;
  fovDeg: number;
  position: {
    x: number;
    y: number;
    z: number;
  };
  target: {
    x: number;
    y: number;
    z: number;
  };
};
type OrbitControlLike = {
  minDistance: number;
  maxDistance: number;
  zoomSpeed: number;
  enableDamping: boolean;
  dampingFactor: number;
  minTargetRadius?: number;
  maxTargetRadius?: number;
};

const MM_TO_M = 0.001;
const SNAPSHOT_WALL_OCCLUSION_OPACITY_THRESHOLD = 0.5;
const MAX_RENDERER_PIXEL_RATIO = 2;
const SNAPSHOT_REFERENCE_MIN_LONG_EDGE_PX = 3072;
const SNAPSHOT_REFERENCE_MAX_LONG_EDGE_PX = 4096;
const SNAPSHOT_REFERENCE_TARGET_ASPECT = 1536 / 1024;
const STRUCTURAL_EDGE_COLOR_HEX = '#111827';
const STRUCTURAL_EDGE_OPACITY = 0.72;
const EDIT_BOX_COLORS = [
  '#f97316',
  '#2563eb',
  '#16a34a',
  '#dc2626',
  '#9333ea',
  '#0891b2',
];
const ORBIT_MIN_DISTANCE_M = 0.35;
const ORBIT_MAX_DISTANCE_ROOM_MULTIPLIER = 1.8;
const ORBIT_MAX_TARGET_RADIUS_ROOM_MULTIPLIER = 1.35;
const REFILL_ASSET_ID_PATTERN = /__(?:reintroduced|accessory_refill)(?:_\d+)?$/i;

function defaultHeightMm(type: string): number {
  const t = type.toLowerCase();
  if (t.includes('rug') || t.includes('mat')) return 10;
  if (t.includes('wall_art') || t.includes('clock') || t.includes('mirror')) return 30;
  if (t.includes('ceiling_light') || t.includes('pendant_light') || t.includes('track_light'))
    return 200;
  if (t.includes('air_conditioner')) return 250;
  if (t.includes('lamp')) return 1400;
  if (t.includes('chair')) return 900;
  if (t.includes('table')) return 500;
  return 800;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function pointsEqualMm(left: RoomPointMm, right: RoomPointMm, epsilon = 1): boolean {
  return Math.abs(left.x - right.x) <= epsilon && Math.abs(left.y - right.y) <= epsilon;
}

function normalizeRoomPolygon(points: RoomPointMm[]): RoomPointMm[] {
  const filtered = points.filter(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y)
  );
  if (filtered.length === 0) {
    return [];
  }

  const deduped: RoomPointMm[] = [];
  for (const point of filtered) {
    const previous = deduped[deduped.length - 1];
    if (!previous || !pointsEqualMm(previous, point)) {
      deduped.push(point);
    }
  }
  if (deduped.length >= 2 && pointsEqualMm(deduped[0], deduped[deduped.length - 1])) {
    deduped.pop();
  }
  if (deduped.length < 3) {
    return deduped;
  }

  return deduped.filter((point, index) => {
    const previous = deduped[(index - 1 + deduped.length) % deduped.length];
    const next = deduped[(index + 1) % deduped.length];
    const toPrevious = { x: previous.x - point.x, y: previous.y - point.y };
    const toNext = { x: next.x - point.x, y: next.y - point.y };
    const cross = toPrevious.x * toNext.y - toPrevious.y * toNext.x;
    const dot = toPrevious.x * toNext.x + toPrevious.y * toNext.y;
    return Math.abs(cross) > 1 || dot >= 0;
  });
}

function normalizeVector2D(vector: RoomPointMm): RoomPointMm | null {
  const length = Math.hypot(vector.x, vector.y);
  if (length <= 1e-6) {
    return null;
  }
  return {
    x: vector.x / length,
    y: vector.y / length,
  };
}

function normalizeVector3D(vector: { x: number; y: number; z: number }):
  | { x: number; y: number; z: number }
  | null {
  const length = Math.hypot(vector.x, vector.y, vector.z);
  if (length <= 1e-6) {
    return null;
  }
  return {
    x: vector.x / length,
    y: vector.y / length,
    z: vector.z / length,
  };
}

function crossVector3D(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
): { x: number; y: number; z: number } {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x,
  };
}

function dotVector3D(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function computePolygonCentroidMm(points: RoomPointMm[]): RoomPointMm {
  if (points.length === 0) {
    return { x: 0, y: 0 };
  }
  if (points.length < 3) {
    const x = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const y = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    return { x, y };
  }

  let signedArea = 0;
  let cx = 0;
  let cy = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const cross = current.x * next.y - next.x * current.y;
    signedArea += cross;
    cx += (current.x + next.x) * cross;
    cy += (current.y + next.y) * cross;
  }
  if (Math.abs(signedArea) <= 1e-6) {
    const x = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const y = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    return { x, y };
  }
  const areaFactor = signedArea * 3;
  return {
    x: cx / areaFactor,
    y: cy / areaFactor,
  };
}

function distancePointToSegmentMm(
  point: RoomPointMm,
  start: RoomPointMm,
  end: RoomPointMm
): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 1e-6) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }
  const rawT =
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared;
  const t = clamp(rawT, 0, 1);
  const projectedX = start.x + dx * t;
  const projectedY = start.y + dy * t;
  return Math.hypot(point.x - projectedX, point.y - projectedY);
}

function distanceToPolygonEdgesMm(point: RoomPointMm, polygon: RoomPointMm[]): number {
  if (polygon.length < 2) {
    return 0;
  }
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    bestDistance = Math.min(
      bestDistance,
      distancePointToSegmentMm(point, start, end)
    );
  }
  return Number.isFinite(bestDistance) ? bestDistance : 0;
}

function dedupeDirections(directions: RoomPointMm[]): RoomPointMm[] {
  const unique: RoomPointMm[] = [];
  for (const direction of directions) {
    const normalized = normalizeVector2D(direction);
    if (!normalized) {
      continue;
    }
    const alreadyPresent = unique.some(
      (candidate) =>
        Math.abs(candidate.x - normalized.x) <= 0.01 &&
        Math.abs(candidate.y - normalized.y) <= 0.01
    );
    if (!alreadyPresent) {
      unique.push(normalized);
    }
  }
  return unique;
}

function findInteriorCornerAnchorMm(params: {
  corner: RoomPointMm;
  previous: RoomPointMm;
  next: RoomPointMm;
  polygon: RoomPointMm[];
  insetMm: number;
  fallbackFocusPointMm: RoomPointMm;
}): RoomPointMm | null {
  const { corner, previous, next, polygon, insetMm, fallbackFocusPointMm } = params;
  const toPrevious = normalizeVector2D({
    x: previous.x - corner.x,
    y: previous.y - corner.y,
  });
  const toNext = normalizeVector2D({
    x: next.x - corner.x,
    y: next.y - corner.y,
  });
  const toFocus = normalizeVector2D({
    x: fallbackFocusPointMm.x - corner.x,
    y: fallbackFocusPointMm.y - corner.y,
  });
  const directions = dedupeDirections([
    toPrevious && toNext
      ? { x: toPrevious.x + toNext.x, y: toPrevious.y + toNext.y }
      : { x: 0, y: 0 },
    toPrevious && toNext
      ? { x: -(toPrevious.x + toNext.x), y: -(toPrevious.y + toNext.y) }
      : { x: 0, y: 0 },
    toPrevious ? { x: -toPrevious.y, y: toPrevious.x } : { x: 0, y: 0 },
    toPrevious ? { x: toPrevious.y, y: -toPrevious.x } : { x: 0, y: 0 },
    toNext ? { x: -toNext.y, y: toNext.x } : { x: 0, y: 0 },
    toNext ? { x: toNext.y, y: -toNext.x } : { x: 0, y: 0 },
    toPrevious && toNext
      ? {
          x: -toPrevious.y - toNext.y,
          y: toPrevious.x + toNext.x,
        }
      : { x: 0, y: 0 },
    toPrevious && toNext
      ? {
          x: toPrevious.y + toNext.y,
          y: -toPrevious.x - toNext.x,
        }
      : { x: 0, y: 0 },
    toFocus ?? { x: 0, y: 0 },
  ]);

  const distanceSteps = [
    insetMm,
    insetMm * 0.82,
    insetMm * 0.66,
    insetMm * 0.5,
    insetMm * 0.36,
    insetMm * 0.24,
    insetMm * 0.12,
  ].filter((distance) => distance > 0);

  let bestCandidate: { point: RoomPointMm; score: number } | null = null;
  for (const direction of directions) {
    for (const distance of distanceSteps) {
      const candidate = {
        x: corner.x + direction.x * distance,
        y: corner.y + direction.y * distance,
      };
      if (!pointInPolygon(candidate, polygon)) {
        continue;
      }
      const edgeClearanceMm = distanceToPolygonEdgesMm(candidate, polygon);
      const focusDistanceMm = Math.hypot(
        fallbackFocusPointMm.x - candidate.x,
        fallbackFocusPointMm.y - candidate.y
      );
      const score = edgeClearanceMm * 4 + distance - focusDistanceMm * 0.05;
      if (!bestCandidate || score > bestCandidate.score) {
        bestCandidate = { point: candidate, score };
      }
    }
  }

  return bestCandidate?.point ?? null;
}

function findInteriorWallAnchorMm(params: {
  start: RoomPointMm;
  end: RoomPointMm;
  polygon: RoomPointMm[];
  insetMm: number;
  fallbackFocusPointMm: RoomPointMm;
}): RoomPointMm | null {
  const { start, end, polygon, insetMm, fallbackFocusPointMm } = params;
  const midpoint = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2,
  };
  const wallDirection = normalizeVector2D({
    x: end.x - start.x,
    y: end.y - start.y,
  });
  const toFocus = normalizeVector2D({
    x: fallbackFocusPointMm.x - midpoint.x,
    y: fallbackFocusPointMm.y - midpoint.y,
  });

  const directions = dedupeDirections([
    toFocus ?? { x: 0, y: 0 },
    wallDirection ? { x: -wallDirection.y, y: wallDirection.x } : { x: 0, y: 0 },
    wallDirection ? { x: wallDirection.y, y: -wallDirection.x } : { x: 0, y: 0 },
    wallDirection && toFocus
      ? { x: -wallDirection.y + toFocus.x, y: wallDirection.x + toFocus.y }
      : { x: 0, y: 0 },
    wallDirection && toFocus
      ? { x: wallDirection.y + toFocus.x, y: -wallDirection.x + toFocus.y }
      : { x: 0, y: 0 },
  ]);
  const distanceSteps = [
    insetMm,
    insetMm * 0.82,
    insetMm * 0.66,
    insetMm * 0.5,
    insetMm * 0.36,
    insetMm * 0.24,
  ].filter((distance) => distance > 0);

  let bestCandidate: { point: RoomPointMm; score: number } | null = null;
  for (const direction of directions) {
    for (const distance of distanceSteps) {
      const candidate = {
        x: midpoint.x + direction.x * distance,
        y: midpoint.y + direction.y * distance,
      };
      if (!pointInPolygon(candidate, polygon)) {
        continue;
      }
      const edgeClearanceMm = distanceToPolygonEdgesMm(candidate, polygon);
      const focusDistanceMm = Math.hypot(
        fallbackFocusPointMm.x - candidate.x,
        fallbackFocusPointMm.y - candidate.y
      );
      const score = edgeClearanceMm * 3 + distance - focusDistanceMm * 0.04;
      if (!bestCandidate || score > bestCandidate.score) {
        bestCandidate = { point: candidate, score };
      }
    }
  }

  return bestCandidate?.point ?? null;
}

function resolveCameraFocusPointMm(params: {
  roomPolygonMm: RoomPointMm[];
  objects: ResolvedRenderObject[];
  fallbackPointMm: RoomPointMm;
}): RoomPointMm {
  const { roomPolygonMm, objects, fallbackPointMm } = params;
  const objectCenters = objects
    .map((object) => ({
      x: (object.bbox.minX + object.bbox.maxX) / 2,
      y: (object.bbox.minY + object.bbox.maxY) / 2,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const averageObjectCenter =
    objectCenters.length > 0
      ? {
          x:
            objectCenters.reduce((sum, point) => sum + point.x, 0) /
            objectCenters.length,
          y:
            objectCenters.reduce((sum, point) => sum + point.y, 0) /
            objectCenters.length,
        }
      : null;
  const averageVertexCenter =
    roomPolygonMm.length > 0
      ? {
          x:
            roomPolygonMm.reduce((sum, point) => sum + point.x, 0) /
            roomPolygonMm.length,
          y:
            roomPolygonMm.reduce((sum, point) => sum + point.y, 0) /
            roomPolygonMm.length,
        }
      : null;
  const candidates = [
    roomPolygonMm.length >= 3 ? computePolygonCentroidMm(roomPolygonMm) : null,
    averageVertexCenter,
    averageObjectCenter,
    fallbackPointMm,
  ].filter((point): point is RoomPointMm => point !== null);

  if (roomPolygonMm.length < 3) {
    return candidates[0] ?? fallbackPointMm;
  }
  for (const candidate of candidates) {
    if (pointInPolygon(candidate, roomPolygonMm)) {
      return candidate;
    }
  }
  return candidates[0] ?? fallbackPointMm;
}

function formatPlanCoordinateMm(valueMm: number): string {
  return `${(valueMm * MM_TO_M).toFixed(1)}m`;
}

function buildRoomCoverageSamplePoints(params: {
  roomPolygonMm: RoomPointMm[];
  roomCenterMm: RoomPointMm;
  wallHeightM: number;
}): Array<{ x: number; y: number; z: number }> {
  const { roomPolygonMm, roomCenterMm, wallHeightM } = params;
  const midWallHeightM = clamp(wallHeightM * 0.58, 0.95, wallHeightM - 0.32);
  return roomPolygonMm.flatMap((point) => {
    const x = (point.x - roomCenterMm.x) * MM_TO_M;
    const z = -(point.y - roomCenterMm.y) * MM_TO_M;
    return [
      { x, y: 0.1, z },
      { x, y: midWallHeightM, z },
    ];
  });
}

function computePresetFovDeg(params: {
  position: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
  aspect: number;
  coveragePoints: Array<{ x: number; y: number; z: number }>;
  frameLimit: number;
}): number {
  const { position, target, aspect, coveragePoints, frameLimit } = params;
  const forward = normalizeVector3D({
    x: target.x - position.x,
    y: target.y - position.y,
    z: target.z - position.z,
  });
  if (!forward) {
    return 68;
  }
  const worldUp = { x: 0, y: 1, z: 0 };
  const right =
    normalizeVector3D(crossVector3D(forward, worldUp)) ??
    normalizeVector3D({ x: 1, y: 0, z: 0 });
  if (!right) {
    return 68;
  }
  const up =
    normalizeVector3D(crossVector3D(right, forward)) ??
    worldUp;

  let maxTanX = 0;
  let maxTanY = 0;
  for (const point of coveragePoints) {
    const relative = {
      x: point.x - position.x,
      y: point.y - position.y,
      z: point.z - position.z,
    };
    const depth = dotVector3D(relative, forward);
    if (depth <= 0.05) {
      continue;
    }
    const horizontal = Math.abs(dotVector3D(relative, right));
    const vertical = Math.abs(dotVector3D(relative, up));
    maxTanX = Math.max(maxTanX, horizontal / depth);
    maxTanY = Math.max(maxTanY, vertical / depth);
  }

  const safeAspect = Math.max(0.75, aspect);
  const safeFrameLimit = clamp(frameLimit, 0.55, 0.92);
  const halfFovFromWidth = Math.atan(maxTanX / (safeAspect * safeFrameLimit));
  const halfFovFromHeight = Math.atan(maxTanY / safeFrameLimit);
  const fovDeg = (Math.max(halfFovFromWidth, halfFovFromHeight) * 360) / Math.PI;
  return clamp(fovDeg, 58, 84);
}

function buildCameraPresets(params: {
  roomPolygonMm: RoomPointMm[];
  roomBoundsMm: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  };
  roomCenterMm: RoomPointMm;
  focusPointMm: RoomPointMm;
  wallHeightMm: number;
  aspect: number;
}): CameraPreset[] {
  const {
    roomPolygonMm,
    roomBoundsMm,
    roomCenterMm,
    focusPointMm,
    wallHeightMm,
    aspect,
  } = params;
  if (roomPolygonMm.length < 3) {
    return [];
  }

  const spanX = Math.max(1, roomBoundsMm.maxX - roomBoundsMm.minX);
  const spanY = Math.max(1, roomBoundsMm.maxY - roomBoundsMm.minY);
  const insetMm = clamp(Math.min(spanX, spanY) * 0.05, 90, 240);
  const wallHeightM = Math.max(1.8, wallHeightMm * MM_TO_M);
  const nearCeilingY = clamp(
    wallHeightM - Math.min(0.18, wallHeightM * 0.08),
    1.45,
    wallHeightM - 0.06
  );
  const midHeightY = clamp(wallHeightM * 0.5, 1.15, wallHeightM - 0.45);
  const focusTargetY = clamp(wallHeightM * 0.38, 0.95, wallHeightM * 0.55);
  const coveragePoints = buildRoomCoverageSamplePoints({
    roomPolygonMm,
    roomCenterMm,
    wallHeightM,
  });

  const cornerPresets = roomPolygonMm.flatMap((corner, index) => {
    const previous = roomPolygonMm[(index - 1 + roomPolygonMm.length) % roomPolygonMm.length];
    const next = roomPolygonMm[(index + 1) % roomPolygonMm.length];
    const anchor = findInteriorCornerAnchorMm({
      corner,
      previous,
      next,
      polygon: roomPolygonMm,
      insetMm,
      fallbackFocusPointMm: focusPointMm,
    });
    if (!anchor) {
      return [];
    }

    const basePosition = {
      x: (anchor.x - roomCenterMm.x) * MM_TO_M,
      z: -(anchor.y - roomCenterMm.y) * MM_TO_M,
    };
    const target = {
      x: (focusPointMm.x - roomCenterMm.x) * MM_TO_M,
      y: focusTargetY,
      z: -(focusPointMm.y - roomCenterMm.y) * MM_TO_M,
    };
    const label = `Corner ${index + 1}`;
    const description = `X ${formatPlanCoordinateMm(corner.x)} · Y ${formatPlanCoordinateMm(corner.y)}`;

    return [
      {
        id: `mid-corner-${index + 1}`,
        label,
        group: 'mid' as const,
        description,
        fovDeg: computePresetFovDeg({
          position: {
            x: basePosition.x,
            y: midHeightY,
            z: basePosition.z,
          },
          target,
          aspect,
          coveragePoints,
          frameLimit: 0.72,
        }),
        position: {
          x: basePosition.x,
          y: midHeightY,
          z: basePosition.z,
        },
        target,
      },
      {
        id: `ceiling-corner-${index + 1}`,
        label,
        group: 'ceiling' as const,
        description,
        fovDeg: computePresetFovDeg({
          position: {
            x: basePosition.x,
            y: nearCeilingY,
            z: basePosition.z,
          },
          target,
          aspect,
          coveragePoints,
          frameLimit: 0.74,
        }),
        position: {
          x: basePosition.x,
          y: nearCeilingY,
          z: basePosition.z,
        },
        target,
      },
    ];
  });
  const wallCenterPresets = roomPolygonMm.flatMap((start, index) => {
    const end = roomPolygonMm[(index + 1) % roomPolygonMm.length];
    const anchor = findInteriorWallAnchorMm({
      start,
      end,
      polygon: roomPolygonMm,
      insetMm,
      fallbackFocusPointMm: focusPointMm,
    });
    if (!anchor) {
      return [];
    }

    const basePosition = {
      x: (anchor.x - roomCenterMm.x) * MM_TO_M,
      z: -(anchor.y - roomCenterMm.y) * MM_TO_M,
    };
    const target = {
      x: (focusPointMm.x - roomCenterMm.x) * MM_TO_M,
      y: focusTargetY,
      z: -(focusPointMm.y - roomCenterMm.y) * MM_TO_M,
    };
    const label = `Wall Center ${index + 1}`;
    const description = 'center of wall looking inward';

    return [
      {
        id: `mid-wall-${index + 1}`,
        label,
        group: 'mid' as const,
        description,
        fovDeg: computePresetFovDeg({
          position: {
            x: basePosition.x,
            y: midHeightY,
            z: basePosition.z,
          },
          target,
          aspect,
          coveragePoints,
          frameLimit: 0.72,
        }),
        position: {
          x: basePosition.x,
          y: midHeightY,
          z: basePosition.z,
        },
        target,
      },
      {
        id: `ceiling-wall-${index + 1}`,
        label,
        group: 'ceiling' as const,
        description,
        fovDeg: computePresetFovDeg({
          position: {
            x: basePosition.x,
            y: nearCeilingY,
            z: basePosition.z,
          },
          target,
          aspect,
          coveragePoints,
          frameLimit: 0.74,
        }),
        position: {
          x: basePosition.x,
          y: nearCeilingY,
          z: basePosition.z,
        },
        target,
      },
    ];
  });

  return [...cornerPresets, ...wallCenterPresets];
}

function applyOrbitCameraPose(
  camera: any,
  controls: any,
  pose: Pick<CameraPreset, 'position' | 'target' | 'fovDeg'>
): void {
  camera.fov = pose.fovDeg;
  camera.updateProjectionMatrix();
  camera.position.set(pose.position.x, pose.position.y, pose.position.z);
  controls.target.set(pose.target.x, pose.target.y, pose.target.z);
  camera.lookAt(pose.target.x, pose.target.y, pose.target.z);
  controls.update();
}

function configureRoomOrbitControls(
  controls: OrbitControlLike,
  roomMaxDimM: number
): void {
  const safeRoomMaxDimM = Math.max(1, Number.isFinite(roomMaxDimM) ? roomMaxDimM : 1);
  const minDistanceM = Math.max(
    ORBIT_MIN_DISTANCE_M,
    Math.min(0.8, safeRoomMaxDimM * 0.06)
  );

  controls.minDistance = minDistanceM;
  controls.maxDistance = Math.max(
    minDistanceM * 4,
    safeRoomMaxDimM * ORBIT_MAX_DISTANCE_ROOM_MULTIPLIER
  );
  controls.zoomSpeed = 0.65;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  if ('minTargetRadius' in controls) {
    controls.minTargetRadius = 0;
  }
  if ('maxTargetRadius' in controls) {
    controls.maxTargetRadius = Math.max(
      controls.maxDistance,
      safeRoomMaxDimM * ORBIT_MAX_TARGET_RADIUS_ROOM_MULTIPLIER
    );
  }
}

function sanitizeSnapshotFileSegment(value: string | null | undefined): string {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return normalized || 'current-layout';
}

function normalizeRenderAssetId(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const raw = value.trim();
  if (!raw) {
    return null;
  }
  const normalized = raw.replace(REFILL_ASSET_ID_PATTERN, '').trim();
  return normalized || raw;
}

function readOpeningAssetId(
  opening: any,
  kind: 'door' | 'window'
): string | null {
  const idKeys =
    kind === 'door'
      ? [
          'asset_id',
          'assetId',
          'model_asset_id',
          'modelAssetId',
          'door_asset_id',
          'doorAssetId',
          'door_id',
          'id',
        ]
      : [
          'asset_id',
          'assetId',
          'model_asset_id',
          'modelAssetId',
          'window_asset_id',
          'windowAssetId',
          'window_id',
          'id',
        ];
  for (const key of idKeys) {
    const value = opening?.[key];
    if (typeof value === 'string' && value.trim()) {
      const normalized = normalizeRenderAssetId(value) ?? value.trim();
      if (kind === 'window' && /^win(?:[_-]|$)/i.test(normalized)) {
        return 'window';
      }
      return normalized;
    }
  }
  return null;
}

function replaceThreeObjectChildren(root: any, nextChild: any): void {
  const previousChildren = [...(root?.children ?? [])];
  for (const child of previousChildren) {
    root.remove(child);
  }
  for (const child of previousChildren) {
    disposeThreeObject(child);
  }
  root.add(nextChild);
}

function addStructuralEdgeOverlay(
  THREE: any,
  root: any,
  options: {
    colorHex?: string;
    opacity?: number;
    renderOrder?: number;
    thresholdAngleDeg?: number;
  } = {}
): void {
  const meshes: any[] = [];
  const collectMesh = (node: any): void => {
    if (!node?.isMesh || !node.geometry || node.userData?.structuralEdgeOverlay) {
      return;
    }
    meshes.push(node);
  };

  if (typeof root?.traverse === 'function') {
    root.traverse(collectMesh);
  } else {
    collectMesh(root);
  }

  for (const mesh of meshes) {
    if (mesh.userData?.structuralEdgeOverlayAdded) {
      continue;
    }
    const geometry = mesh.geometry;
    if (!geometry?.attributes?.position) {
      continue;
    }
    const edgesGeometry = new THREE.EdgesGeometry(
      geometry,
      options.thresholdAngleDeg ?? 18
    );
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(options.colorHex ?? STRUCTURAL_EDGE_COLOR_HEX),
      depthTest: true,
      depthWrite: false,
      opacity: options.opacity ?? STRUCTURAL_EDGE_OPACITY,
      toneMapped: false,
      transparent: true,
    });
    const edges = new THREE.LineSegments(edgesGeometry, edgeMaterial);
    edges.name = 'structural-edge-overlay';
    edges.renderOrder = options.renderOrder ?? 4;
    edges.userData = {
      aiStructuralEdge: true,
      structuralEdgeOverlay: true,
    };
    mesh.add(edges);
    mesh.userData.structuralEdgeOverlayAdded = true;
  }
}

function safeInventoryModelPathSegment(value: string): string {
  return (
    value.trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') ||
    'asset'
  );
}

function canonicalizeInventoryStyle(value: string | null | undefined): string | null {
  const normalized = String(value ?? '').trim().toLowerCase().replace(/\s+/g, ' ');
  return normalized || null;
}

function buildOpeningFallbackModelDescriptor(
  resolution: InventoryPreviewResolution,
  preferredStyle: string | null | undefined
): InventoryModelDescriptor | null {
  const item = resolution.item;
  if (!item) {
    return null;
  }

  const preferredStyleKey = canonicalizeInventoryStyle(preferredStyle);
  const styleTags = item.style_tags
    .map((style) => canonicalizeInventoryStyle(style))
    .filter((style): style is string => style !== null);
  const styleKey =
    (preferredStyleKey && styleTags.includes(preferredStyleKey) ? preferredStyleKey : null) ??
    styleTags[0] ??
    null;
  if (!styleKey) {
    return null;
  }

  const stylePath = safeInventoryModelPathSegment(styleKey);
  const assetPath = safeInventoryModelPathSegment(item.asset_id);
  return {
    url: `/assets/inventory_models/demo_tenant/${stylePath}/${assetPath}.glb`,
    source: 'opening_public_inventory_fallback',
    fit: 'contain',
    scale: 1,
    rotationDeg: 0,
    rotationXDeg: 0,
    rotationZDeg: 0,
    previewCalibration: null,
    previewOverride: null,
  };
}

function buildSnapshotFileStem(variantId: string | null, capturedAt: string): string {
  const normalizedDate = capturedAt.replace(/[:.]/g, '-');
  return `layout-3d-${sanitizeSnapshotFileSegment(variantId)}-${normalizedDate}`;
}

function resolveOpeningSegmentMm(
  item: any
): { p1: { x: number; y: number }; p2: { x: number; y: number } } | null {
  if (item?.start_mm && item?.end_mm) {
    return {
      p1: { x: Number(item.start_mm.x ?? 0), y: Number(item.start_mm.y ?? 0) },
      p2: { x: Number(item.end_mm.x ?? 0), y: Number(item.end_mm.y ?? 0) },
    };
  }
  if (Array.isArray(item?.segment_mm) && item.segment_mm.length >= 2) {
    return {
      p1: {
        x: Number(item.segment_mm[0]?.x ?? 0),
        y: Number(item.segment_mm[0]?.y ?? 0),
      },
      p2: {
        x: Number(item.segment_mm[1]?.x ?? 0),
        y: Number(item.segment_mm[1]?.y ?? 0),
      },
    };
  }
  if (Array.isArray(item?.segmentMm) && item.segmentMm.length >= 2) {
    return {
      p1: {
        x: Number(item.segmentMm[0]?.x ?? 0),
        y: Number(item.segmentMm[0]?.y ?? 0),
      },
      p2: {
        x: Number(item.segmentMm[1]?.x ?? 0),
        y: Number(item.segmentMm[1]?.y ?? 0),
      },
    };
  }
  return null;
}

function normalizeSnapshotOpenings(items: any[]): SnapshotOpeningRecord[] {
  return items
    .map((item) => {
      const segment = resolveOpeningSegmentMm(item);
      if (!segment) return null;
      return {
        id: String(item?.id ?? 'opening'),
        segmentMm: [
          { x: segment.p1.x, y: segment.p1.y },
          { x: segment.p2.x, y: segment.p2.y },
        ],
      };
    })
    .filter((item): item is SnapshotOpeningRecord => item !== null);
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => {
    window.URL.revokeObjectURL(url);
  }, 0);
}

function canvasToDataUrl(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png');
}

type TemporaryVisibilityRecord = {
  node: { visible: boolean };
  visible: boolean;
};

type TemporaryMaterialRecord = {
  material: unknown;
  mesh: { material: unknown };
};

function hideNode(
  records: TemporaryVisibilityRecord[],
  node: unknown
): void {
  if (!node || typeof node !== 'object' || !('visible' in node)) {
    return;
  }
  const visibleNode = node as { visible: boolean };
  records.push({ node: visibleNode, visible: visibleNode.visible });
  visibleNode.visible = false;
}

function hideAiSnapshotHelperNodes(scene: any): TemporaryVisibilityRecord[] {
  const hiddenNodes: TemporaryVisibilityRecord[] = [];
  scene.traverse((node: any) => {
    if (
      (node?.userData?.aiRenderHide === true ||
        node?.userData?.aiStructuralEdge === true ||
        node?.userData?.structuralEdgeOverlay === true) &&
      typeof node.visible === 'boolean'
    ) {
      hideNode(hiddenNodes, node);
    }
  });
  return hiddenNodes;
}

function restoreVisibility(records: TemporaryVisibilityRecord[]): void {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    records[index].node.visible = records[index].visible;
  }
}

function renderDepthMapDataUrl(params: {
  THREE: any;
  renderer: any;
  scene: any;
  camera: any;
}): string {
  const { THREE, renderer, scene, camera } = params;
  const materialRecords: TemporaryMaterialRecord[] = [];
  const hiddenNodes: TemporaryVisibilityRecord[] = [];
  const depthMaterial = new THREE.MeshDepthMaterial({
    depthPacking: THREE.BasicDepthPacking,
    side: THREE.DoubleSide,
  });
  const previousBackground = scene.background ?? null;

  try {
    scene.traverse((node: any) => {
      if (
        node?.userData?.aiStructuralEdge === true &&
        typeof node.visible === 'boolean'
      ) {
        hideNode(hiddenNodes, node);
        return;
      }
      if (!node?.isMesh || !node.material) {
        return;
      }
      materialRecords.push({ mesh: node, material: node.material });
      node.material = depthMaterial;
    });
    scene.background = new THREE.Color('#ffffff');
    renderer.render(scene, camera);
    return canvasToDataUrl(renderer.domElement);
  } finally {
    for (const record of materialRecords) {
      record.mesh.material = record.material;
    }
    restoreVisibility(hiddenNodes);
    depthMaterial.dispose?.();
    scene.background = previousBackground;
  }
}

function renderSegmentationMapDataUrl(params: {
  THREE: any;
  renderer: any;
  scene: any;
  camera: any;
}): string {
  const { THREE, renderer, scene, camera } = params;
  const materialRecords: TemporaryMaterialRecord[] = [];
  const hiddenNodes: TemporaryVisibilityRecord[] = [];
  const materialBySegment = new Map<string, any>();
  const previousBackground = scene.background ?? null;

  const materialForSegment = (segmentKey: string): any => {
    const existing = materialBySegment.get(segmentKey);
    if (existing) {
      return existing;
    }
    const color = colorForSegmentKey(THREE, segmentKey);
    const material = new THREE.MeshBasicMaterial({
      color,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    materialBySegment.set(segmentKey, material);
    return material;
  };

  try {
    scene.traverse((node: any) => {
      if (
        node?.userData?.aiStructuralEdge === true &&
        typeof node.visible === 'boolean'
      ) {
        hideNode(hiddenNodes, node);
        return;
      }
      if (!node?.isMesh || !node.material) {
        return;
      }
      const segmentKey = resolveSegmentationKey(node);
      materialRecords.push({ mesh: node, material: node.material });
      node.material = materialForSegment(segmentKey);
    });
    scene.background = new THREE.Color('#000000');
    renderer.render(scene, camera);
    return canvasToDataUrl(renderer.domElement);
  } finally {
    for (const record of materialRecords) {
      record.mesh.material = record.material;
    }
    restoreVisibility(hiddenNodes);
    for (const material of materialBySegment.values()) {
      material.dispose?.();
    }
    scene.background = previousBackground;
  }
}

function resolveSegmentationKey(node: any): string {
  let current: any = node;
  while (current) {
    if (typeof current.userData?.objectId === 'string') {
      return `object:${current.userData.objectId}`;
    }
    if (typeof current.userData?.wallId === 'string') {
      return `wall:${current.userData.wallId || 'wall'}`;
    }
    if (typeof current.userData?.roomSurface === 'string') {
      return `surface:${current.userData.roomSurface}`;
    }
    if (typeof current.userData?.openingId === 'string') {
      return `opening:${current.userData.openingType ?? 'opening'}:${
        current.userData.openingId
      }`;
    }
    current = current.parent;
  }
  return 'scene:unclassified';
}

function colorForSegmentKey(THREE: any, segmentKey: string): any {
  let hash = 2166136261;
  for (let index = 0; index < segmentKey.length; index += 1) {
    hash ^= segmentKey.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  const hue = ((hash >>> 0) % 360) / 360;
  return new THREE.Color().setHSL(hue, 0.76, 0.56);
}

function resolveSnapshotReferenceRenderSize(canvas: HTMLCanvasElement): {
  height: number;
  width: number;
} {
  const currentWidth = Math.max(1, Math.round(canvas.width || canvas.clientWidth || 1));
  const currentHeight = Math.max(1, Math.round(canvas.height || canvas.clientHeight || 1));
  const currentLongEdge = Math.max(currentWidth, currentHeight);
  const targetLongEdge = clamp(
    Math.max(currentLongEdge, SNAPSHOT_REFERENCE_MIN_LONG_EDGE_PX),
    currentLongEdge,
    SNAPSHOT_REFERENCE_MAX_LONG_EDGE_PX
  );
  const targetAspect = SNAPSHOT_REFERENCE_TARGET_ASPECT;
  return {
    width: Math.max(1, Math.round(targetLongEdge)),
    height: Math.max(1, Math.round(targetLongEdge / targetAspect)),
  };
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, base64Payload] = dataUrl.split(',', 2);
  const mimeMatch = /^data:([^;]+);base64$/i.exec(header);
  const mimeType = mimeMatch?.[1] ?? 'image/png';
  const binary = window.atob(base64Payload ?? '');
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mimeType });
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
        return;
      }
      reject(new Error('Unable to read the reference image.'));
    };
    reader.onerror = () => reject(new Error('Unable to read the reference image.'));
    reader.readAsDataURL(file);
  });
}

function withAlphaColor(hexColor: string, alpha: number): string {
  const hex = hexColor.replace('#', '');
  if (hex.length !== 6) {
    return `rgba(249, 115, 22, ${alpha})`;
  }
  const red = Number.parseInt(hex.slice(0, 2), 16);
  const green = Number.parseInt(hex.slice(2, 4), 16);
  const blue = Number.parseInt(hex.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function editBoxModeLabel(mode: EditBoxMode): string {
  if (mode === 'add') {
    return 'Add';
  }
  if (mode === 'replace') {
    return 'Replace';
  }
  if (mode === 'recolor') {
    return 'Color';
  }
  return 'Prompt';
}

function normalizeEditBoxRect(draft: EditBoxDraft): EditBoxRect | null {
  const x = Math.min(draft.startX, draft.currentX);
  const y = Math.min(draft.startY, draft.currentY);
  const width = Math.abs(draft.currentX - draft.startX);
  const height = Math.abs(draft.currentY - draft.startY);
  if (width < 0.012 || height < 0.012) {
    return null;
  }
  return {
    x: clamp(x, 0, 1),
    y: clamp(y, 0, 1),
    width: clamp(width, 0, 1 - x),
    height: clamp(height, 0, 1 - y),
  };
}

function loadDataUrlImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load the generated image.'));
    image.src = dataUrl;
  });
}

async function buildEditBoxGuideImageDataUrl(params: {
  sourceImageDataUrl: string;
  boxes: RenderEditBox[];
}): Promise<string | null> {
  if (typeof document === 'undefined' || params.boxes.length === 0) {
    return null;
  }

  const sourceImage = await loadDataUrlImage(params.sourceImageDataUrl);
  const width = Math.max(1, sourceImage.naturalWidth || sourceImage.width);
  const height = Math.max(1, sourceImage.naturalHeight || sourceImage.height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  context.drawImage(sourceImage, 0, 0, width, height);
  context.fillStyle = 'rgba(15, 23, 42, 0.16)';
  context.fillRect(0, 0, width, height);

  params.boxes.forEach((box, index) => {
    const color = EDIT_BOX_COLORS[index % EDIT_BOX_COLORS.length];
    const x = box.rect.x * width;
    const y = box.rect.y * height;
    const boxWidth = box.rect.width * width;
    const boxHeight = box.rect.height * height;
    const lineWidth = Math.max(4, Math.min(width, height) * 0.006);
    const label = `${box.id} ${editBoxModeLabel(box.mode)}`;
    const fontSize = Math.max(
      18,
      Math.min(34, Math.round(Math.min(width, height) * 0.032))
    );
    context.save();
    context.lineWidth = lineWidth;
    context.strokeStyle = color;
    context.fillStyle = withAlphaColor(color, 0.14);
    context.fillRect(x, y, boxWidth, boxHeight);
    context.strokeRect(x, y, boxWidth, boxHeight);
    context.font = `800 ${fontSize}px sans-serif`;
    const labelPaddingX = Math.round(fontSize * 0.55);
    const labelPaddingY = Math.round(fontSize * 0.34);
    const labelWidth = Math.ceil(
      context.measureText(label).width + labelPaddingX * 2
    );
    const labelHeight = fontSize + labelPaddingY * 2;
    const labelX = clamp(x + 6, 6, Math.max(6, width - labelWidth - 6));
    const labelY = clamp(y + 6, 6, Math.max(6, height - labelHeight - 6));
    context.fillStyle = 'rgba(255, 255, 255, 0.96)';
    context.fillRect(labelX, labelY, labelWidth, labelHeight);
    context.strokeStyle = color;
    context.lineWidth = Math.max(2, lineWidth * 0.45);
    context.strokeRect(labelX, labelY, labelWidth, labelHeight);
    context.fillStyle = '#111827';
    context.fillText(
      label,
      labelX + labelPaddingX,
      labelY + labelPaddingY + fontSize - 2
    );
    context.restore();
  });
  return canvas.toDataURL('image/png');
}

async function cropObjectAppearanceReference(params: {
  imageDataUrl: string;
  object: VisibleSnapshotObject;
  snapshot: ThreeSnapshotArtifact;
}): Promise<string | null> {
  if (typeof document === 'undefined' || !params.object.screenBboxPx) {
    return null;
  }

  const image = await loadDataUrlImage(params.imageDataUrl);
  const imageWidth = Math.max(1, image.naturalWidth || image.width);
  const imageHeight = Math.max(1, image.naturalHeight || image.height);
  const scaleX = imageWidth / Math.max(1, params.snapshot.canvas.widthPx);
  const scaleY = imageHeight / Math.max(1, params.snapshot.canvas.heightPx);
  const box = params.object.screenBboxPx;
  const rawX = box.minX * scaleX;
  const rawY = box.minY * scaleY;
  const rawWidth = box.width * scaleX;
  const rawHeight = box.height * scaleY;
  if (rawWidth < 10 || rawHeight < 10) {
    return null;
  }

  const padding = Math.max(18, Math.max(rawWidth, rawHeight) * 0.18);
  const cropX = clamp(Math.floor(rawX - padding), 0, imageWidth - 1);
  const cropY = clamp(Math.floor(rawY - padding), 0, imageHeight - 1);
  const cropRight = clamp(Math.ceil(rawX + rawWidth + padding), cropX + 1, imageWidth);
  const cropBottom = clamp(Math.ceil(rawY + rawHeight + padding), cropY + 1, imageHeight);
  const cropWidth = cropRight - cropX;
  const cropHeight = cropBottom - cropY;
  if (cropWidth < 12 || cropHeight < 12) {
    return null;
  }

  const maxOutputSide = 768;
  const outputScale = Math.min(1, maxOutputSide / Math.max(cropWidth, cropHeight));
  const outputWidth = Math.max(1, Math.round(cropWidth * outputScale));
  const outputHeight = Math.max(1, Math.round(cropHeight * outputScale));
  const canvas = document.createElement('canvas');
  canvas.width = outputWidth;
  canvas.height = outputHeight;
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  context.drawImage(
    image,
    cropX,
    cropY,
    cropWidth,
    cropHeight,
    0,
    0,
    outputWidth,
    outputHeight
  );
  return canvas.toDataURL('image/jpeg', 0.86);
}

async function buildLockedGeometryAtlases(params: {
  imageDataUrl: string;
  snapshot: ThreeSnapshotArtifact;
}): Promise<SnapshotImageLockedGeometryAtlas[]> {
  if (typeof document === 'undefined') {
    return [];
  }
  const image = await loadDataUrlImage(params.imageDataUrl);
  const imageWidth = Math.max(1, image.naturalWidth || image.width);
  const imageHeight = Math.max(1, image.naturalHeight || image.height);
  const scaleX = imageWidth / Math.max(1, params.snapshot.canvas.widthPx);
  const scaleY = imageHeight / Math.max(1, params.snapshot.canvas.heightPx);

  const candidates = params.snapshot.visibleObjects
    .filter((object) => {
      const box = object.screenBboxPx;
      return !!box && box.width >= 8 && box.height >= 8;
    })
    .map((object) => {
      const box = object.screenBboxPx;
      const screenArea = box ? Math.max(0, box.width * box.height) : 0;
      const distance = Number.isFinite(object.distanceToCameraM)
        ? object.distanceToCameraM
        : Number.POSITIVE_INFINITY;
      return { object, screenArea, distance };
    })
    .sort((left, right) => right.screenArea - left.screenArea || left.distance - right.distance)
    .slice(0, GEOMETRY_ATLAS_MAX_OBJECTS);

  const atlases: SnapshotImageLockedGeometryAtlas[] = [];
  for (
    let atlasStart = 0;
    atlasStart < candidates.length;
    atlasStart += GEOMETRY_ATLAS_OBJECTS_PER_IMAGE
  ) {
    const atlasIndex = atlases.length + 1;
    const chunk = candidates.slice(
      atlasStart,
      atlasStart + GEOMETRY_ATLAS_OBJECTS_PER_IMAGE
    );
    if (!chunk.length) {
      continue;
    }
    const rows = Math.ceil(chunk.length / GEOMETRY_ATLAS_COLUMNS);
    const atlasWidth = GEOMETRY_ATLAS_COLUMNS * GEOMETRY_ATLAS_CELL_SIZE_PX;
    const atlasHeight = rows * GEOMETRY_ATLAS_CELL_SIZE_PX;
    const canvas = document.createElement('canvas');
    canvas.width = atlasWidth;
    canvas.height = atlasHeight;
    const context = canvas.getContext('2d');
    if (!context) {
      continue;
    }

    context.fillStyle = '#f7f7f5';
    context.fillRect(0, 0, atlasWidth, atlasHeight);
    const objects: SnapshotImageLockedGeometryAtlas['objects'] = [];

    chunk.forEach(({ object }, chunkIndex) => {
      const box = object.screenBboxPx;
      if (!box) {
        return;
      }
      const cellIndex = atlasStart + chunkIndex + 1;
      const localCellIndex = chunkIndex;
      const column = localCellIndex % GEOMETRY_ATLAS_COLUMNS;
      const row = Math.floor(localCellIndex / GEOMETRY_ATLAS_COLUMNS);
      const cellX = column * GEOMETRY_ATLAS_CELL_SIZE_PX;
      const cellY = row * GEOMETRY_ATLAS_CELL_SIZE_PX;
      const label = `G${cellIndex}`;

      context.fillStyle = '#ffffff';
      context.fillRect(
        cellX + 1,
        cellY + 1,
        GEOMETRY_ATLAS_CELL_SIZE_PX - 2,
        GEOMETRY_ATLAS_CELL_SIZE_PX - 2
      );
      context.strokeStyle = '#222222';
      context.lineWidth = 1;
      context.strokeRect(
        cellX + 0.5,
        cellY + 0.5,
        GEOMETRY_ATLAS_CELL_SIZE_PX - 1,
        GEOMETRY_ATLAS_CELL_SIZE_PX - 1
      );
      context.fillStyle = '#111111';
      context.font = 'bold 18px sans-serif';
      context.textBaseline = 'middle';
      context.fillText(label, cellX + 10, cellY + GEOMETRY_ATLAS_CELL_LABEL_HEIGHT_PX / 2);

      const rawX = box.minX * scaleX;
      const rawY = box.minY * scaleY;
      const rawWidth = box.width * scaleX;
      const rawHeight = box.height * scaleY;
      const padding = Math.max(12, Math.max(rawWidth, rawHeight) * 0.16);
      const cropX = clamp(Math.floor(rawX - padding), 0, imageWidth - 1);
      const cropY = clamp(Math.floor(rawY - padding), 0, imageHeight - 1);
      const cropRight = clamp(Math.ceil(rawX + rawWidth + padding), cropX + 1, imageWidth);
      const cropBottom = clamp(Math.ceil(rawY + rawHeight + padding), cropY + 1, imageHeight);
      const cropWidth = cropRight - cropX;
      const cropHeight = cropBottom - cropY;
      if (cropWidth < 8 || cropHeight < 8) {
        return;
      }

      const contentX = cellX + 10;
      const contentY = cellY + GEOMETRY_ATLAS_CELL_LABEL_HEIGHT_PX + 8;
      const contentWidth = GEOMETRY_ATLAS_CELL_SIZE_PX - 20;
      const contentHeight =
        GEOMETRY_ATLAS_CELL_SIZE_PX - GEOMETRY_ATLAS_CELL_LABEL_HEIGHT_PX - 18;
      const drawScale = Math.min(contentWidth / cropWidth, contentHeight / cropHeight);
      const drawWidth = Math.max(1, Math.round(cropWidth * drawScale));
      const drawHeight = Math.max(1, Math.round(cropHeight * drawScale));
      const drawX = Math.round(contentX + (contentWidth - drawWidth) / 2);
      const drawY = Math.round(contentY + (contentHeight - drawHeight) / 2);
      context.drawImage(
        image,
        cropX,
        cropY,
        cropWidth,
        cropHeight,
        drawX,
        drawY,
        drawWidth,
        drawHeight
      );

      objects.push({
        object_id: object.id,
        object_name: visibleObjectName(object),
        screen_bbox_px: object.screenBboxPx,
        atlas_index: atlasIndex,
        cell_index: cellIndex,
        cell_label: label,
        cell_bbox_norm: {
          x: Number((drawX / atlasWidth).toFixed(4)),
          y: Number((drawY / atlasHeight).toFixed(4)),
          width: Number((drawWidth / atlasWidth).toFixed(4)),
          height: Number((drawHeight / atlasHeight).toFixed(4)),
        },
      });
    });

    if (!objects.length) {
      continue;
    }
    atlases.push({
      atlas_index: atlasIndex,
      image_data_url: canvas.toDataURL('image/jpeg', 0.9),
      source: 'clean_3d_visible_object_atlas',
      objects,
    });
  }
  return atlases;
}

function normalizeRootLayoutId(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function readObjectAppearanceReferencesFromRender(
  render: GeneratedRenderRecord
): Record<string, ObjectAppearanceReference> {
  const rawMemory = render.meta[OBJECT_APPEARANCE_REFERENCES_META_KEY];
  const rawReferences = isRecord(rawMemory)
    ? rawMemory.references
    : Array.isArray(rawMemory)
      ? rawMemory
      : null;
  if (!Array.isArray(rawReferences)) {
    return {};
  }

  const rootLayoutId = isRecord(rawMemory)
    ? normalizeRootLayoutId(rawMemory.root_layout_id)
    : null;
  const references: Record<string, ObjectAppearanceReference> = {};
  for (const item of rawReferences) {
    if (!isRecord(item)) {
      continue;
    }
    const objectId =
      typeof item.object_id === 'string' && item.object_id.trim()
        ? item.object_id.trim()
        : typeof item.objectId === 'string' && item.objectId.trim()
          ? item.objectId.trim()
          : '';
    const imageDataUrl =
      typeof item.image_data_url === 'string' && item.image_data_url.startsWith('data:')
        ? item.image_data_url
        : typeof item.imageDataUrl === 'string' && item.imageDataUrl.startsWith('data:')
          ? item.imageDataUrl
          : '';
    if (!objectId || !imageDataUrl) {
      continue;
    }
    const savedAtMs =
      typeof item.saved_at_ms === 'number' && Number.isFinite(item.saved_at_ms)
        ? item.saved_at_ms
        : typeof item.savedAtMs === 'number' && Number.isFinite(item.savedAtMs)
          ? item.savedAtMs
          : Date.now();
    references[objectId] = {
      objectId,
      imageDataUrl,
      fileName:
        typeof item.file_name === 'string' && item.file_name.trim()
          ? item.file_name.trim()
          : typeof item.fileName === 'string' && item.fileName.trim()
            ? item.fileName.trim()
            : `edited-${objectId}.jpg`,
      savedAtMs,
      rootLayoutId: normalizeRootLayoutId(item.root_layout_id) ?? rootLayoutId,
      sourceRenderId: normalizeRootLayoutId(item.source_render_id),
      prompt:
        typeof item.prompt === 'string'
          ? item.prompt
          : typeof item.edit_prompt === 'string'
            ? item.edit_prompt
            : null,
      operationType:
        item.operation_type === 'add' ||
        item.operation_type === 'replace' ||
        item.operation_type === 'recolor' ||
        item.operation_type === 'prompt'
          ? item.operation_type
          : null,
      bboxNorm: isRecord(item.bbox_norm)
        ? {
            x: Number(item.bbox_norm.x ?? 0),
            y: Number(item.bbox_norm.y ?? 0),
            width: Number(item.bbox_norm.width ?? 0),
            height: Number(item.bbox_norm.height ?? 0),
          }
        : null,
    };
  }
  return references;
}

function buildObjectAppearanceReferencesMeta(params: {
  references: Record<string, ObjectAppearanceReference>;
  rootLayoutId: string | null;
}): Record<string, unknown> {
  const { references, rootLayoutId } = params;
  const filteredReferences = Object.values(references)
    .filter((reference) => (reference.rootLayoutId ?? null) === (rootLayoutId ?? null))
    .sort((left, right) => left.objectId.localeCompare(right.objectId));
  return {
    version: OBJECT_APPEARANCE_REFERENCES_VERSION,
    root_layout_id: rootLayoutId,
    references: filteredReferences.map((reference) => ({
      object_id: reference.objectId,
      image_data_url: reference.imageDataUrl,
      file_name: reference.fileName,
      saved_at_ms: reference.savedAtMs,
      root_layout_id: reference.rootLayoutId,
      source_render_id: reference.sourceRenderId,
      prompt: reference.prompt ?? null,
      operation_type: reference.operationType ?? null,
      bbox_norm: reference.bboxNorm ?? null,
    })),
  };
}

function resolveCanvasSizeFromAspectRatio(
  aspectRatio: string | null | undefined
): { widthPx: number; heightPx: number } {
  const rawValue = aspectRatio?.trim() ?? '';
  const [rawWidth, rawHeight] = rawValue.split(':');
  const width = Number(rawWidth);
  const height = Number(rawHeight);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return { widthPx: 1600, heightPx: 1200 };
  }
  return {
    widthPx: Math.round(width * 400),
    heightPx: Math.round(height * 400),
  };
}

function readSavedRenderVisibleObjects(
  render: GeneratedRenderRecord
): VisibleSnapshotObject[] {
  const rawVisibleObjects = render.meta.visible_objects;
  if (!Array.isArray(rawVisibleObjects)) {
    return [];
  }
  return rawVisibleObjects.filter((item): item is VisibleSnapshotObject => {
    if (!isRecord(item)) {
      return false;
    }
    return (
      typeof item.id === 'string' &&
      item.id.trim().length > 0 &&
      typeof item.type === 'string' &&
      item.type.trim().length > 0
    );
  });
}

function notifyRenderFocus(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new Event('tknt:render-focus'));
  }
}

function waitForNextPaint(): Promise<void> {
  if (typeof window === 'undefined') {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => {
      window.setTimeout(resolve, 0);
    });
  });
}

function buildSavedRenderSnapshotArtifact(
  render: GeneratedRenderRecord
): ThreeSnapshotArtifact {
  const visibleObjects = readSavedRenderVisibleObjects(render);

  const cameraPayload = isRecord(render.meta.camera)
    ? (render.meta.camera as ThreeSnapshotArtifact['camera'])
    : {
        positionM: { x: 0, y: 0, z: 0 },
        positionRoomMm: { x: 0, y: 0, z: 0 },
        targetM: null,
        targetRoomMm: null,
        fovDeg: 50,
        aspect: 4 / 3,
        near: 0.1,
        far: 100,
      };
  const { widthPx, heightPx } = resolveCanvasSizeFromAspectRatio(
    typeof render.meta.aspect_ratio === 'string' ? render.meta.aspect_ratio : null
  );

  return {
    capturedAt: render.created_at ?? new Date().toISOString(),
    variantId: null,
    selection: {
      selectedObjectId: visibleObjects[0]?.id ?? null,
      selectedWallId: null,
      editMode: 'idle',
    },
    canvas: {
      widthPx,
      heightPx,
    },
    room: {
      boundsMm: { minX: 0, minY: 0, maxX: 0, maxY: 0 },
      polygonMm: [],
      surfaceColors: {
        wallColorHex: null,
        floorColorHex: null,
        ceilingColorHex: null,
      },
      openings: {
        doors: [],
        windows: [],
      },
    },
    camera: cameraPayload,
    totalObjectCount: visibleObjects.length,
    visibleObjectCount: visibleObjects.length,
    hiddenObjectCount: 0,
    visibilityMethod: 'saved_render_metadata',
    wallOcclusionOpacityThreshold: SNAPSHOT_WALL_OCCLUSION_OPACITY_THRESHOLD,
    allObjects: visibleObjects,
    visibleObjects,
  };
}

function readSavedRenderEditOperations(
  render: GeneratedRenderRecord
): SnapshotImageRenderResult['request']['edit_operations'] {
  const rawOperations = render.meta.edit_operations;
  if (!Array.isArray(rawOperations)) {
    return [];
  }
  return rawOperations.filter(
    (item): item is NonNullable<SnapshotImageRenderResult['request']['edit_operations']>[number] =>
      isRecord(item) && typeof item.object_id === 'string' && item.object_id.trim().length > 0
  );
}

function buildSavedRenderPreviewResult(
  render: GeneratedRenderRecord,
  imageDataUrl: string,
  snapshotArtifact: ThreeSnapshotArtifact
): SnapshotImageRenderResult {
  const rawPresetSelection = isRecord(render.meta.preset_selection)
    ? render.meta.preset_selection
    : {};
  const presetSelection: SnapshotImagePresetSelection = {
    style:
      typeof rawPresetSelection.style === 'string' ? rawPresetSelection.style : null,
    lighting:
      typeof rawPresetSelection.lighting === 'string'
        ? rawPresetSelection.lighting
        : null,
    scenery:
      typeof rawPresetSelection.scenery === 'string'
        ? rawPresetSelection.scenery
        : null,
  };
  const imageBase64 = imageDataUrl.split(',', 2)[1] ?? '';
  const aspectRatio =
    typeof render.meta.aspect_ratio === 'string' ? render.meta.aspect_ratio : '4:3';
  const editOperations = readSavedRenderEditOperations(render);
  const renderMode =
    render.meta.render_mode === 'edit' || render.meta.render_mode === 'camera_rerender'
      ? render.meta.render_mode
      : 'generate';
  const rawUserPrompt =
    typeof render.meta.raw_user_prompt === 'string'
      ? render.meta.raw_user_prompt
      : render.prompt;

  return {
    request: {
      render_mode: renderMode,
      system_prompt: 'Reloaded from saved render.',
      user_prompt: render.prompt,
      raw_user_prompt: rawUserPrompt,
      preset_prompt: null,
      preset_selection: presetSelection,
      aspect_ratio: aspectRatio,
      source_image_mime_type: render.mime_type,
      structural_image_mime_type: render.mime_type,
      layout_reference_enabled: Boolean(render.meta.layout_reference_enabled),
      layout_reference_used: Boolean(render.meta.layout_reference_used),
      layout_reference_image_mime_type: null,
      annotated_reference_used: Boolean(render.meta.annotated_reference_used),
      annotated_reference_image_mime_type: null,
      scene_reference_mode:
        render.meta.scene_reference_mode === 'target_layout_with_scene_reference' ||
        render.meta.scene_reference_mode === 'scene_reference_camera_transfer'
          ? render.meta.scene_reference_mode
          : 'none',
      scene_reference_used: Boolean(render.meta.scene_reference_used),
      reference_only_camera_transfer_used: Boolean(
        render.meta.reference_only_camera_transfer_used
      ),
      scene_reference_image_mime_type: null,
      reference_image_count: 0,
      edit_operations: editOperations,
    },
    image: {
      mime_type: render.mime_type,
      image_base64: imageBase64,
      data_url: imageDataUrl,
      size: 'saved',
      aspect_ratio: aspectRatio,
      raw_response: {},
    },
    models: {
      image_model_name: render.model_name,
      max_tokens: 0,
    },
    metadata: {
      camera: snapshotArtifact.camera,
      visible_objects: snapshotArtifact.visibleObjects as Array<Record<string, unknown>>,
      visible_object_ids: snapshotArtifact.visibleObjects.map((object) => object.id),
      render_mode: renderMode,
      preset_selection: presetSelection,
      edit_operations: editOperations,
    },
  };
}

function resolveRootLayoutIdFromRender(
  render: GeneratedRenderRecord
): string | null {
  return readRenderContextText(render, 'root_layout_id');
}

function readRenderContextText(
  render: GeneratedRenderRecord,
  key: 'root_layout_id' | 'case_id' | 'option_id'
): string | null {
  const topLevelValue = render[key];
  if (typeof topLevelValue === 'string' && topLevelValue.trim()) {
    return topLevelValue.trim();
  }
  const metaValue = render.meta[key];
  if (typeof metaValue === 'string' && metaValue.trim()) {
    return metaValue.trim();
  }
  return null;
}

function resolveSavedLayoutForRender(params: {
  render: GeneratedRenderRecord;
  savedLayouts: SavedLayoutRecord[];
  activeRootLayoutId: string | null;
}): SavedLayoutRecord | null {
  const { render, savedLayouts, activeRootLayoutId } = params;
  const caseId = readRenderContextText(render, 'case_id');
  const optionId = readRenderContextText(render, 'option_id');
  if (caseId && optionId) {
    const layoutOption = savedLayouts.find(
      (layout) => layout.case_id === caseId && layout.option_id === optionId
    );
    if (layoutOption) {
      return layoutOption;
    }
  }
  const explicitRootLayoutId = resolveRootLayoutIdFromRender(render);
  if (explicitRootLayoutId) {
    return savedLayouts.find((layout) => layout.id === explicitRootLayoutId) ?? null;
  }
  if (activeRootLayoutId) {
    return savedLayouts.find((layout) => layout.id === activeRootLayoutId) ?? null;
  }
  return null;
}

function firstPresetKey(
  presets: SnapshotImagePresets | null,
  groupName: RenderPresetGroupName,
  preferred: string
): string {
  const group = presets?.[groupName] ?? {};
  if (preferred && group[preferred]) {
    return preferred;
  }
  return Object.keys(group)[0] ?? '';
}

function optionLabel(rawKey: string): string {
  return rawKey
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function normalizeHexColor(value: unknown, fallback: string): string {
  if (typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value.trim())) {
    return value.trim();
  }
  return fallback;
}

function visibleObjectName(object: VisibleSnapshotObject | null): string {
  if (!object) {
    return 'object';
  }
  const renderIdentityLabel = object.renderIdentityLabel?.trim();
  if (renderIdentityLabel) {
    return renderIdentityLabel;
  }
  if (object.type.trim().toLowerCase() === 'tv_console') {
    return TV_CONSOLE_RENDER_IDENTITY_LABEL;
  }
  return object.type.replace(/_/g, ' ');
}

function visibleObjectLabel(object: VisibleSnapshotObject): string {
  return `${visibleObjectName(object)} - ${object.id}`;
}

function resolveRenderIdentityForObject(type: string): {
  label: string | null;
  hint: string | null;
} {
  const normalizedType = type.trim().toLowerCase();
  if (normalizedType === 'tv_console') {
    return {
      label: TV_CONSOLE_RENDER_IDENTITY_LABEL,
      hint: TV_CONSOLE_RENDER_IDENTITY_HINT,
    };
  }
  return { label: null, hint: null };
}

function hasPresetSelection(selection: SnapshotImagePresetSelection): boolean {
  return Boolean(selection.style || selection.lighting || selection.scenery);
}

function mergePresetSelections(
  base: SnapshotImagePresetSelection | null | undefined,
  delta: SnapshotImagePresetSelection
): SnapshotImagePresetSelection {
  return {
    style: delta.style ?? base?.style ?? null,
    lighting: delta.lighting ?? base?.lighting ?? null,
    scenery: delta.scenery ?? base?.scenery ?? null,
  };
}

function withEffectivePresetSelection(
  render: SnapshotImageRenderResult,
  presetSelection: SnapshotImagePresetSelection,
  rawUserPrompt: string | null
): SnapshotImageRenderResult {
  return {
    ...render,
    request: {
      ...render.request,
      raw_user_prompt: rawUserPrompt,
      preset_selection: presetSelection,
    },
    metadata: {
      ...render.metadata,
      preset_selection: presetSelection,
    },
  };
}

type ReferencePresetDropdownOption = {
  value: string | null;
  label: string;
  referenceImage: string | null;
};

type ObjectReplacementReference = {
  objectId: string;
  imageDataUrl: string;
  fileName: string;
  savedAtMs: number;
};

type EditBoxMode = 'prompt' | 'add' | 'replace' | 'recolor';

type EditBoxRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type RenderEditBox = {
  id: string;
  mode: EditBoxMode;
  rect: EditBoxRect;
  prompt: string;
  imageDataUrl: string | null;
  fileName: string | null;
  targetColor: string;
};

type EditBoxDraft = {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
};

type ObjectAppearanceReference = ObjectReplacementReference & {
  rootLayoutId: string | null;
  sourceRenderId: string | null;
  prompt?: string | null;
  operationType?: EditBoxMode | null;
  bboxNorm?: EditBoxRect | null;
};

type PendingBillingPayment = {
  amountVnd: number;
  chargeKind: BillingChargeKind;
  reason: string;
  reference: string;
};

const OBJECT_APPEARANCE_REFERENCES_META_KEY = 'object_appearance_references';
const OBJECT_APPEARANCE_REFERENCES_VERSION = 1;
const RENDER_STATE_META_KEY = 'render_state';
const EDIT_DELTA_META_KEY = 'edit_delta';
const GEOMETRY_ATLAS_MAX_OBJECTS = 24;
const GEOMETRY_ATLAS_OBJECTS_PER_IMAGE = 12;
const GEOMETRY_ATLAS_COLUMNS = 4;
const GEOMETRY_ATLAS_CELL_SIZE_PX = 256;
const GEOMETRY_ATLAS_CELL_LABEL_HEIGHT_PX = 28;
const TV_CONSOLE_RENDER_IDENTITY_LABEL = 'TV console with integrated flat-screen TV';
const TV_CONSOLE_RENDER_IDENTITY_HINT =
  'Composite media furniture: preserve the black flat-screen TV/screen panel together with the console cabinet; do not render it as a plain box and do not add a separate duplicate TV.';

function buildReferencePresetDropdownOptions(
  group: Record<string, SnapshotImagePresetOption>
): ReferencePresetDropdownOption[] {
  return Object.entries(group).map(([key, option]) => ({
    value: key,
    label: option.label || optionLabel(key),
    referenceImage: option.reference_image ?? null,
  }));
}

type ReferencePresetDropdownProps = {
  label: string;
  options: ReferencePresetDropdownOption[];
  value: string | null | undefined;
  disabled: boolean;
  onChange: (value: string | null) => void;
  allowEmpty?: boolean;
  emptyLabel?: string;
};

type ReferenceDropdownPanelStyle = React.CSSProperties & {
  '--reference-dropdown-panel-max-height'?: string;
};

function ReferencePresetDropdown({
  label,
  options,
  value,
  disabled,
  onChange,
  allowEmpty = false,
  emptyLabel = 'No change',
}: ReferencePresetDropdownProps): React.ReactElement {
  const rootRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [panelStyle, setPanelStyle] = useState<ReferenceDropdownPanelStyle>({
    visibility: 'hidden',
  });
  const normalizedValue = value ?? null;
  const resolvedOptions = allowEmpty
    ? [
        {
          value: null,
          label: emptyLabel,
          referenceImage: null,
        },
        ...options,
      ]
    : options;
  const selectedOption =
    resolvedOptions.find((option) => option.value === normalizedValue) ??
    resolvedOptions[0] ??
    null;

  useEffect(() => {
    if (!open) {
      return;
    }
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!rootRef.current?.contains(target) && !panelRef.current?.contains(target)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  useLayoutEffect(() => {
    if (!open || typeof window === 'undefined') {
      return;
    }

    const updatePanelPosition = (): void => {
      const trigger = rootRef.current;
      if (!trigger) {
        return;
      }
      const triggerRect = trigger.getBoundingClientRect();
      const viewportPadding = 18;
      const panelWidth = Math.min(680, Math.max(300, window.innerWidth - viewportPadding * 2));
      const maxPanelHeight = Math.max(
        180,
        Math.min(420, window.innerHeight - viewportPadding * 2 - 18)
      );
      const estimatedPanelHeight = maxPanelHeight + 18;
      const belowTop = triggerRect.bottom + 6;
      const aboveTop = triggerRect.top - estimatedPanelHeight - 6;
      const shouldOpenUp =
        belowTop + estimatedPanelHeight > window.innerHeight - viewportPadding &&
        triggerRect.top > window.innerHeight - triggerRect.bottom;
      const top = shouldOpenUp
        ? Math.max(viewportPadding, aboveTop)
        : Math.min(belowTop, window.innerHeight - viewportPadding - estimatedPanelHeight);
      const left = Math.min(
        Math.max(viewportPadding, triggerRect.left),
        window.innerWidth - viewportPadding - panelWidth
      );
      setPanelStyle({
        left,
        top: Math.max(viewportPadding, top),
        width: panelWidth,
        visibility: 'visible',
        '--reference-dropdown-panel-max-height': `${maxPanelHeight}px`,
      });
    };

    updatePanelPosition();
    window.addEventListener('resize', updatePanelPosition);
    window.addEventListener('scroll', updatePanelPosition, true);
    return () => {
      window.removeEventListener('resize', updatePanelPosition);
      window.removeEventListener('scroll', updatePanelPosition, true);
    };
  }, [open]);

  useEffect(() => {
    if (disabled && open) {
      setOpen(false);
    }
  }, [disabled, open]);

  const dropdownPanel = open ? (
    <div
      className="reference-dropdown-panel"
      ref={panelRef}
      role="listbox"
      aria-label={label}
      style={panelStyle}
    >
      <div className="reference-dropdown-grid">
        {resolvedOptions.map((option) => {
          const isSelected = option.value === normalizedValue;
          return (
            <button
              key={option.value ?? '__empty__'}
              type="button"
              className={isSelected ? 'reference-preset-card selected' : 'reference-preset-card'}
              aria-selected={isSelected}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              {option.referenceImage ? (
                <img
                  className="reference-preset-image"
                  src={option.referenceImage}
                  alt={`${option.label} reference`}
                  loading="lazy"
                  draggable={false}
                />
              ) : (
                <span className="reference-preset-image is-empty">
                  <span>{option.label}</span>
                </span>
              )}
              <span className="reference-preset-label">{option.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  ) : null;

  return (
    <div className="three-dropdown-field reference-dropdown" ref={rootRef}>
      <span>{label}</span>
      <button
        type="button"
        className="reference-dropdown-trigger"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {selectedOption?.referenceImage ? (
          <img
            className="reference-dropdown-thumb"
            src={selectedOption.referenceImage}
            alt={`${selectedOption.label} reference`}
            loading="lazy"
            draggable={false}
          />
        ) : (
          <span className="reference-dropdown-thumb is-empty">
            <span>No image</span>
          </span>
        )}
        <span className="reference-dropdown-meta">
          <span className="reference-dropdown-value">
            {selectedOption?.label ?? (options.length > 0 ? 'Select option' : 'Loading')}
          </span>
          <span className="reference-dropdown-note">
            {options.length > 0 ? `${options.length} presets` : 'Preset list'}
          </span>
        </span>
        <span className="reference-dropdown-caret">{open ? '▲' : '▼'}</span>
      </button>
      {dropdownPanel && typeof document !== 'undefined'
        ? createPortal(dropdownPanel, document.body)
        : dropdownPanel}
    </div>
  );
}

type CameraPresetSelectProps = {
  value: string | null;
  stepAnchorValue: string | null;
  disabled: boolean;
  midHeightOptions: CameraPreset[];
  ceilingOptions: CameraPreset[];
  onChange: (value: string | null) => void;
};

function CameraPresetStepper({
  value,
  stepAnchorValue,
  disabled,
  midHeightOptions,
  ceilingOptions,
  onChange,
}: CameraPresetSelectProps): React.ReactElement {
  const options = [...midHeightOptions, ...ceilingOptions];
  const activeIndex = options.findIndex((preset) => preset.id === value);
  const stepAnchorIndex =
    activeIndex >= 0 ? activeIndex : options.findIndex((preset) => preset.id === stepAnchorValue);
  const activePreset = activeIndex >= 0 ? options[activeIndex] : null;
  const handleStep = (delta: number): void => {
    if (options.length === 0) {
      return;
    }
    const startIndex = stepAnchorIndex >= 0 ? stepAnchorIndex : delta > 0 ? -1 : 0;
    const nextIndex = (startIndex + delta + options.length) % options.length;
    onChange(options[nextIndex].id);
  };

  return (
    <div className="three-dropdown-field three-dropdown-field-wide three-camera-stepper">
      <span>Quick Camera View</span>
      <div className="three-camera-stepper-row">
        <button
          type="button"
          className="three-camera-stepper-button"
          disabled={disabled || options.length === 0}
          onClick={() => handleStep(-1)}
          aria-label="Previous camera view"
        >
          ‹
        </button>
        <div className="three-camera-stepper-current" aria-live="polite">
          <strong>{activePreset?.label ?? 'Custom orbit'}</strong>
          <span>
            {activePreset
              ? `${activePreset.group === 'mid' ? 'Mid-height' : 'Ceiling'} · ${activePreset.description}`
              : options.length > 0
                ? `${options.length} saved camera views`
                : 'No camera views available'}
          </span>
        </div>
        <button
          type="button"
          className="three-camera-stepper-button"
          disabled={disabled || options.length === 0}
          onClick={() => handleStep(1)}
          aria-label="Next camera view"
        >
          ›
        </button>
      </div>
    </div>
  );
}

type ColorSwatchFieldProps = {
  label: string;
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
};

function ColorSwatchField({
  label,
  value,
  disabled = false,
  onChange,
}: ColorSwatchFieldProps): React.ReactElement {
  return (
    <label className="three-color-field">
      <span>{label}</span>
      <input
        type="color"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
      <span className="three-color-value">{value}</span>
    </label>
  );
}

export const ThreeDPreviewPanel: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    viewMode,
    floorplan,
    design,
    styledResult,
    layoutVariants,
    inventory,
    userInput,
    selectedVariantId,
    selectedWallId,
    savedLayouts,
    savedRenders,
    activeRootLayoutId,
    activeLayoutCaseId,
    selectedSavedRenderId,
    selectWall,
    selectPlacement,
    setActiveRootLayoutId,
    setActiveLayoutCaseId,
    selectSavedRender,
    updatePlacement,
    removePlacement,
    setFloorplan,
    setDesign,
    setStyledResult,
    clearLayoutVariants,
    upsertSavedRender,
    appendRunLog,
    consumeMonthlyImageCredit,
    recordPaidImageUse,
  } = useEditorStore();
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editMode, setEditMode] = useState<'idle' | 'edit'>('idle');
  const initialDesignRef = useRef<any>(null);
  const activeFloorplan = floorplan ?? design?.floorplan_geometry ?? null;
  const editModeRef = useRef<'idle' | 'edit'>('idle');
  const selectedIdRef = useRef<string | null>(null);
  const updateRotatorsVisibilityRef = useRef<(() => void) | null>(null);
  const controlsRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);
  const sceneRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const rotatorGroupRef = useRef<any>(null);
  const renderStateRef = useRef<ThreeSnapshotRenderState | null>(null);
  const [captureStatus, setCaptureStatus] = useState<{
    tone: 'neutral' | 'success' | 'error';
    text: string;
  } | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [renderStatus, setRenderStatus] = useState<{
    tone: 'neutral' | 'success' | 'error';
    text: string;
  } | null>(null);
  const [activeImageRenderAction, setActiveImageRenderAction] =
    useState<ImageRenderAction | null>(null);
  const isRenderingImage = activeImageRenderAction !== null;
  const [pendingBillingPayment, setPendingBillingPayment] =
    useState<PendingBillingPayment | null>(null);
  const [billingCountdownSeconds, setBillingCountdownSeconds] = useState(0);
  const pendingBillingPaymentRef = useRef<PendingBillingPayment | null>(null);
  const pendingBillingResolverRef = useRef<
    ((payment: PendingBillingPayment | null) => void) | null
  >(null);
  const [renderedSnapshotImage, setRenderedSnapshotImage] =
    useState<SnapshotImageRenderResult | null>(null);
  const [savedRenderId, setSavedRenderId] = useState<string | null>(null);
  const [renderPresets, setRenderPresets] =
    useState<SnapshotImagePresets | null>(null);
  const [presetSelection, setPresetSelection] =
    useState<SnapshotImagePresetSelection>({
      style: 'modern',
      lighting: 'daylight_1',
      scenery: 'none',
    });
  const [lastRenderedSnapshot, setLastRenderedSnapshot] =
    useState<ThreeSnapshotArtifact | null>(null);
  const [, setLastRenderedSnapshotImageDataUrl] =
    useState<string | null>(null);
  const [objectAppearanceReferences, setObjectAppearanceReferences] = useState<
    Record<string, ObjectAppearanceReference>
  >({});
  const [editBoxes, setEditBoxes] = useState<RenderEditBox[]>([]);
  const [activeEditBoxId, setActiveEditBoxId] = useState<string | null>(null);
  const [editBoxDraft, setEditBoxDraft] = useState<EditBoxDraft | null>(null);
  const [targetColor, setTargetColor] = useState('#c8a26a');
  const [colorEnabled, setColorEnabled] = useState(false);
  const [editPresetSelection, setEditPresetSelection] =
    useState<SnapshotImagePresetSelection>({
      style: null,
      lighting: null,
      scenery: null,
    });
  const [objectVisualDiagnostics, setObjectVisualDiagnostics] = useState<
    Record<string, ObjectVisualDiagnostic>
  >({});
  const [isThreeSceneLoading, setIsThreeSceneLoading] = useState(false);
  const [cameraPresets, setCameraPresets] = useState<CameraPreset[]>([]);
  const [activeCameraPresetId, setActiveCameraPresetId] = useState<string | null>(null);
  const lastAppliedCameraPresetIdRef = useRef<string | null>(null);
  const applyingCameraPresetRef = useRef(false);
  const userCameraInteractionRef = useRef(false);
  const activeLayoutVariant = useMemo(
    () =>
      layoutVariants.find((variant) => variant.variant_id === selectedVariantId) ??
      null,
    [layoutVariants, selectedVariantId]
  );
  const renderLayoutContext = useMemo(
    () => {
      const caseId = activeLayoutVariant?.saved_layout_case_id ?? activeLayoutCaseId;
      return {
        rootLayoutId:
          activeRootLayoutId ?? activeLayoutVariant?.saved_layout_id ?? null,
        caseId,
        optionId: caseId ? selectedVariantId : null,
      };
    },
    [activeLayoutCaseId, activeLayoutVariant, activeRootLayoutId, selectedVariantId]
  );

  const finishPendingBillingPayment = useCallback((completed: boolean): void => {
    const resolver = pendingBillingResolverRef.current;
    const payment = completed ? pendingBillingPaymentRef.current : null;
    pendingBillingResolverRef.current = null;
    pendingBillingPaymentRef.current = null;
    setPendingBillingPayment(null);
    setBillingCountdownSeconds(0);
    resolver?.(payment);
  }, []);

  const requestBillingPayment = useCallback(
    (payment: PendingBillingPayment): Promise<PendingBillingPayment | null> => {
      pendingBillingPaymentRef.current = payment;
      setPendingBillingPayment(payment);
      setBillingCountdownSeconds(10);
      return new Promise((resolve) => {
        pendingBillingResolverRef.current = resolve;
      });
    },
    []
  );

  const ensureImageBilling = useCallback(
    async (chargeKind: BillingChargeKind): Promise<boolean> => {
      const chargeLabel = chargeKind === 'edit' ? 'edit' : 'generation';
      if (consumeMonthlyImageCredit()) {
        const latestSettings = useEditorStore.getState().billingSettings;
        appendRunLog(
          `Image ${chargeLabel} used one monthly credit (${getMonthlyImagesRemaining(
            latestSettings
          )} left).`
        );
        return true;
      }

      const latestSettings = useEditorStore.getState().billingSettings;
      const payment: PendingBillingPayment = {
        amountVnd: latestSettings.perUsePriceVnd,
        chargeKind,
        reason:
          latestSettings.mode === 'monthly'
            ? 'Monthly image quota is used up. Scan this QR to continue once.'
            : 'Pay per image is selected. Scan this QR to continue.',
        reference: buildPaymentReference(chargeKind),
      };
      setRenderStatus({
        tone: 'neutral',
        text: `Waiting for mock QR payment (${formatVnd(payment.amountVnd)}).`,
      });
      const completedPayment = await requestBillingPayment(payment);
      if (!completedPayment) {
        setRenderStatus({
          tone: 'error',
          text: 'Payment was cancelled. Image rendering was not started.',
        });
        return false;
      }

      recordPaidImageUse();
      appendRunLog(
        `Image ${chargeLabel} unlocked by mock payment ${completedPayment.reference}.`
      );
      return true;
    },
    [
      appendRunLog,
      consumeMonthlyImageCredit,
      recordPaidImageUse,
      requestBillingPayment,
    ]
  );

  useEffect(() => {
    if (!pendingBillingPayment) {
      return;
    }
    const expiresAtMs = Date.now() + 10000;
    const intervalId = window.setInterval(() => {
      const nextSeconds = Math.max(0, Math.ceil((expiresAtMs - Date.now()) / 1000));
      setBillingCountdownSeconds(nextSeconds);
      if (nextSeconds <= 0) {
        finishPendingBillingPayment(true);
      }
    }, 250);
    return () => window.clearInterval(intervalId);
  }, [finishPendingBillingPayment, pendingBillingPayment]);

  useEffect(() => {
    editModeRef.current = editMode;
    selectedIdRef.current = selectedId;
    if (!selectedId && editMode !== 'idle') {
      setEditMode('idle');
      return;
    }
    if (controlsRef.current) {
      controlsRef.current.enabled = editMode === 'idle';
    }
    if (import.meta.env.DEV) {
      console.debug('[3D]', {
        editMode,
        selectedId,
        controlsEnabled: controlsRef.current?.enabled,
      });
    }
    updateRotatorsVisibilityRef.current?.();
  }, [editMode, selectedId]);

  useEffect(() => {
    if (!renderStateRef.current) return;
    renderStateRef.current.variantId = selectedVariantId;
    renderStateRef.current.selectedObjectId = selectedId;
    renderStateRef.current.selectedWallId = selectedWallId;
    renderStateRef.current.editMode = editMode;
  }, [editMode, selectedId, selectedVariantId, selectedWallId]);

  useEffect(() => {
    let cancelled = false;
    fetchRenderPresets()
      .then((presets) => {
        if (cancelled) return;
        setRenderPresets(presets);
        setPresetSelection((current) => ({
          style: firstPresetKey(presets, 'styles', current.style ?? 'modern'),
          lighting: firstPresetKey(presets, 'lights', current.lighting ?? 'daylight_1'),
          scenery: firstPresetKey(presets, 'sceneries', current.scenery ?? 'none'),
        }));
      })
      .catch((presetError) => {
        if (cancelled) return;
        setRenderStatus({
          tone: 'error',
          text:
            presetError instanceof Error
              ? presetError.message
              : 'Failed to load image presets.',
        });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const objects: RenderObject[] = useMemo(() => {
    const out: RenderObject[] = [];
    const placements = design?.placements ?? [];
    const placementsById = new Map<string, any>();
    for (const p of placements) {
      const pid = String(p.placement_id ?? p.id ?? p.object_id ?? '');
      if (!pid) continue;
      placementsById.set(pid, p);
    }

    const rawObjects = styledResult?.objects;
    if (Array.isArray(rawObjects)) {
      const used = new Set<string>();
      for (const obj of rawObjects) {
        const id = String(obj.instance_id ?? obj.id ?? obj.object_type ?? `obj_${out.length + 1}`);
        const objType = String(obj.object_type ?? obj.category ?? 'item');
        const placeOn = obj.place_on
          ? {
              method: String(obj.place_on.method ?? 'floor'),
              targetId: obj.place_on.target_instance_id ?? null,
            }
          : null;
        const placement = placementsById.get(id);
        const styledBbox = getNumericBbox(obj);
        let minX = styledBbox.minX;
        let minY = styledBbox.minY;
        let maxX = styledBbox.maxX;
        let maxY = styledBbox.maxY;
        let rotationCcw = Number(obj.rotation_ccw ?? 0);

        if (placement) {
          used.add(id);
          if (!placementMatchesStyledObject(placement, styledBbox, rotationCcw)) {
            const placementBbox = getWorldBboxFromPlacement(placement);
            minX = placementBbox.minX;
            minY = placementBbox.minY;
            maxX = placementBbox.maxX;
            maxY = placementBbox.maxY;
            rotationCcw = Number(placement.rotation_deg ?? rotationCcw ?? 0);
          }
        }

        const rawAssetId =
          placement?.asset_id ??
          obj.asset_id ??
          obj.assetId ??
          obj.object_id ??
          null;
        const frontWorld = placement?.front_world ?? obj.front_world ?? null;
        const frontSideWorld = placement?.front_side_world ?? obj.front_side_world ?? null;
        const frontSideLocal = placement?.front_side_local ?? obj.front_side_local ?? null;
        out.push({
          id,
          assetId: normalizeRenderAssetId(rawAssetId) ?? null,
          rawAssetId: typeof rawAssetId === 'string' ? rawAssetId : null,
          type: objType,
          colorHex: String(obj.color_hex ?? '#cfcac2'),
          material: obj.material ?? null,
          rotationCcw,
          frontWorld,
          frontSideWorld,
          frontSideLocal,
          bbox: { minX, minY, maxX, maxY },
          placementFootprintMm: placement?.footprint_mm ?? null,
          placeOn,
          draggable:
            placementsById.has(id) &&
            placementsById.get(id)?.constraint_mode !== 'locked',
          reintroduced:
            isReintroducedIdentifier(obj.object_id) ||
            isReintroducedIdentifier(rawAssetId) ||
            isReintroducedIdentifier(id),
        });
      }

      for (const p of placements) {
        const pid = String(p.placement_id ?? p.id ?? p.object_id ?? '');
        if (!pid || used.has(pid)) continue;
        const bbox = getWorldBboxFromPlacement(p);
        out.push({
          id: pid,
          assetId: normalizeRenderAssetId(p.asset_id ?? null) ?? null,
          rawAssetId: typeof p.asset_id === 'string' ? p.asset_id : null,
          type: p.category,
          colorHex: p.color_hex ?? '#cfcac2',
          material: null,
          rotationCcw: Number(p.rotation_deg ?? 0),
          frontWorld: p.front_world ?? null,
          frontSideWorld: p.front_side_world ?? null,
          frontSideLocal: p.front_side_local ?? null,
          bbox,
          placementFootprintMm: p.footprint_mm ?? null,
          placeOn: { method: 'floor', targetId: 'floor' },
          draggable: p.constraint_mode !== 'locked',
          reintroduced:
            isReintroducedIdentifier(p.asset_id) ||
            isReintroducedIdentifier(p.placement_id) ||
            isReintroducedIdentifier(pid),
        });
      }

      return out;
    }

    for (const placement of placements) {
      const pid = String(placement.placement_id ?? placement.id ?? placement.object_id ?? '');
      out.push({
        id: pid || String(out.length + 1),
        assetId: normalizeRenderAssetId(placement.asset_id ?? null) ?? null,
        rawAssetId:
          typeof placement.asset_id === 'string' ? placement.asset_id : null,
        type: placement.category,
        colorHex: placement.color_hex ?? '#cfcac2',
        material: null,
        rotationCcw: Number(placement.rotation_deg ?? 0),
        frontWorld: placement.front_world ?? null,
        frontSideWorld: placement.front_side_world ?? null,
        frontSideLocal: placement.front_side_local ?? null,
        bbox: getWorldBboxFromPlacement(placement),
        placementFootprintMm: placement.footprint_mm ?? null,
        placeOn: { method: 'floor', targetId: 'floor' },
        draggable: placement.constraint_mode !== 'locked',
        reintroduced:
          isReintroducedIdentifier(placement.asset_id) ||
          isReintroducedIdentifier(placement.placement_id),
      });
    }
    return out;
  }, [styledResult, design]);

  const resolvedObjects: ResolvedRenderObject[] = useMemo(
    () =>
      objects.map((obj) => {
        const targetWidthMm = Math.max(
          1,
          Number(obj.placementFootprintMm?.w ?? obj.bbox.maxX - obj.bbox.minX),
        );
        const targetDepthMm = Math.max(
          1,
          Number(obj.placementFootprintMm?.d ?? obj.bbox.maxY - obj.bbox.minY),
        );
        const inventoryResolution = resolveInventoryItemForPreview({
          inventory,
          assetId: obj.assetId ?? obj.rawAssetId ?? null,
          objectType: obj.type,
          material: obj.material ?? null,
          preferredStyle: userInput.style,
          targetWidthMm,
          targetDepthMm,
          targetHeightMm: defaultHeightMm(obj.type),
        });
        return {
          ...obj,
          inventoryResolution,
        };
      }),
    [inventory, objects, userInput.style]
  );

  const reintroducedObjects = useMemo(() => {
    const fromSummary = collectReintroducedObjects(styledResult);
    if (fromSummary.length > 0) {
      return fromSummary;
    }
    return resolvedObjects
      .filter((obj) => obj.reintroduced)
      .map((obj) => ({
        clusterId: null,
        objectId: obj.id,
        objectType: obj.type,
      }));
  }, [resolvedObjects, styledResult]);

  const proxyObjects = useMemo(
    () =>
      Object.values(objectVisualDiagnostics)
        .filter((item) => item.status !== 'glb_loaded' && item.status !== 'glb_loading')
        .sort((left, right) => left.label.localeCompare(right.label)),
    [objectVisualDiagnostics]
  );

  const glbObjects = useMemo(
    () =>
      Object.values(objectVisualDiagnostics)
        .filter((item) => item.status === 'glb_loading' || item.status === 'glb_loaded')
        .sort((left, right) => left.label.localeCompare(right.label)),
    [objectVisualDiagnostics]
  );
  const pendingGlbCount = glbObjects.filter(
    (item) => item.status === 'glb_loading'
  ).length;
  const showThreeLoadingOverlay = isThreeSceneLoading || pendingGlbCount > 0;
  const midHeightCameraPresets = useMemo(
    () => cameraPresets.filter((preset) => preset.group === 'mid'),
    [cameraPresets]
  );
  const ceilingCameraPresets = useMemo(
    () => cameraPresets.filter((preset) => preset.group === 'ceiling'),
    [cameraPresets]
  );
  const activeCameraPreset = useMemo(
    () => cameraPresets.find((preset) => preset.id === activeCameraPresetId) ?? null,
    [activeCameraPresetId, cameraPresets]
  );
  const stylePresetOptions = useMemo(
    () => buildReferencePresetDropdownOptions(renderPresets?.styles ?? {}),
    [renderPresets]
  );
  const lightingPresetOptions = useMemo(
    () => buildReferencePresetDropdownOptions(renderPresets?.lights ?? {}),
    [renderPresets]
  );
  const sceneryPresetOptions = useMemo(
    () => buildReferencePresetDropdownOptions(renderPresets?.sceneries ?? {}),
    [renderPresets]
  );

  const lastVisibleObjects = lastRenderedSnapshot?.visibleObjects ?? [];
  const visibleObjectById = useMemo(
    () => new Map(lastVisibleObjects.map((object) => [object.id, object])),
    [lastVisibleObjects]
  );
  const activeObjectAppearanceEntries = useMemo(
    () =>
      Object.values(objectAppearanceReferences)
        .filter(
          (reference) =>
            (reference.rootLayoutId ?? null) ===
            (renderLayoutContext.rootLayoutId ?? null)
        )
        .sort((left, right) => right.savedAtMs - left.savedAtMs),
    [objectAppearanceReferences, renderLayoutContext.rootLayoutId]
  );
  const selectedObjectIsEditable = Boolean(
    selectedId && visibleObjectById.has(selectedId)
  );
  const editPresetActive = hasPresetSelection(editPresetSelection);
  const configuredEditBoxes = useMemo(
    () =>
      editBoxes.filter((box) => {
        if (box.mode === 'recolor') {
          return Boolean(box.targetColor);
        }
        if (box.mode === 'add' || box.mode === 'replace') {
          return Boolean(box.imageDataUrl || box.prompt.trim());
        }
        return Boolean(box.prompt.trim());
      }),
    [editBoxes]
  );
  const activeEditBox =
    activeEditBoxId !== null
      ? editBoxes.find((box) => box.id === activeEditBoxId) ?? null
      : editBoxes[editBoxes.length - 1] ?? null;
  const roomSurfaces =
    styledResult?.room?.surfaces && typeof styledResult.room.surfaces === 'object'
      ? styledResult.room.surfaces
      : {};
  const wallSurfaceColor = normalizeHexColor(
    roomSurfaces.wall_color_hex ?? roomSurfaces.wallColorHex,
    '#555555'
  );
  const floorSurfaceColor = normalizeHexColor(
    roomSurfaces.floor_color_hex ??
      roomSurfaces.floorColorHex ??
      activeFloorplan?.room?.floor_color_hex,
    '#f0ece4'
  );
  const ceilingSurfaceColor = normalizeHexColor(
    roomSurfaces.ceiling_color_hex ?? roomSurfaces.ceilingColorHex,
    '#fbf7f2'
  );
  const selectedObjectForAppearance =
    selectedId !== null
      ? objects.find((object) => object.id === selectedId) ?? null
      : null;

  function updateRoomSurfaceColor(
    key: 'wall_color_hex' | 'floor_color_hex' | 'ceiling_color_hex',
    value: string
  ): void {
    const nextColor = normalizeHexColor(value, '#ffffff');
    const existingStyledResult =
      styledResult && typeof styledResult === 'object'
        ? (styledResult as Record<string, any>)
        : {};
    const existingRoom =
      existingStyledResult.room && typeof existingStyledResult.room === 'object'
        ? existingStyledResult.room
        : {};
    const existingSurfaces =
      existingRoom.surfaces && typeof existingRoom.surfaces === 'object'
        ? existingRoom.surfaces
        : {};
    setStyledResult({
      ...existingStyledResult,
      room: {
        ...existingRoom,
        surfaces: {
          ...existingSurfaces,
          [key]: nextColor,
        },
      },
    });

    if (key === 'floor_color_hex' && activeFloorplan) {
      setFloorplan({
        ...activeFloorplan,
        room: {
          ...activeFloorplan.room,
          floor_color_hex: nextColor,
        },
      });
    }
  }

  function updateStyledObjectColor(objectId: string, value: string): void {
    const nextColor = normalizeHexColor(value, '#cfcac2');
    if (
      !styledResult ||
      typeof styledResult !== 'object' ||
      !Array.isArray((styledResult as { objects?: unknown[] }).objects)
    ) {
      return;
    }
    setStyledResult({
      ...(styledResult as Record<string, unknown>),
      objects: (styledResult as { objects: unknown[] }).objects.map((item) => {
        if (!item || typeof item !== 'object') {
          return item;
        }
        const objectRow = item as Record<string, unknown>;
        const currentId =
          typeof objectRow.instance_id === 'string'
            ? objectRow.instance_id
            : typeof objectRow.id === 'string'
              ? objectRow.id
              : null;
        if (currentId !== objectId) {
          return item;
        }
        return {
          ...objectRow,
          color_hex: nextColor,
        };
      }),
    });
  }

  function handleObjectColorChange(objectId: string, value: string): void {
    const nextColor = normalizeHexColor(value, '#cfcac2');
    const placement = design?.placements.find(
      (candidate) =>
        String(candidate.placement_id ?? candidate.id ?? candidate.object_id ?? '') === objectId
    );
    if (placement) {
      updatePlacement({
        ...placement,
        color_hex: nextColor,
      });
      return;
    }
    updateStyledObjectColor(objectId, nextColor);
  }

  function buildAppearanceMemoryOperations(
    snapshotArtifact: ThreeSnapshotArtifact
  ): SnapshotImageEditOperation[] {
    const visibleWithReferences = snapshotArtifact.visibleObjects
      .map((object) => {
        const reference = objectAppearanceReferences[object.id] ?? null;
        if (!reference) {
          return null;
        }
        if (
          (reference.rootLayoutId ?? null) !==
          (renderLayoutContext.rootLayoutId ?? null)
        ) {
          return null;
        }
        const box = object.screenBboxPx;
        const screenArea = box ? Math.max(0, box.width * box.height) : 0;
        return { object, reference, screenArea };
      })
      .filter(
        (
          entry
        ): entry is {
          object: VisibleSnapshotObject;
          reference: ObjectAppearanceReference;
          screenArea: number;
        } => entry !== null
      )
      .sort((left, right) => right.screenArea - left.screenArea)
      .slice(0, 6);

    return visibleWithReferences.map(({ object, reference }) => ({
      object_id: object.id,
      object_name: visibleObjectName(object),
      replacement_image_data_url: reference.imageDataUrl,
      target_color: null,
    }));
  }

  function buildEditBoxOperations(boxes: RenderEditBox[]): SnapshotImageEditOperation[] {
    return boxes.map((box) => ({
      object_id: box.id,
      object_name: box.prompt.trim() || editBoxModeLabel(box.mode).toLowerCase(),
      operation_type: box.mode,
      prompt: box.prompt.trim() || null,
      bbox_norm: box.rect,
      replacement_image_data_url: box.imageDataUrl,
      target_color: box.mode === 'recolor' ? box.targetColor : null,
    }));
  }

  function resolveEditedObjectsForMemory(
    boxes: RenderEditBox[],
    snapshot: ThreeSnapshotArtifact | null
  ): Array<{ object: VisibleSnapshotObject; box: RenderEditBox }> {
    if (!snapshot || boxes.length === 0) {
      return [];
    }
    const canvasWidth = Math.max(1, snapshot.canvas.widthPx);
    const canvasHeight = Math.max(1, snapshot.canvas.heightPx);
    const matches = new Map<string, { object: VisibleSnapshotObject; box: RenderEditBox; score: number }>();

    boxes.forEach((box) => {
      snapshot.visibleObjects.forEach((object) => {
        const objectBox = object.screenBboxPx;
        if (!objectBox) {
          return;
        }
        const objectRect = {
          x: objectBox.minX / canvasWidth,
          y: objectBox.minY / canvasHeight,
          width: objectBox.width / canvasWidth,
          height: objectBox.height / canvasHeight,
        };
        const intersectionScore = editBoxObjectOverlapScore(box.rect, objectRect);
        const centerX = objectRect.x + objectRect.width / 2;
        const centerY = objectRect.y + objectRect.height / 2;
        const centerInside =
          centerX >= box.rect.x &&
          centerX <= box.rect.x + box.rect.width &&
          centerY >= box.rect.y &&
          centerY <= box.rect.y + box.rect.height;
        const score = centerInside ? Math.max(intersectionScore, 0.55) : intersectionScore;
        if (score < 0.18) {
          return;
        }
        const existing = matches.get(object.id);
        if (!existing || score > existing.score) {
          matches.set(object.id, { object, box, score });
        }
      });
    });

    return Array.from(matches.values())
      .sort((left, right) => right.score - left.score)
      .map(({ object, box }) => ({ object, box }));
  }

  function editBoxObjectOverlapScore(
    editRect: EditBoxRect,
    objectRect: EditBoxRect
  ): number {
    const left = Math.max(editRect.x, objectRect.x);
    const top = Math.max(editRect.y, objectRect.y);
    const right = Math.min(editRect.x + editRect.width, objectRect.x + objectRect.width);
    const bottom = Math.min(editRect.y + editRect.height, objectRect.y + objectRect.height);
    const intersectionArea = Math.max(0, right - left) * Math.max(0, bottom - top);
    const objectArea = Math.max(0.0001, objectRect.width * objectRect.height);
    return intersectionArea / objectArea;
  }

  function updateEditBox(
    boxId: string,
    updater: (box: RenderEditBox) => RenderEditBox
  ): void {
    setEditBoxes((current) =>
      current.map((box) => (box.id === boxId ? updater(box) : box))
    );
  }

  function resolveEditBoxPoint(event: React.PointerEvent<HTMLDivElement>): {
    x: number;
    y: number;
  } {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1),
      y: clamp((event.clientY - rect.top) / Math.max(1, rect.height), 0, 1),
    };
  }

  function handleEditBoxPointerDown(
    event: React.PointerEvent<HTMLDivElement>
  ): void {
    if (!renderedSnapshotImage || isRenderingImage) {
      return;
    }
    event.preventDefault();
    const point = resolveEditBoxPoint(event);
    setEditBoxDraft({
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
    });
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function handleEditBoxPointerMove(
    event: React.PointerEvent<HTMLDivElement>
  ): void {
    if (!editBoxDraft) {
      return;
    }
    const point = resolveEditBoxPoint(event);
    setEditBoxDraft((current) =>
      current
        ? {
            ...current,
            currentX: point.x,
            currentY: point.y,
          }
        : null
    );
  }

  function handleEditBoxPointerUp(
    event: React.PointerEvent<HTMLDivElement>
  ): void {
    if (!editBoxDraft) {
      return;
    }
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    const rect = normalizeEditBoxRect(editBoxDraft);
    setEditBoxDraft(null);
    if (!rect) {
      return;
    }
    const nextIndex =
      Math.max(
        0,
        ...editBoxes.map((box) => Number(/^R(\d+)$/i.exec(box.id)?.[1] ?? 0))
      ) + 1;
    const nextBox: RenderEditBox = {
      id: `R${nextIndex}`,
      mode: 'prompt',
      rect,
      prompt: '',
      imageDataUrl: null,
      fileName: null,
      targetColor,
    };
    setEditBoxes((current) => [...current, nextBox]);
    setActiveEditBoxId(nextBox.id);
  }

  async function persistObjectAppearanceReferencesForRender(
    render: GeneratedRenderRecord,
    references: Record<string, ObjectAppearanceReference>,
    metaPatch: Record<string, unknown> = {}
  ): Promise<GeneratedRenderRecord> {
      const memoryMeta = buildObjectAppearanceReferencesMeta({
        references,
        rootLayoutId: renderLayoutContext.rootLayoutId,
      });
    const rawReferences = memoryMeta.references;
    const referenceCount = Array.isArray(rawReferences) ? rawReferences.length : 0;
    return updateGeneratedRenderMeta(render.id, {
      ...metaPatch,
      [OBJECT_APPEARANCE_REFERENCES_META_KEY]: memoryMeta,
      object_appearance_reference_count: referenceCount,
    });
  }

  useEffect(() => {
    const nextDiagnostics = Object.fromEntries(
      resolvedObjects.map((obj) => {
        const modelResolution = obj.inventoryResolution.modelResolution;
        let status: ObjectVisualDiagnostic['status'] = 'proxy_no_inventory_match';
        let reason = obj.inventoryResolution.reason;
        if (obj.inventoryResolution.item) {
          if (obj.inventoryResolution.glbEligible && modelResolution?.descriptor) {
            status = 'glb_loading';
            reason = obj.inventoryResolution.glbReason;
          } else if (modelResolution?.descriptor) {
            status = 'proxy_low_confidence_match';
            reason = obj.inventoryResolution.glbReason;
          } else {
            status = 'proxy_no_model_descriptor';
            reason = modelResolution?.reason ?? 'Matched inventory asset has no GLB metadata.';
          }
        }
        return [
          obj.id,
          {
            objectId: obj.id,
            label: obj.type,
            rawAssetId: obj.rawAssetId ?? obj.assetId ?? null,
            resolvedAssetId: obj.inventoryResolution.item?.asset_id ?? null,
            reintroduced: Boolean(obj.reintroduced),
            modelUrl: modelResolution?.descriptor?.url ?? null,
            previewRotationOffsetDeg: null,
            finalDescriptorRotationDeg: modelResolution?.descriptor?.rotationDeg ?? null,
            status,
            reason,
          } satisfies ObjectVisualDiagnostic,
        ];
      })
    );
    setObjectVisualDiagnostics(nextDiagnostics);
  }, [resolvedObjects]);

  function handleApplyCameraPreset(preset: CameraPreset): void {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    if (!camera || !controls) {
      return;
    }

    applyingCameraPresetRef.current = true;
    userCameraInteractionRef.current = false;
    applyOrbitCameraPose(camera, controls, preset);
    renderer?.render?.(scene, camera);
    lastAppliedCameraPresetIdRef.current = preset.id;
    setActiveCameraPresetId(preset.id);
    window.requestAnimationFrame(() => {
      applyingCameraPresetRef.current = false;
    });
  }

  function handleCameraPresetSelection(presetId: string | null): void {
    if (!presetId) {
      setActiveCameraPresetId(null);
      return;
    }
    const preset = cameraPresets.find((candidate) => candidate.id === presetId) ?? null;
    if (!preset) {
      return;
    }
    handleApplyCameraPreset(preset);
  }

  async function handleCaptureSnapshot(): Promise<void> {
    setIsCapturing(true);
    setCaptureStatus({
      tone: 'neutral',
      text: 'Capturing the current 3D view...',
    });

    try {
      const { snapshotArtifact, snapshotImageDataUrl } =
        await captureCurrentAiSnapshot();
      const pngBlob = dataUrlToBlob(snapshotImageDataUrl);
      const jsonBlob = new Blob([JSON.stringify(snapshotArtifact, null, 2)], {
        type: 'application/json',
      });
      const fileStem = buildSnapshotFileStem(
        snapshotArtifact.variantId ?? selectedVariantId,
        snapshotArtifact.capturedAt
      );

      downloadBlob(pngBlob, `${fileStem}.png`);
      downloadBlob(jsonBlob, `${fileStem}.json`);

      setCaptureStatus({
        tone: 'success',
        text: `Saved snapshot PNG and JSON with ${snapshotArtifact.visibleObjectCount} visible objects.`,
      });
    } catch (captureError) {
      setCaptureStatus({
        tone: 'error',
        text:
          captureError instanceof Error
            ? captureError.message
            : 'Failed to capture the current 3D snapshot.',
      });
    } finally {
      setIsCapturing(false);
    }
  }

  async function captureCurrentAiSnapshot(): Promise<{
    snapshotArtifact: ThreeSnapshotArtifact;
    snapshotImageDataUrl: string;
    cleanSnapshotImageDataUrl: string;
    depthMapImageDataUrl: string;
    edgeMapImageDataUrl: string;
    segmentationMapImageDataUrl: string;
  }> {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const renderState = renderStateRef.current;
    if (
      !renderer ||
      !scene ||
      !camera ||
      !renderState ||
      !(renderer.domElement instanceof HTMLCanvasElement)
    ) {
      throw new Error(
        '3D preview is not ready yet. Please wait for the scene to finish loading.'
      );
    }

    const hiddenNodes: TemporaryVisibilityRecord[] = [];
    const previousPixelRatio =
      typeof renderer.getPixelRatio === 'function' ? renderer.getPixelRatio() : null;
    const previousCameraAspect =
      typeof camera.aspect === 'number' ? camera.aspect : null;
    const restoreRect = renderer.domElement.parentElement?.getBoundingClientRect();
    const restoreWidth = Math.max(
      1,
      Math.round(restoreRect?.width || renderer.domElement.clientWidth || 1)
    );
    const restoreHeight = Math.max(
      1,
      Math.round(restoreRect?.height || renderer.domElement.clientHeight || 1)
    );

    try {
      const rotatorGroup = rotatorGroupRef.current;
      if (rotatorGroup && typeof rotatorGroup.visible === 'boolean') {
        hideNode(hiddenNodes, rotatorGroup);
      }
      hiddenNodes.push(...hideAiSnapshotHelperNodes(scene));
      const renderSize = resolveSnapshotReferenceRenderSize(renderer.domElement);

      if (typeof renderer.setPixelRatio === 'function') {
        renderer.setPixelRatio(1);
      }
      renderer.setSize(renderSize.width, renderSize.height, false);
      if (previousCameraAspect !== null) {
        camera.aspect = renderSize.width / renderSize.height;
        camera.updateProjectionMatrix?.();
      }
      controlsRef.current?.update?.();
      const snapshotArtifact = buildThreeSnapshotArtifact(renderState);
      renderer.render(scene, camera);
      const sourceImageDataUrl = canvasToDataUrl(renderer.domElement);
      const depthMapImageDataUrl = renderDepthMapDataUrl({
        THREE: renderState.THREE,
        renderer,
        scene,
        camera,
      });
      const segmentationMapImageDataUrl = renderSegmentationMapDataUrl({
        THREE: renderState.THREE,
        renderer,
        scene,
        camera,
      });
      const edgeMapImageDataUrl = await buildEdgeMapReferenceImageDataUrl({
        sourceImageDataUrl,
      });
      if (!edgeMapImageDataUrl) {
        throw new Error('Unable to build the edge-map 3D snapshot image.');
      }
      const labeledImageDataUrl = await buildStructuralPerspectiveReferenceImageDataUrl({
        sourceImageDataUrl,
        snapshot: snapshotArtifact,
      });
      if (!labeledImageDataUrl) {
        throw new Error('Unable to build the labeled 3D snapshot image.');
      }
      return {
        snapshotArtifact,
        snapshotImageDataUrl: labeledImageDataUrl,
        cleanSnapshotImageDataUrl: sourceImageDataUrl,
        depthMapImageDataUrl,
        edgeMapImageDataUrl,
        segmentationMapImageDataUrl,
      };
    } finally {
      restoreVisibility(hiddenNodes);
      if (previousPixelRatio !== null && typeof renderer.setPixelRatio === 'function') {
        renderer.setPixelRatio(previousPixelRatio);
      }
      renderer.setSize(restoreWidth, restoreHeight, false);
      if (previousCameraAspect !== null) {
        camera.aspect = previousCameraAspect;
        camera.updateProjectionMatrix?.();
      }
      controlsRef.current?.update?.();
      renderer.render(scene, camera);
    }
  }

  async function handleRenderSnapshotImage(): Promise<void> {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const renderState = renderStateRef.current;
    if (
      !renderer ||
      !scene ||
      !camera ||
      !renderState ||
      !(renderer.domElement instanceof HTMLCanvasElement)
    ) {
      setRenderStatus({
        tone: 'error',
        text: '3D preview is not ready yet. Please wait for the scene to finish loading.',
      });
      return;
    }

    flushSync(() => {
      setActiveImageRenderAction('generate');
      setRenderStatus({
        tone: 'neutral',
        text: 'Starting image generation...',
      });
    });
    try {
      await waitForNextPaint();
      const billingApproved = await ensureImageBilling('generate');
      if (!billingApproved) {
        return;
      }
      setRenderStatus({
        tone: 'neutral',
        text: 'Capturing clean 3D view, depth, edge, segmentation, and structural guide...',
      });
      const {
        snapshotArtifact,
        snapshotImageDataUrl,
        cleanSnapshotImageDataUrl,
        depthMapImageDataUrl,
        edgeMapImageDataUrl,
        segmentationMapImageDataUrl,
      } =
        await captureCurrentAiSnapshot();
      const appearanceMemoryOperations = buildAppearanceMemoryOperations(snapshotArtifact);
      const lockedGeometryAtlases = await buildLockedGeometryAtlases({
        imageDataUrl: cleanSnapshotImageDataUrl,
        snapshot: snapshotArtifact,
      });
      const renderResult: SnapshotImageRenderApiResponse = await renderSnapshotImage(
        {
          snapshotPayload: snapshotArtifact as Record<string, unknown>,
          snapshotImageDataUrl: cleanSnapshotImageDataUrl,
          depthMapImageDataUrl,
          edgeMapImageDataUrl,
          segmentationMapImageDataUrl,
          layoutReferenceImageDataUrl: null,
          annotatedReferenceImageDataUrl: snapshotImageDataUrl,
          sceneReferenceImageDataUrl: null,
          sceneReferenceMode: 'none',
          userPrompt: '',
          renderMode: 'generate',
          presetSelection,
          rootLayoutId: renderLayoutContext.rootLayoutId,
          caseId: renderLayoutContext.caseId,
          optionId: renderLayoutContext.optionId,
          editOperations: appearanceMemoryOperations,
          lockedGeometryAtlases,
        }
      );
      setRenderedSnapshotImage(renderResult.render);
      notifyRenderFocus();
      setLastRenderedSnapshot(snapshotArtifact);
      setLastRenderedSnapshotImageDataUrl(cleanSnapshotImageDataUrl);
      setEditBoxes([]);
      setActiveEditBoxId(null);
      setEditBoxDraft(null);
      if (renderResult.saved_render) {
        const savedWithMemory = await persistObjectAppearanceReferencesForRender(
          renderResult.saved_render,
          objectAppearanceReferences
        );
        upsertSavedRender(savedWithMemory);
        setSavedRenderId(savedWithMemory.id);
        selectSavedRender(savedWithMemory.id);
      } else {
        setSavedRenderId(null);
        selectSavedRender(null);
      }
      const successMessage =
        lockedGeometryAtlases.length > 0
          ? `Generated image with ${lockedGeometryAtlases.length} geometry atlas reference(s).`
          : appearanceMemoryOperations.length > 0
          ? `Generated image with ${appearanceMemoryOperations.length} object memory reference(s).`
          : `Generated image with ${renderResult.render.models.image_model_name}.`;
      setRenderStatus({
        tone: 'success',
        text: successMessage,
      });
    } catch (renderError) {
      setRenderStatus({
        tone: 'error',
        text:
          renderError instanceof Error
            ? renderError.message
            : 'Failed to generate an image from the current snapshot.',
      });
    } finally {
      setActiveImageRenderAction(null);
    }
  }

  async function handleCameraRerenderSnapshotImage(): Promise<void> {
    if (!renderedSnapshotImage) {
      setRenderStatus({
        tone: 'error',
        text: 'Generate an image before re-rendering from a new camera.',
      });
      return;
    }

    flushSync(() => {
      setActiveImageRenderAction('camera_rerender');
      setRenderStatus({
        tone: 'neutral',
        text: 'Starting camera re-render...',
      });
    });
    try {
      await waitForNextPaint();
      const billingApproved = await ensureImageBilling('generate');
      if (!billingApproved) {
        return;
      }
      setRenderStatus({
        tone: 'neutral',
        text: 'Capturing new 3D camera view, depth, edge, segmentation, and structural guide...',
      });
      const {
        snapshotArtifact,
        snapshotImageDataUrl,
        cleanSnapshotImageDataUrl,
        depthMapImageDataUrl,
        edgeMapImageDataUrl,
        segmentationMapImageDataUrl,
      } = await captureCurrentAiSnapshot();
      const appearanceMemoryOperations = buildAppearanceMemoryOperations(snapshotArtifact);
      const lockedGeometryAtlases = await buildLockedGeometryAtlases({
        imageDataUrl: cleanSnapshotImageDataUrl,
        snapshot: snapshotArtifact,
      });
      const renderResult: SnapshotImageRenderApiResponse = await renderSnapshotImage(
        {
          snapshotPayload: snapshotArtifact as Record<string, unknown>,
          snapshotImageDataUrl: cleanSnapshotImageDataUrl,
          depthMapImageDataUrl,
          edgeMapImageDataUrl,
          segmentationMapImageDataUrl,
          layoutReferenceImageDataUrl: null,
          annotatedReferenceImageDataUrl: snapshotImageDataUrl,
          sceneReferenceImageDataUrl: renderedSnapshotImage.image.data_url,
          sceneReferenceMode: 'target_layout_with_scene_reference',
          userPrompt: renderedSnapshotImage.request.raw_user_prompt ?? '',
          renderMode: 'camera_rerender',
          presetSelection:
            renderedSnapshotImage.request.preset_selection ?? presetSelection,
          rootLayoutId: renderLayoutContext.rootLayoutId,
          caseId: renderLayoutContext.caseId,
          optionId: renderLayoutContext.optionId,
          editOperations: appearanceMemoryOperations,
          lockedGeometryAtlases,
        }
      );
      setRenderedSnapshotImage(renderResult.render);
      notifyRenderFocus();
      setLastRenderedSnapshot(snapshotArtifact);
      setLastRenderedSnapshotImageDataUrl(cleanSnapshotImageDataUrl);
      setEditBoxes([]);
      setActiveEditBoxId(null);
      setEditBoxDraft(null);
      if (renderResult.saved_render) {
        const savedWithMemory = await persistObjectAppearanceReferencesForRender(
          renderResult.saved_render,
          objectAppearanceReferences,
          {
            [RENDER_STATE_META_KEY]: {
              version: 1,
              preset_selection:
                renderedSnapshotImage.request.preset_selection ?? presetSelection,
              raw_user_prompt: renderedSnapshotImage.request.raw_user_prompt ?? null,
              root_layout_id: renderLayoutContext.rootLayoutId,
              source_render_id: savedRenderId,
            },
          }
        );
        upsertSavedRender(savedWithMemory);
        setSavedRenderId(savedWithMemory.id);
        selectSavedRender(savedWithMemory.id);
      } else {
        setSavedRenderId(null);
        selectSavedRender(null);
      }
      setRenderStatus({
        tone: 'success',
        text:
          lockedGeometryAtlases.length > 0
            ? `Re-rendered camera with ${lockedGeometryAtlases.length} geometry atlas reference(s).`
            : appearanceMemoryOperations.length > 0
            ? `Re-rendered camera with ${appearanceMemoryOperations.length} object memory reference(s).`
            : `Re-rendered camera with ${renderResult.render.models.image_model_name}.`,
      });
    } catch (renderError) {
      setRenderStatus({
        tone: 'error',
        text:
          renderError instanceof Error
            ? renderError.message
            : 'Failed to re-render the image from the current camera.',
      });
    } finally {
      setActiveImageRenderAction(null);
    }
  }

  async function handleEditSnapshotImage(): Promise<void> {
    if (!renderedSnapshotImage) {
      setRenderStatus({
        tone: 'error',
        text: 'Generate an image before applying edits.',
      });
      return;
    }
    if (configuredEditBoxes.length === 0 && !editPresetActive) {
      setRenderStatus({
        tone: 'error',
        text: 'Draw an edit box or choose an edit preset before applying edits.',
      });
      return;
    }
    flushSync(() => {
      setActiveImageRenderAction('edit');
      setRenderStatus({
        tone: 'neutral',
        text: 'Starting image edit...',
      });
    });
    try {
      await waitForNextPaint();
      const billingApproved = await ensureImageBilling('edit');
      if (!billingApproved) {
        return;
      }
      setRenderStatus({
        tone: 'neutral',
        text: 'Applying the requested edit...',
      });
      const {
        snapshotArtifact: targetSnapshotArtifact,
        cleanSnapshotImageDataUrl: targetCleanSnapshotImageDataUrl,
      } = await captureCurrentAiSnapshot();
      const sourceSnapshotArtifact = lastRenderedSnapshot ?? targetSnapshotArtifact;
      const editOperations = buildEditBoxOperations(configuredEditBoxes);
      const editBoxGuideImageDataUrl = await buildEditBoxGuideImageDataUrl({
        sourceImageDataUrl: renderedSnapshotImage.image.data_url,
        boxes: configuredEditBoxes,
      });
      const effectivePresetSelection = mergePresetSelections(
        renderedSnapshotImage.request.preset_selection,
        editPresetSelection
      );
      const effectiveRawUserPrompt =
        renderedSnapshotImage.request.raw_user_prompt || null;

      const renderResult: SnapshotImageRenderApiResponse = await renderSnapshotImage(
        {
          snapshotPayload: sourceSnapshotArtifact as unknown as Record<string, unknown>,
          snapshotImageDataUrl: targetCleanSnapshotImageDataUrl,
          layoutReferenceImageDataUrl: null,
          annotatedReferenceImageDataUrl: editBoxGuideImageDataUrl,
          editSourceImageDataUrl: renderedSnapshotImage.image.data_url,
          userPrompt: '',
          renderMode: 'edit',
          presetSelection: editPresetSelection,
          rootLayoutId: renderLayoutContext.rootLayoutId,
          caseId: renderLayoutContext.caseId,
          optionId: renderLayoutContext.optionId,
          editOperations,
        }
      );
      const savedAtMs = Date.now();
      const memorySnapshot = sourceSnapshotArtifact;
      const editedObjectsForMemory = resolveEditedObjectsForMemory(
        configuredEditBoxes,
        memorySnapshot
      );
      const memoryEntries = (
        await Promise.all(
          editedObjectsForMemory.map(async ({ object, box }) => {
            const imageDataUrl =
              box.imageDataUrl && (box.mode === 'replace' || box.mode === 'add')
                ? box.imageDataUrl
                : await cropObjectAppearanceReference({
                    imageDataUrl: renderResult.render.image.data_url,
                    object,
                    snapshot: memorySnapshot,
                  });
            if (!imageDataUrl) {
              return null;
            }
            return {
              objectId: object.id,
              imageDataUrl,
              fileName: box.fileName ?? `edited-${object.id}.jpg`,
              savedAtMs,
              rootLayoutId: renderLayoutContext.rootLayoutId,
              sourceRenderId: renderResult.saved_render?.id ?? savedRenderId,
              prompt: box.prompt.trim() || null,
              operationType: box.mode,
              bboxNorm: box.rect,
            } satisfies ObjectAppearanceReference;
          })
        )
      ).filter(
        (entry): entry is ObjectAppearanceReference => entry !== null
      );
      const mergedObjectAppearanceReferences = {
        ...objectAppearanceReferences,
        ...Object.fromEntries(
          memoryEntries.map((reference) => [reference.objectId, reference])
        ),
      };
      const renderWithEffectiveState = withEffectivePresetSelection(
        renderResult.render,
        effectivePresetSelection,
        effectiveRawUserPrompt
      );
      setRenderedSnapshotImage(renderWithEffectiveState);
      notifyRenderFocus();
      setLastRenderedSnapshot(memorySnapshot);
      setLastRenderedSnapshotImageDataUrl(targetCleanSnapshotImageDataUrl);
      setObjectAppearanceReferences(mergedObjectAppearanceReferences);
      setEditBoxes([]);
      setActiveEditBoxId(null);
      setEditBoxDraft(null);
      setColorEnabled(false);
      setEditPresetSelection({
        style: null,
        lighting: null,
        scenery: null,
      });
      if (renderResult.saved_render) {
        const savedWithMemory = await persistObjectAppearanceReferencesForRender(
          renderResult.saved_render,
          mergedObjectAppearanceReferences,
          {
            preset_selection: effectivePresetSelection,
            raw_user_prompt: effectiveRawUserPrompt,
            [RENDER_STATE_META_KEY]: {
              version: 1,
              preset_selection: effectivePresetSelection,
              raw_user_prompt: effectiveRawUserPrompt,
              root_layout_id: renderLayoutContext.rootLayoutId,
              source_render_id: savedRenderId,
            },
            [EDIT_DELTA_META_KEY]: {
              user_prompt: null,
              preset_selection: editPresetSelection,
              edit_operations: editOperations,
              layout_reference_used: false,
            },
          }
        );
        upsertSavedRender(savedWithMemory);
        setSavedRenderId(savedWithMemory.id);
        selectSavedRender(savedWithMemory.id);
      } else {
        setSavedRenderId(null);
        selectSavedRender(null);
      }
      setRenderStatus({
        tone: 'success',
        text:
          memoryEntries.length > 0
            ? `Edited image and saved ${memoryEntries.length} object appearance memory reference(s).`
            : `Edited image with ${renderResult.render.models.image_model_name}.`,
      });
    } catch (editError) {
      setRenderStatus({
        tone: 'error',
        text:
          editError instanceof Error
            ? editError.message
            : 'Failed to edit the generated image.',
      });
    } finally {
      setActiveImageRenderAction(null);
    }
  }

  useEffect(() => {
    if (!selectedSavedRenderId) {
      return;
    }

    const selectedRender =
      savedRenders.find((render) => render.id === selectedSavedRenderId) ?? null;
    if (!selectedRender || selectedRender.source !== 'snapshot_render') {
      return;
    }
    const inferredRootLayout = resolveSavedLayoutForRender({
      render: selectedRender,
      savedLayouts,
      activeRootLayoutId,
    });
    if (
      savedRenderId === selectedRender.id &&
      renderedSnapshotImage !== null &&
      lastRenderedSnapshot !== null &&
      (!inferredRootLayout || activeRootLayoutId === inferredRootLayout.id)
    ) {
      return;
    }

    let cancelled = false;
    const loadSavedRender = async () => {
      setRenderStatus({
        tone: 'neutral',
        text: `Loading saved render ${selectedRender.id} for editing...`,
      });
      try {
        const renderDetail = await getGeneratedRenderDetail(selectedRender.id);
        if (cancelled) {
          return;
        }
        const hydratedRender = renderDetail.render;
        const snapshotArtifact = buildSavedRenderSnapshotArtifact(hydratedRender);
        if (inferredRootLayout) {
          const nextFloorplan =
            inferredRootLayout.floorplan_json as unknown as FloorplanGeometry;
          const nextDesign =
            inferredRootLayout.design_json as unknown as DesignPlanV1 | null;
          setFloorplan(nextFloorplan);
          setDesign(
            nextDesign ?? {
              floorplan_geometry: nextFloorplan,
              placements: [],
            }
          );
          setStyledResult(inferredRootLayout.styled_result_json ?? null);
          clearLayoutVariants();
          setActiveRootLayoutId(inferredRootLayout.id);
          setActiveLayoutCaseId(inferredRootLayout.case_id ?? null);
        }
        const imageDataUrl = renderDetail.image_data_url;
        setRenderedSnapshotImage(
          buildSavedRenderPreviewResult(
            hydratedRender,
            imageDataUrl,
            snapshotArtifact
          )
        );
        notifyRenderFocus();
        setLastRenderedSnapshot(snapshotArtifact);
        setLastRenderedSnapshotImageDataUrl(imageDataUrl);
        setSavedRenderId(hydratedRender.id);
        setEditBoxes([]);
        setActiveEditBoxId(null);
        setEditBoxDraft(null);
        setObjectAppearanceReferences(
          readObjectAppearanceReferencesFromRender(hydratedRender)
        );
        setColorEnabled(false);
        setEditPresetSelection({
          style: null,
          lighting: null,
          scenery: null,
        });
        setRenderStatus({
          tone: 'success',
          text: inferredRootLayout
            ? `Loaded saved render ${hydratedRender.id} with its root layout.`
            : `Loaded saved render ${hydratedRender.id}. No linked root layout was found, so the current room stays in place.`,
        });
      } catch (loadError) {
        if (cancelled) {
          return;
        }
        setRenderStatus({
          tone: 'error',
          text:
            loadError instanceof Error
              ? loadError.message
              : 'Failed to load the saved render.',
        });
      }
    };

    void loadSavedRender();
    return () => {
      cancelled = true;
    };
  }, [
    activeRootLayoutId,
    clearLayoutVariants,
    lastRenderedSnapshot,
    renderedSnapshotImage,
    savedRenderId,
    savedLayouts,
    savedRenders,
    setActiveLayoutCaseId,
    setActiveRootLayoutId,
    setFloorplan,
    setStyledResult,
    setDesign,
    selectedSavedRenderId,
  ]);

  useEffect(() => {
    if (viewMode !== '3d') {
      setIsThreeSceneLoading(false);
      return;
    }
    if (!containerRef.current) {
      setIsThreeSceneLoading(false);
      return;
    }
    if (!activeFloorplan) {
      setError('Missing floorplan data for 3D preview.');
      setCameraPresets([]);
      setIsThreeSceneLoading(false);
      return;
    }

    let disposed = false;
    let raf = 0;
    let renderer: any = null;
    let controls: any = null;
    let resizeObserver: ResizeObserver | null = null;

    const el = containerRef.current;
    setError(null);
    setIsThreeSceneLoading(true);
    const finishSceneLoading = () => {
      if (!disposed) {
        setIsThreeSceneLoading(false);
      }
    };

    const run = async (width: number, height: number) => {
      try {
        const THREE_URL = new URL('../vendor/three/three.module.js', import.meta.url).toString();
        const ORBIT_URL = new URL('../vendor/three/OrbitControls.js', import.meta.url).toString();
        const GLTF_LOADER_URL = new URL(
          '../vendor/three/loaders/GLTFLoader.js',
          import.meta.url
        ).toString();
        const KTX2_LOADER_URL = new URL(
          '../vendor/three/loaders/KTX2Loader.js',
          import.meta.url
        ).toString();

        const THREE: any = await import(/* @vite-ignore */ THREE_URL);
        const controlsMod: any = await import(/* @vite-ignore */ ORBIT_URL);
        const OrbitControls = controlsMod.OrbitControls;
        let GLTFLoader: any = null;
        let KTX2Loader: any = null;
        try {
          const gltfLoaderMod: any = await import(/* @vite-ignore */ GLTF_LOADER_URL);
          GLTFLoader = gltfLoaderMod.GLTFLoader ?? null;
        } catch {
          GLTFLoader = null;
        }
        try {
          const ktx2LoaderMod: any = await import(/* @vite-ignore */ KTX2_LOADER_URL);
          KTX2Loader = ktx2LoaderMod.KTX2Loader ?? null;
        } catch {
          KTX2Loader = null;
        }

        if (disposed || !el.isConnected) {
          finishSceneLoading();
          return;
        }
        if (import.meta.env.DEV) {
          console.debug('[3D] init', { width, height });
        }

        renderer = new THREE.WebGLRenderer({
          antialias: true,
          preserveDrawingBuffer: true,
        });
        renderer.setPixelRatio(
          Math.min(window.devicePixelRatio || 1, MAX_RENDERER_PIXEL_RATIO)
        );
        renderer.setSize(width, height);
        renderer.setClearColor(0xf6f4ef, 1);
        renderer.domElement.style.touchAction = 'none';
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.domElement.style.width = '100%';
        renderer.domElement.style.height = '100%';
        renderer.domElement.style.display = 'block';
        const gl = renderer.getContext?.();
        if (!gl || (typeof gl.isContextLost === 'function' && gl.isContextLost())) {
          setError('WebGL context unavailable. Please enable WebGL in your browser.');
          finishSceneLoading();
          return;
        }
        while (el.firstChild) {
          el.removeChild(el.firstChild);
        }
        el.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        sceneRef.current = scene;
        rendererRef.current = renderer;
        const ambient = new THREE.AmbientLight(0xffffff, 0.85);
        scene.add(ambient);
        const dir = new THREE.DirectionalLight(0xffffff, 0.9);
        dir.position.set(6, 10, 4);
        dir.castShadow = true;
        dir.shadow.mapSize.width = 2048;
        dir.shadow.mapSize.height = 2048;
        dir.shadow.camera.near = 0.1;
        dir.shadow.camera.far = 60;
        scene.add(dir);
        const hemi = new THREE.HemisphereLight(0xffffff, 0xf3efe7, 0.6);
        scene.add(hemi);

        const roomPtsRaw =
          activeFloorplan.room?.polygon_mm ??
          activeFloorplan.room?.polygon_ccw ??
          styledResult?.room?.polygon_mm ??
          styledResult?.room?.polygon_ccw ??
          [];
        const roomPts = normalizeRoomPolygon(
          Array.isArray(roomPtsRaw)
            ? roomPtsRaw
                .map((p: any) => ({ x: Number(p?.x ?? 0), y: Number(p?.y ?? 0) }))
                .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
            : []
        );
        if (!initialDesignRef.current && design) {
          initialDesignRef.current = JSON.parse(JSON.stringify(design));
        }
        const roomBounds = roomPts.length >= 3 ? polygonBounds(roomPts) : null;
        let minX = roomBounds?.minX ?? 0;
        let minY = roomBounds?.minY ?? 0;
        let maxX = roomBounds?.maxX ?? 6000;
        let maxY = roomBounds?.maxY ?? 4000;
        if (!roomBounds && resolvedObjects.length) {
          minX = Infinity;
          minY = Infinity;
          maxX = -Infinity;
          maxY = -Infinity;
        }

        const roomCenterX = roomBounds ? (roomBounds.minX + roomBounds.maxX) / 2 : (minX + maxX) / 2;
        const roomCenterY = roomBounds ? (roomBounds.minY + roomBounds.maxY) / 2 : (minY + maxY) / 2;
        const roomSpan = roomBounds
          ? Math.max(1, roomBounds.maxX - roomBounds.minX, roomBounds.maxY - roomBounds.minY)
          : null;
        const outlierLimit = roomSpan ? roomSpan * 4 : null;

        for (const obj of resolvedObjects) {
          const { minX: ox1, minY: oy1, maxX: ox2, maxY: oy2 } = obj.bbox;
          if (![ox1, oy1, ox2, oy2].every((v) => Number.isFinite(v))) continue;
          if (outlierLimit !== null) {
            const cx = (ox1 + ox2) / 2;
            const cy = (oy1 + oy2) / 2;
            const dist = Math.hypot(cx - roomCenterX, cy - roomCenterY);
            if (dist > outlierLimit) continue;
          }
          minX = Math.min(minX, ox1);
          minY = Math.min(minY, oy1);
          maxX = Math.max(maxX, ox2);
          maxY = Math.max(maxY, oy2);
        }

        if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
          minX = 0;
          minY = 0;
          maxX = 6000;
          maxY = 4000;
        }

        const bounds = { minX, minY, maxX, maxY };
        const spanX = Math.max(1, maxX - minX);
        const spanY = Math.max(1, maxY - minY);
        const centerXmm = (minX + maxX) / 2;
        const centerYmm = (minY + maxY) / 2;

        const maxDimM = Math.max(spanX, spanY) * MM_TO_M;
        if (!Number.isFinite(maxDimM) || !Number.isFinite(centerXmm) || !Number.isFinite(centerYmm)) {
          setError('Invalid floorplan bounds for 3D preview.');
          finishSceneLoading();
          return;
        }
        const wallHeightMm = Number(userInput.height ?? 2800);
        const wallHeight = wallHeightMm * MM_TO_M;
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 500);
        camera.far = Math.max(500, maxDimM * 8);
        camera.updateProjectionMatrix();
        camera.position.set(0, Math.max(2.5, maxDimM * 0.9), Math.max(3.5, maxDimM * 1.2));
        cameraRef.current = camera;
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enabled = editModeRef.current === 'idle';
        controls.enableRotate = true;
        controls.enablePan = true;
        controls.enableZoom = true;
        configureRoomOrbitControls(controls, maxDimM);
        controlsRef.current = controls;
        controls.target.set(0, clamp(wallHeight * 0.42, 1.0, wallHeight * 0.6), 0);
        controls.update();
        camera.lookAt(0, clamp(wallHeight * 0.42, 1.0, wallHeight * 0.6), 0);

        const floorColorHex = normalizeHexColor(
          styledResult?.room?.surfaces?.floor_color_hex ??
          styledResult?.room?.surfaces?.floorColorHex ??
          activeFloorplan.room?.floor_color_hex,
          '#f0ece4'
        );
        const ceilingColorHex = normalizeHexColor(
          styledResult?.room?.surfaces?.ceiling_color_hex ??
          styledResult?.room?.surfaces?.ceilingColorHex ??
          null,
          '#fbf7f2'
        );

        const floorPts =
          roomPts.length >= 3
            ? roomPts
            : [
                { x: minX, y: minY },
                { x: maxX, y: minY },
                { x: maxX, y: maxY },
                { x: minX, y: maxY },
              ];
        const roomPolygonMm = normalizeRoomPolygon(floorPts);
        const focusPointMm = resolveCameraFocusPointMm({
          roomPolygonMm,
          objects: resolvedObjects,
          fallbackPointMm: { x: centerXmm, y: centerYmm },
        });
        const nextCameraPresets = buildCameraPresets({
          roomPolygonMm,
          roomBoundsMm: roomBounds ?? bounds,
          roomCenterMm: { x: centerXmm, y: centerYmm },
          focusPointMm,
          wallHeightMm,
          aspect: width / Math.max(height, 1),
        });
        if (!disposed) {
          setCameraPresets(nextCameraPresets);
        }
        const lastAppliedCameraPresetId = lastAppliedCameraPresetIdRef.current;
        const preferredCameraPreset =
          nextCameraPresets.find((preset) => preset.id === lastAppliedCameraPresetId) ??
          nextCameraPresets.find((preset) => preset.group === 'mid') ??
          nextCameraPresets[0] ??
          null;
        if (preferredCameraPreset) {
          applyingCameraPresetRef.current = true;
          userCameraInteractionRef.current = false;
          applyOrbitCameraPose(camera, controls, preferredCameraPreset);
          lastAppliedCameraPresetIdRef.current = preferredCameraPreset.id;
          setActiveCameraPresetId(preferredCameraPreset.id);
          window.requestAnimationFrame(() => {
            applyingCameraPresetRef.current = false;
          });
        } else {
          userCameraInteractionRef.current = false;
          lastAppliedCameraPresetIdRef.current = null;
          setActiveCameraPresetId(null);
        }
        if (floorPts.length >= 3) {
          const shapePts = floorPts.map(
            (p) => new THREE.Vector2((p.x - centerXmm) * MM_TO_M, (p.y - centerYmm) * MM_TO_M)
          );
          const shape = new THREE.Shape(shapePts);
          const geom = new THREE.ShapeGeometry(shape);
          geom.rotateX(-Math.PI / 2);
          const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(floorColorHex), side: THREE.DoubleSide });
          const floor = new THREE.Mesh(geom, mat);
          floor.receiveShadow = true;
          floor.castShadow = false;
          floor.userData = {
            roomSurface: 'floor',
          };
          scene.add(floor);
          const outlinePoints = floorPts.map(
            (p) =>
              new THREE.Vector3(
                (p.x - centerXmm) * MM_TO_M,
                0.006,
                -(p.y - centerYmm) * MM_TO_M
              )
          );
          const outlineGeometry = new THREE.BufferGeometry().setFromPoints(outlinePoints);
          const outline = new THREE.LineLoop(
            outlineGeometry,
            new THREE.LineBasicMaterial({
              color: new THREE.Color('#111827'),
              depthWrite: false,
              opacity: 0.72,
              transparent: true,
            })
          );
          outline.renderOrder = 5;
          scene.add(outline);
        }

        const grid = new THREE.GridHelper(Math.max(spanX, spanY) * MM_TO_M * 1.2, 20, 0xd6d1c7, 0xd6d1c7);
        grid.position.y = 0.001;
        grid.userData.aiRenderHide = true;
        scene.add(grid);

        const wallColorHex = normalizeHexColor(
          styledResult?.room?.surfaces?.wall_color_hex ??
          styledResult?.room?.surfaces?.wallColorHex ??
          '#555555',
          '#555555'
        );

        const wallPts = roomPts.length >= 2 ? roomPts : floorPts;
        const wallsRaw = activeFloorplan.walls ?? [];
        const walls =
          wallsRaw.length > 0
            ? wallsRaw
            : wallPts.length >= 2
              ? wallPts.map((p, idx) => ({
                  wall_id: `wall_${idx + 1}`,
                  p1_mm: p,
                  p2_mm: wallPts[(idx + 1) % wallPts.length],
                  thickness_mm: 100,
                }))
              : [];
        const wallOpacityById = new Map<string, number>();
        for (const wall of walls) {
          const wallId = String(wall.wall_id ?? '');
          if (!wallId) continue;
          wallOpacityById.set(wallId, clamp(Number(wall.opacity ?? 1), 0, 1));
        }
        const openings =
          activeFloorplan.openings ??
          styledResult?.room?.openings ?? {
            doors: [],
            windows: [],
          };
        const resolveSegment = (item: any): { p1: any; p2: any } | null => resolveOpeningSegmentMm(item);
        const wallGroup = new THREE.Group();
        scene.add(wallGroup);
        for (const wall of walls) {
          const x1 = Number(wall.p1_mm?.x ?? 0);
          const y1 = Number(wall.p1_mm?.y ?? 0);
          const x2 = Number(wall.p2_mm?.x ?? 0);
          const y2 = Number(wall.p2_mm?.y ?? 0);
          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenMm = Math.hypot(dx, dy);
          if (lenMm <= 0) continue;
          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          const angle = Math.atan2(dy, dx);
          const thicknessMm = Number(wall.thickness_mm ?? 100);
          const wallId = String(wall.wall_id ?? '');
          const wallOpacity = clamp(Number(wall.opacity ?? 1), 0, 1);
          const isSelectedWall = !!wallId && selectedWallId === wallId;
          const wallMeshColorHex = normalizeHexColor(wall.color_hex, wallColorHex);

          const geom = new THREE.BoxGeometry(lenMm * MM_TO_M, wallHeight, thicknessMm * MM_TO_M);
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(isSelectedWall ? '#d58c54' : wallMeshColorHex),
            transparent: wallOpacity < 1,
            opacity: wallOpacity,
          });
          const mesh = new THREE.Mesh(geom, mat);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          mesh.userData = {
            wallId,
            wallMesh: true,
            wallOpacity,
          };
          // 2D plan uses +Y "up". Our 3D preview maps plan Y to -Z so 3D matches the 2D canvas orientation.
          mesh.position.set((midX - centerXmm) * MM_TO_M, wallHeight / 2, -(midY - centerYmm) * MM_TO_M);
          mesh.rotation.y = angle;
          if (wallOpacity > 0.05) {
            addStructuralEdgeOverlay(THREE, mesh, {
              opacity: Math.max(0.62, Math.min(0.94, wallOpacity * 0.9)),
              renderOrder: 6,
              thresholdAngleDeg: 1,
            });
          }
          wallGroup.add(mesh);
        }

        type WallSeg = {
          x1: number;
          y1: number;
          x2: number;
          y2: number;
          dx: number;
          dy: number;
          len2: number;
          thicknessMm: number;
          angle: number;
        };

        const wallSegs: WallSeg[] = (walls ?? [])
          .map((wall) => {
            const x1 = Number(wall.p1_mm?.x ?? 0);
            const y1 = Number(wall.p1_mm?.y ?? 0);
            const x2 = Number(wall.p2_mm?.x ?? 0);
            const y2 = Number(wall.p2_mm?.y ?? 0);
            const dx = x2 - x1;
            const dy = y2 - y1;
            const len2 = dx * dx + dy * dy;
            return {
              x1,
              y1,
              x2,
              y2,
              dx,
              dy,
              len2,
              thicknessMm: Number(wall.thickness_mm ?? 100),
              angle: Math.atan2(dy, dx),
            };
          })
          .filter((w) => w.len2 > 1e-6);

        const findWallMount = (
          pxMm: number,
          pyMm: number
        ): { xMm: number; yMm: number; rotY: number } | null => {
          if (wallSegs.length === 0) return null;
          let best: { seg: WallSeg; t: number; projX: number; projY: number; dist2: number } | null = null;

          for (const seg of wallSegs) {
            const vx = pxMm - seg.x1;
            const vy = pyMm - seg.y1;
            const t = clamp((vx * seg.dx + vy * seg.dy) / seg.len2, 0, 1);
            const projX = seg.x1 + seg.dx * t;
            const projY = seg.y1 + seg.dy * t;
            const dx = pxMm - projX;
            const dy = pyMm - projY;
            const dist2 = dx * dx + dy * dy;
            if (!best || dist2 < best.dist2) {
              best = { seg, t, projX, projY, dist2 };
            }
          }

          if (!best) return null;
          const seg = best.seg;
          const len = Math.sqrt(seg.len2) || 1;
          const leftNx = -seg.dy / len;
          const leftNy = seg.dx / len;
          const rightNx = -leftNx;
          const rightNy = -leftNy;
          const offsetMm = seg.thicknessMm / 2 + 35;

          const leftX = best.projX + leftNx * offsetMm;
          const leftY = best.projY + leftNy * offsetMm;
          const rightX = best.projX + rightNx * offsetMm;
          const rightY = best.projY + rightNy * offsetMm;

          const dl2 = (leftX - centerXmm) ** 2 + (leftY - centerYmm) ** 2;
          const dr2 = (rightX - centerXmm) ** 2 + (rightY - centerYmm) ** 2;

          // Rotation convention: our proxy meshes face +Z as "front".
          // With planY -> -Z mapping, rotY = angle makes local +Z point to wall's RIGHT normal,
          // and rotY = angle + PI makes local +Z point to wall's LEFT normal.
          const useLeft = dl2 <= dr2;
          const rotY = useLeft ? seg.angle + Math.PI : seg.angle;
          return { xMm: useLeft ? leftX : rightX, yMm: useLeft ? leftY : rightY, rotY };
        };

        const findInteriorOffsetPoint = (
          pxMm: number,
          pyMm: number,
          extraMm: number
        ): { xMm: number; yMm: number; seg: WallSeg } | null => {
          if (wallSegs.length === 0) return null;
          let best: { seg: WallSeg; t: number; projX: number; projY: number; dist2: number } | null = null;

          for (const seg of wallSegs) {
            const vx = pxMm - seg.x1;
            const vy = pyMm - seg.y1;
            const t = clamp((vx * seg.dx + vy * seg.dy) / seg.len2, 0, 1);
            const projX = seg.x1 + seg.dx * t;
            const projY = seg.y1 + seg.dy * t;
            const dx = pxMm - projX;
            const dy = pyMm - projY;
            const dist2 = dx * dx + dy * dy;
            if (!best || dist2 < best.dist2) {
              best = { seg, t, projX, projY, dist2 };
            }
          }
          if (!best) return null;

          const seg = best.seg;
          const len = Math.sqrt(seg.len2) || 1;
          const leftNx = -seg.dy / len;
          const leftNy = seg.dx / len;
          const rightNx = -leftNx;
          const rightNy = -leftNy;
          const offsetMm = seg.thicknessMm / 2 + extraMm;

          const leftX = best.projX + leftNx * offsetMm;
          const leftY = best.projY + leftNy * offsetMm;
          const rightX = best.projX + rightNx * offsetMm;
          const rightY = best.projY + rightNy * offsetMm;

          const dl2 = (leftX - centerXmm) ** 2 + (leftY - centerYmm) ** 2;
          const dr2 = (rightX - centerXmm) ** 2 + (rightY - centerYmm) ** 2;
          const useLeft = dl2 <= dr2;
          return { xMm: useLeft ? leftX : rightX, yMm: useLeft ? leftY : rightY, seg };
        };

        if (!GLTFLoader) {
          setObjectVisualDiagnostics((current) => {
            const next = { ...current };
            for (const obj of resolvedObjects) {
              if (next[obj.id]?.status !== 'glb_loading') {
                continue;
              }
              next[obj.id] = {
                ...next[obj.id],
                status: 'proxy_no_loader',
                reason: 'GLTFLoader is unavailable in the current frontend bundle.',
              };
            }
            return next;
          });
        }

        const heightById = new Map<string, number>();
        for (const obj of resolvedObjects) {
          const inventoryDimensions = getInventoryDimensionsMm(obj.inventoryResolution.item);
          const hMm =
            inventoryDimensions?.height ??
            defaultHeightMm(obj.type);
          heightById.set(obj.id, hMm);
        }

        const objectGroup = new THREE.Group();
        scene.add(objectGroup);

        const openingsGroup = new THREE.Group();
        scene.add(openingsGroup);

        const rotatorGroup = new THREE.Group();
        scene.add(rotatorGroup);

        const openingColors = styledResult?.room?.opening_colors ?? styledResult?.room?.openingColors ?? null;
        const doorColorById = new Map<string, string>();
        const winColorById = new Map<string, string>();
        if (openingColors?.doors && Array.isArray(openingColors.doors)) {
          for (const d of openingColors.doors) {
            if (d?.id && typeof d.id === 'string' && d?.color_hex) {
              doorColorById.set(d.id, String(d.color_hex));
            }
          }
        }
        if (openingColors?.windows && Array.isArray(openingColors.windows)) {
          for (const w of openingColors.windows) {
            if (w?.id && typeof w.id === 'string' && w?.color_hex) {
              winColorById.set(w.id, String(w.color_hex));
            }
          }
        }

        const tryLoadOpeningGlb = (params: {
          root: any;
          resolution: InventoryPreviewResolution;
          wM: number;
          dM: number;
          hM: number;
          fitStrategy: GlbFitStrategy;
        }): void => {
          const descriptor =
            params.resolution.modelResolution?.descriptor ??
            buildOpeningFallbackModelDescriptor(params.resolution, userInput.style);
          if (!GLTFLoader || !descriptor) {
            return;
          }
          const previewOverride = applyInventoryModelPreviewOverride({
            descriptor,
            context: 'panel',
            fitStrategy: params.fitStrategy,
            wM: params.wM,
            dM: params.dM,
            hM: params.hM,
          });
          void buildGlbPreviewModel({
            THREE,
            GLTFLoader,
            KTX2Loader,
            renderer,
            descriptor: previewOverride.descriptor,
            wM: previewOverride.wM,
            dM: previewOverride.dM,
            hM: previewOverride.hM,
            fitStrategy: previewOverride.fitStrategy,
            colorHex: null,
            preserveMaterialTextures: true,
          })
            .then((model) => {
              if (disposed || !params.root.parent) {
                disposeThreeObject(model);
                return;
              }
              addStructuralEdgeOverlay(THREE, model, {
                opacity: 0.58,
                renderOrder: 7,
                thresholdAngleDeg: 18,
              });
              replaceThreeObjectChildren(params.root, model);
            })
            .catch(() => {
              // Keep the procedural opening proxy when the model cannot be loaded.
            });
        };

        // Doors and windows start as procedural proxies, then upgrade to GLB models when available.
        for (const door of openings.doors ?? []) {
          const seg = resolveSegment(door);
          const p1 = seg?.p1 ?? null;
          const p2 = seg?.p2 ?? null;
          if (!p1 || !p2) continue;
          const x1 = Number(p1.x ?? 0);
          const y1 = Number(p1.y ?? 0);
          const x2 = Number(p2.x ?? 0);
          const y2 = Number(p2.y ?? 0);
          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenMm = Math.hypot(dx, dy);
          if (!(lenMm > 1)) continue;

          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          const angle = Math.atan2(dy, dx);
          const doorId = String(door?.door_id ?? door?.id ?? 'door');
          const doorColor = normalizeHexColor(
            door?.color_hex ?? doorColorById.get(doorId),
            '#8c7a63'
          );
          const attachedWallOpacity = clamp(
            Number(wallOpacityById.get(String(door?.wall_id ?? '')) ?? 1),
            0,
            1,
          );

          const thicknessM = 0.04;
          const extraMm = thicknessM * 0.5 / MM_TO_M + 6; // fully inside wall face + a small gap
          const anchor = findInteriorOffsetPoint(midX, midY, extraMm);
          const pxMm = anchor?.xMm ?? midX;
          const pyMm = anchor?.yMm ?? midY;

          const hM = Math.min(2.05, Math.max(1.8, wallHeight - 0.1));
          const wM = lenMm * MM_TO_M;

          const root = new THREE.Group();
          root.userData = {
            openingId: doorId,
            openingType: 'door',
          };
          root.position.set((pxMm - centerXmm) * MM_TO_M, 0, -(pyMm - centerYmm) * MM_TO_M);
          root.rotation.y = angle;
          openingsGroup.add(root);

          const leaf = new THREE.Mesh(
            new THREE.BoxGeometry(wM, hM, thicknessM),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color(doorColor),
              roughness: 0.65,
              metalness: 0.08,
              transparent: attachedWallOpacity < 1,
              opacity: attachedWallOpacity,
            })
          );
          leaf.castShadow = true;
          leaf.receiveShadow = true;
          leaf.position.y = hM / 2;
          addStructuralEdgeOverlay(THREE, leaf, {
            opacity: 0.7,
            renderOrder: 7,
            thresholdAngleDeg: 1,
          });
          root.add(leaf);

          const resolution = resolveInventoryItemForPreview({
            inventory,
            assetId: readOpeningAssetId(door, 'door') ?? doorId,
            objectType: 'door',
            material: typeof door?.material === 'string' ? door.material : null,
            preferredStyle: userInput.style,
            targetWidthMm: lenMm,
            targetDepthMm: thicknessM / MM_TO_M,
            targetHeightMm: hM / MM_TO_M,
          });
          tryLoadOpeningGlb({
            root,
            resolution,
            wM,
            dM: thicknessM,
            hM,
            fitStrategy: 'wall-plane',
          });
        }

        for (const win of openings.windows ?? []) {
          const seg = resolveSegment(win);
          const p1 = seg?.p1 ?? null;
          const p2 = seg?.p2 ?? null;
          if (!p1 || !p2) continue;
          const x1 = Number(p1.x ?? 0);
          const y1 = Number(p1.y ?? 0);
          const x2 = Number(p2.x ?? 0);
          const y2 = Number(p2.y ?? 0);
          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenMm = Math.hypot(dx, dy);
          if (!(lenMm > 1)) continue;

          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          const angle = Math.atan2(dy, dx);
          const winId = String(win?.window_id ?? win?.id ?? 'window');
          const frameColor = normalizeHexColor(
            win?.color_hex ?? winColorById.get(winId),
            '#c0c0c0'
          );
          const attachedWallOpacity = clamp(
            Number(wallOpacityById.get(String(win?.wall_id ?? '')) ?? 1),
            0,
            1,
          );

          const sillMm = Number(win?.sill_height_mm ?? 900);
          const heightMm = Number(win?.height_mm ?? 1200);
          const bottomMm = clamp(sillMm, 0, wallHeightMm - 200);
          const maxHeightMm = Math.max(200, wallHeightMm - bottomMm - 120);
          const actualHeightMm = clamp(heightMm, 200, maxHeightMm);

          const frameDepthM = 0.08;
          const extraMm = frameDepthM * 0.5 / MM_TO_M + 6;
          const anchor = findInteriorOffsetPoint(midX, midY, extraMm);
          const pxMm = anchor?.xMm ?? midX;
          const pyMm = anchor?.yMm ?? midY;

          const wM = lenMm * MM_TO_M;
          const hM = actualHeightMm * MM_TO_M;
          const group = new THREE.Group();
          group.userData = {
            openingId: winId,
            openingType: 'window',
          };
          group.position.set(
            (pxMm - centerXmm) * MM_TO_M,
            bottomMm * MM_TO_M,
            -(pyMm - centerYmm) * MM_TO_M
          );
          group.rotation.y = angle;

          const frame = new THREE.Mesh(
            new THREE.BoxGeometry(wM, hM, frameDepthM),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color(frameColor),
              roughness: 0.35,
              metalness: 0.55,
              transparent: attachedWallOpacity < 1,
              opacity: attachedWallOpacity,
            })
          );
          frame.castShadow = true;
          frame.receiveShadow = true;
          frame.position.y = hM / 2;
          addStructuralEdgeOverlay(THREE, frame, {
            opacity: 0.7,
            renderOrder: 7,
            thresholdAngleDeg: 1,
          });
          group.add(frame);

          const glass = new THREE.Mesh(
            new THREE.BoxGeometry(wM * 0.92, hM * 0.9, 0.01),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color('#9fc9e8'),
              roughness: 0.12,
              metalness: 0.02,
              transparent: true,
              opacity: attachedWallOpacity >= 1 ? 0.35 : attachedWallOpacity * 0.35,
            })
          );
          glass.position.set(0, hM / 2, frameDepthM / 2 - 0.006);
          glass.castShadow = false;
          glass.receiveShadow = false;
          addStructuralEdgeOverlay(THREE, glass, {
            opacity: 0.48,
            renderOrder: 7,
            thresholdAngleDeg: 1,
          });
          group.add(glass);

          openingsGroup.add(group);

          const resolution = resolveInventoryItemForPreview({
            inventory,
            assetId: readOpeningAssetId(win, 'window') ?? winId,
            objectType: 'window',
            material: typeof win?.material === 'string' ? win.material : null,
            preferredStyle: userInput.style,
            targetWidthMm: lenMm,
            targetDepthMm: frameDepthM / MM_TO_M,
            targetHeightMm: actualHeightMm,
          });
          tryLoadOpeningGlb({
            root: group,
            resolution,
            wM,
            dM: frameDepthM,
            hM,
            fitStrategy: 'wall-plane',
          });
        }

        const draggableById = new Map<string, boolean>();
        for (const obj of resolvedObjects) {
          draggableById.set(obj.id, !!obj.draggable);
        }

        const proxyById = new Map<string, any>();
        const objectMetadataById = new Map<string, SnapshotObjectRecord>();
        const rotatorById = new Map<string, any>();

        for (const obj of resolvedObjects) {
          const wMmPlan = Math.max(1, obj.bbox.maxX - obj.bbox.minX);
          const dMmPlan = Math.max(1, obj.bbox.maxY - obj.bbox.minY);
          const cxMm = (obj.bbox.minX + obj.bbox.maxX) / 2;
          const cyMm = (obj.bbox.minY + obj.bbox.maxY) / 2;

          const method = String(obj.placeOn?.method ?? 'floor');
          const targetId = obj.placeOn?.targetId ?? null;
          const typeKey = obj.type.toLowerCase();
          const isCeiling =
            method === 'hang_on' &&
            (String(targetId ?? '').toLowerCase() === 'ceiling' ||
              typeKey.includes('ceiling_light') ||
              typeKey.includes('pendant_light') ||
              typeKey.includes('track_light'));
          const rotCcwDeg = ((Number(obj.rotationCcw ?? 0) % 360) + 360) % 360;

          const inventoryDimensions = getInventoryDimensionsMm(obj.inventoryResolution.item);
          const isWallHung = method === 'hang_on' && !isCeiling;
          const placementFootprintMm = obj.placementFootprintMm;
          const hasPlacementFootprint =
            !isWallHung &&
            Number.isFinite(Number(placementFootprintMm?.w)) &&
            Number.isFinite(Number(placementFootprintMm?.d)) &&
            Number(placementFootprintMm?.w) > 0 &&
            Number(placementFootprintMm?.d) > 0;
          let wMm = hasPlacementFootprint
            ? Math.max(1, Number(placementFootprintMm?.w))
            : wMmPlan;
          let dMm = hasPlacementFootprint
            ? Math.max(1, Number(placementFootprintMm?.d))
            : dMmPlan;
          if (!isWallHung && !hasPlacementFootprint && (rotCcwDeg === 90 || rotCcwDeg === 270)) {
            const tmp = wMm;
            wMm = dMm;
            dMm = tmp;
          }

          // Keep placement-backed visuals on their local footprint; bbox remains the
          // world-space envelope after rotation.
          const hMmDefault = heightById.get(obj.id) ?? defaultHeightMm(obj.type);
          let wM = wMm * MM_TO_M;
          let dM = dMm * MM_TO_M;
          let hM = hMmDefault * MM_TO_M;

          // Wall-hung items in a 2D plan typically use bbox as (width x height).
          if (isWallHung) {
            wM = Math.max(0.06, wMmPlan * MM_TO_M);
            hM = Math.max(0.06, dMmPlan * MM_TO_M);
            const thicknessMm = resolveWallHungPreviewDepthMm({
              objectType: obj.type,
              planWidthMm: wMmPlan,
              planHeightMm: dMmPlan,
              inventoryDimensions,
            });
            dM = Math.max(0.02, thicknessMm * MM_TO_M);
          }

          let baseY = 0;
          if (method === 'on_top' && typeof targetId === 'string' && heightById.has(targetId)) {
            baseY = (heightById.get(targetId) ?? 0) * MM_TO_M + 0.004;
          } else if (method === 'hang_on') {
            if (isCeiling) {
              baseY = wallHeight - hM;
            } else {
              baseY = clamp(wallHeight * 0.6 - hM / 2, 0.6, wallHeight - hM);
            }
          }

          let pxMm = cxMm;
          let pyMm = cyMm;
          // Plan Y is mapped to -Z so 3D matches the 2D canvas orientation.
          // Prefer the same explicit front metadata that drives the 2D white marker.
          let rotY = resolvePlanObjectYawRad({
            rotationDeg: rotCcwDeg,
            frontWorld: obj.frontWorld,
            frontSideWorld: obj.frontSideWorld,
            frontSideLocal: obj.frontSideLocal,
          });
          if (method === 'hang_on' && !isCeiling) {
            const mount = findWallMount(cxMm, cyMm);
            if (mount) {
              pxMm = mount.xMm;
              pyMm = mount.yMm;
              rotY = mount.rotY;
            }
          }

          const objectRoot = new THREE.Group();
          objectRoot.position.set(
            (pxMm - centerXmm) * MM_TO_M,
            baseY,
            -(pyMm - centerYmm) * MM_TO_M
          );
          objectRoot.rotation.y = rotY;
          objectRoot.userData = {
            objectId: obj.id,
            draggable: draggableById.get(obj.id) ?? false,
            baseY,
            method,
            wMm,
            dMm,
          };

          const fallbackProxy = buildObjectProxy(THREE, {
            type: obj.type,
            wM,
            dM,
            hM,
            colorHex: obj.colorHex,
            material: obj.material ?? null,
          });
          addStructuralEdgeOverlay(THREE, fallbackProxy, {
            opacity: 0.66,
            renderOrder: 7,
            thresholdAngleDeg: 10,
          });
          objectRoot.add(fallbackProxy);
          objectGroup.add(objectRoot);
          proxyById.set(obj.id, objectRoot);
          const renderIdentity = resolveRenderIdentityForObject(obj.type);
          objectMetadataById.set(obj.id, {
            id: obj.id,
            assetId: obj.assetId ?? null,
            rawAssetId: obj.rawAssetId ?? null,
            type: obj.type,
            renderIdentityLabel: renderIdentity.label,
            renderIdentityHint: renderIdentity.hint,
            colorHex: obj.colorHex,
            material: obj.material ?? null,
            bboxMm: {
              minX: Math.round(obj.bbox.minX),
              minY: Math.round(obj.bbox.minY),
              maxX: Math.round(obj.bbox.maxX),
              maxY: Math.round(obj.bbox.maxY),
            },
            planPositionMm: {
              x: Math.round(pxMm),
              y: Math.round(pyMm),
              z: Math.round((baseY + hM / 2) / MM_TO_M),
            },
            dimensionsMm: {
              width: Math.round(wM / MM_TO_M),
              depth: Math.round(dM / MM_TO_M),
              height: Math.round(hM / MM_TO_M),
            },
            planRotationDeg: Math.round(rotCcwDeg * 100) / 100,
            sceneYawDeg: Math.round(yawRadToPlanRotationDeg(rotY) * 100) / 100,
            placeOn: obj.placeOn
              ? {
                  method: String(obj.placeOn.method ?? 'floor'),
                  targetId: obj.placeOn.targetId ?? null,
                }
              : null,
          });

          const modelDescriptor = obj.inventoryResolution.glbEligible
            ? obj.inventoryResolution.modelResolution?.descriptor ?? null
            : null;
          const preserveGeneratedMeshShape = isUserGeneratedPreviewItem(
            obj.inventoryResolution.item
          );
          const stretchWallHungToLayoutBox =
            isWallHung &&
            isShelfLikePreviewType(obj.type) &&
            !preserveGeneratedMeshShape;
          if (GLTFLoader && modelDescriptor) {
            const resolvedFitStrategy = isWallHung
              ? stretchWallHungToLayoutBox
                ? 'wall-plane'
                : preserveGeneratedMeshShape
                  ? 'wall-plane-uniform'
                  : 'wall-plane-uniform-or-proxy'
              : preserveGeneratedMeshShape
                ? 'footprint-uniform'
                : 'footprint-uniform-or-proxy';
            const previewOverride = applyInventoryModelPreviewOverride({
              descriptor: modelDescriptor,
              context: 'panel',
              fitStrategy: resolvedFitStrategy,
              wM,
              dM,
              hM,
            });
            const resolvedAssetId =
              obj.inventoryResolution.item?.asset_id?.toLowerCase() ?? null;
            const isSectionalSofaFamily =
              obj.type.toLowerCase().includes('sectional_sofa') ||
              resolvedAssetId?.startsWith('sectional_sofa_') === true;
            const needsSectionalFacingFallback =
              isSectionalSofaFamily &&
              !(previewOverride.applied?.rotationDegOffset === 180);
            const descriptorForPanel = needsSectionalFacingFallback
              ? {
                  ...previewOverride.descriptor,
                  rotationDeg: previewOverride.descriptor.rotationDeg + 180,
                }
              : previewOverride.descriptor;
            void buildGlbPreviewModel({
                THREE,
                GLTFLoader,
                KTX2Loader,
                renderer,
                descriptor: descriptorForPanel,
                wM: previewOverride.wM,
                dM: previewOverride.dM,
                hM: previewOverride.hM,
                fitStrategy: previewOverride.fitStrategy,
                colorHex: obj.colorHex,
                preserveMaterialTextures: preserveGeneratedMeshShape,
                minimumHorizontalAspectRatio: preserveGeneratedMeshShape ? 0.45 : null,
              })
              .then((model) => {
                if (disposed || !objectRoot.parent) {
                  disposeThreeObject(model);
                  return;
                }
                setObjectVisualDiagnostics((current) => ({
                  ...current,
                  [obj.id]: {
                    ...(current[obj.id] ?? {
                      objectId: obj.id,
                      label: obj.type,
                      rawAssetId: obj.rawAssetId ?? obj.assetId ?? null,
                      resolvedAssetId: obj.inventoryResolution.item?.asset_id ?? null,
                      reintroduced: Boolean(obj.reintroduced),
                    }),
                    modelUrl: descriptorForPanel.url,
                    previewRotationOffsetDeg:
                      previewOverride.applied?.rotationDegOffset ?? 0,
                    finalDescriptorRotationDeg: descriptorForPanel.rotationDeg,
                    status: 'glb_loaded',
                    reason: previewOverride.applied
                      ? `Loaded GLB with preview override for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`
                      : needsSectionalFacingFallback
                        ? `Loaded GLB with sectional sofa panel facing fallback for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`
                      : `Loaded GLB for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`,
                  },
                }));
                const previousChildren = [...objectRoot.children];
                for (const child of previousChildren) {
                  objectRoot.remove(child);
                }
                for (const child of previousChildren) {
                  disposeThreeObject(child);
                }
                addStructuralEdgeOverlay(THREE, model, {
                  opacity: 0.56,
                  renderOrder: 7,
                  thresholdAngleDeg: 18,
                });
                objectRoot.add(model);
              })
              .catch((modelError: unknown) => {
                const reason =
                  modelError instanceof Error
                    ? modelError.message
                    : 'Failed to load GLB, so the object stayed on the proxy mesh.';
                const isAspectMismatch = reason.includes('proxy fallback stays in place');
                setObjectVisualDiagnostics((current) => ({
                  ...current,
                  [obj.id]: {
                    ...(current[obj.id] ?? {
                      objectId: obj.id,
                      label: obj.type,
                      rawAssetId: obj.rawAssetId ?? obj.assetId ?? null,
                      resolvedAssetId: obj.inventoryResolution.item?.asset_id ?? null,
                      reintroduced: Boolean(obj.reintroduced),
                    }),
                    status: isAspectMismatch ? 'proxy_aspect_mismatch' : 'proxy_model_load_error',
                    reason,
                  },
                }));
                // Keep the procedural proxy as fallback when model loading fails.
              });
          }

          // Rotation handle (torus + knob) for draggable floor objects.
          if (draggableById.get(obj.id) && method !== 'hang_on') {
            const ringR = Math.max(0.12, Math.max(wM, dM) * 0.65);
            const ringTube = Math.max(0.008, ringR * 0.06);
            const ring = new THREE.Mesh(
              new THREE.TorusGeometry(ringR, ringTube, 12, 36),
              new THREE.MeshStandardMaterial({ color: new THREE.Color('#6b6b6b'), roughness: 0.5, metalness: 0.1 })
            );
            ring.rotation.x = Math.PI / 2;
            ring.position.set(0, Math.max(0.04, hM + 0.04), 0);
            const knob = new THREE.Mesh(
              new THREE.SphereGeometry(Math.max(0.01, ringTube * 1.2), 12, 12),
              new THREE.MeshStandardMaterial({ color: new THREE.Color('#2f2f2f'), roughness: 0.35, metalness: 0.2 })
            );
            knob.position.set(ringR, ring.position.y, 0);
            const rotator = new THREE.Group();
            rotator.add(ring);
            rotator.add(knob);
            rotator.position.copy(objectRoot.position);
            rotator.rotation.y = objectRoot.rotation.y;
            rotator.userData = { objectId: obj.id, rotator: true };
            rotator.visible = false;
            rotatorGroup.add(rotator);
            rotatorById.set(obj.id, rotator);
          }
        }

        rotatorGroupRef.current = rotatorGroup;
        renderStateRef.current = {
          THREE,
          renderer,
          camera,
          controls,
          objectGroup,
          wallGroup,
          openingsGroup,
          proxyById,
          objectMetadataById,
          roomBoundsMm: {
            minX: Math.round(bounds.minX),
            minY: Math.round(bounds.minY),
            maxX: Math.round(bounds.maxX),
            maxY: Math.round(bounds.maxY),
          },
          roomPolygonMm: roomPolygonMm.map((point) => ({
            x: Number(point.x),
            y: Number(point.y),
          })),
          roomCenterMm: {
            x: centerXmm,
            y: centerYmm,
          },
          roomOpenings: {
            doors: normalizeSnapshotOpenings(Array.isArray(openings.doors) ? openings.doors : []),
            windows: normalizeSnapshotOpenings(
              Array.isArray(openings.windows) ? openings.windows : []
            ),
          },
          roomSurfaceColors: {
            wallColorHex,
            floorColorHex,
            ceilingColorHex,
          },
          variantId: selectedVariantId,
          selectedObjectId: selectedIdRef.current,
          selectedWallId,
          editMode: editModeRef.current,
          wallOcclusionOpacityThreshold: SNAPSHOT_WALL_OCCLUSION_OPACITY_THRESHOLD,
        };

        const updateRotatorsVisibility = () => {
          const mode = editModeRef.current;
          const selected = selectedIdRef.current;
          for (const [oid, rotator] of rotatorById.entries()) {
            rotator.visible = mode === 'edit' && !!selected && oid === selected;
          }
        };
        updateRotatorsVisibilityRef.current = updateRotatorsVisibility;
        updateRotatorsVisibility();

        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const gridMm = 50;
        let dragState:
          | {
              mesh: any;
              objectId: string;
              plane: any;
              offset: { x: number; z: number };
              lastValid: { x: number; z: number };
            }
          | null = null;
        let rotateState:
          | {
              mesh: any;
              objectId: string;
              plane: any;
              baseY: number;
            }
          | null = null;
        let orbitClickStart: { x: number; y: number } | null = null;
        let orbitMoved = false;
        const handleControlsStart = () => {
          if (!applyingCameraPresetRef.current) {
            userCameraInteractionRef.current = true;
          }
        };
        const handleControlsChange = () => {
          if (applyingCameraPresetRef.current || !userCameraInteractionRef.current) {
            return;
          }
          setActiveCameraPresetId((current) => (current === null ? current : null));
        };
        const handleControlsEnd = () => {
          userCameraInteractionRef.current = false;
        };
        controls.addEventListener?.('start', handleControlsStart);
        controls.addEventListener?.('change', handleControlsChange);
        controls.addEventListener?.('end', handleControlsEnd);

        const getPointer = (evt: PointerEvent) => {
          const rect = renderer.domElement.getBoundingClientRect();
          pointer.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
          pointer.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
        };

        const findDraggableRoot = (obj: any): any => {
          let cur = obj;
          while (cur) {
            if (cur.userData?.objectId) return cur;
            cur = cur.parent;
          }
          return null;
        };

        const handlePointerDown = (evt: PointerEvent) => {
          if (evt.button !== 0) return;
          if (!renderer) return;
          getPointer(evt);
          raycaster.setFromCamera(pointer, camera);

          const mode = editModeRef.current;
          if (mode === 'idle') {
            orbitClickStart = { x: evt.clientX, y: evt.clientY };
            orbitMoved = false;
            return;
          }

          if (mode === 'edit') {
            const rotIntersects = raycaster.intersectObjects(rotatorGroup.children, true);
            if (rotIntersects.length) {
              const hitRot = findDraggableRoot(rotIntersects[0].object);
              if (hitRot && hitRot.userData?.rotator) {
                const objectId = hitRot.userData.objectId;
                const target = proxyById.get(objectId);
                if (target) {
                  const baseY = Number(target.userData.baseY ?? 0);
                  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -baseY);
                  rotateState = { mesh: target, objectId, plane, baseY };
                  setSelectedId(objectId);
                  if (controls) controls.enabled = false;
                  return;
                }
              }
            }
          }

          const intersects = raycaster.intersectObjects(objectGroup.children, true);
          if (!intersects.length) return;
          const hit = findDraggableRoot(intersects[0].object);
          if (mode !== 'edit') return;
          if (!hit || !hit.userData?.draggable) return;
          if (hit.userData?.objectId !== selectedIdRef.current) return;

          const method = String(hit.userData.method ?? 'floor');
          if (method === 'hang_on') return;

          const baseY = Number(hit.userData.baseY ?? 0);
          const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -baseY);
          const intersectPoint = new THREE.Vector3();
          raycaster.ray.intersectPlane(plane, intersectPoint);
          const offset = {
            x: hit.position.x - intersectPoint.x,
            z: hit.position.z - intersectPoint.z,
          };
          dragState = {
            mesh: hit,
            objectId: hit.userData.objectId,
            plane,
            offset,
            lastValid: { x: hit.position.x, z: hit.position.z },
          };
          if (controls) controls.enabled = false;
        };

        const handlePointerMove = (evt: PointerEvent) => {
          if (editModeRef.current === 'idle') {
            if (orbitClickStart) {
              const dx = evt.clientX - orbitClickStart.x;
              const dy = evt.clientY - orbitClickStart.y;
              if (Math.hypot(dx, dy) > 3) orbitMoved = true;
            }
            return;
          }
          if (rotateState) {
            getPointer(evt);
            raycaster.setFromCamera(pointer, camera);
            const intersectPoint = new THREE.Vector3();
            raycaster.ray.intersectPlane(rotateState.plane, intersectPoint);
            const dx = intersectPoint.x - rotateState.mesh.position.x;
            const dz = intersectPoint.z - rotateState.mesh.position.z;
            const angle = Math.atan2(dz, dx);
            rotateState.mesh.rotation.y = angle + Math.PI / 2;
            const rotator = rotatorById.get(rotateState.objectId);
            if (rotator) {
              rotator.rotation.y = rotateState.mesh.rotation.y;
              rotator.position.copy(rotateState.mesh.position);
            }
            return;
          }

          if (!dragState) return;
          getPointer(evt);
          raycaster.setFromCamera(pointer, camera);
          const intersectPoint = new THREE.Vector3();
          raycaster.ray.intersectPlane(dragState.plane, intersectPoint);
          let nextX = intersectPoint.x + dragState.offset.x;
          let nextZ = intersectPoint.z + dragState.offset.z;

          const worldXmm = nextX / MM_TO_M + centerXmm;
          const worldYmm = -(nextZ / MM_TO_M) + centerYmm;
          const placement = design?.placements?.find((p) => p.placement_id === dragState?.objectId);
          const wMm = Number(dragState.mesh.userData?.wMm ?? 0);
          const dMm = Number(dragState.mesh.userData?.dMm ?? 0);
          const halfW = Math.max(0, wMm / 2);
          const halfD = Math.max(0, dMm / 2);

          const snap = (valueMm: number) => Math.round(valueMm / gridMm) * gridMm;
          let snappedXmm = snap(worldXmm);
          let snappedYmm = snap(worldYmm);

          snappedXmm = clamp(snappedXmm, bounds.minX + halfW, bounds.maxX - halfW);
          snappedYmm = clamp(snappedYmm, bounds.minY + halfD, bounds.maxY - halfD);

          const inside = pointInPolygon({ x: snappedXmm, y: snappedYmm }, roomPolygonMm);
          if (inside) {
            const snappedX = (snappedXmm - centerXmm) * MM_TO_M;
            const snappedZ = -(snappedYmm - centerYmm) * MM_TO_M;
            nextX = snappedX;
            nextZ = snappedZ;
            dragState.lastValid = { x: nextX, z: nextZ };
          } else {
            nextX = dragState.lastValid.x;
            nextZ = dragState.lastValid.z;
          }

          dragState.mesh.position.x = nextX;
          dragState.mesh.position.z = nextZ;
          const rotator = rotatorById.get(dragState.objectId);
          if (rotator) {
            rotator.position.copy(dragState.mesh.position);
          }
        };

        const handlePointerUp = () => {
          const activeRotate = rotateState;
          const activeDrag = dragState;

          if (activeRotate) {
            const placement = design?.placements?.find(
              (p) => String(p.placement_id ?? p.id ?? p.object_id ?? '') === activeRotate?.objectId
            );
            if (placement) {
              const rot = yawRadToPlanRotationDeg(activeRotate.mesh.rotation.y);
              updatePlacement({ ...placement, rotation_deg: rot });
            }
          } else if (activeDrag) {
            const worldXmm = activeDrag.mesh.position.x / MM_TO_M + centerXmm;
            const worldYmm = -(activeDrag.mesh.position.z / MM_TO_M) + centerYmm;
            const placement = design?.placements?.find(
              (p) => String(p.placement_id ?? p.id ?? p.object_id ?? '') === activeDrag?.objectId
            );
            if (placement) {
              updatePlacement({ ...placement, position_mm: { x: worldXmm, y: worldYmm } });
            }
          } else if (editModeRef.current === 'idle') {
            if (orbitClickStart && !orbitMoved) {
              raycaster.setFromCamera(pointer, camera);
              const intersects = raycaster.intersectObjects(objectGroup.children, true);
              if (intersects.length) {
                const hit = findDraggableRoot(intersects[0].object);
                if (hit?.userData?.objectId) {
                  const objectId = String(hit.userData.objectId);
                  setSelectedId(objectId);
                  selectPlacement(objectId);
                  selectWall(null);
                  return;
                }
              }
              const wallIntersects = raycaster.intersectObjects(wallGroup.children, true);
              if (wallIntersects.length) {
                const wallHit = wallIntersects[0].object;
                const wallId = String(wallHit?.userData?.wallId ?? '');
                if (wallId) {
                  setSelectedId(null);
                  selectPlacement(null);
                  setEditMode('idle');
                  selectWall(wallId);
                  return;
                }
              }
              setSelectedId(null);
              selectPlacement(null);
              selectWall(null);
            }
          }

          rotateState = null;
          dragState = null;
          orbitClickStart = null;
          orbitMoved = false;
          if (controls) controls.enabled = editModeRef.current === 'idle';
        };

        renderer.domElement.addEventListener('pointerdown', handlePointerDown);
        renderer.domElement.addEventListener('pointermove', handlePointerMove);
        renderer.domElement.addEventListener('pointerup', handlePointerUp);
        renderer.domElement.addEventListener('pointerleave', handlePointerUp);
        window.addEventListener('pointerup', handlePointerUp);
        window.addEventListener('pointercancel', handlePointerUp);
        window.addEventListener('blur', handlePointerUp);

        const handleResize = () => {
          if (!renderer || disposed) return;
          const w = el.clientWidth;
          const h = el.clientHeight;
          if (w < 2 || h < 2) return;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        };
        resizeObserver = new ResizeObserver(handleResize);
        resizeObserver.observe(el);
        window.addEventListener('resize', handleResize);

        const animate = () => {
          if (disposed) return;
          raf = window.requestAnimationFrame(animate);
          controls?.update?.();
          renderer.render(scene, camera);
        };
        renderer.render(scene, camera);
        animate();
        finishSceneLoading();

        return () => {
          window.removeEventListener('resize', handleResize);
          controls.removeEventListener?.('start', handleControlsStart);
          controls.removeEventListener?.('change', handleControlsChange);
          controls.removeEventListener?.('end', handleControlsEnd);
          userCameraInteractionRef.current = false;
          renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
          renderer.domElement.removeEventListener('pointermove', handlePointerMove);
          renderer.domElement.removeEventListener('pointerup', handlePointerUp);
          renderer.domElement.removeEventListener('pointerleave', handlePointerUp);
          window.removeEventListener('pointerup', handlePointerUp);
          window.removeEventListener('pointercancel', handlePointerUp);
          window.removeEventListener('blur', handlePointerUp);
          if (updateRotatorsVisibilityRef.current === updateRotatorsVisibility) {
            updateRotatorsVisibilityRef.current = null;
          }
        };
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load Three.js');
        finishSceneLoading();
      }
    };

    let cleanup: (() => void) | null = null;
    const start = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w < 2 || h < 2) return false;
      run(w, h)
        .then((fn) => {
          if (disposed) {
            if (typeof fn === 'function') {
              try {
                fn();
              } catch {
                // ignore
              }
            }
            return;
          }
          cleanup = typeof fn === 'function' ? fn : null;
        })
        .catch(() => {
          cleanup = null;
          finishSceneLoading();
        });
      return true;
    };
    if (!start()) {
      resizeObserver = new ResizeObserver(() => {
        if (disposed || renderer) return;
        if (start() && resizeObserver) {
          resizeObserver.disconnect();
          resizeObserver = null;
        }
      });
      resizeObserver.observe(el);
    }

    return () => {
      disposed = true;
      setIsThreeSceneLoading(false);
      window.cancelAnimationFrame(raf);
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
      try {
        cleanup?.();
      } catch {
        // ignore
      }
      try {
        controls?.dispose?.();
      } catch {
        // ignore
      }
      controlsRef.current = null;
      try {
        renderer?.dispose?.();
      } catch {
        // ignore
      }
      try {
        if (renderer?.domElement && el.contains(renderer.domElement)) {
          el.removeChild(renderer.domElement);
        }
      } catch {
        // ignore
      }
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      rotatorGroupRef.current = null;
      renderStateRef.current = null;
      if (import.meta.env.DEV) {
        console.debug('[3D] cleanup');
      }
    };
  }, [
    viewMode,
    activeFloorplan,
    resolvedObjects,
    styledResult,
    inventory,
    userInput.height,
    userInput.style,
    design,
    updatePlacement,
    selectedWallId,
    selectWall,
    selectPlacement,
  ]);

  if (viewMode !== '3d') return null;

  const controlsSlot =
    typeof document === 'undefined' ? null : document.getElementById('three-controls-slot');

  const threePanelHeader = (
    <div className="three-panel-header">
        <div className="three-panel-title">3D Preview</div>
        <div className="three-panel-actions">
          <button
            className="ghost"
            disabled={isCapturing}
            onClick={() => {
              void handleCaptureSnapshot();
            }}
          >
            {isCapturing ? 'Capturing...' : 'Capture'}
          </button>
          <button
            className={`ghost three-image-action-button${
              activeImageRenderAction === 'generate' ? ' loading' : ''
            }`}
            disabled={isRenderingImage}
            onClick={() => {
              void handleRenderSnapshotImage();
            }}
          >
            {activeImageRenderAction === 'generate' ? (
              <>
                <span className="toolbar-spinner" aria-hidden="true" />
                <span>Generating...</span>
              </>
            ) : (
              'Generate Image'
            )}
          </button>
          <button
            className="ghost"
            onClick={() => {
              if (!initialDesignRef.current) return;
              setDesign(JSON.parse(JSON.stringify(initialDesignRef.current)));
            }}
          >
            Reset
          </button>
          <button
            className="danger"
            disabled={!selectedId}
            onClick={() => {
              if (!selectedId) return;
              const placement = design?.placements?.find(
                (p) => String(p.placement_id ?? p.id ?? p.object_id ?? '') === selectedId
              );
              if (placement) {
                removePlacement(placement.placement_id);
              }
              setSelectedId(null);
              selectPlacement(null);
            }}
          >
            Delete
          </button>
          {renderedSnapshotImage ? (
            <button
              className={`ghost three-image-action-button${
                activeImageRenderAction === 'camera_rerender' ? ' loading' : ''
              }`}
              disabled={isRenderingImage}
              onClick={() => {
                void handleCameraRerenderSnapshotImage();
              }}
            >
              {activeImageRenderAction === 'camera_rerender' ? (
                <>
                  <span className="toolbar-spinner" aria-hidden="true" />
                  <span>Re-rendering...</span>
                </>
              ) : (
                'Rerender Camera'
              )}
            </button>
          ) : null}
          {renderedSnapshotImage ? (
            <button
              className={`ghost three-image-action-button three-apply-edit-action${
                activeImageRenderAction === 'edit' ? ' loading' : ''
              }`}
              disabled={isRenderingImage}
              onClick={() => {
                void handleEditSnapshotImage();
              }}
            >
              {activeImageRenderAction === 'edit' ? (
                <>
                  <span className="toolbar-spinner" aria-hidden="true" />
                  <span>Editing...</span>
                </>
              ) : (
                'Apply Edit'
              )}
            </button>
          ) : null}
          {selectedId ? <div className="three-panel-hint">Selected object: {selectedId}</div> : null}
          {!selectedId && selectedWallId ? (
            <div className="three-panel-hint">Selected wall: {selectedWallId}</div>
          ) : null}
          {captureStatus ? (
            <div
              aria-live="polite"
              className={`three-panel-status ${captureStatus.tone}`}
            >
              {captureStatus.text}
            </div>
          ) : null}
          {renderStatus ? (
            <div
              aria-live="polite"
              className={`three-panel-status ${renderStatus.tone}`}
            >
              {renderStatus.text}
            </div>
          ) : null}
          <div className="three-panel-prompt">
            <div className="three-control-grid">
              {midHeightCameraPresets.length > 0 || ceilingCameraPresets.length > 0 ? (
                <CameraPresetStepper
                  value={activeCameraPresetId}
                  stepAnchorValue={lastAppliedCameraPresetIdRef.current}
                  disabled={isRenderingImage}
                  midHeightOptions={midHeightCameraPresets}
                  ceilingOptions={ceilingCameraPresets}
                  onChange={handleCameraPresetSelection}
                />
              ) : null}
              <ReferencePresetDropdown
                label="Style"
                options={stylePresetOptions}
                value={presetSelection.style}
                disabled={isRenderingImage || !renderPresets}
                onChange={(value) =>
                  setPresetSelection((current) => ({
                    ...current,
                    style: value,
                  }))
                }
              />
              <ReferencePresetDropdown
                label="Lighting"
                options={lightingPresetOptions}
                value={presetSelection.lighting}
                disabled={isRenderingImage || !renderPresets}
                onChange={(value) =>
                  setPresetSelection((current) => ({
                    ...current,
                    lighting: value,
                  }))
                }
              />
              <ReferencePresetDropdown
                label="Scenery"
                options={sceneryPresetOptions}
                value={presetSelection.scenery}
                disabled={isRenderingImage || !renderPresets}
                onChange={(value) =>
                  setPresetSelection((current) => ({
                    ...current,
                    scenery: value,
                  }))
                }
              />
            </div>
            <details className="three-appearance-panel">
              <summary className="three-appearance-summary">
                <span>Appearance</span>
                <span>{objects.length > 0 ? `3 surfaces - ${objects.length} objects` : '3 surfaces'}</span>
              </summary>
              <div className="three-appearance-content">
              <div className="three-color-grid">
                <ColorSwatchField
                  label="Walls"
                  value={wallSurfaceColor}
                  disabled={isRenderingImage}
                  onChange={(value) => updateRoomSurfaceColor('wall_color_hex', value)}
                />
                <ColorSwatchField
                  label="Floor"
                  value={floorSurfaceColor}
                  disabled={isRenderingImage}
                  onChange={(value) => updateRoomSurfaceColor('floor_color_hex', value)}
                />
                <ColorSwatchField
                  label="Ceiling"
                  value={ceilingSurfaceColor}
                  disabled={isRenderingImage}
                  onChange={(value) => updateRoomSurfaceColor('ceiling_color_hex', value)}
                />
              </div>
              {selectedObjectForAppearance ? (
                <ColorSwatchField
                  label={`Selected: ${selectedObjectForAppearance.type.replace(/_/g, ' ')}`}
                  value={normalizeHexColor(selectedObjectForAppearance.colorHex, '#cfcac2')}
                  disabled={isRenderingImage}
                  onChange={(value) =>
                    handleObjectColorChange(selectedObjectForAppearance.id, value)
                  }
                />
              ) : null}
              {objects.length > 0 ? (
                <div className="three-object-color-list" aria-label="Object colors">
                  {objects.map((object) => (
                    <ColorSwatchField
                      key={object.id}
                      label={`${object.type.replace(/_/g, ' ')} · ${object.id}`}
                      value={normalizeHexColor(object.colorHex, '#cfcac2')}
                      disabled={isRenderingImage}
                      onChange={(value) => handleObjectColorChange(object.id, value)}
                    />
                  ))}
                </div>
              ) : null}
              </div>
            </details>
            {activeObjectAppearanceEntries.length > 0 ? (
              <div className="three-memory-strip">
                <span>
                  Object memory: {activeObjectAppearanceEntries.length} saved
                </span>
                <button
                  className="ghost"
                  disabled={isRenderingImage}
                  onClick={() => setObjectAppearanceReferences({})}
                  type="button"
                >
                  Clear memory
                </button>
              </div>
            ) : null}
            {renderedSnapshotImage ? (
              <div className="three-edit-controls">
                <div className="three-edit-object-field">
                  <div className="three-edit-object-label-row">
                    <label>Drawn edit boxes</label>
                    <span className="three-edit-object-count">
                      {configuredEditBoxes.length}/{editBoxes.length} configured
                    </span>
                  </div>
                  {editBoxes.length === 0 ? (
                    <div className="three-panel-prompt-note">
                      Draw one or more boxes directly on the generated image below.
                    </div>
                  ) : (
                    <div className="three-edit-box-list">
                      {editBoxes.map((box) => (
                        <div
                          key={box.id}
                          className={[
                            'three-edit-box-card',
                            activeEditBox?.id === box.id ? 'selected' : '',
                          ]
                            .filter(Boolean)
                            .join(' ')}
                        >
                          <div className="three-edit-box-card-header">
                            <button
                              className="ghost"
                              disabled={isRenderingImage}
                              onClick={() => setActiveEditBoxId(box.id)}
                              type="button"
                            >
                              {box.id}
                            </button>
                            <select
                              disabled={isRenderingImage}
                              value={box.mode}
                              onChange={(event) => {
                                updateEditBox(box.id, (current) => ({
                                  ...current,
                                  mode: event.target.value as EditBoxMode,
                                }));
                              }}
                            >
                              <option value="prompt">Prompt edit</option>
                              <option value="add">Add object</option>
                              <option value="replace">Replace object</option>
                              <option value="recolor">Change color</option>
                            </select>
                            <button
                              className="ghost"
                              disabled={isRenderingImage}
                              onClick={() => {
                                setEditBoxes((current) =>
                                  current.filter((candidate) => candidate.id !== box.id)
                                );
                                if (activeEditBoxId === box.id) {
                                  setActiveEditBoxId(null);
                                }
                              }}
                              type="button"
                            >
                              Remove
                            </button>
                          </div>
                          <textarea
                            className="three-panel-prompt-input three-edit-box-prompt"
                            disabled={isRenderingImage}
                            placeholder={
                              box.mode === 'add'
                                ? 'Describe the object to add, or upload a reference image.'
                                : box.mode === 'replace'
                                  ? 'Describe what to replace this region with.'
                                  : box.mode === 'recolor'
                                    ? 'Optional color/material note for this region.'
                                    : 'Describe the edit for this boxed region.'
                            }
                            rows={2}
                            value={box.prompt}
                            onChange={(event) => {
                              updateEditBox(box.id, (current) => ({
                                ...current,
                                prompt: event.target.value,
                              }));
                            }}
                          />
                          {box.mode === 'add' || box.mode === 'replace' ? (
                            <div className="three-edit-box-reference-row">
                              {box.imageDataUrl ? (
                                <img
                                  className="three-edit-reference-thumb"
                                  src={box.imageDataUrl}
                                  alt={`${box.id} reference`}
                                  draggable={false}
                                />
                              ) : (
                                <span className="three-edit-reference-thumb is-empty">
                                  No image
                                </span>
                              )}
                              <input
                                type="file"
                                accept="image/png,image/jpeg,image/webp"
                                disabled={isRenderingImage}
                                onChange={(event) => {
                                  const file = event.target.files?.[0] ?? null;
                                  if (!file) {
                                    return;
                                  }
                                  void fileToDataUrl(file).then((imageDataUrl) => {
                                    updateEditBox(box.id, (current) => ({
                                      ...current,
                                      imageDataUrl,
                                      fileName: file.name,
                                    }));
                                  });
                                }}
                              />
                            </div>
                          ) : null}
                          {box.mode === 'recolor' ? (
                            <div className="three-edit-box-color-row">
                              <input
                                type="color"
                                disabled={isRenderingImage}
                                value={box.targetColor}
                                onChange={(event) => {
                                  updateEditBox(box.id, (current) => ({
                                    ...current,
                                    targetColor: event.target.value,
                                  }));
                                }}
                                aria-label={`${box.id} target color`}
                              />
                              <span>{box.targetColor}</span>
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="three-edit-object-actions">
                    <button
                      className="ghost"
                      disabled={isRenderingImage || editBoxes.length === 0}
                      onClick={() => {
                        setEditBoxes([]);
                        setActiveEditBoxId(null);
                        setEditBoxDraft(null);
                      }}
                      type="button"
                    >
                      Clear boxes
                    </button>
                  </div>
                </div>
                <div className="three-control-grid three-control-grid-compact">
                  <ReferencePresetDropdown
                    label="Edit Style"
                    options={stylePresetOptions}
                    value={editPresetSelection.style}
                    disabled={isRenderingImage || !renderPresets}
                    allowEmpty
                    emptyLabel="No change"
                    onChange={(value) =>
                      setEditPresetSelection((current) => ({
                        ...current,
                        style: value,
                      }))
                    }
                  />
                  <ReferencePresetDropdown
                    label="Edit Lighting"
                    options={lightingPresetOptions}
                    value={editPresetSelection.lighting}
                    disabled={isRenderingImage || !renderPresets}
                    allowEmpty
                    emptyLabel="No change"
                    onChange={(value) =>
                      setEditPresetSelection((current) => ({
                        ...current,
                        lighting: value,
                      }))
                    }
                  />
                  <ReferencePresetDropdown
                    label="Edit Scenery"
                    options={sceneryPresetOptions}
                    value={editPresetSelection.scenery}
                    disabled={isRenderingImage || !renderPresets}
                    allowEmpty
                    emptyLabel="No change"
                    onChange={(value) =>
                      setEditPresetSelection((current) => ({
                        ...current,
                        scenery: value,
                      }))
                    }
                  />
                </div>
                {selectedId && !selectedObjectIsEditable ? (
                  <div className="three-panel-prompt-note">
                    Selected 3D object is not visible in this generated image.
                  </div>
                ) : null}
                {configuredEditBoxes.length === 0 && !editPresetActive ? (
                  <div className="three-panel-prompt-note">
                    Draw an edit box or choose an edit preset before applying edits.
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
          <div className="three-panel-hint">
            Click-drag to orbit. Click an object to select it, or click a wall to adjust its opacity.
          </div>
        </div>
      </div>
  );

  return (
    <div className="three-panel">
      {controlsSlot ? createPortal(threePanelHeader, controlsSlot) : threePanelHeader}
      <div className="three-panel-body">
        <div ref={containerRef} className="three-panel-canvas" />
        {!error && showThreeLoadingOverlay ? (
          <div className="three-loading-overlay" role="status" aria-live="polite">
            <span className="three-loading-spinner" aria-hidden="true" />
            <span>
              {isThreeSceneLoading
                ? 'Loading 3D preview...'
                : `Loading ${pendingGlbCount} 3D model${pendingGlbCount === 1 ? '' : 's'}...`}
            </span>
          </div>
        ) : null}
        {error ? (
          <div className="three-preview-overlay">
            <div className="muted">{error}</div>
          </div>
        ) : null}
        {!error && objects.length === 0 ? (
          <div className="three-preview-overlay">
            <div className="muted">No objects to render (only room shell).</div>
          </div>
        ) : null}
      </div>
      {reintroducedObjects.length > 0 || proxyObjects.length > 0 || glbObjects.length > 0 ? (
        <div className="three-debug-grid">
          {reintroducedObjects.length > 0 ? (
            <div className="three-debug-card">
              <div className="three-debug-title">
                Reintroduced Objects ({reintroducedObjects.length})
              </div>
              <div className="three-debug-list">
                {reintroducedObjects.map((item) => (
                  <div
                    key={`${item.clusterId ?? 'no-cluster'}-${item.objectId}`}
                    className="three-debug-item"
                  >
                    <strong>{item.objectType}</strong>
                    <span>{item.objectId}</span>
                    {item.clusterId ? <span>cluster: {item.clusterId}</span> : null}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {glbObjects.length > 0 ? (
            <div className="three-debug-card">
              <div className="three-debug-title">Loaded GLBs ({glbObjects.length})</div>
              <div className="three-debug-list">
                {glbObjects.map((item) => {
                  const modelName = item.modelUrl?.split('/').pop() ?? 'n/a';
                  const offsetLabel =
                    typeof item.previewRotationOffsetDeg === 'number'
                      ? `${Math.round(item.previewRotationOffsetDeg)}deg`
                      : 'pending';
                  const finalRotationLabel =
                    typeof item.finalDescriptorRotationDeg === 'number'
                      ? `${Math.round(item.finalDescriptorRotationDeg)}deg`
                      : 'n/a';
                  return (
                    <div key={item.objectId} className="three-debug-item">
                      <strong>{item.label}</strong>
                      <span>
                        raw: {item.rawAssetId ?? 'n/a'} | inventory:{' '}
                        {item.resolvedAssetId ?? 'not matched'}
                      </span>
                      <span>
                        model: {modelName} | offset: {offsetLabel} | final rot:{' '}
                        {finalRotationLabel}
                      </span>
                      <span>{item.reason}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
          {proxyObjects.length > 0 ? (
            <div className="three-debug-card">
              <div className="three-debug-title">
                Proxy Fallbacks ({proxyObjects.length})
              </div>
              <div className="three-debug-list">
                {proxyObjects.map((item) => (
                  <div key={item.objectId} className="three-debug-item">
                    <strong>{item.label}</strong>
                    <span>
                      raw: {item.rawAssetId ?? 'n/a'} | inventory:{' '}
                      {item.resolvedAssetId ?? 'not matched'}
                    </span>
                    <span>{item.reason}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      {renderedSnapshotImage ? (
        <div className="three-render-output">
          <div className="three-render-output-header">
            <div>
              <strong>AI Render</strong>
              <div className="muted">
                Image: {renderedSnapshotImage.models.image_model_name} | Aspect:{' '}
                {renderedSnapshotImage.image.aspect_ratio} | Max tokens:{' '}
                {renderedSnapshotImage.models.max_tokens} | Target ref:{' '}
                {renderedSnapshotImage.request.layout_reference_used ? 'on' : 'off'} | Object refs:{' '}
                {renderedSnapshotImage.request.reference_image_count ?? 0}
              </div>
            </div>
            <div className="three-render-chip-row">
              <span className="three-render-chip">{renderedSnapshotImage.models.image_model_name}</span>
              <span className="three-render-chip fallback">
                {renderedSnapshotImage.request.render_mode === 'edit'
                  ? 'Edit'
                  : renderedSnapshotImage.request.render_mode === 'camera_rerender'
                    ? 'Camera'
                    : 'Generate'}
              </span>
              {savedRenderId ? (
                <span className="three-render-chip fallback">
                  Saved to account: {savedRenderId}
                </span>
              ) : null}
            </div>
          </div>
            <div className="three-render-output-grid">
              <div className="three-render-image-wrap">
                <img
                  className="three-render-image"
                  src={renderedSnapshotImage.image.data_url}
                  alt="Generated architectural render from the current 3D snapshot"
                  draggable={false}
                />
                <div
                  className="three-edit-box-overlay"
                  onPointerDown={handleEditBoxPointerDown}
                  onPointerMove={handleEditBoxPointerMove}
                  onPointerUp={handleEditBoxPointerUp}
                  onPointerCancel={() => setEditBoxDraft(null)}
                >
                  {editBoxes.map((box, index) => (
                    <button
                      key={box.id}
                      className={[
                        'three-edit-box-rect',
                        activeEditBox?.id === box.id ? 'selected' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      onClick={(event) => {
                        event.stopPropagation();
                        setActiveEditBoxId(box.id);
                      }}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                      }}
                      style={{
                        borderColor: EDIT_BOX_COLORS[index % EDIT_BOX_COLORS.length],
                        left: `${box.rect.x * 100}%`,
                        top: `${box.rect.y * 100}%`,
                        width: `${box.rect.width * 100}%`,
                        height: `${box.rect.height * 100}%`,
                      }}
                      type="button"
                    >
                      <span>{box.id}</span>
                    </button>
                  ))}
                  {editBoxDraft ? (
                    <div
                      className="three-edit-box-rect draft"
                      style={{
                        left: `${Math.min(editBoxDraft.startX, editBoxDraft.currentX) * 100}%`,
                        top: `${Math.min(editBoxDraft.startY, editBoxDraft.currentY) * 100}%`,
                        width: `${Math.abs(editBoxDraft.currentX - editBoxDraft.startX) * 100}%`,
                        height: `${Math.abs(editBoxDraft.currentY - editBoxDraft.startY) * 100}%`,
                      }}
                    />
                  ) : null}
                </div>
              </div>
          </div>
        </div>
      ) : null}
      {pendingBillingPayment ? (
        <BillingPaymentModal
          amountVnd={pendingBillingPayment.amountVnd}
          chargeKind={pendingBillingPayment.chargeKind}
          countdownSeconds={billingCountdownSeconds}
          reason={pendingBillingPayment.reason}
          reference={pendingBillingPayment.reference}
          onCancel={() => finishPendingBillingPayment(false)}
        />
      ) : null}
    </div>
  );
};

function pointInPolygon(point: { x: number; y: number }, polygon: Array<{ x: number; y: number }>): boolean {
  if (!polygon || polygon.length < 3) return true;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
