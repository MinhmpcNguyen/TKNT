import {
  DesignPlanV1,
  DirectionVector,
  FloorplanGeometry,
  LayoutVariant,
  Point,
  Wall,
} from '../types';
import { normalizePoints } from './geometry';
import {
  frontWorldToLocalSide,
  normalizeCardinalSide,
  vecToSide,
} from './planOrientation';
import {
  type NumericBbox,
  getBboxCenter,
  getLocalFootprintFromWorldBbox,
  getNumericBbox,
} from './styledLayout';

type PreparedPipelineResult = {
  floorplan: FloorplanGeometry | null;
  styledResult: any | null;
  design: DesignPlanV1 | null;
  variants: LayoutVariant[];
  selectedVariantId: string | null;
};

type FloorplanPreparation = {
  floorplan: FloorplanGeometry | null;
  shiftDx: number;
  shiftDy: number;
};

const REFILL_OBJECT_ID_PATTERN = /__(?:reintroduced|accessory_refill)(?:_\d+)?$/i;
const DEFAULT_DOOR_ASSET_ID = 'door_002';
const FLOOR_COVERING_WALL_CLEARANCE_MM = 40;
const FLOOR_COVERING_MIN_SIZE_MM = 320;

export function preparePipelineResultLayouts(params: {
  roomPayload: any;
  resultPayload: any;
  fallbackFloorplan: FloorplanGeometry | null;
}): PreparedPipelineResult {
  const { roomPayload, resultPayload, fallbackFloorplan } = params;
  const { floorplan, shiftDx, shiftDy } = buildFloorplanFromRoomArtifact(
    roomPayload,
    fallbackFloorplan,
  );

  const baseStyledPayload =
    resultPayload && typeof resultPayload === 'object'
      ? { ...resultPayload }
      : resultPayload;
  if (baseStyledPayload && typeof baseStyledPayload === 'object') {
    delete baseStyledPayload.variants;
    delete baseStyledPayload.selected_variant_id;
  }
  const baseAbsoluteLayout =
    resultPayload?.absolute_layout && typeof resultPayload.absolute_layout === 'object'
      ? resultPayload.absolute_layout
      : {};
  const enrichedBaseStyledPayload = enrichRotationCcw(
    baseStyledPayload,
    baseAbsoluteLayout,
  );
  const mergedBaseStyledPayload = mergeAbsoluteObjectsIntoStyledResult(
    enrichedBaseStyledPayload,
    baseAbsoluteLayout,
  );
  const shiftedStyledResult = shiftStyledLayout(
    mergedBaseStyledPayload,
    shiftDx,
    shiftDy,
  );
  const styledFloorplan = applyStyledRoomAppearance(floorplan, shiftedStyledResult);
  const postProcessedStyledResult = postProcessFloorCoverings(
    shiftedStyledResult,
    styledFloorplan,
  );
  const design = buildDesignFromStyledLayout(postProcessedStyledResult, styledFloorplan);

  const rawVariants = Array.isArray(resultPayload?.variants) ? resultPayload.variants : [];
  const variants = rawVariants
    .map((variant: any) =>
      prepareLayoutVariant({
        variant,
        floorplan,
        shiftDx,
        shiftDy,
      }),
    )
    .filter(
      (variant: LayoutVariant | null): variant is LayoutVariant =>
        variant !== null,
    );

  const selectedVariantId = resolveSelectedVariantId(
    resultPayload?.selected_variant_id,
    variants,
  );

  return {
    floorplan: styledFloorplan,
    styledResult: postProcessedStyledResult,
    design,
    variants,
    selectedVariantId,
  };
}

function prepareLayoutVariant(params: {
  variant: any;
  floorplan: FloorplanGeometry | null;
  shiftDx: number;
  shiftDy: number;
}): LayoutVariant | null {
  const { variant, floorplan, shiftDx, shiftDy } = params;
  if (!variant || typeof variant !== 'object') {
    return null;
  }

  const variantId = String(variant.variant_id ?? '').trim();
  if (!variantId) {
    return null;
  }

  const rawStyledResult =
    variant.styled_result && typeof variant.styled_result === 'object'
      ? variant.styled_result
      : {};
  const rawAbsoluteLayout =
    variant.absolute_layout && typeof variant.absolute_layout === 'object'
      ? variant.absolute_layout
      : {};
  const enrichedStyledResult = enrichRotationCcw(rawStyledResult, rawAbsoluteLayout);
  const mergedStyledResult = mergeAbsoluteObjectsIntoStyledResult(
    enrichedStyledResult,
    rawAbsoluteLayout,
  );
  const shiftedStyledResult = shiftStyledLayout(
    mergedStyledResult,
    shiftDx,
    shiftDy,
  );
  const shiftedAbsoluteLayout = shiftAbsoluteLayout(
    rawAbsoluteLayout,
    shiftDx,
    shiftDy,
  );
  const styledFloorplan = applyStyledRoomAppearance(floorplan, shiftedStyledResult);
  const postProcessedStyledResult = postProcessFloorCoverings(
    shiftedStyledResult,
    styledFloorplan,
  );
  const postProcessedAbsoluteLayout = postProcessFloorCoverings(
    shiftedAbsoluteLayout,
    styledFloorplan,
  );

  return {
    variant_id: variantId,
    label: String(variant.label ?? variantId),
    source: String(variant.source ?? ''),
    reason: String(variant.reason ?? ''),
    layout_score: Number(variant.layout_score ?? 0),
    hard_valid: Boolean(variant.hard_valid),
    complete:
      variant.complete === undefined ? undefined : Boolean(variant.complete),
    gallery_eligible:
      variant.gallery_eligible === undefined
        ? undefined
        : Boolean(variant.gallery_eligible),
    coverage_ratio:
      variant.coverage_ratio === undefined
        ? undefined
        : Number(variant.coverage_ratio ?? 0),
    missing_cluster_ids: Array.isArray(variant.missing_cluster_ids)
      ? variant.missing_cluster_ids
          .map((item: unknown) => String(item ?? '').trim())
          .filter(Boolean)
      : undefined,
    notes: Array.isArray(variant.notes)
      ? variant.notes
          .map((item: unknown) => String(item ?? '').trim())
          .filter(Boolean)
      : [],
    absolute_layout: postProcessedAbsoluteLayout,
    styled_result: postProcessedStyledResult,
    design: buildDesignFromStyledLayout(postProcessedStyledResult, styledFloorplan),
  };
}

function resolveSelectedVariantId(
  selectedVariantId: unknown,
  variants: LayoutVariant[],
): string | null {
  if (
    typeof selectedVariantId === 'string' &&
    variants.some((variant) => variant.variant_id === selectedVariantId)
  ) {
    return selectedVariantId;
  }
  return variants[0]?.variant_id ?? null;
}

function buildFloorplanFromRoomArtifact(
  roomPayload: any,
  fallbackFloorplan: FloorplanGeometry | null,
): FloorplanPreparation {
  const room = roomPayload?.room ?? {};
  const polygonRaw = Array.isArray(room?.polygon_ccw) ? room.polygon_ccw : [];
  if (!polygonRaw.length) {
    return {
      floorplan: fallbackFloorplan,
      shiftDx: 0,
      shiftDy: 0,
    };
  }

  const normalized = normalizePoints(polygonRaw);
  const shiftPoint = (point: any): Point => ({
    x: Number(point?.x ?? 0) - normalized.dx,
    y: Number(point?.y ?? 0) - normalized.dy,
  });
  const roomPoints = normalized.points;
  const wallThickness = 100;
  const walls: Wall[] = roomPoints.map((point, index) => {
    const next = roomPoints[(index + 1) % roomPoints.length];
    return {
      wall_id: `wall_${index + 1}`,
      p1_mm: point,
      p2_mm: next,
      thickness_mm: wallThickness,
      room_id: room?.room_id ?? 'room_1',
    };
  });

  const doors = Array.isArray(roomPayload?.openings?.doors)
    ? roomPayload.openings.doors.map((door: any, index: number) => {
        const segment = Array.isArray(door?.segment_mm) ? door.segment_mm : [];
        return {
          door_id: door?.id ?? `door_${index + 1}`,
          asset_id:
            readString(door?.asset_id) ??
            readString(door?.assetId) ??
            readString(door?.door_asset_id) ??
            readString(door?.doorAssetId) ??
            DEFAULT_DOOR_ASSET_ID,
          start_mm: shiftPoint(segment[0]),
          end_mm: shiftPoint(segment[1]),
        };
      })
    : [];
  const windows = Array.isArray(roomPayload?.openings?.windows)
    ? roomPayload.openings.windows.map((windowRow: any, index: number) => {
        const segment = Array.isArray(windowRow?.segment_mm)
          ? windowRow.segment_mm
          : [];
        return {
          window_id: windowRow?.id ?? `window_${index + 1}`,
          start_mm: shiftPoint(segment[0]),
          end_mm: shiftPoint(segment[1]),
        };
      })
    : [];

  return {
    floorplan: {
      room: {
        room_id: room?.room_id ?? 'room_1',
        name: room?.name ?? 'Room',
        polygon_mm: roomPoints,
      },
      walls,
      openings: { doors, windows },
    },
    shiftDx: normalized.dx,
    shiftDy: normalized.dy,
  };
}

function shiftStyledLayout(
  styledPayload: any,
  shiftDx: number,
  shiftDy: number,
): any {
  if (!styledPayload || typeof styledPayload !== 'object') {
    return styledPayload;
  }

  const shiftPoint = (point: any): Point => ({
    x: Number(point?.x ?? 0) - shiftDx,
    y: Number(point?.y ?? 0) - shiftDy,
  });
  const shiftBbox = (bbox: any) => {
    if (!bbox || typeof bbox !== 'object') {
      return bbox;
    }
    return {
      ...bbox,
      min_x: Number(bbox.min_x ?? 0) - shiftDx,
      min_y: Number(bbox.min_y ?? 0) - shiftDy,
      max_x: Number(bbox.max_x ?? 0) - shiftDx,
      max_y: Number(bbox.max_y ?? 0) - shiftDy,
    };
  };
  const shiftManualInput = (manualInput: any) => {
    if (!manualInput || typeof manualInput !== 'object') {
      return manualInput;
    }
    return {
      ...manualInput,
      position_mm:
        manualInput.position_mm && typeof manualInput.position_mm === 'object'
          ? shiftPoint(manualInput.position_mm)
          : manualInput.position_mm,
    };
  };

  const room = styledPayload.room;
  const shiftedRoom =
    room && typeof room === 'object'
      ? {
          ...room,
          polygon_mm: Array.isArray(room.polygon_mm)
            ? room.polygon_mm.map(shiftPoint)
            : room.polygon_mm,
          polygon_ccw: Array.isArray(room.polygon_ccw)
            ? room.polygon_ccw.map(shiftPoint)
            : room.polygon_ccw,
          obstacles: Array.isArray(room.obstacles)
            ? room.obstacles.map((obstacle: any) => ({
                ...obstacle,
                polygon_ccw: Array.isArray(obstacle?.polygon_ccw)
                  ? obstacle.polygon_ccw.map(shiftPoint)
                  : obstacle?.polygon_ccw,
              }))
            : room.obstacles,
          openings:
            room.openings && typeof room.openings === 'object'
              ? {
                  ...room.openings,
                  doors: Array.isArray(room.openings.doors)
                    ? room.openings.doors.map((door: any) => ({
                        ...door,
                        segment_mm: Array.isArray(door?.segment_mm)
                          ? door.segment_mm.map(shiftPoint)
                          : door?.segment_mm,
                      }))
                    : room.openings.doors,
                  windows: Array.isArray(room.openings.windows)
                    ? room.openings.windows.map((windowRow: any) => ({
                        ...windowRow,
                        segment_mm: Array.isArray(windowRow?.segment_mm)
                          ? windowRow.segment_mm.map(shiftPoint)
                          : windowRow?.segment_mm,
                      }))
                    : room.openings.windows,
                }
              : room.openings,
        }
      : room;

  const shiftedObjects = Array.isArray(styledPayload.objects)
    ? styledPayload.objects.map((objectRow: any) => ({
        ...objectRow,
        polygon_ccw: Array.isArray(objectRow?.polygon_ccw)
          ? objectRow.polygon_ccw.map(shiftPoint)
          : objectRow?.polygon_ccw,
        bbox: shiftBbox(objectRow?.bbox),
        manual_input: shiftManualInput(objectRow?.manual_input),
      }))
    : styledPayload.objects;

  return {
    ...styledPayload,
    room: shiftedRoom,
    objects: shiftedObjects,
  };
}

function mergeAbsoluteObjectsIntoStyledResult(
  styledPayload: any,
  absoluteLayoutPayload: any,
): any {
  if (!styledPayload || typeof styledPayload !== 'object') {
    return styledPayload;
  }

  const absoluteObjects = Array.isArray(absoluteLayoutPayload?.objects)
    ? absoluteLayoutPayload.objects
    : [];
  if (absoluteObjects.length === 0) {
    return styledPayload;
  }

  const styledObjects = Array.isArray(styledPayload.objects)
    ? styledPayload.objects
    : [];
  const styledByKey = new Map<string, Record<string, unknown>>();
  for (const objectRow of styledObjects) {
    if (!isRecordValue(objectRow)) {
      continue;
    }
    const key = getLayoutObjectKey(objectRow);
    if (key) {
      styledByKey.set(key, objectRow);
    }
  }

  const mergedObjects: Array<Record<string, unknown>> = [];
  for (let index = 0; index < absoluteObjects.length; index += 1) {
    const absoluteRow = absoluteObjects[index];
    if (!isRecordValue(absoluteRow)) {
      continue;
    }
    const key = getLayoutObjectKey(absoluteRow);
    const styledRow = key ? styledByKey.get(key) ?? null : null;
    mergedObjects.push(mergeAbsoluteObjectRow(absoluteRow, styledRow, index));
  }

  const seenKeys = new Set(
    mergedObjects
      .map((objectRow) => getLayoutObjectKey(objectRow))
      .filter((value): value is string => Boolean(value)),
  );
  for (const styledRow of styledObjects) {
    if (!isRecordValue(styledRow)) {
      continue;
    }
    const key = getLayoutObjectKey(styledRow);
    if (key && !seenKeys.has(key)) {
      mergedObjects.push(styledRow);
    }
  }

  return {
    ...styledPayload,
    objects: mergedObjects,
  };
}

function mergeAbsoluteObjectRow(
  absoluteRow: Record<string, unknown>,
  styledRow: Record<string, unknown> | null,
  index: number,
): Record<string, unknown> {
  const instanceId =
    readString(absoluteRow.instance_id) ??
    readString(styledRow?.instance_id) ??
    readString(absoluteRow.object_id) ??
    readString(styledRow?.object_id) ??
    readString(absoluteRow.id) ??
    readString(styledRow?.id) ??
    `obj_${index}`;
  const objectType =
    readString(absoluteRow.object_type) ??
    readString(styledRow?.object_type) ??
    readString(absoluteRow.category) ??
    readString(styledRow?.category) ??
    'item';

  return {
    ...(styledRow ?? {}),
    ...absoluteRow,
    instance_id: instanceId,
    object_type: objectType,
    color_hex: absoluteRow.color_hex ?? styledRow?.color_hex ?? null,
    material: absoluteRow.material ?? styledRow?.material ?? null,
    place_on: absoluteRow.place_on ?? styledRow?.place_on ?? null,
    rotation_ccw:
      absoluteRow.rotation_ccw ??
      absoluteRow.rot ??
      styledRow?.rotation_ccw ??
      styledRow?.rotation_deg ??
      0,
    front_world: absoluteRow.front_world ?? styledRow?.front_world ?? null,
    front_side_world:
      absoluteRow.front_side_world ?? styledRow?.front_side_world ?? null,
    axis_world: absoluteRow.axis_world ?? styledRow?.axis_world ?? null,
    bbox: absoluteRow.bbox ?? styledRow?.bbox ?? null,
    collision_layer: absoluteRow.collision_layer ?? styledRow?.collision_layer ?? null,
  };
}

function shiftAbsoluteLayout(
  absoluteLayoutPayload: any,
  shiftDx: number,
  shiftDy: number,
): Record<string, unknown> {
  if (!isRecordValue(absoluteLayoutPayload)) {
    return {};
  }
  if (shiftDx === 0 && shiftDy === 0) {
    return absoluteLayoutPayload;
  }

  const output: Record<string, unknown> = { ...absoluteLayoutPayload };
  output.room = shiftAbsoluteRoom(output.room, shiftDx, shiftDy);
  output.openings = shiftOpeningCollections(output.openings, shiftDx, shiftDy);
  for (const key of ['objects', 'placements', 'object_placements']) {
    const rows = output[key];
    if (Array.isArray(rows)) {
      output[key] = rows.map((row) => shiftAbsoluteObjectRow(row, shiftDx, shiftDy));
    }
  }
  return output;
}

function shiftAbsoluteRoom(value: unknown, shiftDx: number, shiftDy: number): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  return {
    ...value,
    polygon_ccw: shiftPointArray(value.polygon_ccw, shiftDx, shiftDy),
    polygon_mm: shiftPointArray(value.polygon_mm, shiftDx, shiftDy),
    centroid_mm: shiftPointRecord(value.centroid_mm, shiftDx, shiftDy),
    bbox_mm: shiftBboxRecord(value.bbox_mm, shiftDx, shiftDy),
    openings: shiftOpeningCollections(value.openings, shiftDx, shiftDy),
  };
}

function shiftOpeningCollections(
  value: unknown,
  shiftDx: number,
  shiftDy: number,
): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  const output: Record<string, unknown> = { ...value };
  for (const key of ['doors', 'windows']) {
    const rows = output[key];
    if (Array.isArray(rows)) {
      output[key] = rows.map((row) => shiftOpeningRow(row, shiftDx, shiftDy));
    }
  }
  return output;
}

function shiftOpeningRow(value: unknown, shiftDx: number, shiftDy: number): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  return {
    ...value,
    segment_mm: shiftPointArray(value.segment_mm, shiftDx, shiftDy),
    start_mm: shiftPointRecord(value.start_mm, shiftDx, shiftDy),
    end_mm: shiftPointRecord(value.end_mm, shiftDx, shiftDy),
  };
}

function shiftAbsoluteObjectRow(
  value: unknown,
  shiftDx: number,
  shiftDy: number,
): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  const output: Record<string, unknown> = {
    ...value,
    bbox: shiftBboxRecord(value.bbox, shiftDx, shiftDy),
    center: shiftPointRecord(value.center, shiftDx, shiftDy),
    position_mm: shiftPointRecord(value.position_mm, shiftDx, shiftDy),
    polygon_ccw: shiftPointArray(value.polygon_ccw, shiftDx, shiftDy),
    polygon_mm: shiftPointArray(value.polygon_mm, shiftDx, shiftDy),
  };
  if (Object.prototype.hasOwnProperty.call(value, 'x')) {
    output.x = shiftCoordinate(value.x, shiftDx);
  }
  if (Object.prototype.hasOwnProperty.call(value, 'y')) {
    output.y = shiftCoordinate(value.y, shiftDy);
  }
  if (Array.isArray(value.rect) && value.rect.length >= 4) {
    output.rect = value.rect.map((item, index) =>
      index % 2 === 0
        ? shiftCoordinate(item, shiftDx)
        : shiftCoordinate(item, shiftDy),
    );
  }
  return output;
}

function shiftPointArray(value: unknown, shiftDx: number, shiftDy: number): unknown {
  if (!Array.isArray(value)) {
    return value;
  }
  return value.map((point) => shiftPointRecord(point, shiftDx, shiftDy));
}

function shiftPointRecord(value: unknown, shiftDx: number, shiftDy: number): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  return {
    ...value,
    x: shiftCoordinate(value.x, shiftDx),
    y: shiftCoordinate(value.y, shiftDy),
  };
}

function shiftBboxRecord(value: unknown, shiftDx: number, shiftDy: number): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  const output: Record<string, unknown> = { ...value };
  for (const key of ['min_x', 'max_x', 'minX', 'maxX']) {
    if (Object.prototype.hasOwnProperty.call(value, key)) {
      output[key] = shiftCoordinate(value[key], shiftDx);
    }
  }
  for (const key of ['min_y', 'max_y', 'minY', 'maxY']) {
    if (Object.prototype.hasOwnProperty.call(value, key)) {
      output[key] = shiftCoordinate(value[key], shiftDy);
    }
  }
  return output;
}

function shiftCoordinate(value: unknown, delta: number): unknown {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue - delta : value;
}

function getLayoutObjectKey(objectRow: Record<string, unknown>): string | null {
  return (
    readString(objectRow.instance_id) ??
    readString(objectRow.object_id) ??
    readString(objectRow.id)
  );
}

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function isRecordValue(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function postProcessFloorCoverings(
  layoutPayload: any,
  floorplan: FloorplanGeometry | null,
): any {
  if (!isRecordValue(layoutPayload) || !floorplan) {
    return layoutPayload;
  }
  const objects = layoutPayload.objects;
  if (!Array.isArray(objects)) {
    return layoutPayload;
  }

  let changed = false;
  const processedObjects = objects.map((objectRow) => {
    if (!isRecordValue(objectRow)) {
      return objectRow;
    }
    const processedRow = postProcessFloorCoveringRow(objectRow, floorplan);
    if (processedRow !== objectRow) {
      changed = true;
    }
    return processedRow;
  });

  if (!changed) {
    return layoutPayload;
  }
  return {
    ...layoutPayload,
    objects: processedObjects,
  };
}

function postProcessFloorCoveringRow(
  objectRow: Record<string, unknown>,
  floorplan: FloorplanGeometry,
): Record<string, unknown> {
  if (!isFloorCoveringRow(objectRow) || isLockedLayoutObject(objectRow)) {
    return objectRow;
  }
  const bbox = getNumericBbox(objectRow);
  if (!isUsableBbox(bbox)) {
    return objectRow;
  }
  const fittedBbox = fitFloorCoveringBboxToRoom(bbox, floorplan);
  if (bboxAlmostEqual(bbox, fittedBbox)) {
    return objectRow;
  }
  return transformLayoutObjectToBbox(objectRow, bbox, fittedBbox);
}

function isFloorCoveringRow(objectRow: Record<string, unknown>): boolean {
  const categoryParts = [
    objectRow.object_type,
    objectRow.category,
    objectRow.type,
    isRecordValue(objectRow.manual_input)
      ? objectRow.manual_input.object_type ??
        objectRow.manual_input.category ??
        objectRow.manual_input.type
      : null,
  ];
  const category = categoryParts
    .map((value) => (typeof value === 'string' ? value.toLowerCase() : ''))
    .join(' ');
  const collisionLayer =
    typeof objectRow.collision_layer === 'string'
      ? objectRow.collision_layer.trim().toLowerCase()
      : '';
  return (
    collisionLayer === 'floor_underlay' ||
    category.includes('rug') ||
    category.includes('carpet') ||
    category.includes('floor_mat') ||
    category.includes('area_mat') ||
    category.includes('runner')
  );
}

function isLockedLayoutObject(objectRow: Record<string, unknown>): boolean {
  if (objectRow.constraint_mode === 'locked') {
    return true;
  }
  const manualInput = objectRow.manual_input;
  return isRecordValue(manualInput) && manualInput.constraint_mode === 'locked';
}

function fitFloorCoveringBboxToRoom(
  bbox: NumericBbox,
  floorplan: FloorplanGeometry,
): NumericBbox {
  let best = pushBboxInsideWallClearance(bbox, floorplan);
  if (floorCoveringBboxFits(best, floorplan)) {
    return best;
  }

  for (let step = 1; step <= 18; step += 1) {
    const scale = Math.max(0.52, 1 - step * 0.035);
    const scaled = scaleBboxAroundCenter(best, scale, scale);
    if (!isUsableBbox(scaled)) {
      break;
    }
    const adjusted = pushBboxInsideWallClearance(scaled, floorplan);
    if (floorCoveringBboxFits(adjusted, floorplan)) {
      return adjusted;
    }
    best = adjusted;
  }

  return best;
}

function pushBboxInsideWallClearance(
  bbox: NumericBbox,
  floorplan: FloorplanGeometry,
): NumericBbox {
  const polygon = floorplan.room?.polygon_mm ?? [];
  if (polygon.length < 3 || floorplan.walls.length === 0) {
    return bbox;
  }

  let next = { ...bbox };
  for (let pass = 0; pass < 4; pass += 1) {
    let moved = false;
    for (const wall of floorplan.walls) {
      const constraint = getWallInteriorConstraint(wall, polygon, next);
      if (!constraint) {
        continue;
      }
      const requiredDistance =
        Math.max(0, Number(wall.thickness_mm ?? 0)) / 2 +
        FLOOR_COVERING_WALL_CLEARANCE_MM;
      const minDistance = Math.min(
        ...bboxCorners(next).map((corner) =>
          dot(
            {
              x: corner.x - constraint.point.x,
              y: corner.y - constraint.point.y,
            },
            constraint.interiorNormal,
          ),
        ),
      );
      if (minDistance + 1e-6 >= requiredDistance) {
        continue;
      }
      const push = requiredDistance - minDistance;
      next = translateBbox(next, {
        x: constraint.interiorNormal.x * push,
        y: constraint.interiorNormal.y * push,
      });
      moved = true;
    }
    if (!moved) {
      break;
    }
  }
  return next;
}

function getWallInteriorConstraint(
  wall: Wall,
  polygon: Point[],
  bbox: NumericBbox,
): { point: Point; interiorNormal: Point } | null {
  const center = bboxCenter(bbox);
  const projection = projectPointToSegment(center, wall.p1_mm, wall.p2_mm);
  if (!projection || projection.t <= 1e-6 || projection.t >= 1 - 1e-6) {
    return null;
  }
  const leftNormal = { x: -projection.dir.y, y: projection.dir.x };
  return {
    point: projection.point,
    interiorNormal: interiorNormalForWall(projection.point, leftNormal, polygon),
  };
}

function floorCoveringBboxFits(
  bbox: NumericBbox,
  floorplan: FloorplanGeometry,
): boolean {
  const polygon = floorplan.room?.polygon_mm ?? [];
  if (polygon.length < 3) {
    return true;
  }
  return (
    bboxCorners(bbox).every((corner) => pointInPolygon(corner, polygon)) &&
    bboxAlmostEqual(bbox, pushBboxInsideWallClearance(bbox, floorplan))
  );
}

function transformLayoutObjectToBbox(
  objectRow: Record<string, unknown>,
  sourceBbox: NumericBbox,
  targetBbox: NumericBbox,
): Record<string, unknown> {
  const center = bboxCenter(targetBbox);
  const output: Record<string, unknown> = {
    ...objectRow,
    bbox: writeBboxRecord(objectRow.bbox, targetBbox),
    center,
  };
  if (Object.prototype.hasOwnProperty.call(objectRow, 'x')) {
    output.x = transformCoordinate(
      objectRow.x,
      sourceBbox.minX,
      sourceBbox.maxX,
      targetBbox.minX,
      targetBbox.maxX,
    );
  }
  if (Object.prototype.hasOwnProperty.call(objectRow, 'y')) {
    output.y = transformCoordinate(
      objectRow.y,
      sourceBbox.minY,
      sourceBbox.maxY,
      targetBbox.minY,
      targetBbox.maxY,
    );
  }
  output.polygon_ccw = transformPointArrayWithinBbox(
    objectRow.polygon_ccw,
    sourceBbox,
    targetBbox,
  );
  output.polygon_mm = transformPointArrayWithinBbox(
    objectRow.polygon_mm,
    sourceBbox,
    targetBbox,
  );
  output.rect = transformRectArrayWithinBbox(objectRow.rect, sourceBbox, targetBbox);
  if (isRecordValue(objectRow.manual_input)) {
    output.manual_input = transformManualInputWithinBbox(
      objectRow.manual_input,
      sourceBbox,
      targetBbox,
    );
  }
  return output;
}

function transformManualInputWithinBbox(
  manualInput: Record<string, unknown>,
  sourceBbox: NumericBbox,
  targetBbox: NumericBbox,
): Record<string, unknown> {
  const widthScale = bboxWidth(targetBbox) / Math.max(1, bboxWidth(sourceBbox));
  const depthScale = bboxDepth(targetBbox) / Math.max(1, bboxDepth(sourceBbox));
  const next: Record<string, unknown> = {
    ...manualInput,
    position_mm: transformPointRecordWithinBbox(
      manualInput.position_mm,
      sourceBbox,
      targetBbox,
    ),
  };
  if (isRecordValue(manualInput.footprint_mm)) {
    next.footprint_mm = {
      ...manualInput.footprint_mm,
      w: Math.max(
        FLOOR_COVERING_MIN_SIZE_MM,
        Number(manualInput.footprint_mm.w ?? 0) * widthScale,
      ),
      d: Math.max(
        FLOOR_COVERING_MIN_SIZE_MM,
        Number(manualInput.footprint_mm.d ?? 0) * depthScale,
      ),
    };
  }
  return next;
}

function transformPointArrayWithinBbox(
  value: unknown,
  sourceBbox: NumericBbox,
  targetBbox: NumericBbox,
): unknown {
  if (!Array.isArray(value)) {
    return value;
  }
  return value.map((point) =>
    transformPointRecordWithinBbox(point, sourceBbox, targetBbox),
  );
}

function transformPointRecordWithinBbox(
  value: unknown,
  sourceBbox: NumericBbox,
  targetBbox: NumericBbox,
): unknown {
  if (!isRecordValue(value)) {
    return value;
  }
  return {
    ...value,
    x: transformCoordinate(
      value.x,
      sourceBbox.minX,
      sourceBbox.maxX,
      targetBbox.minX,
      targetBbox.maxX,
    ),
    y: transformCoordinate(
      value.y,
      sourceBbox.minY,
      sourceBbox.maxY,
      targetBbox.minY,
      targetBbox.maxY,
    ),
  };
}

function transformRectArrayWithinBbox(
  value: unknown,
  sourceBbox: NumericBbox,
  targetBbox: NumericBbox,
): unknown {
  if (!Array.isArray(value)) {
    return value;
  }
  return value.map((item, index) =>
    index % 2 === 0
      ? transformCoordinate(
          item,
          sourceBbox.minX,
          sourceBbox.maxX,
          targetBbox.minX,
          targetBbox.maxX,
        )
      : transformCoordinate(
          item,
          sourceBbox.minY,
          sourceBbox.maxY,
          targetBbox.minY,
          targetBbox.maxY,
        ),
  );
}

function transformCoordinate(
  value: unknown,
  sourceMin: number,
  sourceMax: number,
  targetMin: number,
  targetMax: number,
): unknown {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return value;
  }
  const sourceSpan = Math.max(1, sourceMax - sourceMin);
  const ratio = (numeric - sourceMin) / sourceSpan;
  return targetMin + ratio * (targetMax - targetMin);
}

function writeBboxRecord(value: unknown, bbox: NumericBbox): Record<string, unknown> {
  return {
    ...(isRecordValue(value) ? value : {}),
    min_x: bbox.minX,
    min_y: bbox.minY,
    max_x: bbox.maxX,
    max_y: bbox.maxY,
  };
}

function isUsableBbox(bbox: NumericBbox): boolean {
  return (
    [bbox.minX, bbox.minY, bbox.maxX, bbox.maxY].every(Number.isFinite) &&
    bboxWidth(bbox) >= FLOOR_COVERING_MIN_SIZE_MM &&
    bboxDepth(bbox) >= FLOOR_COVERING_MIN_SIZE_MM
  );
}

function bboxCorners(bbox: NumericBbox): Point[] {
  return [
    { x: bbox.minX, y: bbox.minY },
    { x: bbox.maxX, y: bbox.minY },
    { x: bbox.maxX, y: bbox.maxY },
    { x: bbox.minX, y: bbox.maxY },
  ];
}

function bboxCenter(bbox: NumericBbox): Point {
  return {
    x: (bbox.minX + bbox.maxX) / 2,
    y: (bbox.minY + bbox.maxY) / 2,
  };
}

function bboxWidth(bbox: NumericBbox): number {
  return Math.max(0, bbox.maxX - bbox.minX);
}

function bboxDepth(bbox: NumericBbox): number {
  return Math.max(0, bbox.maxY - bbox.minY);
}

function translateBbox(bbox: NumericBbox, delta: Point): NumericBbox {
  return {
    minX: bbox.minX + delta.x,
    minY: bbox.minY + delta.y,
    maxX: bbox.maxX + delta.x,
    maxY: bbox.maxY + delta.y,
  };
}

function scaleBboxAroundCenter(
  bbox: NumericBbox,
  widthScale: number,
  depthScale: number,
): NumericBbox {
  const center = bboxCenter(bbox);
  const halfWidth = Math.max(
    FLOOR_COVERING_MIN_SIZE_MM / 2,
    (bboxWidth(bbox) * widthScale) / 2,
  );
  const halfDepth = Math.max(
    FLOOR_COVERING_MIN_SIZE_MM / 2,
    (bboxDepth(bbox) * depthScale) / 2,
  );
  return {
    minX: center.x - halfWidth,
    minY: center.y - halfDepth,
    maxX: center.x + halfWidth,
    maxY: center.y + halfDepth,
  };
}

function bboxAlmostEqual(a: NumericBbox, b: NumericBbox, tolerance = 0.5): boolean {
  return (
    Math.abs(a.minX - b.minX) <= tolerance &&
    Math.abs(a.minY - b.minY) <= tolerance &&
    Math.abs(a.maxX - b.maxX) <= tolerance &&
    Math.abs(a.maxY - b.maxY) <= tolerance
  );
}

function projectPointToSegment(
  point: Point,
  start: Point,
  end: Point,
): { t: number; point: Point; dir: Point } | null {
  const vx = end.x - start.x;
  const vy = end.y - start.y;
  const lengthSquared = vx * vx + vy * vy;
  if (lengthSquared < 1e-6) {
    return null;
  }
  const rawT =
    ((point.x - start.x) * vx + (point.y - start.y) * vy) / lengthSquared;
  const t = Math.max(0, Math.min(1, rawT));
  const length = Math.sqrt(lengthSquared) || 1;
  return {
    t,
    point: { x: start.x + t * vx, y: start.y + t * vy },
    dir: { x: vx / length, y: vy / length },
  };
}

function interiorNormalForWall(
  boundaryPoint: Point,
  leftNormal: Point,
  polygon: Point[],
): Point {
  const eps = 1;
  const leftTest = {
    x: boundaryPoint.x + leftNormal.x * eps,
    y: boundaryPoint.y + leftNormal.y * eps,
  };
  const rightTest = {
    x: boundaryPoint.x - leftNormal.x * eps,
    y: boundaryPoint.y - leftNormal.y * eps,
  };
  const leftInside = pointInPolygon(leftTest, polygon);
  const rightInside = pointInPolygon(rightTest, polygon);
  if (leftInside && !rightInside) {
    return leftNormal;
  }
  if (rightInside && !leftInside) {
    return { x: -leftNormal.x, y: -leftNormal.y };
  }
  const center = polygonCentroid(polygon);
  const toCenter = {
    x: center.x - boundaryPoint.x,
    y: center.y - boundaryPoint.y,
  };
  return dot(toCenter, leftNormal) >= 0
    ? leftNormal
    : { x: -leftNormal.x, y: -leftNormal.y };
}

function polygonCentroid(points: Point[]): Point {
  if (points.length === 0) {
    return { x: 0, y: 0 };
  }
  const sum = points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function dot(a: Point, b: Point): number {
  return a.x * b.x + a.y * b.y;
}

function pointInPolygon(point: Point, polygon: Point[]): boolean {
  if (polygon.length < 3) {
    return true;
  }
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

function buildDesignFromStyledLayout(
  styledPayload: any,
  floorplan: FloorplanGeometry | null,
): DesignPlanV1 | null {
  if (!floorplan) {
    return null;
  }
  const objects = Array.isArray(styledPayload?.objects) ? styledPayload.objects : [];
  const placements = objects.map((objectRow: any, index: number) => {
    const category = String(objectRow?.object_type ?? 'item');
    const rawInstanceId = String(
      objectRow?.instance_id ?? objectRow?.id ?? objectRow?.object_id ?? `obj_${index}`,
    );
    const placeOn = normalizePlaceOn(objectRow?.place_on);
    const manualInput =
      objectRow?.manual_input && typeof objectRow.manual_input === 'object'
        ? objectRow.manual_input
        : null;
    const frontWorld = normalizeFrontWorld(objectRow?.front_world);
    const frontSideWorld =
      normalizeCardinalSide(objectRow?.front_side_world) ??
      vecToSide(frontWorld);
    const normalizedAssetId =
      normalizeLayoutAssetId(
        manualInput?.asset_id ??
          rawInstanceId ??
          objectRow?.object_id ??
          category,
      ) ?? rawInstanceId;
    if (manualInput) {
      return {
        placement_id: String(
          manualInput.placement_id ??
            rawInstanceId ??
            `manual_${index}`,
        ),
        asset_id: normalizedAssetId,
        category: String(manualInput.category ?? category),
        position_mm: {
          x: Number(manualInput.position_mm?.x ?? 0),
          y: Number(manualInput.position_mm?.y ?? 0),
        },
        rotation_deg: Number(
          manualInput.rotation_deg ?? getObjectRotationCcw(objectRow),
        ),
        front_world: frontWorld,
        front_side_world: frontSideWorld,
        front_side_local: frontWorldToLocalSide(
          frontWorld,
          Number(manualInput.rotation_deg ?? getObjectRotationCcw(objectRow)),
        ),
        footprint_mm: {
          w: Number(manualInput.footprint_mm?.w ?? 0),
          d: Number(manualInput.footprint_mm?.d ?? 0),
        },
        height_mm: Number(manualInput.height_mm ?? 0),
        color_hex: String(
          objectRow?.color_hex ?? manualInput.color_hex ?? '#cfcac2',
        ),
        anchor: 'center' as const,
        source: manualInput.source === 'manual' ? 'manual' : 'generated',
        constraint_mode:
          manualInput.constraint_mode === 'locked'
            ? 'locked'
            : manualInput.constraint_mode === 'pinned'
              ? 'pinned'
              : 'free',
        place_on: placeOn,
        collision_layer: normalizeCollisionLayer(objectRow?.collision_layer),
      };
    }
    const rotationDeg = getObjectRotationCcw(objectRow);
    const bbox = getNumericBbox(objectRow);
    const footprint = normalizePlacementFootprint(
      category,
      getLocalFootprintFromWorldBbox(bbox, rotationDeg),
      placeOn,
    );
    const center = getBboxCenter(bbox);
    return {
      placement_id: rawInstanceId,
      asset_id: normalizedAssetId,
      category,
      position_mm: center,
      rotation_deg: rotationDeg,
      front_world: frontWorld,
      front_side_world: frontSideWorld,
      front_side_local: frontWorldToLocalSide(frontWorld, rotationDeg),
      footprint_mm: footprint,
      height_mm: 0,
      color_hex: String(objectRow?.color_hex ?? '#cfcac2'),
      anchor: 'center' as const,
      source: 'generated' as const,
      constraint_mode: 'free' as const,
      place_on: placeOn,
      collision_layer: normalizeCollisionLayer(objectRow?.collision_layer),
    };
  });

  return {
    floorplan_geometry: floorplan,
    placements,
  };
}

function normalizeCollisionLayer(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const layer = value.trim();
  return layer || null;
}

function normalizeLayoutAssetId(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const raw = value.trim();
  if (!raw) {
    return null;
  }
  const normalized = raw.replace(REFILL_OBJECT_ID_PATTERN, '').trim();
  return normalized || raw;
}

function normalizePlaceOn(
  placeOn: any,
): {
  method: string;
  target_instance_id?: string | null;
} | null {
  if (!placeOn || typeof placeOn !== 'object') {
    return null;
  }
  const method = String(placeOn.method ?? '').trim();
  if (!method) {
    return null;
  }
  const targetInstanceId =
    typeof placeOn.target_instance_id === 'string' && placeOn.target_instance_id.trim()
      ? placeOn.target_instance_id.trim()
      : null;
  return {
    method,
    target_instance_id: targetInstanceId,
  };
}

function normalizePlacementFootprint(
  category: string,
  footprint: { w: number; d: number },
  placeOn: { method: string; target_instance_id?: string | null } | null,
): { w: number; d: number } {
  if (
    placeOn?.method === 'hang_on' &&
    isSlimWallMountedCategory(category)
  ) {
    const longSide = Math.max(footprint.w, footprint.d, 1);
    const shortSide = Math.max(24, Math.min(72, Math.min(footprint.w, footprint.d, 60)));
    return { w: longSide, d: shortSide };
  }
  return footprint;
}

function isSlimWallMountedCategory(category: string): boolean {
  const lowered = category.toLowerCase();
  return (
    lowered.includes('mirror') ||
    lowered.includes('wall_art') ||
    lowered.includes('clock') ||
    lowered.includes('sconce') ||
    lowered.includes('whiteboard') ||
    lowered.includes('screen')
  );
}

function applyStyledRoomAppearance(
  floorplan: FloorplanGeometry | null,
  styledPayload: any,
): FloorplanGeometry | null {
  if (!floorplan) {
    return null;
  }
  const surfaces = styledPayload?.room?.surfaces;
  const openingColors =
    styledPayload?.room?.opening_colors ?? styledPayload?.room?.openingColors;
  const wallColor =
    typeof surfaces?.wall_color_hex === 'string'
      ? surfaces.wall_color_hex
      : null;
  const floorColor =
    typeof surfaces?.floor_color_hex === 'string'
      ? surfaces.floor_color_hex
      : null;
  const doorColors = new Map<string, string>();
  const windowColors = new Map<string, string>();

  if (Array.isArray(openingColors?.doors)) {
    for (const row of openingColors.doors) {
      const id =
        typeof row?.id === 'string' && row.id.trim() ? row.id.trim() : null;
      const color =
        typeof row?.color_hex === 'string' && row.color_hex.trim()
          ? row.color_hex.trim()
          : null;
      if (id && color) {
        doorColors.set(id, color);
      }
    }
  }

  if (Array.isArray(openingColors?.windows)) {
    for (const row of openingColors.windows) {
      const id =
        typeof row?.id === 'string' && row.id.trim() ? row.id.trim() : null;
      const color =
        typeof row?.color_hex === 'string' && row.color_hex.trim()
          ? row.color_hex.trim()
          : null;
      if (id && color) {
        windowColors.set(id, color);
      }
    }
  }

  return {
    ...floorplan,
    room: {
      ...floorplan.room,
      floor_color_hex: floorColor ?? floorplan.room.floor_color_hex,
    },
    walls: floorplan.walls.map((wall) => ({
      ...wall,
      color_hex: wallColor ?? wall.color_hex,
    })),
    openings: {
      doors: floorplan.openings.doors.map((door) => ({
        ...door,
        color_hex: doorColors.get(door.door_id) ?? door.color_hex,
      })),
      windows: floorplan.openings.windows.map((windowRow) => ({
        ...windowRow,
        color_hex: windowColors.get(windowRow.window_id) ?? windowRow.color_hex,
      })),
    },
  };
}

function enrichRotationCcw(styledPayload: any, absoluteLayoutPayload: any): any {
  if (!styledPayload || typeof styledPayload !== 'object') {
    return styledPayload;
  }
  if (!absoluteLayoutPayload || typeof absoluteLayoutPayload !== 'object') {
    return styledPayload;
  }

  const absoluteObjects = Array.isArray(absoluteLayoutPayload.objects)
    ? absoluteLayoutPayload.objects
    : [];
  const byId = new Map<string, Record<string, unknown>>();
  const byBbox = new Map<string, Record<string, unknown>>();

  for (const objectRow of absoluteObjects) {
    if (!objectRow || typeof objectRow !== 'object') {
      continue;
    }
    const objectId = String(objectRow.object_id ?? '').trim();
    const rotation = getObjectRotationCcw(objectRow);
    const bbox = objectRow.bbox;
    if (!Number.isFinite(rotation)) {
      continue;
    }
    const orientationPayload = {
      rotation_ccw: ((rotation % 360) + 360) % 360,
      front_world: normalizeFrontWorld(objectRow.front_world),
      front_side_world:
        normalizeCardinalSide(objectRow.front_side_world) ??
        vecToSide(normalizeFrontWorld(objectRow.front_world)),
      axis_world:
        objectRow.axis_world && typeof objectRow.axis_world === 'object'
          ? objectRow.axis_world
          : null,
    };
    if (objectId) {
      byId.set(objectId, orientationPayload);
    }
    const key = bboxKey(bbox);
    if (key) {
      byBbox.set(key, orientationPayload);
    }
  }

  const objects = Array.isArray(styledPayload.objects) ? styledPayload.objects : [];
  return {
    ...styledPayload,
    objects: objects.map((objectRow: any) => {
      if (!objectRow || typeof objectRow !== 'object') {
        return objectRow;
      }
      const instanceId = String(
        objectRow.instance_id ?? objectRow.id ?? '',
      ).trim();
      const bbox = objectRow.bbox;
      const bboxOrientationPayload = byBbox.get(bboxKey(bbox) ?? '') ?? null;
      const orientationPayload: Record<string, unknown> | null = instanceId
        ? byId.get(instanceId) ?? bboxOrientationPayload
        : bboxOrientationPayload;
      const nextRotation = Number(
        orientationPayload?.rotation_ccw ?? getObjectRotationCcw(objectRow),
      );
      return {
        ...objectRow,
        rotation_ccw: Number.isFinite(nextRotation) ? nextRotation : 0,
        front_world:
          normalizeFrontWorld(orientationPayload?.front_world) ??
          normalizeFrontWorld(objectRow.front_world),
        front_side_world:
          normalizeCardinalSide(orientationPayload?.front_side_world) ??
          normalizeCardinalSide(objectRow.front_side_world),
        axis_world:
          orientationPayload?.axis_world && typeof orientationPayload.axis_world === 'object'
            ? orientationPayload.axis_world
            : objectRow.axis_world,
      };
    }),
  };
}

function normalizeFrontWorld(value: unknown): DirectionVector | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const dx = Number((value as { dx?: unknown }).dx ?? NaN);
  const dy = Number((value as { dy?: unknown }).dy ?? NaN);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return null;
  }
  return { dx, dy };
}

function getObjectRotationCcw(objectRow: any): number {
  const rotation = Number(
    objectRow?.rotation_ccw ?? objectRow?.rot ?? objectRow?.rotation_deg ?? 0,
  );
  if (!Number.isFinite(rotation)) {
    return 0;
  }
  return ((rotation % 360) + 360) % 360;
}

function bboxKey(bbox: any): string | null {
  if (!bbox || typeof bbox !== 'object') {
    return null;
  }
  return [
    Number(bbox.min_x ?? 0),
    Number(bbox.min_y ?? 0),
    Number(bbox.max_x ?? 0),
    Number(bbox.max_y ?? 0),
  ].join(':');
}
