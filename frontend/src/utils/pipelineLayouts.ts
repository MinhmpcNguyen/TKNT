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
  const design = buildDesignFromStyledLayout(shiftedStyledResult, styledFloorplan);

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
    styledResult: shiftedStyledResult,
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
    absolute_layout: shiftedAbsoluteLayout,
    styled_result: shiftedStyledResult,
    design: buildDesignFromStyledLayout(shiftedStyledResult, styledFloorplan),
    cluster_previews: buildClusterPreviewsFromAbsoluteLayout(shiftedAbsoluteLayout),
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

function buildClusterPreviewsFromAbsoluteLayout(
  absoluteLayout: Record<string, unknown>,
): Record<string, unknown> {
  const objects = Array.isArray(absoluteLayout?.objects)
    ? absoluteLayout.objects
    : [];
  const grouped = new Map<
    string,
    Array<{ x: number; y: number; w: number; h: number }>
  >();

  for (const row of objects) {
    if (!row || typeof row !== 'object') {
      continue;
    }
    const clusterId = String((row as any).cluster_id ?? '').trim();
    const bbox = (row as any).bbox;
    if (!clusterId || !bbox || typeof bbox !== 'object') {
      continue;
    }
    const minX = Number((bbox as any).min_x ?? 0);
    const minY = Number((bbox as any).min_y ?? 0);
    const maxX = Number((bbox as any).max_x ?? minX);
    const maxY = Number((bbox as any).max_y ?? minY);
    const width = Math.max(1, maxX - minX);
    const height = Math.max(1, maxY - minY);
    const rects = grouped.get(clusterId) ?? [];
    rects.push({ x: minX, y: minY, w: width, h: height });
    grouped.set(clusterId, rects);
  }

  return Object.fromEntries(
    Array.from(grouped.entries()).map(([clusterId, rects]) => [
      clusterId,
      {
        payload: {
          cluster_id: clusterId,
          cluster_footprint: {
            type: 'union_of_rects',
            rects,
          },
        },
      },
    ]),
  );
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
