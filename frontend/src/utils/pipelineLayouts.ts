import { DesignPlanV1, FloorplanGeometry, LayoutVariant, Point, Wall } from '../types';
import { normalizePoints } from './geometry';
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
  const shiftedStyledResult = shiftStyledLayout(
    baseStyledPayload,
    shiftDx,
    shiftDy,
  );
  const design = buildDesignFromStyledLayout(shiftedStyledResult, floorplan);

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
    .filter((variant): variant is LayoutVariant => variant !== null);

  const selectedVariantId = resolveSelectedVariantId(
    resultPayload?.selected_variant_id,
    variants,
  );

  return {
    floorplan,
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
  const shiftedStyledResult = shiftStyledLayout(
    enrichedStyledResult,
    shiftDx,
    shiftDy,
  );

  return {
    variant_id: variantId,
    label: String(variant.label ?? variantId),
    source: String(variant.source ?? ''),
    reason: String(variant.reason ?? ''),
    layout_score: Number(variant.layout_score ?? 0),
    hard_valid: Boolean(variant.hard_valid),
    notes: Array.isArray(variant.notes)
      ? variant.notes
          .map((item: unknown) => String(item ?? '').trim())
          .filter(Boolean)
      : [],
    absolute_layout: rawAbsoluteLayout,
    styled_result: shiftedStyledResult,
    design: buildDesignFromStyledLayout(shiftedStyledResult, floorplan),
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

function buildDesignFromStyledLayout(
  styledPayload: any,
  floorplan: FloorplanGeometry | null,
): DesignPlanV1 | null {
  if (!floorplan) {
    return null;
  }
  const objects = Array.isArray(styledPayload?.objects) ? styledPayload.objects : [];
  const placements = objects.map((objectRow: any, index: number) => {
    const manualInput =
      objectRow?.manual_input && typeof objectRow.manual_input === 'object'
        ? objectRow.manual_input
        : null;
    if (manualInput) {
      return {
        placement_id: String(
          manualInput.placement_id ??
            objectRow?.instance_id ??
            `manual_${index}`,
        ),
        asset_id: String(
          manualInput.asset_id ??
            objectRow?.instance_id ??
            objectRow?.object_type ??
            `manual_${index}`,
        ),
        category: String(manualInput.category ?? objectRow?.object_type ?? 'item'),
        position_mm: {
          x: Number(manualInput.position_mm?.x ?? 0),
          y: Number(manualInput.position_mm?.y ?? 0),
        },
        rotation_deg: Number(
          manualInput.rotation_deg ?? objectRow?.rotation_ccw ?? 0,
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
      };
    }
    const rotationDeg = Number(objectRow?.rotation_ccw ?? 0);
    const bbox = getNumericBbox(objectRow);
    const footprint = getLocalFootprintFromWorldBbox(bbox, rotationDeg);
    const center = getBboxCenter(bbox);
    return {
      placement_id: String(objectRow?.instance_id ?? `obj_${index}`),
      asset_id: String(
        objectRow?.instance_id ?? objectRow?.object_type ?? `obj_${index}`,
      ),
      category: String(objectRow?.object_type ?? 'item'),
      position_mm: center,
      rotation_deg: rotationDeg,
      footprint_mm: footprint,
      height_mm: 0,
      color_hex: String(objectRow?.color_hex ?? '#cfcac2'),
      anchor: 'center' as const,
      source: 'generated' as const,
      constraint_mode: 'free' as const,
    };
  });

  return {
    floorplan_geometry: floorplan,
    placements,
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
  const byId = new Map<string, number>();
  const byBbox = new Map<string, number>();

  for (const objectRow of absoluteObjects) {
    if (!objectRow || typeof objectRow !== 'object') {
      continue;
    }
    const objectId = String(objectRow.object_id ?? '').trim();
    const rotation = Number(objectRow.rotation_ccw ?? NaN);
    const bbox = objectRow.bbox;
    if (!Number.isFinite(rotation)) {
      continue;
    }
    if (objectId) {
      byId.set(objectId, ((rotation % 360) + 360) % 360);
    }
    const key = bboxKey(bbox);
    if (key) {
      byBbox.set(key, ((rotation % 360) + 360) % 360);
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
      const nextRotation =
        (instanceId && byId.get(instanceId)) ??
        byBbox.get(bboxKey(bbox) ?? '') ??
        Number(objectRow.rotation_ccw ?? 0);
      return {
        ...objectRow,
        rotation_ccw: Number.isFinite(nextRotation) ? nextRotation : 0,
      };
    }),
  };
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
