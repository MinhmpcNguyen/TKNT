import React from 'react';
import { LayoutVariant, Point } from '../types';
import { useEditorStore } from '../store';

type PreviewRect = {
  kind: 'rect';
  id: string;
  color: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type PreviewLine = {
  kind: 'line';
  id: string;
  color: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  strokeWidth: number;
};

type PreviewFrontVector = {
  dx: number;
  dy: number;
};

type PreviewScene = {
  polygon: Point[];
  shapes: Array<PreviewRect | PreviewLine>;
  viewBox: string;
};

export const LayoutVariantsGallery: React.FC = () => {
  const layoutVariants = useEditorStore((state) => state.layoutVariants);
  const selectedVariantId = useEditorStore((state) => state.selectedVariantId);
  const selectLayoutVariant = useEditorStore((state) => state.selectLayoutVariant);
  const setFloorplan = useEditorStore((state) => state.setFloorplan);
  const setDesign = useEditorStore((state) => state.setDesign);
  const setStyledResult = useEditorStore((state) => state.setStyledResult);
  const setViewMode = useEditorStore((state) => state.setViewMode);
  const setVariantPickerOpen = useEditorStore((state) => state.setVariantPickerOpen);
  const setActiveRootLayoutId = useEditorStore((state) => state.setActiveRootLayoutId);
  const setActiveLayoutCaseId = useEditorStore((state) => state.setActiveLayoutCaseId);
  const setActiveSavedLayoutId = useEditorStore((state) => state.setActiveSavedLayoutId);
  const selectSavedRender = useEditorStore((state) => state.selectSavedRender);
  const appendRunLog = useEditorStore((state) => state.appendRunLog);

  const applyVariantToEditor = (variant: LayoutVariant) => {
    if (variant.design) {
      setFloorplan(variant.design.floorplan_geometry);
      setDesign(variant.design);
    }
    setStyledResult(variant.styled_result);
    selectLayoutVariant(variant.variant_id);
    setActiveLayoutCaseId(variant.saved_layout_case_id ?? null);
    setActiveSavedLayoutId(variant.saved_layout_id ?? null);
    setActiveRootLayoutId(variant.saved_layout_id ?? null);
    selectSavedRender(null);
    setViewMode('2d');
    setVariantPickerOpen(false);
    appendRunLog(`Selected layout: ${variant.label}`);
  };

  if (!layoutVariants.length) {
    return (
      <section className="layout-variants-gallery empty">
        <div className="layout-variants-header">
          <div>
            <h2>Design Options</h2>
            <p>No layout variants were returned for this run.</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="layout-variants-gallery">
      <div className="layout-variants-header">
        <div>
          <h2>Design Options</h2>
          <p>
            Compare {layoutVariants.length} layouts. Click to inspect, double-click
            to load one into the editor.
          </p>
        </div>
        <div className="layout-variants-count">{layoutVariants.length} options</div>
      </div>
      <div className="layout-variants-grid">
        {layoutVariants.map((variant) => {
          const scene = buildPreviewScene(variant);
          const isSelected = variant.variant_id === selectedVariantId;
          const hasBlockingIssues = variantHasBlockingFunctionalIssues(variant);
          const isEligible =
            !hasBlockingIssues && (variant.gallery_eligible ?? variant.hard_valid);
          const badgeLabel =
            variant.complete === false ? 'Incomplete' : isEligible ? 'Valid' : 'Needs review';
          return (
            <button
              key={variant.variant_id}
              type="button"
              className={`layout-variant-card${isSelected ? ' selected' : ''}`}
              onClick={() => selectLayoutVariant(variant.variant_id)}
              onDoubleClick={() => applyVariantToEditor(variant)}
            >
              <div className="layout-variant-card-header">
                <div>
                  <div className="layout-variant-title">{variant.label}</div>
                  <div className="layout-variant-source">{variant.source}</div>
                </div>
                <div className="layout-variant-score">
                  <span className={isEligible ? 'ok' : 'warn'}>
                    {badgeLabel}
                  </span>
                </div>
              </div>
              <svg className="layout-variant-svg" viewBox={scene.viewBox}>
                <polygon
                  points={scene.polygon.map((point) => `${point.x},${point.y}`).join(' ')}
                  fill="#faf7f1"
                  stroke="#b8b0a2"
                  strokeWidth={60}
                />
                {scene.shapes.map((shape) =>
                  shape.kind === 'rect' ? (
                    <rect
                      key={shape.id}
                      x={shape.x}
                      y={shape.y}
                      width={shape.width}
                      height={shape.height}
                      rx={40}
                      fill={shape.color}
                      stroke="#2f2f2f"
                      strokeWidth={28}
                    />
                  ) : (
                    <line
                      key={shape.id}
                      x1={shape.x1}
                      y1={shape.y1}
                      x2={shape.x2}
                      y2={shape.y2}
                      stroke={shape.color}
                      strokeWidth={shape.strokeWidth}
                      strokeLinecap="round"
                    />
                  ),
                )}
              </svg>
              <div className="layout-variant-card-body">
                <p>{variant.reason || 'Alternative layout generated from the same room brief.'}</p>
                <div className="layout-variant-metrics">
                  <span>Score {variant.layout_score}</span>
                  <span>{variant.notes.length} notes</span>
                </div>
                {variant.notes.length > 0 ? (
                  <div className="layout-variant-notes">
                    {variant.notes.slice(0, 3).map((note) => (
                      <span key={note}>{note}</span>
                    ))}
                  </div>
                ) : null}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
};

function variantHasBlockingFunctionalIssues(variant: LayoutVariant): boolean {
  const verifySummary = getVariantVerifySummary(variant);
  const blockingCount = Number(verifySummary?.blocking_issue_count ?? 0);
  if (Number.isFinite(blockingCount) && blockingCount > 0) {
    return true;
  }
  return hasNonEmptyIssueList(verifySummary?.blocking_protected_region_issues);
}

function getVariantVerifySummary(variant: LayoutVariant): Record<string, unknown> | null {
  const direct = variant.absolute_layout?.solver_debug;
  if (direct && typeof direct === 'object') {
    const verifySummary = (direct as { verify_summary?: unknown }).verify_summary;
    if (verifySummary && typeof verifySummary === 'object') {
      return verifySummary as Record<string, unknown>;
    }
  }
  const fallback = variant.absolute_layout?.verify_summary;
  if (fallback && typeof fallback === 'object') {
    return fallback as Record<string, unknown>;
  }
  return null;
}

function hasNonEmptyIssueList(value: unknown): boolean {
  return Array.isArray(value) && value.length > 0;
}

function buildPreviewScene(variant: LayoutVariant): PreviewScene {
  const room = variant.styled_result?.room ?? {};
  const polygonRaw = Array.isArray(room?.polygon_mm)
    ? room.polygon_mm
    : Array.isArray(room?.polygon_ccw)
      ? room.polygon_ccw
      : [];
  const polygon = polygonRaw
    .map((point: any) => ({
      x: Number(point?.x ?? 0),
      y: Number(point?.y ?? 0),
    }))
    .filter((point: Point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  const windows = extractPreviewWindows(variant);
  const windowShapes = windows.map((windowRow, index) => ({
    kind: 'line' as const,
    id: `window_${index + 1}`,
    color: '#8eaad6',
    x1: windowRow.start.x,
    y1: windowRow.start.y,
    x2: windowRow.end.x,
    y2: windowRow.end.y,
      strokeWidth: 54,
  }));
  const rawObjects = buildPreviewObjectRows(variant);
  const objectShapes = rawObjects.flatMap((objectRow: any, index: number) =>
    buildObjectPreviewShapes(objectRow, index, windows),
  );

  const transformed = flipPreviewSceneY({
    polygon,
    shapes: [...windowShapes, ...objectShapes],
  });
  const bounds = computeSceneBounds(transformed.polygon, transformed.shapes);
  return {
    polygon: transformed.polygon,
    shapes: transformed.shapes,
    viewBox: `${bounds.minX} ${bounds.minY} ${bounds.width} ${bounds.height}`,
  };
}

function buildPreviewObjectRows(variant: LayoutVariant): any[] {
  const absoluteObjects: any[] = Array.isArray(variant.absolute_layout?.objects)
    ? variant.absolute_layout.objects
    : [];
  const styledObjects: any[] = Array.isArray(variant.styled_result?.objects)
    ? variant.styled_result.objects
    : [];
  if (absoluteObjects.length === 0) {
    return styledObjects;
  }

  const styledByKey = new Map<string, any>();
  for (const objectRow of styledObjects) {
    const key = getPreviewObjectKey(objectRow);
    if (key) {
      styledByKey.set(key, objectRow);
    }
  }

  const merged = absoluteObjects.map((objectRow) => {
    const key = getPreviewObjectKey(objectRow);
    const styledRow = key ? styledByKey.get(key) : null;
    if (!styledRow) {
      return objectRow;
    }
    return {
      ...styledRow,
      ...objectRow,
      instance_id:
        objectRow?.instance_id ?? styledRow?.instance_id ?? objectRow?.object_id ?? null,
      object_type:
        objectRow?.object_type ??
        styledRow?.object_type ??
        styledRow?.category ??
        objectRow?.category ??
        null,
      color_hex: objectRow?.color_hex ?? styledRow?.color_hex ?? null,
      material: objectRow?.material ?? styledRow?.material ?? null,
      place_on: objectRow?.place_on ?? styledRow?.place_on ?? null,
      rotation_ccw:
        objectRow?.rotation_ccw ?? objectRow?.rotation_deg ?? styledRow?.rotation_ccw ?? 0,
      bbox: objectRow?.bbox ?? styledRow?.bbox ?? null,
    };
  });

  const seenKeys = new Set(
    merged
      .map((objectRow) => getPreviewObjectKey(objectRow))
      .filter((value): value is string => Boolean(value)),
  );
  for (const styledRow of styledObjects) {
    const key = getPreviewObjectKey(styledRow);
    if (key && !seenKeys.has(key)) {
      merged.push(styledRow);
    }
  }
  return merged;
}

function getPreviewObjectKey(objectRow: any): string | null {
  const raw =
    objectRow?.instance_id ??
    objectRow?.object_id ??
    objectRow?.id ??
    null;
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim();
  return normalized.length > 0 ? normalized : null;
}

function computeSceneBounds(
  polygon: Point[],
  shapes: Array<PreviewRect | PreviewLine>,
): { minX: number; minY: number; width: number; height: number } {
  const extents = computeSceneExtents(polygon, shapes);
  if (!extents) {
    return { minX: 0, minY: 0, width: 1000, height: 1000 };
  }
  const padding = 120;
  return {
    minX: extents.minX - padding,
    minY: extents.minY - padding,
    width: Math.max(1, extents.maxX - extents.minX + padding * 2),
    height: Math.max(1, extents.maxY - extents.minY + padding * 2),
  };
}

function computeSceneExtents(
  polygon: Point[],
  shapes: Array<PreviewRect | PreviewLine>,
): { minX: number; minY: number; maxX: number; maxY: number } | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const point of polygon) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  for (const shape of shapes) {
    if (shape.kind === 'rect') {
      minX = Math.min(minX, shape.x);
      minY = Math.min(minY, shape.y);
      maxX = Math.max(maxX, shape.x + shape.width);
      maxY = Math.max(maxY, shape.y + shape.height);
      continue;
    }
    minX = Math.min(minX, shape.x1, shape.x2);
    minY = Math.min(minY, shape.y1, shape.y2);
    maxX = Math.max(maxX, shape.x1, shape.x2);
    maxY = Math.max(maxY, shape.y1, shape.y2);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null;
  }

  return { minX, minY, maxX, maxY };
}

function buildObjectPreviewShapes(
  objectRow: any,
  index: number,
  windows: PreviewWindow[],
): Array<PreviewRect | PreviewLine> {
  const bbox = objectRow?.bbox ?? {};
  const minX = Number(bbox.min_x ?? 0);
  const minY = Number(bbox.min_y ?? 0);
  const maxX = Number(bbox.max_x ?? minX);
  const maxY = Number(bbox.max_y ?? minY);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const id = String(objectRow?.instance_id ?? `obj_${index}`);
  const color = String(objectRow?.color_hex ?? '#cfcac2');
  const category = String(objectRow?.object_type ?? objectRow?.category ?? '').trim();
  const frontWorld = normalizePreviewFrontWorld(objectRow?.front_world);
  if (isCurtainLike(category)) {
    const matchedWindow = findNearestPreviewWindow(
      {
        x: minX + width / 2,
        y: minY + height / 2,
      },
      windows,
    );
    if (matchedWindow) {
      return [
        {
          kind: 'line',
          id,
          color,
          x1: matchedWindow.start.x,
          y1: matchedWindow.start.y,
          x2: matchedWindow.end.x,
          y2: matchedWindow.end.y,
          strokeWidth: 92,
        },
      ];
    }
  }
  if (isMirrorLike(category) && String(objectRow?.place_on?.method ?? '') === 'hang_on') {
    if (width >= height) {
      const midY = minY + height / 2;
      return [
        {
          kind: 'line',
          id,
          color,
          x1: minX,
          y1: midY,
          x2: maxX,
          y2: midY,
          strokeWidth: 42,
        },
      ];
    }
    const midX = minX + width / 2;
    return [
      {
        kind: 'line',
        id,
        color,
        x1: midX,
        y1: minY,
        x2: midX,
        y2: maxY,
        strokeWidth: 42,
      },
    ];
  }
  const shapes: Array<PreviewRect | PreviewLine> = [
    {
      kind: 'rect',
      id,
      color,
      x: minX,
      y: minY,
      width,
      height,
    },
  ];
  const frontMarker = buildFrontMarkerShape({
    id,
    minX,
    minY,
    maxX,
    maxY,
    frontWorld,
  });
  if (frontMarker) {
    shapes.push(frontMarker);
  }
  return shapes;
}

function buildFrontMarkerShape(params: {
  id: string;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  frontWorld: PreviewFrontVector | null;
}): PreviewLine | null {
  const { id, minX, minY, maxX, maxY, frontWorld } = params;
  const frontSide = dominantPreviewFrontSide(frontWorld);
  if (!frontSide) {
    return null;
  }
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const insetX = Math.max(50, Math.min(width * 0.16, 120));
  const insetY = Math.max(50, Math.min(height * 0.16, 120));
  if (frontSide === 'top') {
    const y = minY + insetY;
    return {
      kind: 'line',
      id: `${id}__front`,
      color: 'rgba(255,255,255,0.95)',
      x1: minX + insetX,
      y1: y,
      x2: maxX - insetX,
      y2: y,
      strokeWidth: 52,
    };
  }
  if (frontSide === 'right') {
    const x = maxX - insetX;
    return {
      kind: 'line',
      id: `${id}__front`,
      color: 'rgba(255,255,255,0.95)',
      x1: x,
      y1: minY + insetY,
      x2: x,
      y2: maxY - insetY,
      strokeWidth: 52,
    };
  }
  if (frontSide === 'bottom') {
    const y = maxY - insetY;
    return {
      kind: 'line',
      id: `${id}__front`,
      color: 'rgba(255,255,255,0.95)',
      x1: minX + insetX,
      y1: y,
      x2: maxX - insetX,
      y2: y,
      strokeWidth: 52,
    };
  }
  if (frontSide === 'left') {
    const x = minX + insetX;
    return {
      kind: 'line',
      id: `${id}__front`,
      color: 'rgba(255,255,255,0.95)',
      x1: x,
      y1: minY + insetY,
      x2: x,
      y2: maxY - insetY,
      strokeWidth: 52,
    };
  }
  return null;
}

function normalizePreviewFrontWorld(value: unknown): PreviewFrontVector | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const raw = value as { dx?: unknown; dy?: unknown };
  const dx = Number(raw.dx);
  const dy = Number(raw.dy);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return null;
  }
  if (Math.hypot(dx, dy) <= 1e-6) {
    return null;
  }
  return { dx, dy };
}

function dominantPreviewFrontSide(
  frontWorld: PreviewFrontVector | null,
): 'top' | 'right' | 'bottom' | 'left' | null {
  if (!frontWorld) {
    return null;
  }
  if (Math.abs(frontWorld.dx) >= Math.abs(frontWorld.dy)) {
    return frontWorld.dx >= 0 ? 'right' : 'left';
  }
  return frontWorld.dy >= 0 ? 'bottom' : 'top';
}

type PreviewWindow = {
  start: Point;
  end: Point;
};

function extractPreviewWindows(variant: LayoutVariant): PreviewWindow[] {
  const floorplanWindows = variant.design?.floorplan_geometry?.openings?.windows;
  if (Array.isArray(floorplanWindows) && floorplanWindows.length > 0) {
    return floorplanWindows
      .map((windowRow) => ({
        start: {
          x: Number(windowRow.start_mm?.x ?? 0),
          y: Number(windowRow.start_mm?.y ?? 0),
        },
        end: {
          x: Number(windowRow.end_mm?.x ?? 0),
          y: Number(windowRow.end_mm?.y ?? 0),
        },
      }))
      .filter((windowRow) => isFinite(windowRow.start.x) && isFinite(windowRow.end.x));
  }
  const roomWindows = variant.styled_result?.room?.openings?.windows;
  if (!Array.isArray(roomWindows)) {
    return [];
  }
  return roomWindows
    .map((windowRow: any) => {
      const segment = Array.isArray(windowRow?.segment_mm) ? windowRow.segment_mm : [];
      return {
        start: {
          x: Number(segment[0]?.x ?? 0),
          y: Number(segment[0]?.y ?? 0),
        },
        end: {
          x: Number(segment[1]?.x ?? 0),
          y: Number(segment[1]?.y ?? 0),
        },
      };
    })
    .filter((windowRow) => isFinite(windowRow.start.x) && isFinite(windowRow.end.x));
}

function findNearestPreviewWindow(
  target: Point,
  windows: PreviewWindow[],
): PreviewWindow | null {
  let best: { row: PreviewWindow; distance: number } | null = null;
  for (const windowRow of windows) {
    const center = {
      x: (windowRow.start.x + windowRow.end.x) / 2,
      y: (windowRow.start.y + windowRow.end.y) / 2,
    };
    const distance = Math.hypot(target.x - center.x, target.y - center.y);
    if (!best || distance < best.distance) {
      best = { row: windowRow, distance };
    }
  }
  return best?.row ?? null;
}

function flipPreviewSceneY(input: {
  polygon: Point[];
  shapes: Array<PreviewRect | PreviewLine>;
}): PreviewScene {
  const worldBounds = computeSceneExtents(input.polygon, input.shapes);
  const maxY = worldBounds ? worldBounds.maxY : 0;
  const polygon = input.polygon.map((point) => ({
    x: point.x,
    y: maxY - point.y,
  }));
  const shapes = input.shapes.map((shape) => {
    if (shape.kind === 'rect') {
      return {
        ...shape,
        y: maxY - (shape.y + shape.height),
      };
    }
    return {
      ...shape,
      y1: maxY - shape.y1,
      y2: maxY - shape.y2,
    };
  });
  return {
    polygon,
    shapes,
    viewBox: '0 0 1 1',
  };
}

function isCurtainLike(category: string): boolean {
  const lowered = category.toLowerCase();
  return (
    lowered.includes('curtain') ||
    lowered.includes('blind') ||
    lowered.includes('drape')
  );
}

function isMirrorLike(category: string): boolean {
  return category.toLowerCase().includes('mirror');
}
