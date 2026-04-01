import React from 'react';
import { LayoutVariant, Point } from '../types';
import { useEditorStore } from '../store';

type PreviewRect = {
  id: string;
  color: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type PreviewScene = {
  polygon: Point[];
  rects: PreviewRect[];
  viewBox: string;
};

export const LayoutVariantsGallery: React.FC = () => {
  const layoutVariants = useEditorStore((state) => state.layoutVariants);
  const selectedVariantId = useEditorStore((state) => state.selectedVariantId);
  const selectLayoutVariant = useEditorStore((state) => state.selectLayoutVariant);

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
            Compare {layoutVariants.length} layouts, click one card to inspect it,
            then choose it from the toolbar.
          </p>
        </div>
        <div className="layout-variants-count">{layoutVariants.length} options</div>
      </div>
      <div className="layout-variants-grid">
        {layoutVariants.map((variant) => {
          const scene = buildPreviewScene(variant);
          const isSelected = variant.variant_id === selectedVariantId;
          return (
            <button
              key={variant.variant_id}
              type="button"
              className={`layout-variant-card${isSelected ? ' selected' : ''}`}
              onClick={() => selectLayoutVariant(variant.variant_id)}
            >
              <div className="layout-variant-card-header">
                <div>
                  <div className="layout-variant-title">{variant.label}</div>
                  <div className="layout-variant-source">{variant.source}</div>
                </div>
                <div className="layout-variant-score">
                  <span className={variant.hard_valid ? 'ok' : 'warn'}>
                    {variant.hard_valid ? 'Valid' : 'Needs review'}
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
                {scene.rects.map((rect) => (
                  <rect
                    key={rect.id}
                    x={rect.x}
                    y={rect.y}
                    width={rect.width}
                    height={rect.height}
                    rx={40}
                    fill={rect.color}
                    stroke="#2f2f2f"
                    strokeWidth={28}
                  />
                ))}
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

  const rects = Array.isArray(variant.styled_result?.objects)
    ? variant.styled_result.objects
        .map((objectRow: any, index: number) => {
          const bbox = objectRow?.bbox ?? {};
          const minX = Number(bbox.min_x ?? 0);
          const minY = Number(bbox.min_y ?? 0);
          const maxX = Number(bbox.max_x ?? minX);
          const maxY = Number(bbox.max_y ?? minY);
          return {
            id: String(objectRow?.instance_id ?? `obj_${index}`),
            color: String(objectRow?.color_hex ?? '#cfcac2'),
            x: minX,
            y: minY,
            width: Math.max(1, maxX - minX),
            height: Math.max(1, maxY - minY),
          } satisfies PreviewRect;
        })
        .filter((rect: PreviewRect) => rect.width > 0 && rect.height > 0)
    : [];

  const bounds = computeSceneBounds(polygon, rects);
  return {
    polygon,
    rects,
    viewBox: `${bounds.minX} ${bounds.minY} ${bounds.width} ${bounds.height}`,
  };
}

function computeSceneBounds(
  polygon: Point[],
  rects: PreviewRect[],
): { minX: number; minY: number; width: number; height: number } {
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
  for (const rect of rects) {
    minX = Math.min(minX, rect.x);
    minY = Math.min(minY, rect.y);
    maxX = Math.max(maxX, rect.x + rect.width);
    maxY = Math.max(maxY, rect.y + rect.height);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return { minX: 0, minY: 0, width: 1000, height: 1000 };
  }

  const padding = 120;
  return {
    minX: minX - padding,
    minY: minY - padding,
    width: Math.max(1, maxX - minX + padding * 2),
    height: Math.max(1, maxY - minY + padding * 2),
  };
}
