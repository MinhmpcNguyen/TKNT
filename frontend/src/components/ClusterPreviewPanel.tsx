import React from 'react';
import { useEditorStore } from '../store';

const colors = ['#dbeafe', '#fee2e2', '#dcfce7', '#fef3c7', '#e9d5ff', '#cffafe'];

function bboxFromRects(rects: any[]) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  rects.forEach((r) => {
    const x = Number(r.x ?? 0);
    const y = Number(r.y ?? 0);
    const w = Number(r.w ?? 0);
    const h = Number(r.h ?? 0);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + w);
    maxY = Math.max(maxY, y + h);
  });
  if (!isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}

export const ClusterPreviewPanel: React.FC = () => {
  const clusterPreviews = useEditorStore((s) => s.clusterPreviews);
  const entries = Object.entries(clusterPreviews);

  return (
    <aside className="panel cluster-preview-panel">
      <h3>Cluster Preview</h3>
      {entries.length === 0 ? (
        <p className="muted">No cluster previews yet.</p>
      ) : (
        <div className="cluster-preview-list">
          {entries.map(([clusterId, data], idx) => {
            const cluster = (data as any)?.payload ?? data;
            const rects = cluster?.cluster_footprint?.rects ?? [];
            const bbox = bboxFromRects(rects);
            const w = Math.max(1, bbox.maxX - bbox.minX);
            const h = Math.max(1, bbox.maxY - bbox.minY);
            const color = colors[idx % colors.length];
            return (
              <div key={clusterId} className="cluster-preview-card">
                <div className="cluster-title">{clusterId}</div>
                <svg viewBox={`${bbox.minX} ${bbox.minY} ${w} ${h}`} className="cluster-svg">
                  {rects.map((r: any, i: number) => (
                    <rect
                      key={`${clusterId}-${i}`}
                      x={r.x}
                      y={r.y}
                      width={r.w}
                      height={r.h}
                      fill={color}
                      stroke="#333"
                      strokeWidth="20"
                    />
                  ))}
                </svg>
              </div>
            );
          })}
        </div>
      )}
    </aside>
  );
};
