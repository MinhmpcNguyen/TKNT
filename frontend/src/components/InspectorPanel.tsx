import React from 'react';
import { useEditorStore } from '../store';

export const InspectorPanel: React.FC = () => {
  const {
    design,
    floorplan,
    selectedPlacementId,
    selectedOpeningId,
    selectedOpeningType,
    updatePlacement,
    removePlacement,
    removeOpening,
  } = useEditorStore();
  const placement = design?.placements.find((p) => p.placement_id === selectedPlacementId);
  const opening =
    selectedOpeningType === 'door'
      ? floorplan?.openings.doors.find((d) => d.door_id === selectedOpeningId)
      : selectedOpeningType === 'window'
        ? floorplan?.openings.windows.find((w) => w.window_id === selectedOpeningId)
        : null;

  if (!placement && !opening) {
    return (
      <aside className="panel inspector-panel">
        <h3>Inspector</h3>
        <p>Select an item</p>
      </aside>
    );
  }

  if (opening && selectedOpeningType) {
    const length = Math.hypot(
      opening.end_mm.x - opening.start_mm.x,
      opening.end_mm.y - opening.start_mm.y,
    );
    return (
      <aside className="panel inspector-panel">
        <h3>Inspector</h3>
        <div className="inspector-row">
          <label>Type</label>
          <div className="inspector-readonly">{selectedOpeningType}</div>
        </div>
        <div className="inspector-row">
          <label>Length (mm)</label>
          <div className="inspector-readonly">{Math.round(length)}</div>
        </div>
        <button
          className="danger"
          onClick={() =>
            removeOpening(
              selectedOpeningType,
              selectedOpeningType === 'door' ? (opening as any).door_id : (opening as any).window_id,
            )
          }
        >
          Remove
        </button>
      </aside>
    );
  }

  return (
    <aside className="panel inspector-panel">
      <h3>Inspector</h3>
      <div className="inspector-row">
        <label>Category</label>
        <div className="inspector-readonly">{placement?.category}</div>
      </div>
      <div className="inspector-row">
        <label>X (mm)</label>
        <input
          type="number"
          value={Math.round(placement.position_mm.x)}
          onChange={(evt) =>
            updatePlacement({
              ...placement,
              position_mm: { ...placement.position_mm, x: Number(evt.target.value) },
            })
          }
        />
      </div>
      <div className="inspector-row">
        <label>Y (mm)</label>
        <input
          type="number"
          value={Math.round(placement.position_mm.y)}
          onChange={(evt) =>
            updatePlacement({
              ...placement,
              position_mm: { ...placement.position_mm, y: Number(evt.target.value) },
            })
          }
        />
      </div>
      <div className="inspector-row">
        <label>Rotation</label>
        <input
          type="number"
          value={Math.round(placement.rotation_deg)}
          onChange={(evt) =>
            updatePlacement({
              ...placement,
              rotation_deg: Number(evt.target.value),
            })
          }
        />
      </div>
      <div className="inspector-row">
        <label>Color</label>
        <input
          type="color"
          value={placement.color_hex}
          onChange={(evt) =>
            updatePlacement({
              ...placement,
              color_hex: evt.target.value,
            })
          }
        />
      </div>
      <button className="danger" onClick={() => removePlacement(placement.placement_id)}>
        Remove
      </button>
    </aside>
  );
};
