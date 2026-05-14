import React from 'react';
import { useEditorStore } from '../store';

export const InspectorPanel: React.FC = () => {
  const {
    design,
    floorplan,
    selectedPlacementId,
    selectedWallId,
    selectedOpeningId,
    selectedOpeningType,
    updateWall,
    updateOpening,
    updatePlacement,
    removeWall,
    removePlacement,
    removeOpening,
  } = useEditorStore();
  const wall = floorplan?.walls.find((item) => item.wall_id === selectedWallId) ?? null;
  const placement = design?.placements.find((p) => p.placement_id === selectedPlacementId);
  const isManualPlacement = placement?.source === 'manual';
  const isLockedPlacement = placement?.constraint_mode === 'locked';
  const manualConstraintMode = isLockedPlacement ? 'locked' : 'pinned';
  const opening =
    selectedOpeningType === 'door'
      ? floorplan?.openings.doors.find((d) => d.door_id === selectedOpeningId)
      : selectedOpeningType === 'window'
        ? floorplan?.openings.windows.find((w) => w.window_id === selectedOpeningId)
        : null;

  if (!placement && !opening && !wall) {
    return (
      <aside className="panel inspector-panel">
        <h3>Inspector</h3>
        <p>Select an item</p>
      </aside>
    );
  }

  if (opening && selectedOpeningType && selectedOpeningId) {
    const length = Math.hypot(
      opening.end_mm.x - opening.start_mm.x,
      opening.end_mm.y - opening.start_mm.y,
    );
    const openingId = selectedOpeningId;
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
        <div className="inspector-row">
          <label>Color</label>
          <input
            type="color"
            value={opening.color_hex ?? (selectedOpeningType === 'door' ? '#8c7a63' : '#c0c0c0')}
            onChange={(evt) =>
              updateOpening(selectedOpeningType, openingId, {
                color_hex: evt.target.value,
              })
            }
          />
        </div>
        <button
          className="danger"
          onClick={() =>
            removeOpening(
              selectedOpeningType,
              openingId,
            )
          }
        >
          Remove
        </button>
      </aside>
    );
  }

  if (wall) {
    const lengthMm = Math.hypot(
      wall.p2_mm.x - wall.p1_mm.x,
      wall.p2_mm.y - wall.p1_mm.y,
    );
    const opacityPercent = Math.round((wall.opacity ?? 1) * 100);
    return (
      <aside className="panel inspector-panel">
        <h3>Inspector</h3>
        <div className="inspector-row">
          <label>Type</label>
          <div className="inspector-readonly">wall</div>
        </div>
        <div className="inspector-row">
          <label>Length (mm)</label>
          <div className="inspector-readonly">{Math.round(lengthMm)}</div>
        </div>
        <div className="inspector-row">
          <label>Color</label>
          <input
            type="color"
            value={wall.color_hex ?? '#555555'}
            onChange={(evt) =>
              updateWall(wall.wall_id, {
                color_hex: evt.target.value,
              })
            }
          />
        </div>
        <div className="inspector-row">
          <label>Opacity</label>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={opacityPercent}
            onChange={(evt) =>
              updateWall(wall.wall_id, {
                opacity: Number(evt.target.value) / 100,
              })
            }
          />
        </div>
        <div className="inspector-row">
          <label>Opacity (%)</label>
          <input
            type="number"
            min={0}
            max={100}
            step={5}
            value={opacityPercent}
            onChange={(evt) =>
              updateWall(wall.wall_id, {
                opacity: Math.max(0, Math.min(1, Number(evt.target.value) / 100)),
              })
            }
          />
        </div>
        <div className="inspector-row">
          <label>Tip</label>
          <div className="inspector-readonly">
            Lower opacity to reveal the room interior for screenshots.
          </div>
        </div>
        <button
          className="danger"
          disabled={!!wall.locked}
          onClick={() => removeWall(wall.wall_id)}
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
        <label>Source</label>
        <div className="inspector-readonly">{placement?.source ?? 'generated'}</div>
      </div>
      {isManualPlacement ? (
        <>
          <div className="inspector-row">
            <label>Constraint</label>
            <select
              value={manualConstraintMode}
              onChange={(evt) =>
                updatePlacement({
                  ...placement,
                  constraint_mode: evt.target.value === 'locked' ? 'locked' : 'pinned',
                })
              }
            >
              <option value="pinned">Pinned</option>
              <option value="locked">Locked</option>
            </select>
          </div>
          <div className="inspector-row">
            <label>Mode</label>
            <div className="inspector-readonly">
              {manualConstraintMode === 'locked'
                ? 'Locked stays fixed during editing.'
                : 'Pinned is preserved for generation but still editable.'}
            </div>
          </div>
        </>
      ) : null}
      <div className="inspector-row">
        <label>X (mm)</label>
        <input
          type="number"
          value={Math.round(placement.position_mm.x)}
          disabled={isLockedPlacement}
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
          disabled={isLockedPlacement}
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
          disabled={isLockedPlacement}
          onChange={(evt) =>
            updatePlacement({
              ...placement,
              rotation_deg: Number(evt.target.value),
            })
          }
        />
        <div className="inspector-button-row">
          <button
            disabled={isLockedPlacement}
            onClick={() =>
              updatePlacement({
                ...placement,
                rotation_deg: placement.rotation_deg - 90,
              })
            }
            type="button"
          >
            Rotate -90
          </button>
          <button
            disabled={isLockedPlacement}
            onClick={() =>
              updatePlacement({
                ...placement,
                rotation_deg: placement.rotation_deg + 90,
              })
            }
            type="button"
          >
            Rotate +90
          </button>
        </div>
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
