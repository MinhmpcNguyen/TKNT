import React from 'react';
import { useEditorStore } from '../store';

export const UserInputPanel: React.FC = () => {
  const {
    userInput,
    updateUserInput,
    selectedCornerIndex,
    selectedWallId,
    floorplan,
    updateWall,
    removeWall,
  } = useEditorStore();

  const wall = floorplan?.walls.find((w) => w.wall_id === selectedWallId) ?? null;

  return (
    <aside className="panel user-input-panel">
      <h3>User Input</h3>
      <div className="inspector-row">
        <label>Description</label>
        <textarea
          value={userInput.description}
          onChange={(evt) => updateUserInput({ description: evt.target.value })}
        />
      </div>
      <div className="inspector-row">
        <label>Room Type</label>
        <select
          value={userInput.room_type}
          onChange={(evt) => updateUserInput({ room_type: evt.target.value })}
        >
          <option value="office">Office</option>
          <option value="bedroom">Bedroom</option>
          <option value="living_room">Living Room</option>
          <option value="studio">Studio</option>
          <option value="kitchen">Kitchen</option>
        </select>
      </div>
      <div className="inspector-row">
        <label>Style</label>
        <input
          value={userInput.style}
          onChange={(evt) => updateUserInput({ style: evt.target.value })}
        />
      </div>
      <div className="inspector-row">
        <label>Special Notes</label>
        <textarea
          value={userInput.special_description}
          onChange={(evt) => updateUserInput({ special_description: evt.target.value })}
        />
      </div>
      <div className="inspector-row">
        <label>Feng Shui</label>
        <textarea
          value={userInput.feng_shui}
          onChange={(evt) => updateUserInput({ feng_shui: evt.target.value })}
        />
      </div>
      <div className="inspector-row">
        <label>Floor Area (m²)</label>
        <input
          type="number"
          value={userInput.floor_area_m2}
          onChange={(evt) => updateUserInput({ floor_area_m2: Number(evt.target.value) })}
        />
      </div>
      <div className="inspector-row">
        <label>Height (mm)</label>
        <input
          type="number"
          value={userInput.height}
          onChange={(evt) => updateUserInput({ height: Number(evt.target.value) })}
        />
      </div>
      <div className="inspector-row">
        <label>Location Floor</label>
        <input
          type="number"
          value={userInput.location_floor}
          onChange={(evt) => updateUserInput({ location_floor: Number(evt.target.value) })}
        />
      </div>
      <div className="inspector-row">
        <label>Bathrooms</label>
        <input
          type="number"
          value={userInput.bathrooms}
          onChange={(evt) => updateUserInput({ bathrooms: Number(evt.target.value) })}
        />
      </div>
      <div className="inspector-row">
        <label>Windows</label>
        <input
          type="number"
          value={userInput.windows}
          onChange={(evt) => updateUserInput({ windows: Number(evt.target.value) })}
        />
      </div>
      <div className="inspector-row">
        <label>Window Direction</label>
        <input
          value={userInput.window_direction}
          onChange={(evt) => updateUserInput({ window_direction: evt.target.value })}
        />
      </div>
      <div className="inspector-row">
        <label>Room Points</label>
        <div className="point-list">
          {userInput.shape_points.length === 0 && <div className="muted">No points yet.</div>}
          {userInput.shape_points.map((point, idx) => (
            <div
              key={`${point.x}-${point.y}-${idx}`}
              className={`point-item ${selectedCornerIndex === idx ? 'active' : ''}`}
            >
              <span>#{idx + 1}</span>
              <span>x={Math.round(point.x)}</span>
              <span>y={Math.round(point.y)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="inspector-row">
        <label>Selected Wall</label>
        {wall ? (
          <div className="wall-panel">
            <div className="muted">{wall.wall_id}</div>
            <div className="wall-row">
              <label>Thickness (mm)</label>
              <input
                type="number"
                value={wall.thickness_mm}
                onChange={(evt) =>
                  updateWall(wall.wall_id, { thickness_mm: Number(evt.target.value) })
                }
              />
            </div>
            <div className="wall-actions">
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(wall.locked)}
                  onChange={(evt) =>
                    updateWall(wall.wall_id, { locked: evt.target.checked })
                  }
                />
                Locked
              </label>
              <button
                className="danger"
                disabled={Boolean(wall.locked)}
                onClick={() => removeWall(wall.wall_id)}
              >
                Delete Wall
              </button>
            </div>
          </div>
        ) : (
          <div className="muted">None</div>
        )}
      </div>
    </aside>
  );
};
