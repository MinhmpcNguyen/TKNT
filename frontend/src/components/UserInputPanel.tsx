import React, { useEffect, useRef, useState } from 'react';
import {
  ROOM_TYPE_OPTIONS,
  STYLE_OPTIONS,
  normalizeRoomType,
  normalizeStyle,
} from '../options';
import { useEditorStore } from '../store';

export const UserInputPanel: React.FC = () => {
  const {
    userInput,
    updateUserInput,
    design,
    styledResult,
    selectedCornerIndex,
    selectedWallId,
    floorplan,
    updateWall,
    removeWall,
  } = useEditorStore();

  const wall = floorplan?.walls.find((w) => w.wall_id === selectedWallId) ?? null;
  const hasLayout = Boolean(styledResult || (design?.placements.length ?? 0) > 0);
  const roomTypeValue = normalizeRoomType(userInput.room_type);
  const styleValue = normalizeStyle(userInput.style);
  const roomTypeLabel =
    ROOM_TYPE_OPTIONS.find((option) => option.value === roomTypeValue)?.label ?? roomTypeValue;
  const styleLabel =
    STYLE_OPTIONS.find((option) => option.value === styleValue)?.label ?? styleValue;
  const [isOpen, setIsOpen] = useState(!hasLayout);
  const hadLayoutRef = useRef(hasLayout);

  useEffect(() => {
    if (roomTypeValue !== userInput.room_type || styleValue !== userInput.style) {
      updateUserInput({ room_type: roomTypeValue, style: styleValue });
    }
  }, [roomTypeValue, styleValue, updateUserInput, userInput.room_type, userInput.style]);

  useEffect(() => {
    if (!hadLayoutRef.current && hasLayout) {
      setIsOpen(false);
    }
    if (hadLayoutRef.current && !hasLayout) {
      setIsOpen(true);
    }
    hadLayoutRef.current = hasLayout;
  }, [hasLayout]);

  return (
    <aside className={`panel user-input-panel collapsible-panel${isOpen ? ' open' : ''}`}>
      <button
        className="panel-collapse-trigger"
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
      >
        <span>
          <span className="panel-collapse-title">User Input</span>
          <span className="panel-collapse-subtitle">
            {userInput.case_name.trim() || `${roomTypeLabel} · ${styleLabel}`}
          </span>
        </span>
        <span className="panel-collapse-caret">{isOpen ? 'Hide' : 'Edit'}</span>
      </button>
      {isOpen ? (
        <div className="panel-collapse-body">
          <div className="inspector-row">
            <label>Case Name</label>
            <input
              value={userInput.case_name}
              onChange={(evt) => updateUserInput({ case_name: evt.target.value })}
              placeholder="Living room movie night"
            />
          </div>
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
              value={roomTypeValue}
              onChange={(evt) => updateUserInput({ room_type: evt.target.value })}
            >
              {ROOM_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="inspector-row">
            <label>Style</label>
            <select
              value={styleValue}
              onChange={(evt) => updateUserInput({ style: evt.target.value })}
            >
              {STYLE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
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
            <label>Windows</label>
            <input
              type="number"
              value={userInput.windows}
              onChange={(evt) => updateUserInput({ windows: Number(evt.target.value) })}
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
        </div>
      ) : null}
    </aside>
  );
};
