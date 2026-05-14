import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Layer, Line, Rect, Stage, Group, Circle, Text } from 'react-konva';
import { useEditorStore } from '../store';
import { clamp, polygonBounds, toCanvas, fromCanvas } from '../utils/geometry';
import { DirectionVector, Door, InventoryItem, Placement, Point, Window } from '../types';
import { saveUiSnapshot } from '../api';

const GRID_SIZE = 500;

function frontMarkerPoints(
  width: number,
  depth: number,
  frontSide: 'top' | 'bottom' | 'left' | 'right',
): number[] {
  const lineHalf =
    frontSide === 'top' || frontSide === 'bottom'
      ? Math.max(12, Math.min(width * 0.24, 64))
      : Math.max(12, Math.min(depth * 0.24, 64));
  const inset =
    frontSide === 'top' || frontSide === 'bottom'
      ? Math.max(8, Math.min(depth * 0.12, 18))
      : Math.max(8, Math.min(width * 0.12, 18));

  switch (frontSide) {
    case 'bottom':
      return [-lineHalf, depth / 2 - inset, lineHalf, depth / 2 - inset];
    case 'left':
      return [-width / 2 + inset, -lineHalf, -width / 2 + inset, lineHalf];
    case 'right':
      return [width / 2 - inset, -lineHalf, width / 2 - inset, lineHalf];
    case 'top':
    default:
      return [-lineHalf, -depth / 2 + inset, lineHalf, -depth / 2 + inset];
  }
}

function frontWorldToCanvasLocalSide(
  frontWorld: DirectionVector | null | undefined,
  rotationDeg: number,
): 'top' | 'bottom' | 'left' | 'right' | null {
  if (!frontWorld) {
    return null;
  }
  const dx = Number(frontWorld.dx);
  const dy = Number(frontWorld.dy);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return null;
  }
  const radians = (((rotationDeg % 360) + 360) % 360 * Math.PI) / 180;
  const targetX = dx;
  const targetY = -dy;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const localX = cos * targetX + sin * targetY;
  const localY = -sin * targetX + cos * targetY;
  if (Math.abs(localX) >= Math.abs(localY)) {
    return localX >= 0 ? 'right' : 'left';
  }
  return localY >= 0 ? 'bottom' : 'top';
}

export const Canvas2D: React.FC = () => {
  const stageRef = useRef<any>(null);
  const dropZoneRef = useRef<HTMLDivElement | null>(null);
  const {
    floorplan,
    design,
    inventory,
    selectedPlacementId,
    selectedCornerIndex,
    selectPlacement,
    selectCorner,
    selectedWallId,
    selectWall,
    selectOpening,
    updatePlacement,
    removePlacement,
    updateWall,
    violations,
    setDesign,
    setFloorplan,
    roomEditMode,
    drawWallMode,
    lastDesignMeta,
    snapshotEnabled,
    tenantId,
  } = useEditorStore();
  const [scale, setScale] = useState(0.12);
  const [offset, setOffset] = useState<Point>({ x: 80, y: 80 });
  const [drawingLine, setDrawingLine] = useState<{ startIndex: number; end: Point } | null>(null);
  const [lastSnapshotKey, setLastSnapshotKey] = useState<string | null>(null);

  const bounds = useMemo(() => {
    if (!floorplan) {
      return { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
    }
    return polygonBounds(floorplan.room.polygon_mm);
  }, [floorplan]);

  const toCanvasPoint = (point: Point) => toCanvas(point, scale, offset, bounds);
  const fromCanvasPoint = (point: Point) => fromCanvas(point, scale, offset, bounds);

  const width = 900;
  const height = 650;

  useEffect(() => {
    if (!snapshotEnabled || !design || !lastDesignMeta) return;
    const iteration = lastDesignMeta.iteration ?? 0;
    const attempt = lastDesignMeta.attempt ?? 0;
    const key = `${iteration}:${attempt}:${lastDesignMeta.receivedAt}`;
    if (lastSnapshotKey === key) return;
    const stage = stageRef.current;
    if (!stage) return;
    const image = stage.toDataURL({ pixelRatio: 2 });
    saveUiSnapshot({
      tenant_id: tenantId,
      iteration,
      attempt,
      image_base64: image,
    }).catch(() => {
      // ignore snapshot errors
    });
    setLastSnapshotKey(key);
  }, [design, lastDesignMeta, snapshotEnabled, lastSnapshotKey]);

  const handleWheel = (evt: any) => {
    evt.evt.preventDefault();
    const delta = evt.evt.deltaY > 0 ? 0.9 : 1.1;
    setScale((prev) => clamp(prev * delta, 0.05, 0.4));
  };

  const handleDrop = (evt: React.DragEvent<HTMLDivElement>) => {
    evt.preventDefault();
    const asset = resolveDroppedAsset(evt.dataTransfer, inventory);
    if (!asset) return;
    const boundsRect = dropZoneRef.current?.getBoundingClientRect();
    if (!boundsRect) return;
    const pos = fromCanvasPoint({
      x: evt.clientX - boundsRect.left,
      y: evt.clientY - boundsRect.top,
    });

    if (asset.category === 'door' || asset.category === 'window') {
      if (!floorplan) return;
      const length = asset.footprint_mm?.w ?? (asset.category === 'door' ? 900 : 1200);
      const snap = snapToWall(pos, length, floorplan.walls);
      if (!snap) return;
      if (asset.category === 'door') {
        const nextDoor: Door = {
          door_id: `door_${Date.now()}`,
          wall_id: snap.wall_id,
          start_mm: snap.start,
          end_mm: snap.end,
          leaf_width_mm: length,
          hinge_side: 'left',
          open_direction: 'in',
        };
        setFloorplan({
          ...floorplan,
          openings: { ...floorplan.openings, doors: [...floorplan.openings.doors, nextDoor] },
        });
        selectOpening('door', nextDoor.door_id);
      } else {
        const nextWindow: Window = {
          window_id: `window_${Date.now()}`,
          wall_id: snap.wall_id,
          start_mm: snap.start,
          end_mm: snap.end,
          sill_height_mm: 900,
          height_mm: 1200,
        };
        setFloorplan({
          ...floorplan,
          openings: { ...floorplan.openings, windows: [...floorplan.openings.windows, nextWindow] },
        });
        selectOpening('window', nextWindow.window_id);
      }
      return;
    }

    const placement: Placement = {
      placement_id: crypto.randomUUID(),
      asset_id: asset.asset_id,
      category: asset.category,
      position_mm: pos,
      rotation_deg: 0,
      footprint_mm: asset.footprint_mm,
      height_mm: asset.height_mm ?? 0,
      color_hex: asset.color_hex ?? '#cfcac2',
      anchor: 'center',
      source: 'manual',
      constraint_mode: 'pinned',
    };

    const nextDesign = design
      ? {
          ...design,
          placements: [...design.placements, placement],
        }
      : floorplan
        ? {
            floorplan_geometry: floorplan,
            placements: [placement],
          }
        : null;
    if (!nextDesign) return;
    setDesign(nextDesign);
    selectPlacement(placement.placement_id);
  };

  const handleStageMouseMove = () => {
    if (!drawingLine || !floorplan) return;
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return;
      const nextEnd = fromCanvasPoint({ x: pointer.x, y: pointer.y });
    setDrawingLine({ ...drawingLine, end: nextEnd });
  };

  const handleStageMouseUp = () => {
    if (!drawingLine || !floorplan) return;
    const startPoint = floorplan.room.polygon_mm[drawingLine.startIndex];
    if (!startPoint) {
      setDrawingLine(null);
      return;
    }
    const nearest = findNearestPoint(
      floorplan.room.polygon_mm,
      drawingLine.end,
      drawingLine.startIndex,
      200,
    );
    if (nearest) {
      const newWall = {
        wall_id: `wall_${Date.now()}`,
        p1_mm: startPoint,
        p2_mm: nearest.point,
        thickness_mm: floorplan.walls[0]?.thickness_mm ?? 100,
        room_id: floorplan.room.room_id,
      };
      const exists = floorplan.walls.some((wall) =>
        (pointsEqual(wall.p1_mm, newWall.p1_mm) && pointsEqual(wall.p2_mm, newWall.p2_mm)) ||
        (pointsEqual(wall.p1_mm, newWall.p2_mm) && pointsEqual(wall.p2_mm, newWall.p1_mm))
      );
      if (!exists) {
        setFloorplan({
          ...floorplan,
          walls: [...floorplan.walls, newWall],
        });
        selectWall(newWall.wall_id);
      }
    }
    setDrawingLine(null);
  };

  const handleStageDoubleClick = (evt: any) => {
    if (!roomEditMode || !floorplan) return;
    if (evt.target !== evt.target.getStage()) return;
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return;
      const point = fromCanvasPoint({ x: pointer.x, y: pointer.y });
    const points = floorplan.room.polygon_mm;
    const nextPoints = [...points, point];
    setFloorplan({
      ...floorplan,
      room: { ...floorplan.room, polygon_mm: nextPoints },
    });
  };

  const drawGrid = () => {
    const lines = [];
    for (let x = bounds.minX; x <= bounds.maxX + GRID_SIZE; x += GRID_SIZE) {
      const start = toCanvasPoint({ x, y: bounds.minY });
      const end = toCanvasPoint({ x, y: bounds.maxY });
      lines.push(
        <Line key={`gx-${x}`} points={[start.x, start.y, end.x, end.y]} stroke="#e6e1d8" strokeWidth={1} />
      );
    }
    for (let y = bounds.minY; y <= bounds.maxY + GRID_SIZE; y += GRID_SIZE) {
      const start = toCanvasPoint({ x: bounds.minX, y });
      const end = toCanvasPoint({ x: bounds.maxX, y });
      lines.push(
        <Line key={`gy-${y}`} points={[start.x, start.y, end.x, end.y]} stroke="#e6e1d8" strokeWidth={1} />
      );
    }
    return lines;
  };

  const renderWalls = () => {
    if (!floorplan) return null;
    return floorplan.walls.map((wall) => {
      const p1 = toCanvasPoint(wall.p1_mm);
      const p2 = toCanvasPoint(wall.p2_mm);
      const isSelected = selectedWallId === wall.wall_id;
      const strokeColor = isSelected ? "#d9534f" : (wall.color_hex || "#555");
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.hypot(dx, dy) || 1;
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      const nx = -dy / len;
      const ny = dx / len;
      const labelPos = { x: mid.x + nx * 12, y: mid.y + ny * 12 };
      const lengthMm = Math.hypot(
        wall.p2_mm.x - wall.p1_mm.x,
        wall.p2_mm.y - wall.p1_mm.y,
      );
      const lengthLabel = `${(lengthMm / 1000).toFixed(2)} m`;
      return (
        <Group key={wall.wall_id}>
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke={strokeColor}
            strokeWidth={Math.max(2, wall.thickness_mm * scale)}
            lineCap="round"
            onClick={() => {
              selectWall(wall.wall_id);
              selectCorner(null);
            }}
            onDblClick={() => {
              if (!roomEditMode || !floorplan) return;
              if (wall.locked) return;
              const nextWalls = floorplan.walls.filter((w) => w.wall_id !== wall.wall_id);
              setFloorplan({ ...floorplan, walls: nextWalls });
              selectWall(null);
            }}
            draggable={false}
          />
          <Text
            text={lengthLabel}
            x={labelPos.x + 4}
            y={labelPos.y - 8}
            fontSize={11}
            fill="#6d665c"
            fontStyle="bold"
          />
        </Group>
      );
    });
  };

  const renderRoomOutline = () => {
    if (!floorplan || roomEditMode) return null;
    const points = floorplan.room.polygon_mm.flatMap((p) => {
      const c = toCanvasPoint(p);
      return [c.x, c.y];
    });
    const floorFill = floorplan.room.floor_color_hex ?? 'rgba(246, 244, 239, 0.6)';
    return (
      <Line
        points={points}
        closed={!roomEditMode && points.length >= 6}
        fill={floorFill}
        stroke="#bdb4a7"
        strokeWidth={2}
        onClick={() => {
          selectCorner(null);
          selectWall(null);
        }}
      />
    );
  };

  const renderRoomHandles = () => {
    if (!floorplan || !roomEditMode) return null;
    return floorplan.room.polygon_mm.map((point, idx) => {
      const pos = toCanvasPoint(point);
      const isSelected = selectedCornerIndex === idx;
      return (
        <Group key={`corner-${idx}`}>
          <Circle
            x={pos.x}
            y={pos.y}
            radius={8}
            fill={isSelected ? "#ffd28b" : "#ffffff"}
            stroke="#2f2f2f"
            strokeWidth={1.5}
            draggable={!drawWallMode}
            onClick={() => {
              selectCorner(idx);
              selectWall(null);
            }}
            onMouseDown={() => {
              if (!roomEditMode || !drawWallMode || !floorplan) return;
              if (selectedCornerIndex !== idx) {
                selectCorner(idx);
                return;
              }
              setDrawingLine({ startIndex: idx, end: floorplan.room.polygon_mm[idx] });
            }}
            onDragMove={(evt) => {
              const nextPoint = fromCanvasPoint({ x: evt.target.x(), y: evt.target.y() });
              const oldPoint = floorplan.room.polygon_mm[idx];
              const nextPoints = [...floorplan.room.polygon_mm];
              nextPoints[idx] = nextPoint;
              const updatedWalls = floorplan.walls.map((wall) => {
                if (wall.locked) {
                  return wall;
                }
                const p1 = wall.p1_mm;
                const p2 = wall.p2_mm;
                if (pointsEqual(p1, oldPoint)) {
                  return { ...wall, p1_mm: nextPoint };
                }
                if (pointsEqual(p2, oldPoint)) {
                  return { ...wall, p2_mm: nextPoint };
                }
                return wall;
              });
              setFloorplan({
                ...floorplan,
                room: { ...floorplan.room, polygon_mm: nextPoints },
                walls: updatedWalls,
              });
            }}
            onDragEnd={() => {
              return;
            }}
          />
          <Text
            text={`${idx + 1}`}
            x={pos.x + 10}
            y={pos.y - 10}
            fontSize={12}
            fill="#6d665c"
          />
        </Group>
      );
    });
  };

  const renderOpenings = () => {
    if (!floorplan) return null;
    const doorShapes = floorplan.openings.doors.map((door) => {
      const p1 = toCanvasPoint(door.start_mm);
      const p2 = toCanvasPoint(door.end_mm);
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      return (
        <Group
          key={door.door_id}
          draggable
          onDragEnd={(evt) => {
            if (!floorplan) return;
            const dx = evt.target.x() / scale;
            const dy = -evt.target.y() / scale;
            evt.target.position({ x: 0, y: 0 });
            const moved = {
              start_mm: { x: door.start_mm.x + dx, y: door.start_mm.y + dy },
              end_mm: { x: door.end_mm.x + dx, y: door.end_mm.y + dy },
            };
            const length = Math.hypot(
              moved.end_mm.x - moved.start_mm.x,
              moved.end_mm.y - moved.start_mm.y,
            );
            const center = {
              x: (moved.start_mm.x + moved.end_mm.x) / 2,
              y: (moved.start_mm.y + moved.end_mm.y) / 2,
            };
            const snap = snapToWall(center, length, floorplan.walls);
            const updated = floorplan.openings.doors.map((d) =>
              d.door_id === door.door_id && snap
                ? {
                    ...d,
                    start_mm: snap.start,
                    end_mm: snap.end,
                    wall_id: snap.wall_id,
                  }
                : d
            );
            setFloorplan({
              ...floorplan,
              openings: { ...floorplan.openings, doors: updated },
            });
          }}
          onClick={() => selectOpening('door', door.door_id)}
        >
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke={door.color_hex || '#2f2f2f'}
            strokeWidth={4}
          />
          <Text
            text="door"
            fontSize={11}
            fill="#2f2f2f"
            x={mid.x + 6}
            y={mid.y - 10}
          />
        </Group>
      );
    });
    const windowShapes = floorplan.openings.windows.map((window) => {
      const p1 = toCanvasPoint(window.start_mm);
      const p2 = toCanvasPoint(window.end_mm);
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      return (
        <Group
          key={window.window_id}
          draggable
          onDragEnd={(evt) => {
            if (!floorplan) return;
            const dx = evt.target.x() / scale;
            const dy = -evt.target.y() / scale;
            evt.target.position({ x: 0, y: 0 });
            const moved = {
              start_mm: { x: window.start_mm.x + dx, y: window.start_mm.y + dy },
              end_mm: { x: window.end_mm.x + dx, y: window.end_mm.y + dy },
            };
            const length = Math.hypot(
              moved.end_mm.x - moved.start_mm.x,
              moved.end_mm.y - moved.start_mm.y,
            );
            const center = {
              x: (moved.start_mm.x + moved.end_mm.x) / 2,
              y: (moved.start_mm.y + moved.end_mm.y) / 2,
            };
            const snap = snapToWall(center, length, floorplan.walls);
            const updated = floorplan.openings.windows.map((w) =>
              w.window_id === window.window_id && snap
                ? {
                    ...w,
                    start_mm: snap.start,
                    end_mm: snap.end,
                    wall_id: snap.wall_id,
                  }
                : w
            );
            setFloorplan({
              ...floorplan,
              openings: { ...floorplan.openings, windows: updated },
            });
          }}
          onClick={() => selectOpening('window', window.window_id)}
        >
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke={window.color_hex || '#4f7c9b'}
            strokeWidth={4}
            dash={[6, 6]}
          />
          <Text
            text="window"
            fontSize={11}
            fill="#4f7c9b"
            x={mid.x + 6}
            y={mid.y - 10}
          />
        </Group>
      );
    });
    return (
      <>
        {doorShapes}
        {windowShapes}
      </>
    );
  };

  const isPlacementErrored = (placementId: string) =>
    violations.some((v) => v.placement_ids.includes(placementId) && v.severity === 'error');

  const renderPlacements = () => {
    if (!design) return null;
    const placements = design.placements.map((placement, index) => ({ placement, index }));
    placements.sort((a, b) => {
      const aLayer = placementLayer(a.placement);
      const bLayer = placementLayer(b.placement);
      if (aLayer !== bLayer) return aLayer - bLayer;
      if (aLayer === 0) {
        const areaA = a.placement.footprint_mm.w * a.placement.footprint_mm.d;
        const areaB = b.placement.footprint_mm.w * b.placement.footprint_mm.d;
        if (areaA !== areaB) return areaB - areaA;
      }
      return a.index - b.index;
    });
    return placements.map(({ placement }) => {
      const isLocked = placement.constraint_mode === 'locked';
      const isCurtain = isCurtainLike(placement.category);
      const isMirror = isMirrorLike(placement.category);
      const isWallMounted = isWallMountedLike(placement);
      const isCeilingMounted = isCeilingMountedLike(placement);
      const matchedWindow = isCurtain
        ? findNearestWindowForPlacement(
            placement.position_mm,
            floorplan?.openings.windows ?? [],
          )
        : null;
      const matchedWall = isWallMounted && placement.place_on?.method === 'hang_on'
        ? snapToWall(
            placement.position_mm,
            Math.max(
              placement.footprint_mm.w,
              placement.footprint_mm.d,
              160,
            ),
            floorplan?.walls ?? [],
          )
        : null;
      const pos = toCanvasPoint(placement.position_mm);
      const w = placement.footprint_mm.w * scale;
      const d = placement.footprint_mm.d * scale;
      const isSelected = placement.placement_id === selectedPlacementId;
      const stroke = isPlacementErrored(placement.placement_id) ? '#d9534f' : '#333';
      const labelText = isCurtain ? 'curtain' : placement.category;
      const labelColor = isCurtain ? '#b65a3a' : '#1f1f1f';
      const showInlineLabel = true;
      const showHoverTag = false;
      const tagText = labelText;
      const frontSide =
        frontWorldToCanvasLocalSide(placement.front_world, placement.rotation_deg) ??
        placement.front_side_local ??
        'top';
      const frontPoints = frontMarkerPoints(w, d, frontSide);
      const baseSize = Math.max(0, Math.min(w, d));
      const maxLabelWidth = Math.max(0, w - 10);
      const maxLabelHeight = Math.max(0, d - 10);
      const tagHeight = Math.min(maxLabelHeight, clamp(baseSize / 7, 12, 24));
      const tagFontSize = clamp(tagHeight * 0.55, 8, 14);
      const tagPaddingX = Math.max(4, tagFontSize * 0.55);
      const tagPaddingY = Math.max(2, tagFontSize * 0.25);
      const estTextW = tagText.length * tagFontSize * 0.58;
      const tagWidth = Math.min(maxLabelWidth, Math.max(24, estTextW + tagPaddingX * 2));

      if (isCurtain && matchedWindow) {
        const start = toCanvasPoint(matchedWindow.start_mm);
        const end = toCanvasPoint(matchedWindow.end_mm);
        const mid = {
          x: (start.x + end.x) / 2,
          y: (start.y + end.y) / 2,
        };
        const curtainStrokeWidth = clamp(
          placement.footprint_mm.d * scale * 0.14,
          8,
          16,
        );
        return (
          <Group
            key={placement.placement_id}
            draggable={!isLocked}
            onDragEnd={(evt) => {
              if (isLocked) return;
              const dx = evt.target.x() / scale;
              const dy = -evt.target.y() / scale;
              evt.target.position({ x: 0, y: 0 });
              updatePlacement({
                ...placement,
                position_mm: {
                  x: placement.position_mm.x + dx,
                  y: placement.position_mm.y + dy,
                },
              });
            }}
            onClick={() => selectPlacement(placement.placement_id)}
            onTap={() => selectPlacement(placement.placement_id)}
          >
            <Line
              points={[start.x, start.y, end.x, end.y]}
              stroke={stroke}
              strokeWidth={curtainStrokeWidth + (isSelected ? 3 : 1)}
              opacity={0.18}
              lineCap="round"
              listening={false}
            />
            <Line
              points={[start.x, start.y, end.x, end.y]}
              stroke={placement.color_hex || '#ece7dc'}
              strokeWidth={curtainStrokeWidth}
              opacity={0.95}
              lineCap="round"
              hitStrokeWidth={Math.max(18, curtainStrokeWidth + 10)}
            />
            {showInlineLabel && (
              <Group
                x={mid.x - tagWidth / 2}
                y={mid.y - curtainStrokeWidth - tagHeight - 4}
                listening={false}
              >
                <Rect
                  width={tagWidth}
                  height={tagHeight}
                  fill="rgba(255,255,255,0.78)"
                  stroke="rgba(0,0,0,0.18)"
                  cornerRadius={5}
                />
                <Text
                  text={tagText}
                  fontSize={tagFontSize}
                  fill={labelColor}
                  x={tagPaddingX}
                  y={tagPaddingY - 1}
                  width={Math.max(0, tagWidth - tagPaddingX * 2)}
                  ellipsis
                />
              </Group>
            )}
            {isSelected && (
              <Group x={mid.x + 12} y={mid.y - curtainStrokeWidth - 10}>
                <Rect
                  width={Math.max(70, placement.category.length * 8)}
                  height={20}
                  fill="#fff"
                  stroke="#333"
                  cornerRadius={4}
                  shadowColor="#333"
                  shadowBlur={2}
                  shadowOpacity={0.15}
                />
                <Text
                  text={placement.category}
                  fontSize={12}
                  fill="#111"
                  x={6}
                  y={4}
                />
              </Group>
            )}
            {isSelected && (
              <Group
                x={mid.x + 12}
                y={mid.y + 14}
                onClick={(evt) => {
                  evt.cancelBubble = true;
                  removePlacement(placement.placement_id);
                }}
                onTap={(evt) => {
                  evt.cancelBubble = true;
                  removePlacement(placement.placement_id);
                }}
              >
                <Rect
                  width={60}
                  height={18}
                  fill="#d9534f"
                  stroke="#b52b27"
                  cornerRadius={4}
                  shadowColor="#000"
                  shadowBlur={2}
                  shadowOpacity={0.15}
                />
                <Text
                  text="Delete"
                  fontSize={11}
                  fill="#fff"
                  x={10}
                  y={3}
                />
              </Group>
            )}
          </Group>
        );
      }

      if (isWallMounted && matchedWall) {
        const start = toCanvasPoint(matchedWall.start);
        const end = toCanvasPoint(matchedWall.end);
        const mid = {
          x: (start.x + end.x) / 2,
          y: (start.y + end.y) / 2,
        };
        const wallItemStrokeWidth = clamp(
          placement.footprint_mm.d * scale,
          6,
          isMirror ? 12 : 10,
        );
        return (
          <Group
            key={placement.placement_id}
            draggable={!isLocked}
            onDragEnd={(evt) => {
              if (isLocked) return;
              const dx = evt.target.x() / scale;
              const dy = -evt.target.y() / scale;
              evt.target.position({ x: 0, y: 0 });
              updatePlacement({
                ...placement,
                position_mm: {
                  x: placement.position_mm.x + dx,
                  y: placement.position_mm.y + dy,
                },
              });
            }}
            onClick={() => selectPlacement(placement.placement_id)}
            onTap={() => selectPlacement(placement.placement_id)}
          >
            <Line
              points={[start.x, start.y, end.x, end.y]}
              stroke={stroke}
              strokeWidth={wallItemStrokeWidth + (isSelected ? 3 : 1)}
              opacity={0.2}
              lineCap="round"
              listening={false}
            />
            <Line
              points={[start.x, start.y, end.x, end.y]}
              stroke={placement.color_hex || (isMirror ? '#dfe6ec' : '#b48a63')}
              strokeWidth={wallItemStrokeWidth}
              opacity={0.96}
              lineCap="round"
              hitStrokeWidth={Math.max(16, wallItemStrokeWidth + 8)}
            />
            <Line
              points={[start.x, start.y, end.x, end.y]}
              stroke="rgba(255,255,255,0.7)"
              strokeWidth={Math.max(1.5, wallItemStrokeWidth * 0.22)}
              opacity={0.7}
              lineCap="round"
              listening={false}
            />
            {showInlineLabel && (
              <Group
                x={mid.x - tagWidth / 2}
                y={mid.y - wallItemStrokeWidth - tagHeight - 4}
                listening={false}
              >
                <Rect
                  width={tagWidth}
                  height={tagHeight}
                  fill="rgba(255,255,255,0.78)"
                  stroke="rgba(0,0,0,0.18)"
                  cornerRadius={5}
                />
                <Text
                  text={tagText}
                  fontSize={tagFontSize}
                  fill={isMirror ? '#647789' : '#8a5a34'}
                  x={tagPaddingX}
                  y={tagPaddingY - 1}
                  width={Math.max(0, tagWidth - tagPaddingX * 2)}
                  ellipsis
                />
              </Group>
            )}
            {isSelected && (
              <Group x={mid.x + 12} y={mid.y - wallItemStrokeWidth - 10}>
                <Rect
                  width={Math.max(70, placement.category.length * 8)}
                  height={20}
                  fill="#fff"
                  stroke="#333"
                  cornerRadius={4}
                  shadowColor="#333"
                  shadowBlur={2}
                  shadowOpacity={0.15}
                />
                <Text
                  text={placement.category}
                  fontSize={12}
                  fill="#111"
                  x={6}
                  y={4}
                />
              </Group>
            )}
            {isSelected && (
              <Group
                x={mid.x + 12}
                y={mid.y + 14}
                onClick={(evt) => {
                  evt.cancelBubble = true;
                  removePlacement(placement.placement_id);
                }}
                onTap={(evt) => {
                  evt.cancelBubble = true;
                  removePlacement(placement.placement_id);
                }}
              >
                <Rect
                  width={60}
                  height={18}
                  fill="#d9534f"
                  stroke="#b52b27"
                  cornerRadius={4}
                  shadowColor="#000"
                  shadowBlur={2}
                  shadowOpacity={0.15}
                />
                <Text
                  text="Delete"
                  fontSize={11}
                  fill="#fff"
                  x={10}
                  y={3}
                />
              </Group>
            )}
          </Group>
        );
      }

      if (isCeilingMounted) {
        const radius = Math.max(8, Math.min(Math.max(w, d) * 0.28, 24));
        return (
          <Group
            key={placement.placement_id}
            x={pos.x}
            y={pos.y}
            draggable={!isLocked}
            onDragMove={(evt) => {
              if (isLocked) return;
              const newPos = fromCanvasPoint({ x: evt.target.x(), y: evt.target.y() });
              updatePlacement({ ...placement, position_mm: newPos });
            }}
            onClick={() => selectPlacement(placement.placement_id)}
            onTap={() => selectPlacement(placement.placement_id)}
          >
            <Circle
              radius={radius}
              fill={placement.color_hex || '#fff3b0'}
              stroke={stroke}
              strokeWidth={isSelected ? 3 : 1}
              opacity={0.92}
              shadowColor="#d8b84f"
              shadowBlur={isSelected ? 8 : 4}
              shadowOpacity={0.25}
            />
            <Line
              points={[-radius * 0.55, 0, radius * 0.55, 0]}
              stroke="rgba(255,255,255,0.95)"
              strokeWidth={Math.max(2, radius * 0.16)}
              lineCap="round"
              listening={false}
            />
            <Line
              points={[0, -radius * 0.55, 0, radius * 0.55]}
              stroke="rgba(255,255,255,0.95)"
              strokeWidth={Math.max(2, radius * 0.16)}
              lineCap="round"
              listening={false}
            />
            {showInlineLabel && (
              <Group x={-tagWidth / 2} y={radius + 5} listening={false}>
                <Rect
                  width={tagWidth}
                  height={tagHeight}
                  fill="rgba(255,255,255,0.78)"
                  stroke="rgba(0,0,0,0.18)"
                  cornerRadius={5}
                />
                <Text
                  text={tagText}
                  fontSize={tagFontSize}
                  fill="#7a671d"
                  x={tagPaddingX}
                  y={tagPaddingY - 1}
                  width={Math.max(0, tagWidth - tagPaddingX * 2)}
                  ellipsis
                />
              </Group>
            )}
          </Group>
        );
      }

      return (
        <Group
          key={placement.placement_id}
          x={pos.x}
          y={pos.y}
          rotation={placement.rotation_deg}
          draggable={!isLocked}
          onDragMove={(evt) => {
            if (isLocked) return;
            const newPos = fromCanvasPoint({ x: evt.target.x(), y: evt.target.y() });
            updatePlacement({ ...placement, position_mm: newPos });
          }}
          onClick={() => selectPlacement(placement.placement_id)}
          onTap={() => selectPlacement(placement.placement_id)}
        >
          <Rect
            x={-w / 2}
            y={-d / 2}
            width={w}
            height={d}
            fill={placement.color_hex || '#cfcac2'}
            stroke={stroke}
            strokeWidth={isSelected ? 3 : 1}
            cornerRadius={6}
            shadowColor="#333"
            shadowBlur={isSelected ? 6 : 0}
            shadowOffset={{ x: 0, y: 0 }}
            shadowOpacity={0.2}
          />
          <Line
            points={frontPoints}
            stroke="rgba(255,255,255,0.96)"
            strokeWidth={Math.max(3, Math.min(Math.min(w, d) * 0.06, 8))}
            lineCap="round"
            listening={false}
          />
          {showInlineLabel && (
            <Group x={-w / 2 + 6} y={-d / 2 + 6} listening={false}>
              <Rect
                width={tagWidth}
                height={tagHeight}
                fill="rgba(255,255,255,0.75)"
                stroke="rgba(0,0,0,0.18)"
                cornerRadius={5}
              />
              <Text
                text={tagText}
                fontSize={tagFontSize}
                fill={labelColor}
                x={tagPaddingX}
                y={tagPaddingY - 1}
                width={Math.max(0, tagWidth - tagPaddingX * 2)}
                ellipsis
              />
            </Group>
          )}
          {showHoverTag && (
            <Group x={w / 2 + 10} y={-d / 2} listening={false}>
              <Rect
                width={Math.max(70, tagText.length * 7 + 22)}
                height={20}
                fill="rgba(255,255,255,0.9)"
                stroke="rgba(0,0,0,0.25)"
                cornerRadius={4}
                shadowColor="#333"
                shadowBlur={2}
                shadowOpacity={0.12}
              />
              <Text text={tagText} fontSize={12} fill="#111" x={6} y={4} />
            </Group>
          )}
          {isSelected && (
            <Group x={w / 2 + 14} y={-d / 2}>
              <Rect
                width={Math.max(70, placement.category.length * 8)}
                height={20}
                fill="#fff"
                stroke="#333"
                cornerRadius={4}
                shadowColor="#333"
                shadowBlur={2}
                shadowOpacity={0.15}
              />
              <Text
                text={placement.category}
                fontSize={12}
                fill="#111"
                x={6}
                y={4}
              />
            </Group>
          )}
          {isSelected && (
            <Group
              x={w / 2 + 14}
              y={-d / 2 + 26}
              onClick={(evt) => {
                evt.cancelBubble = true;
                removePlacement(placement.placement_id);
              }}
              onTap={(evt) => {
                evt.cancelBubble = true;
                removePlacement(placement.placement_id);
              }}
            >
              <Rect
                width={60}
                height={18}
                fill="#d9534f"
                stroke="#b52b27"
                cornerRadius={4}
                shadowColor="#000"
                shadowBlur={2}
                shadowOpacity={0.15}
              />
              <Text
                text="Delete"
                fontSize={11}
                fill="#fff"
                x={10}
                y={3}
              />
            </Group>
          )}
          {isSelected && !isLocked && (
            <>
              <Line
                points={[0, 0, w / 2 + 18, 0]}
                stroke="rgba(51, 51, 51, 0.45)"
                strokeWidth={1.5}
                dash={[5, 5]}
                listening={false}
              />
              <Text
                text="Rotate"
                x={w / 2 + 24}
                y={-8}
                fontSize={11}
                fill="#6d665c"
                listening={false}
              />
              <Circle
                x={w / 2 + 18}
                y={0}
                radius={10}
                fill="#fff7eb"
                stroke="#a46a2a"
                strokeWidth={2}
                draggable
                onDragMove={(evt) => {
                  const localX = evt.target.x();
                  const localY = evt.target.y();
                  const rad = Math.atan2(localY, localX);
                  updatePlacement({ ...placement, rotation_deg: (rad * 180) / Math.PI });
                }}
              />
            </>
          )}
        </Group>
      );
    });
  };

  return (
    <div
      ref={dropZoneRef}
      className="canvas-drop-zone"
      onDrop={handleDrop}
      onDragOver={(evt) => evt.preventDefault()}
    >
      <Stage
        width={width}
        height={height}
        ref={stageRef}
        onWheel={handleWheel}
        onDblClick={handleStageDoubleClick}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
        style={{ background: '#f8f6f1', borderRadius: 12, border: '1px solid #e0dbd2' }}
      >
        <Layer>
          {drawGrid()}
          {renderRoomOutline()}
          {renderWalls()}
          {drawingLine && floorplan && floorplan.room.polygon_mm[drawingLine.startIndex] && (() => {
            const start = toCanvasPoint(floorplan.room.polygon_mm[drawingLine.startIndex]);
            const end = toCanvasPoint(drawingLine.end);
            return (
              <Line
                points={[start.x, start.y, end.x, end.y]}
                stroke="#d4a15d"
                strokeWidth={2}
                dash={[6, 6]}
              />
            );
          })()}
          {renderOpenings()}
          {renderPlacements()}
          {renderRoomHandles()}
        </Layer>
      </Stage>
    </div>
  );
};

function pointsEqual(a: Point, b: Point, tolerance = 1): boolean {
  return Math.abs(a.x - b.x) <= tolerance && Math.abs(a.y - b.y) <= tolerance;
}

function resolveDroppedAsset(
  dataTransfer: DataTransfer,
  inventory: InventoryItem[],
): PlacementAsset | null {
  const rawCustom = dataTransfer.getData('application/x-asset');
  const rawPlain = dataTransfer.getData('text/plain');
  const rawAssetId = dataTransfer.getData('text/x-asset-id');

  for (const candidate of [rawCustom, rawPlain]) {
    if (!candidate) continue;
    try {
      const parsed = JSON.parse(candidate);
      if (isPlacementAsset(parsed)) {
        return parsed;
      }
    } catch {
      // fall through to inventory lookup
    }
  }

  const lookupId = rawAssetId || rawPlain;
  if (!lookupId) {
    return null;
  }
  const matched = inventory.find((item) => item.asset_id === lookupId);
  return matched ?? null;
}

type PlacementAsset = {
  asset_id: string;
  category: string;
  footprint_mm: { w: number; d: number };
  height_mm?: number;
  color_hex?: string | null;
};

function isPlacementAsset(value: unknown): value is PlacementAsset {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const row = value as Record<string, unknown>;
  const footprint = row.footprint_mm;
  return (
    typeof row.asset_id === 'string' &&
    typeof row.category === 'string' &&
    !!footprint &&
    typeof footprint === 'object' &&
    typeof (footprint as Record<string, unknown>).w === 'number' &&
    typeof (footprint as Record<string, unknown>).d === 'number'
  );
}

function isRugLike(category: string): boolean {
  const cat = category.toLowerCase();
  return (
    cat.includes('rug') ||
    cat.includes('carpet') ||
    cat.includes('mat') ||
    cat.includes('runner')
  );
}

function isFloorCovering(category: string): boolean {
  const cat = category.toLowerCase();
  return (
    isRugLike(cat) ||
    cat.includes('mattress') ||
    cat.includes('floor_rug') ||
    cat.includes('area_rug')
  );
}

function placementLayer(placement: Placement): number {
  const explicitLayer = String((placement as any).collision_layer ?? '').trim();
  if (explicitLayer === 'floor_underlay') return 0;
  if (explicitLayer === 'surface_child') return 2;
  if (explicitLayer === 'wall_mounted') return 3;
  if (explicitLayer === 'ceiling') return 4;
  if (isFloorCovering(placement.category)) return 0;
  if (isCeilingMountedLike(placement)) return 4;
  if (isWallMountedLike(placement)) return 3;
  if (placement.place_on?.method === 'on_top') return 2;
  return 1;
}

function isCurtainLike(category: string): boolean {
  const cat = category.toLowerCase();
  return cat.includes('curtain') || cat.includes('blind') || cat.includes('drape');
}

function isMirrorLike(category: string): boolean {
  return category.toLowerCase().includes('mirror');
}

function isWallMountedLike(placement: Placement): boolean {
  const cat = placement.category.toLowerCase();
  return (
    placement.place_on?.method === 'hang_on' &&
    placement.place_on.target_instance_id !== 'ceiling' &&
    (
      isMirrorLike(cat) ||
      cat.includes('wall_art') ||
      cat.includes('art') ||
      cat.includes('clock') ||
      cat.includes('sconce') ||
      cat.includes('whiteboard') ||
      cat.includes('screen')
    )
  );
}

function isCeilingMountedLike(placement: Placement): boolean {
  const cat = placement.category.toLowerCase();
  return (
    placement.place_on?.target_instance_id === 'ceiling' ||
    cat.includes('ceiling_light') ||
    cat.includes('pendant_light') ||
    cat.includes('track_light')
  );
}

function findNearestWindowForPlacement(
  target: Point,
  windows: Array<{ start_mm: Point; end_mm: Point }>,
): { start_mm: Point; end_mm: Point } | null {
  let best: {
    windowRow: { start_mm: Point; end_mm: Point };
    distance: number;
  } | null = null;
  for (const windowRow of windows) {
    const center = {
      x: (windowRow.start_mm.x + windowRow.end_mm.x) / 2,
      y: (windowRow.start_mm.y + windowRow.end_mm.y) / 2,
    };
    const distance = Math.hypot(target.x - center.x, target.y - center.y);
    if (!best || distance < best.distance) {
      best = { windowRow, distance };
    }
  }
  return best?.windowRow ?? null;
}

function findNearestPoint(
  points: Point[],
  target: Point,
  excludeIndex: number,
  maxDistance: number,
): { point: Point; index: number } | null {
  let best: { point: Point; index: number; dist: number } | null = null;
  for (const [idx, point] of points.entries()) {
    if (idx === excludeIndex) continue;
    const dist = Math.hypot(point.x - target.x, point.y - target.y);
    if (dist <= maxDistance && (!best || dist < best.dist)) {
      best = { point, index: idx, dist };
    }
  }
  if (!best) return null;
  return { point: best.point, index: best.index };
}

type WallSnap = { start: Point; end: Point; wall_id: string };

function snapToWall(point: Point, length: number, walls: { wall_id: string; p1_mm: Point; p2_mm: Point }[]): WallSnap | null {
  if (!walls || walls.length === 0) return null;
  let best: { wall: typeof walls[number]; t: number; dist: number; len: number; dir: Point } | null = null;
  for (const wall of walls) {
    const proj = projectPointToSegment(point, wall.p1_mm, wall.p2_mm);
    if (!proj) continue;
    if (!best || proj.dist < best.dist) {
      best = { wall, t: proj.t, dist: proj.dist, len: proj.len, dir: proj.dir };
    }
  }
  if (!best) return null;
  const half = Math.min(length / 2, Math.max(10, best.len / 2 - 10));
  const tLen = best.t * best.len;
  const clamped = clamp(tLen, half, best.len - half);
  const center = {
    x: best.wall.p1_mm.x + best.dir.x * clamped,
    y: best.wall.p1_mm.y + best.dir.y * clamped,
  };
  return {
    wall_id: best.wall.wall_id,
    start: { x: center.x - best.dir.x * half, y: center.y - best.dir.y * half },
    end: { x: center.x + best.dir.x * half, y: center.y + best.dir.y * half },
  };
}

function projectPointToSegment(p: Point, a: Point, b: Point): { t: number; dist: number; len: number; dir: Point } | null {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const len2 = vx * vx + vy * vy;
  if (len2 < 1e-6) return null;
  let t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2;
  t = clamp(t, 0, 1);
  const proj = { x: a.x + t * vx, y: a.y + t * vy };
  const dist = Math.hypot(p.x - proj.x, p.y - proj.y);
  const len = Math.sqrt(len2);
  const dir = { x: vx / len, y: vy / len };
  return { t, dist, len, dir };
}
