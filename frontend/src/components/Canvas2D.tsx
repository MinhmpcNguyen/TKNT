import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Layer, Line, Rect, Stage, Group, Circle, Text } from 'react-konva';
import { useEditorStore } from '../store';
import { clamp, polygonBounds, toCanvas, fromCanvas } from '../utils/geometry';
import { Door, InventoryItem, Placement, Point, Window } from '../types';
import {
  frontWorldToLocalSide,
  planRotationToCanvasDeg,
} from '../utils/planOrientation';

const GRID_SIZE = 500;
const DOOR_MARKER_COLOR = '#2f3437';
const DOOR_LABEL_FILL = '#fff3e7';
const DOOR_LABEL_STROKE = '#d2a06a';
const DOOR_LABEL_TEXT = '#704417';
const WINDOW_MARKER_COLOR = '#4f9fbd';
const WINDOW_LABEL_FILL = '#edf8ff';
const WINDOW_LABEL_STROKE = '#8ec6d9';
const WINDOW_LABEL_TEXT = '#2e6f86';
const OPENING_WALL_CUT_FILL = '#f8f6f1';
const ROTATE_CONTROL_FILL = '#fff7eb';
const ROTATE_CONTROL_STROKE = '#a46a2a';
const ROTATE_CONTROL_HIGHLIGHT = '#f4a13c';
const OBJECT_COLOR_POPOVER_WIDTH = 252;
const CANVAS_MAX_WIDTH = 900;
const CANVAS_MAX_HEIGHT = 650;
const CANVAS_ASPECT_RATIO = CANVAS_MAX_WIDTH / CANVAS_MAX_HEIGHT;
const CANVAS_PADDING_MIN = 24;
const CANVAS_PADDING_MAX = 80;
const OBJECT_COLOR_SWATCHES = [
  '#cfcac2',
  '#6b7280',
  '#9f7a68',
  '#d9a441',
  '#4f7f6f',
  '#4f6f9f',
];

type OpeningOverlay = {
  normal: Point;
  tangent: Point;
  label: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

function segmentFrameTowardPoint(
  start: Point,
  end: Point,
  target: Point,
): { tangent: Point; normal: Point } {
  const tangent = normalizedVector({
    x: end.x - start.x,
    y: end.y - start.y,
  });
  const firstNormal = { x: -tangent.y, y: tangent.x };
  const mid = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2,
  };
  const towardTarget = {
    x: target.x - mid.x,
    y: target.y - mid.y,
  };
  const normal =
    firstNormal.x * towardTarget.x + firstNormal.y * towardTarget.y >= 0
      ? firstNormal
      : { x: -firstNormal.x, y: -firstNormal.y };
  return { tangent, normal };
}

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

function openingOverlay(
  start: Point,
  end: Point,
  roomCenter: Point,
  labelWidth: number,
  labelHeight: number,
): OpeningOverlay {
  const mid = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2,
  };
  const { tangent, normal } = segmentFrameTowardPoint(start, end, roomCenter);
  const labelNormal = { x: -normal.x, y: -normal.y };
  const labelGap = 28;
  const labelCenter = {
    x: mid.x + labelNormal.x * labelGap,
    y: mid.y + labelNormal.y * labelGap,
  };
  return {
    normal,
    tangent,
    label: {
      x: labelCenter.x - labelWidth / 2,
      y: labelCenter.y - labelHeight / 2,
      width: labelWidth,
      height: labelHeight,
    },
  };
}

function buildDoorSwing(
  door: Door,
  start: Point,
  end: Point,
  overlay: OpeningOverlay,
): {
  arc: number[];
  leaf: number[];
} {
  const segmentLength = Math.hypot(end.x - start.x, end.y - start.y);
  const radius = clamp(segmentLength * 0.86, 28, 92);
  const hingeSide = door.hinge_side === 'right' ? 'right' : 'left';
  const hinge = hingeSide === 'right' ? end : start;
  const closedVector =
    hingeSide === 'right'
      ? { x: -overlay.tangent.x, y: -overlay.tangent.y }
      : overlay.tangent;
  const swingNormal =
    door.open_direction === 'out'
      ? { x: -overlay.normal.x, y: -overlay.normal.y }
      : overlay.normal;
  const startAngle = Math.atan2(closedVector.y, closedVector.x);
  const endAngle = Math.atan2(swingNormal.y, swingNormal.x);
  const angleDelta = normalizedAngleDelta(endAngle - startAngle);
  const arc: number[] = [];
  const steps = 12;
  for (let index = 0; index <= steps; index += 1) {
    const angle = startAngle + (angleDelta * index) / steps;
    arc.push(hinge.x + Math.cos(angle) * radius, hinge.y + Math.sin(angle) * radius);
  }
  return {
    arc,
    leaf: [
      hinge.x,
      hinge.y,
      hinge.x + swingNormal.x * radius,
      hinge.y + swingNormal.y * radius,
    ],
  };
}

function normalizedVector(vector: Point): Point {
  const length = Math.hypot(vector.x, vector.y);
  if (length <= 1e-6) {
    return { x: 1, y: 0 };
  }
  return {
    x: vector.x / length,
    y: vector.y / length,
  };
}

function normalizedAngleDelta(value: number): number {
  let delta = value;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return delta;
}

function normalizeColorInput(value: string | null | undefined): string {
  return /^#[0-9a-f]{6}$/i.test(value ?? '') ? value ?? '#cfcac2' : '#cfcac2';
}

function polygonCenter(points: Point[]): Point {
  if (points.length === 0) {
    return { x: 0, y: 0 };
  }
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
}

function isEditableKeyboardTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.tagName === 'SELECT')
  );
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
    selectedOpeningId,
    selectedOpeningType,
    selectWall,
    selectOpening,
    updatePlacement,
    removePlacement,
    removeOpening,
    updateWall,
    violations,
    setDesign,
    setFloorplan,
    roomEditMode,
    drawWallMode,
    setFloorplanCapture,
  } = useEditorStore();
  const [stageSize, setStageSize] = useState({
    width: CANVAS_MAX_WIDTH,
    height: CANVAS_MAX_HEIGHT,
  });
  const [zoomMultiplier, setZoomMultiplier] = useState(1);
  const [drawingLine, setDrawingLine] = useState<{ startIndex: number; end: Point } | null>(null);

  const bounds = useMemo(() => {
    if (!floorplan) {
      return { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
    }
    return polygonBounds(floorplan.room.polygon_mm);
  }, [floorplan]);

  useEffect(() => {
    const node = dropZoneRef.current;
    if (!node) {
      return undefined;
    }

    const updateStageSize = () => {
      const measuredWidth = node.getBoundingClientRect().width;
      const nextWidth = clamp(measuredWidth, 1, CANVAS_MAX_WIDTH);
      const nextHeight = Math.round(nextWidth / CANVAS_ASPECT_RATIO);
      setStageSize((current) =>
        Math.abs(current.width - nextWidth) < 1 && Math.abs(current.height - nextHeight) < 1
          ? current
          : { width: nextWidth, height: nextHeight }
      );
    };

    updateStageSize();
    const observer = new ResizeObserver(updateStageSize);
    observer.observe(node);
    window.addEventListener('resize', updateStageSize);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateStageSize);
    };
  }, []);

  const roomWidth = Math.max(1, bounds.maxX - bounds.minX);
  const roomHeight = Math.max(1, bounds.maxY - bounds.minY);
  const canvasPadding = clamp(
    Math.min(stageSize.width, stageSize.height) * 0.12,
    CANVAS_PADDING_MIN,
    CANVAS_PADDING_MAX,
  );
  const availableCanvasWidth = Math.max(1, stageSize.width - canvasPadding * 2);
  const availableCanvasHeight = Math.max(1, stageSize.height - canvasPadding * 2);
  const baseScale = Math.max(
    0.01,
    Math.min(availableCanvasWidth / roomWidth, availableCanvasHeight / roomHeight),
  );
  const scale = baseScale * zoomMultiplier;
  const renderedRoomWidth = roomWidth * scale;
  const renderedRoomHeight = roomHeight * scale;
  const offset = {
    x: (stageSize.width - renderedRoomWidth) / 2 - bounds.minX * scale,
    y: (stageSize.height - renderedRoomHeight) / 2,
  };
  const toCanvasPoint = (point: Point) => toCanvas(point, scale, offset, bounds);
  const fromCanvasPoint = (point: Point) => fromCanvas(point, scale, offset, bounds);

  const width = stageSize.width;
  const height = stageSize.height;
  const selectedPlacement =
    design?.placements.find((placement) => placement.placement_id === selectedPlacementId) ?? null;
  const selectedPlacementCanvasPoint = selectedPlacement
    ? toCanvasPoint(selectedPlacement.position_mm)
    : null;
  const selectedPlacementColor = normalizeColorInput(selectedPlacement?.color_hex);
  const selectedColorControlPosition = selectedPlacementCanvasPoint
    ? {
        left: clamp(selectedPlacementCanvasPoint.x + 18, 12, width - OBJECT_COLOR_POPOVER_WIDTH),
        top: clamp(selectedPlacementCanvasPoint.y - 52, 12, height - 48),
      }
    : null;
  const selectedCornerPoint =
    roomEditMode && floorplan && selectedCornerIndex !== null
      ? floorplan.room.polygon_mm[selectedCornerIndex] ?? null
      : null;
  const selectedCornerCanvasPoint = selectedCornerPoint
    ? toCanvasPoint(selectedCornerPoint)
    : null;
  const selectedCornerDeletePosition =
    selectedCornerCanvasPoint && floorplan && floorplan.room.polygon_mm.length > 3
      ? {
          left: clamp(selectedCornerCanvasPoint.x + 16, 8, width - 74),
          top: clamp(selectedCornerCanvasPoint.y + 12, 8, height - 34),
        }
      : null;

  useEffect(() => {
    setFloorplanCapture(() => {
      const stage = stageRef.current;
      if (!stage) return null;
      try {
        return stage.toDataURL({ pixelRatio: 1.5 });
      } catch {
        return null;
      }
    });
    return () => setFloorplanCapture(null);
  }, [setFloorplanCapture]);

  const deleteRoomCorner = useCallback(
    (index: number) => {
      if (!floorplan) return;
      const points = floorplan.room.polygon_mm;
      if (points.length <= 3 || index < 0 || index >= points.length) return;
      const removed = points[index];
      const nextPoints = points.filter((_, pointIndex) => pointIndex !== index);
      const nextWalls = floorplan.walls.filter((wall) => {
        const p1 = wall.p1_mm;
        const p2 = wall.p2_mm;
        return !pointsEqual(p1, removed) && !pointsEqual(p2, removed);
      });
      setFloorplan({
        ...floorplan,
        room: { ...floorplan.room, polygon_mm: nextPoints },
        walls: nextWalls,
      });
      selectCorner(null);
      selectWall(null);
    },
    [floorplan, selectCorner, selectWall, setFloorplan],
  );

  useEffect(() => {
    if (!selectedOpeningId || !selectedOpeningType) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;
      if (isEditableKeyboardTarget(event.target)) {
        return;
      }
      event.preventDefault();
      removeOpening(selectedOpeningType, selectedOpeningId);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [removeOpening, selectedOpeningId, selectedOpeningType]);

  useEffect(() => {
    if (!roomEditMode || selectedCornerIndex === null) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;
      if (isEditableKeyboardTarget(event.target)) {
        return;
      }
      event.preventDefault();
      deleteRoomCorner(selectedCornerIndex);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteRoomCorner, roomEditMode, selectedCornerIndex]);

  const handleWheel = (evt: any) => {
    evt.evt.preventDefault();
    const delta = evt.evt.deltaY > 0 ? 0.9 : 1.1;
    setZoomMultiplier((prev) => clamp(prev * delta, 0.4, 3.5));
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

  const selectNearestRoomCorner = (canvasPoint: Point) => {
    if (!roomEditMode || !floorplan) return false;
    const nearest = findNearestCanvasPoint(
      floorplan.room.polygon_mm,
      canvasPoint,
      toCanvasPoint,
      30,
    );
    if (!nearest) return false;
    selectCorner(nearest.index);
    if (drawWallMode && selectedCornerIndex === nearest.index) {
      setDrawingLine({
        startIndex: nearest.index,
        end: floorplan.room.polygon_mm[nearest.index],
      });
    }
    return true;
  };

  const handleStageMouseDown = () => {
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return;
    selectNearestRoomCorner(pointer);
  };

  const handleCanvasMouseDown = (evt: React.MouseEvent<HTMLDivElement>) => {
    if (!dropZoneRef.current) return;
    const target = evt.target;
    if (
      target instanceof Element &&
      target.closest('.canvas-corner-delete-button, .canvas-object-color-popover')
    ) {
      return;
    }
    const boundsRect = dropZoneRef.current.getBoundingClientRect();
    selectNearestRoomCorner({
      x: evt.clientX - boundsRect.left,
      y: evt.clientY - boundsRect.top,
    });
  };

  const handleCanvasTouchStart = (evt: React.TouchEvent<HTMLDivElement>) => {
    if (!dropZoneRef.current) return;
    const touch = evt.touches[0];
    if (!touch) return;
    const target = evt.target;
    if (
      target instanceof Element &&
      target.closest('.canvas-corner-delete-button, .canvas-object-color-popover')
    ) {
      return;
    }
    const boundsRect = dropZoneRef.current.getBoundingClientRect();
    selectNearestRoomCorner({
      x: touch.clientX - boundsRect.left,
      y: touch.clientY - boundsRect.top,
    });
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
      return (
        <Group key={wall.wall_id}>
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke={strokeColor}
            strokeWidth={Math.max(2, wall.thickness_mm * scale)}
            lineCap="round"
            onClick={() => {
              selectWall(wall.wall_id);
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
        </Group>
      );
    });
  };

  const renderWallLabels = () => {
    if (!floorplan) return null;
    const roomCenter = toCanvasPoint(polygonCenter(floorplan.room.polygon_mm));
    return floorplan.walls.map((wall) => {
      const p1 = toCanvasPoint(wall.p1_mm);
      const p2 = toCanvasPoint(wall.p2_mm);
      const { normal } = segmentFrameTowardPoint(p1, p2, roomCenter);
      const mid = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
      const lengthMm = Math.hypot(
        wall.p2_mm.x - wall.p1_mm.x,
        wall.p2_mm.y - wall.p1_mm.y,
      );
      const lengthLabel = `${(lengthMm / 1000).toFixed(2)} m`;
      const labelWidth = Math.max(48, lengthLabel.length * 6.4 + 10);
      const labelHeight = 16;
      const labelCenter = {
        x: mid.x + normal.x * 24,
        y: mid.y + normal.y * 24,
      };
      return (
        <Group
          key={`wall-label-${wall.wall_id}`}
          x={labelCenter.x - labelWidth / 2}
          y={labelCenter.y - labelHeight / 2}
          listening={false}
        >
          <Rect
            width={labelWidth}
            height={labelHeight}
            fill="rgba(248,246,241,0.82)"
            stroke="rgba(109,102,92,0.16)"
            strokeWidth={1}
            cornerRadius={4}
          />
          <Text
            text={lengthLabel}
            x={0}
            y={2}
            width={labelWidth}
            fontSize={11}
            fill="#6d665c"
            fontStyle="bold"
            align="center"
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

  const renderCornerDeleteButton = (index: number, position: Point) => (
    <Group
      x={position.x}
      y={position.y}
      onMouseDown={(evt) => {
        evt.cancelBubble = true;
      }}
      onClick={(evt) => {
        evt.cancelBubble = true;
        deleteRoomCorner(index);
      }}
      onTap={(evt) => {
        evt.cancelBubble = true;
        deleteRoomCorner(index);
      }}
    >
      <Rect
        width={58}
        height={20}
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
        y={4}
        listening={false}
      />
    </Group>
  );

  const renderRoomHandles = () => {
    if (!floorplan || !roomEditMode) return null;
    return floorplan.room.polygon_mm.map((point, idx) => {
      const pos = toCanvasPoint(point);
      const isSelected = selectedCornerIndex === idx;
      const selectRoomCorner = (evt: any) => {
        evt.cancelBubble = true;
        selectCorner(idx);
        if (!drawWallMode || !floorplan) return;
        if (selectedCornerIndex !== idx) {
          return;
        }
        setDrawingLine({ startIndex: idx, end: floorplan.room.polygon_mm[idx] });
      };
      return (
        <Group
          key={`corner-${idx}`}
          x={pos.x}
          y={pos.y}
          draggable={!drawWallMode}
          onMouseDown={selectRoomCorner}
          onTouchStart={selectRoomCorner}
          onClick={selectRoomCorner}
          onTap={selectRoomCorner}
          onDragStart={(evt) => {
            evt.cancelBubble = true;
            selectCorner(idx);
          }}
          onDragMove={(evt) => {
            evt.cancelBubble = true;
            selectCorner(idx);
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
        >
          <Circle
            radius={18}
            fill="rgba(213, 140, 84, 0.01)"
            stroke="rgba(213, 140, 84, 0)"
          />
          <Circle
            radius={8}
            fill={isSelected ? "#ffd28b" : "#ffffff"}
            stroke="#2f2f2f"
            strokeWidth={1.5}
            listening={false}
          />
          <Text
            text={`${idx + 1}`}
            x={10}
            y={-10}
            fontSize={12}
            fill="#6d665c"
            listening={false}
          />
          {isSelected && floorplan.room.polygon_mm.length > 3
            ? renderCornerDeleteButton(idx, { x: 14, y: 10 })
            : null}
        </Group>
      );
    });
  };

  const renderOpeningDeleteButton = (
    type: 'door' | 'window',
    id: string,
    position: Point,
  ) => (
    <Group
      x={position.x}
      y={position.y}
      onClick={(evt) => {
        evt.cancelBubble = true;
        removeOpening(type, id);
      }}
      onTap={(evt) => {
        evt.cancelBubble = true;
        removeOpening(type, id);
      }}
    >
      <Rect
        width={58}
        height={20}
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
        y={4}
      />
    </Group>
  );

  const renderOpenings = () => {
    if (!floorplan) return null;
    const roomCenter = toCanvasPoint(polygonCenter(floorplan.room.polygon_mm));
    const doorShapes = floorplan.openings.doors.map((door) => {
      const p1 = toCanvasPoint(door.start_mm);
      const p2 = toCanvasPoint(door.end_mm);
      const overlay = openingOverlay(p1, p2, roomCenter, 38, 18);
      const swing = buildDoorSwing(door, p1, p2, overlay);
      const wall = floorplan.walls.find((row) => row.wall_id === door.wall_id);
      const wallCutStrokeWidth = Math.max(11, (wall?.thickness_mm ?? 100) * scale + 4);
      const isSelected =
        selectedOpeningType === 'door' && selectedOpeningId === door.door_id;
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
            stroke={OPENING_WALL_CUT_FILL}
            strokeWidth={wallCutStrokeWidth}
            lineCap="butt"
            listening={false}
          />
          <Line
            points={swing.leaf}
            stroke={door.color_hex || DOOR_MARKER_COLOR}
            strokeWidth={2}
            opacity={0.78}
            lineCap="round"
            listening={false}
          />
          <Line
            points={swing.arc}
            stroke={door.color_hex || DOOR_MARKER_COLOR}
            strokeWidth={1.6}
            opacity={0.58}
            dash={[4, 3]}
            lineCap="round"
            listening={false}
          />
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke={door.color_hex || DOOR_MARKER_COLOR}
            strokeWidth={isSelected ? 7 : 5}
            lineCap="butt"
          />
          <Group x={overlay.label.x} y={overlay.label.y} listening={false}>
            <Rect
              width={overlay.label.width}
              height={overlay.label.height}
              fill={DOOR_LABEL_FILL}
              stroke={DOOR_LABEL_STROKE}
              strokeWidth={1}
              cornerRadius={4}
              shadowColor="#6b4c2a"
              shadowBlur={3}
              shadowOpacity={0.12}
            />
            <Text
              text="door"
              fontSize={10}
              fontStyle="bold"
              fill={DOOR_LABEL_TEXT}
              x={0}
              y={2.5}
              width={overlay.label.width}
              align="center"
            />
          </Group>
          {isSelected
            ? renderOpeningDeleteButton('door', door.door_id, {
                x: overlay.label.x + overlay.label.width + 6,
                y: overlay.label.y - 1,
              })
            : null}
        </Group>
      );
    });
    const windowShapes = floorplan.openings.windows.map((window) => {
      const p1 = toCanvasPoint(window.start_mm);
      const p2 = toCanvasPoint(window.end_mm);
      const overlay = openingOverlay(p1, p2, roomCenter, 48, 18);
      const wall = floorplan.walls.find((row) => row.wall_id === window.wall_id);
      const wallCutStrokeWidth = Math.max(11, (wall?.thickness_mm ?? 100) * scale + 4);
      const isSelected =
        selectedOpeningType === 'window' && selectedOpeningId === window.window_id;
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
            stroke={OPENING_WALL_CUT_FILL}
            strokeWidth={wallCutStrokeWidth}
            lineCap="butt"
            listening={false}
          />
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke={window.color_hex || WINDOW_MARKER_COLOR}
            strokeWidth={isSelected ? 7 : 5}
            dash={[8, 6]}
            lineCap="butt"
          />
          <Line
            points={[p1.x, p1.y, p2.x, p2.y]}
            stroke="rgba(255,255,255,0.82)"
            strokeWidth={1.4}
            lineCap="butt"
            listening={false}
          />
          <Group x={overlay.label.x} y={overlay.label.y} listening={false}>
            <Rect
              width={overlay.label.width}
              height={overlay.label.height}
              fill={WINDOW_LABEL_FILL}
              stroke={WINDOW_LABEL_STROKE}
              strokeWidth={1}
              cornerRadius={4}
              shadowColor="#2e6f86"
              shadowBlur={3}
              shadowOpacity={0.12}
            />
            <Text
              text="window"
              fontSize={10}
              fontStyle="bold"
              fill={WINDOW_LABEL_TEXT}
              x={0}
              y={2.5}
              width={overlay.label.width}
              align="center"
            />
          </Group>
          {isSelected
            ? renderOpeningDeleteButton('window', window.window_id, {
                x: overlay.label.x + overlay.label.width + 6,
                y: overlay.label.y - 1,
              })
            : null}
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
      const canvasRotationDeg = planRotationToCanvasDeg(placement.rotation_deg);
      const stroke = isPlacementErrored(placement.placement_id) ? '#d9534f' : '#333';
      const labelText = isCurtain ? 'curtain' : placement.category;
      const labelColor = isCurtain ? '#b65a3a' : '#1f1f1f';
      const selectedLabelWidth = Math.max(70, placement.category.length * 8);
      const showInlineLabel = true;
      const showHoverTag = false;
      const tagText = labelText;
      const frontSide =
        placement.front_side_local ??
        frontWorldToLocalSide(placement.front_world, placement.rotation_deg) ??
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
                  width={selectedLabelWidth}
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
                  width={selectedLabelWidth}
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

      const selectedOverlaySize = Math.max(w, d);
      const selectedOverlayX = clamp(
        pos.x + selectedOverlaySize / 2 + 14,
        4,
        width - selectedLabelWidth - 4
      );
      const selectedOverlayY = clamp(
        pos.y - selectedOverlaySize / 2,
        4,
        height - 48
      );

      return (
        <React.Fragment key={placement.placement_id}>
          <Group
            x={pos.x}
            y={pos.y}
            rotation={canvasRotationDeg}
            draggable={!isLocked}
            onDragMove={(evt) => {
              if (isLocked) return;
              if (evt.target !== evt.currentTarget) return;
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
            {isSelected && !isLocked && (
              <Group
                x={w / 2 + 18}
                y={0}
                rotation={-canvasRotationDeg}
              >
                <Line
                  points={[-18, 0, 0, 0]}
                  stroke="rgba(51, 51, 51, 0.45)"
                  strokeWidth={1.5}
                  dash={[5, 5]}
                  listening={false}
                />
                <Group
                  draggable
                  y={0}
                  onMouseDown={(evt) => {
                    evt.cancelBubble = true;
                  }}
                  onTouchStart={(evt) => {
                    evt.cancelBubble = true;
                  }}
                  onDragStart={(evt) => {
                    evt.cancelBubble = true;
                  }}
                  onDragMove={(evt) => {
                    evt.cancelBubble = true;
                    const pointer = stageRef.current?.getPointerPosition();
                    if (!pointer) return;
                    const worldPoint = fromCanvasPoint({ x: pointer.x, y: pointer.y });
                    const localX = worldPoint.x - placement.position_mm.x;
                    const localY = worldPoint.y - placement.position_mm.y;
                    const rad = Math.atan2(localY, localX);
                    updatePlacement({ ...placement, rotation_deg: (rad * 180) / Math.PI });
                    evt.target.position({ x: 0, y: 0 });
                  }}
                >
                  <Circle
                    radius={16}
                    fill={ROTATE_CONTROL_FILL}
                    stroke={ROTATE_CONTROL_STROKE}
                    strokeWidth={2.5}
                    shadowColor="#000"
                    shadowBlur={4}
                    shadowOpacity={0.18}
                  />
                  <Line
                    points={[-5, -2, 0, -7, 6, -2]}
                    stroke={ROTATE_CONTROL_HIGHLIGHT}
                    strokeWidth={2.5}
                    lineCap="round"
                    lineJoin="round"
                    closed
                    listening={false}
                  />
                  <Line
                    points={[-6, 5, 0, 9, 7, 5]}
                    stroke={ROTATE_CONTROL_HIGHLIGHT}
                    strokeWidth={2.5}
                    lineCap="round"
                    lineJoin="round"
                    closed
                    listening={false}
                  />
                </Group>
              </Group>
            )}
          </Group>
          {isSelected && (
            <Group
              x={selectedOverlayX}
              y={selectedOverlayY}
              onClick={(evt) => {
                evt.cancelBubble = true;
              }}
              onTap={(evt) => {
                evt.cancelBubble = true;
              }}
            >
              <Group>
                <Rect
                  width={selectedLabelWidth}
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
              <Group
                y={26}
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
            </Group>
          )}
        </React.Fragment>
      );
    });
  };

  return (
    <div
      ref={dropZoneRef}
      className="canvas-drop-zone"
      onDrop={handleDrop}
      onDragOver={(evt) => evt.preventDefault()}
      onMouseDown={handleCanvasMouseDown}
      onTouchStart={handleCanvasTouchStart}
    >
      {selectedPlacement && selectedColorControlPosition ? (
        <div
          className="canvas-object-color-popover"
          style={{
            left: selectedColorControlPosition.left,
            top: selectedColorControlPosition.top,
          }}
          onClick={(evt) => evt.stopPropagation()}
          onMouseDown={(evt) => evt.stopPropagation()}
        >
          <span>Color</span>
          <div className="canvas-object-color-swatches" aria-label="Object color presets">
            {OBJECT_COLOR_SWATCHES.map((color) => (
              <button
                key={color}
                type="button"
                className={
                  color.toLowerCase() === selectedPlacementColor.toLowerCase() ? 'selected' : ''
                }
                style={{ backgroundColor: color }}
                aria-label={`Set object color ${color}`}
                onClick={() => updatePlacement({ ...selectedPlacement, color_hex: color })}
              />
            ))}
          </div>
          <input
            type="color"
            aria-label="Object color"
            value={selectedPlacementColor}
            onChange={(evt) =>
              updatePlacement({ ...selectedPlacement, color_hex: evt.target.value })
            }
          />
        </div>
      ) : null}
      {selectedCornerDeletePosition && selectedCornerIndex !== null ? (
        <button
          type="button"
          className="canvas-corner-delete-button"
          style={{
            left: selectedCornerDeletePosition.left,
            top: selectedCornerDeletePosition.top,
          }}
          onClick={(evt) => {
            evt.stopPropagation();
            deleteRoomCorner(selectedCornerIndex);
          }}
          onMouseDown={(evt) => evt.stopPropagation()}
        >
          Delete
        </button>
      ) : null}
      <Stage
        width={width}
        height={height}
        ref={stageRef}
        onWheel={handleWheel}
        onMouseDown={handleStageMouseDown}
        onTouchStart={handleStageMouseDown}
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
          {renderPlacements()}
          {renderOpenings()}
          {renderWallLabels()}
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

function findNearestCanvasPoint(
  points: Point[],
  target: Point,
  toCanvasPoint: (point: Point) => Point,
  maxDistancePx: number,
): { point: Point; index: number } | null {
  let best: { point: Point; index: number; distance: number } | null = null;
  for (const [index, point] of points.entries()) {
    const canvasPoint = toCanvasPoint(point);
    const distance = Math.hypot(canvasPoint.x - target.x, canvasPoint.y - target.y);
    if (distance <= maxDistancePx && (!best || distance < best.distance)) {
      best = { point, index, distance };
    }
  }
  return best ? { point: best.point, index: best.index } : null;
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
