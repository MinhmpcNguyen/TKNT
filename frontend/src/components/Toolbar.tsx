import React, { useState } from 'react';
import { checkDesign, getClusterComposer, getClusterList, getPipelineArtifact, getPipelineResult, getPipelineStatus, runPipeline } from '../api';
import { useEditorStore } from '../store';
import { FloorplanGeometry, Point, Wall } from '../types';
import { normalizePoints } from '../utils/geometry';
import { getBboxCenter, getLocalFootprintFromWorldBbox, getNumericBbox } from '../utils/styledLayout';

const DEFAULT_PIPELINE_POLL_TIMEOUT_SECONDS = 1800;
const PIPELINE_POLL_TIMEOUT_SECONDS = Math.max(
  60,
  Math.floor(
    Number(
      (import.meta as any).env?.VITE_PIPELINE_POLL_TIMEOUT_SECONDS ??
      DEFAULT_PIPELINE_POLL_TIMEOUT_SECONDS
    )
  )
);

export const Toolbar: React.FC = () => {
  const {
    design,
    floorplan,
    tenantId,
    setDesign,
    setStyledResult,
    setViolations,
    userInput,
    updateUserInput,
    roomEditMode,
    setRoomEditMode,
    baseFloorplan,
    selectedCornerIndex,
    selectCorner,
    selectWall,
    setFloorplan,
    drawWallMode,
    setDrawWallMode,
    setRunLogs,
    appendRunLog,
    clearRunLogs,
    setRunStatus,
    setLastDesignMeta,
    setAdditionRequests,
    setCaseId,
    setPipelineStage,
    setClusterPreviews,
    updateClusterPreview,
    styledResult,
    viewMode,
    setViewMode,
  } = useEditorStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!floorplan) return;
    setLoading(true);
    setError(null);
    clearRunLogs();
    setAdditionRequests([]);
    setRunStatus('running', null);
    setStyledResult(null);
    setViewMode('2d');
    try {
      const caseResp = await runPipeline({
        user_id: tenantId || 'demo_user',
        input_payload: {
          tenant_id: tenantId,
          user_input: userInput,
          floorplan_geometry: floorplan,
          constraints: {
            description: userInput.description,
            room_type: userInput.room_type,
            floor_area_m2: userInput.floor_area_m2,
            height: userInput.height,
            shape_points: userInput.shape_points,
            location_floor: userInput.location_floor,
            bathrooms: userInput.bathrooms,
            windows: userInput.windows,
            window_direction: userInput.window_direction,
          },
          max_iterations: 12,
        },
        description: userInput.description,
        special_notes: userInput.special_description,
      });
      appendRunLog(`Case started: ${caseResp.case_id}`);
      setCaseId(caseResp.case_id);
      setPipelineStage('queued');
      setClusterPreviews({});
      const caseId = caseResp.case_id;
      let done = false;
      let lastStage = '';
      const loadFinal = async (caseId: string) => {
        let roomFloorplan: FloorplanGeometry | null = null;
        let shiftDx = 0;
        let shiftDy = 0;
        try {
          const roomArtifact = await getPipelineArtifact(caseId, 'room_interpreter');
          const roomPayload = roomArtifact?.payload ?? roomArtifact;
          const room = roomPayload?.room ?? {};
          const polyRaw = Array.isArray(room?.polygon_ccw) ? room.polygon_ccw : [];
          const normalized = normalizePoints(polyRaw);
          shiftDx = normalized.dx;
          shiftDy = normalized.dy;
          const shiftPoint = (p: Point) => ({ x: p.x - normalized.dx, y: p.y - normalized.dy });
          const roomPoints = normalized.points;
          const wallThickness = 100;
          const walls: Wall[] = roomPoints.map((p, idx) => {
            const next = roomPoints[(idx + 1) % roomPoints.length];
            return {
              wall_id: `wall_${idx + 1}`,
              p1_mm: p,
              p2_mm: next,
              thickness_mm: wallThickness,
              room_id: room?.room_id ?? 'room_1',
            };
          });
          const doors = (roomPayload?.openings?.doors ?? []).map((door: any, idx: number) => {
            const seg = door?.segment_mm ?? [];
            const p1 = seg?.[0] ?? { x: 0, y: 0 };
            const p2 = seg?.[1] ?? { x: 0, y: 0 };
            return {
              door_id: door?.id ?? `door_${idx + 1}`,
              start_mm: shiftPoint(p1),
              end_mm: shiftPoint(p2),
            };
          });
          const windows = (roomPayload?.openings?.windows ?? []).map((win: any, idx: number) => {
            const seg = win?.segment_mm ?? [];
            const p1 = seg?.[0] ?? { x: 0, y: 0 };
            const p2 = seg?.[1] ?? { x: 0, y: 0 };
            return {
              window_id: win?.id ?? `window_${idx + 1}`,
              start_mm: shiftPoint(p1),
              end_mm: shiftPoint(p2),
            };
          });
          roomFloorplan = {
            room: {
              room_id: room?.room_id ?? 'room_1',
              name: room?.name ?? 'Room',
              polygon_mm: roomPoints,
            },
            walls,
            openings: { doors, windows },
          };
          setFloorplan(roomFloorplan);
        } catch {
          roomFloorplan = floorplan;
        }
        const result = await getPipelineResult(caseId);
        const stylist = result.result ?? result;
        const objects = stylist?.objects ?? [];
        // Store a shifted version of stylist output for 3D preview (match UI-normalized floorplan coordinates).
        try {
          const shiftPoint = (p: any): Point => ({
            x: Number(p?.x ?? 0) - shiftDx,
            y: Number(p?.y ?? 0) - shiftDy,
          });
          const shiftBBox = (bbox: any) => {
            if (!bbox) return bbox;
            return {
              ...bbox,
              min_x: Number(bbox.min_x ?? 0) - shiftDx,
              min_y: Number(bbox.min_y ?? 0) - shiftDy,
              max_x: Number(bbox.max_x ?? 0) - shiftDx,
              max_y: Number(bbox.max_y ?? 0) - shiftDy,
            };
          };
          const shiftedRoom = stylist?.room
            ? {
              ...stylist.room,
              polygon_ccw: Array.isArray(stylist.room.polygon_ccw)
                ? stylist.room.polygon_ccw.map(shiftPoint)
                : stylist.room.polygon_ccw,
              obstacles: Array.isArray(stylist.room.obstacles)
                ? stylist.room.obstacles.map((ob: any) => ({
                  ...ob,
                  polygon_ccw: Array.isArray(ob?.polygon_ccw) ? ob.polygon_ccw.map(shiftPoint) : ob?.polygon_ccw,
                }))
                : stylist.room.obstacles,
              openings: stylist.room.openings
                ? {
                  ...stylist.room.openings,
                  doors: Array.isArray(stylist.room.openings?.doors)
                    ? stylist.room.openings.doors.map((d: any) => ({
                      ...d,
                      segment_mm: Array.isArray(d?.segment_mm)
                        ? d.segment_mm.map(shiftPoint)
                        : d?.segment_mm,
                    }))
                    : stylist.room.openings?.doors,
                  windows: Array.isArray(stylist.room.openings?.windows)
                    ? stylist.room.openings.windows.map((w: any) => ({
                      ...w,
                      segment_mm: Array.isArray(w?.segment_mm)
                        ? w.segment_mm.map(shiftPoint)
                        : w?.segment_mm,
                    }))
                    : stylist.room.openings?.windows,
                }
                : stylist.room.openings,
            }
            : null;
          const shiftedObjects = Array.isArray(objects)
            ? objects.map((obj: any) => ({
              ...obj,
              polygon_ccw: Array.isArray(obj?.polygon_ccw) ? obj.polygon_ccw.map(shiftPoint) : obj?.polygon_ccw,
              bbox: shiftBBox(obj?.bbox),
            }))
            : objects;
          setStyledResult({ ...stylist, room: shiftedRoom, objects: shiftedObjects });
        } catch {
          setStyledResult(stylist);
        }
        const placements = objects.map((obj: any, idx: number) => {
          const rotationDeg = Number(obj.rotation_ccw ?? 0);
          const bbox = getNumericBbox(obj);
          const footprint = getLocalFootprintFromWorldBbox(bbox, rotationDeg);
          const center = getBboxCenter(bbox);
          return {
            placement_id: obj.instance_id ?? `obj_${idx}`,
            asset_id: obj.instance_id ?? obj.object_type ?? `obj_${idx}`,
            category: obj.object_type ?? 'item',
            position_mm: { x: center.x - shiftDx, y: center.y - shiftDy },
            rotation_deg: rotationDeg,
            footprint_mm: footprint,
            height_mm: 0,
            color_hex: obj.color_hex ?? '#cfcac2',
            anchor: 'center',
          };
        });
        setDesign({ floorplan_geometry: roomFloorplan ?? floorplan, placements });
        setViolations([]);
        setLastDesignMeta({ iteration: null, attempt: null });
      };
      const maxPollSeconds = PIPELINE_POLL_TIMEOUT_SECONDS;
      for (let i = 0; i < maxPollSeconds && !done; i += 1) {
        const status = await getPipelineStatus(caseId);
        setRunStatus(status.stage ?? null, null);
        if (status.stage && status.stage !== lastStage) {
          appendRunLog(`Stage: ${status.stage}`);
          setPipelineStage(status.stage);
          lastStage = status.stage;
        }
        if (status.stage === 'error') {
          setError(status.error ?? 'Pipeline error');
          done = true;
          break;
        }
        if (status.stage === 'cluster_composer' || status.stage === 'cluster_placer' || status.stage === 'absolute_layout' || status.stage === 'stylist' || status.stage === 'done') {
          try {
            const list = await getClusterList(caseId);
            for (const clusterId of list.clusters ?? []) {
              const preview = await getClusterComposer(caseId, clusterId);
              updateClusterPreview(clusterId, preview);
            }
          } catch {
            // ignore preview errors
          }
        }
        if (status.stage === 'done') {
          await loadFinal(caseId);
          done = true;
          break;
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (!done) {
        appendRunLog(
          `Pipeline still running after ${PIPELINE_POLL_TIMEOUT_SECONDS}s of polling. You can wait and refresh, or re-open this case.`
        );
        const status = await getPipelineStatus(caseId);
        if (status.stage === 'done') {
          setRunStatus(status.stage ?? null, null);
          appendRunLog('Stage: done');
          setPipelineStage('done');
          await loadFinal(caseId);
          done = true;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generate failed');
      setRunStatus('error', null);
    } finally {
      setLoading(false);
    }
  };

  const handleCheck = async () => {
    if (!floorplan || !design) return;
    setLoading(true);
    setError(null);
    try {
      const violations = await checkDesign({
        floorplan_geometry: floorplan,
        placements: design.placements,
        constraints: {
          description: userInput.description,
          room_type: userInput.room_type,
          floor_area_m2: userInput.floor_area_m2,
          height: userInput.height,
          shape_points: userInput.shape_points,
          location_floor: userInput.location_floor,
          bathrooms: userInput.bathrooms,
          windows: userInput.windows,
          window_direction: userInput.window_direction,
        },
      });
      setViolations(violations);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Check failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="toolbar">
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
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? 'Working…' : 'Generate'}
      </button>
      <button onClick={handleCheck} disabled={loading || !design}>
        Check
      </button>
      <button
        onClick={() => setViewMode('2d')}
        className={viewMode === '2d' ? 'active' : ''}
        disabled={loading}
      >
        2D
      </button>
      <button
        onClick={() => setViewMode('3d')}
        className={viewMode === '3d' ? 'active' : ''}
        disabled={loading || !floorplan || (!styledResult && !design)}
      >
        3D
      </button>
      <button
        onClick={() => {
          if (roomEditMode) {
            setRoomEditMode(false);
            setDrawWallMode(false);
            if (floorplan) {
              setTimeout(() => setFloorplan({ ...floorplan }), 0);
            }
          } else {
            setRoomEditMode(true);
          }
        }}
        className={roomEditMode ? 'active' : ''}
      >
        {roomEditMode ? 'Finish Room Edit' : 'Edit Room'}
      </button>
      <button
        onClick={() => setDrawWallMode(!drawWallMode)}
        disabled={!roomEditMode}
        className={drawWallMode ? 'active' : ''}
      >
        Draw Line
      </button>
      <button
        onClick={() => {
          if (!floorplan) return;
          const points = floorplan.room.polygon_mm;
          if (points.length <= 3) return;
          if (selectedCornerIndex === null) return;
          const index = selectedCornerIndex;
          const nextPoints = points.filter((_, i) => i !== index);
          const removed = points[index];
          const nextWalls = floorplan.walls.filter((wall) => {
            const p1 = wall.p1_mm;
            const p2 = wall.p2_mm;
            return !pointsEqual(p1, removed) && !pointsEqual(p2, removed);
          });
          setFloorplan({ ...floorplan, room: { ...floorplan.room, polygon_mm: nextPoints }, walls: nextWalls });
          selectCorner(null);
          selectWall(null);
        }}
        disabled={!roomEditMode || selectedCornerIndex === null}
      >
        Delete Point
      </button>
      <button
        onClick={() => {
          if (baseFloorplan) {
            setFloorplan(baseFloorplan);
            selectCorner(null);
            selectWall(null);
          }
        }}
      >
        Reset Room
      </button>
      {error && <span className="toolbar-status error">{error}</span>}
    </div>
  );
};

function pointsEqual(a: Point, b: Point, tolerance = 1): boolean {
  return Math.abs(a.x - b.x) <= tolerance && Math.abs(a.y - b.y) <= tolerance;
}
