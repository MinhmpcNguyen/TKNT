import React, { useState } from 'react';
import { checkDesign, getClusterComposer, getClusterList, getPipelineArtifact, getPipelineResult, getPipelineStatus, runPipeline } from '../api';
import { useEditorStore } from '../store';
import { Placement, Point } from '../types';
import { preparePipelineResultLayouts } from '../utils/pipelineLayouts';

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
    layoutVariants,
    selectedVariantId,
    variantPickerOpen,
    setLayoutVariants,
    selectLayoutVariant,
    setVariantPickerOpen,
    clearLayoutVariants,
    styledResult,
    viewMode,
    setViewMode,
  } = useEditorStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const galleryOpen = variantPickerOpen && layoutVariants.length > 0;

  const applyVariantToEditor = (variantId: string | null) => {
    const variant =
      layoutVariants.find((item) => item.variant_id === variantId) ?? layoutVariants[0];
    if (!variant) {
      return;
    }
    if (variant.design) {
      setFloorplan(variant.design.floorplan_geometry);
      setDesign(variant.design);
    }
    setStyledResult(variant.styled_result);
    selectLayoutVariant(variant.variant_id);
    setViewMode('2d');
    setVariantPickerOpen(false);
    appendRunLog(`Selected layout: ${variant.label}`);
  };

  const handleGenerate = async () => {
    if (!floorplan) return;
    const manualPlacements = collectManualPlacements(design?.placements ?? []);
    setLoading(true);
    setError(null);
    clearRunLogs();
    setAdditionRequests([]);
    setRunStatus('running', null);
    clearLayoutVariants();
    setStyledResult(null);
    setVariantPickerOpen(false);
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
          manual_placements: manualPlacements,
          max_iterations: 12,
        },
        description: userInput.description,
        special_notes: userInput.special_description,
      });
      appendRunLog(`Case started: ${caseResp.case_id}`);
      if (manualPlacements.length > 0) {
        appendRunLog(`Protected ${manualPlacements.length} manual furniture placement(s) during generation.`);
      }
      setCaseId(caseResp.case_id);
      setPipelineStage('queued');
      setClusterPreviews({});
      const caseId = caseResp.case_id;
      let done = false;
      let lastStage = '';
      const loadFinal = async (caseId: string) => {
        let roomPayload: any = null;
        try {
          const roomArtifact = await getPipelineArtifact(caseId, 'room_interpreter');
          roomPayload = roomArtifact?.payload ?? roomArtifact;
        } catch {
          roomPayload = null;
        }
        const result = await getPipelineResult(caseId);
        const stylist = result.result ?? result;
        const prepared = preparePipelineResultLayouts({
          roomPayload,
          resultPayload: stylist,
          fallbackFloorplan: floorplan ?? null,
        });
        if (prepared.floorplan) {
          setFloorplan(prepared.floorplan);
        }
        setStyledResult(prepared.styledResult);
        if (prepared.design) {
          setDesign(prepared.design);
        }
        if (prepared.variants.length > 0) {
          setLayoutVariants(prepared.variants, prepared.selectedVariantId);
          setVariantPickerOpen(true);
          appendRunLog(`Loaded ${prepared.variants.length} layout variants`);
        } else {
          clearLayoutVariants();
          setVariantPickerOpen(false);
        }
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
      <button onClick={handleCheck} disabled={loading || galleryOpen || !design}>
        Check
      </button>
      {layoutVariants.length > 0 && galleryOpen ? (
        <button
          onClick={() => applyVariantToEditor(selectedVariantId)}
          disabled={loading || !selectedVariantId}
        >
          Choose Layout
        </button>
      ) : null}
      {layoutVariants.length > 0 && !galleryOpen ? (
        <button
          onClick={() => {
            setVariantPickerOpen(true);
            setViewMode('2d');
          }}
          disabled={loading}
        >
          Back to {layoutVariants.length} Layouts
        </button>
      ) : null}
      <button
        onClick={() => setViewMode('2d')}
        className={viewMode === '2d' ? 'active' : ''}
        disabled={loading || galleryOpen}
      >
        2D
      </button>
      <button
        onClick={() => setViewMode('3d')}
        className={viewMode === '3d' ? 'active' : ''}
        disabled={loading || galleryOpen || !floorplan || (!styledResult && !design)}
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
        disabled={galleryOpen}
      >
        {roomEditMode ? 'Finish Room Edit' : 'Edit Room'}
      </button>
      <button
        onClick={() => setDrawWallMode(!drawWallMode)}
        disabled={galleryOpen || !roomEditMode}
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
        disabled={galleryOpen || !roomEditMode || selectedCornerIndex === null}
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
        disabled={galleryOpen}
      >
        Reset Room
      </button>
      {error && <span className="toolbar-status error">{error}</span>}
    </div>
  );
};

function collectManualPlacements(placements: Placement[]): Placement[] {
  return placements
    .filter((placement) => placement.source === 'manual')
    .map((placement) => ({
      ...placement,
      source: 'manual',
      constraint_mode:
        placement.constraint_mode === 'locked' ? 'locked' : 'pinned',
    }));
}

function pointsEqual(a: Point, b: Point, tolerance = 1): boolean {
  return Math.abs(a.x - b.x) <= tolerance && Math.abs(a.y - b.y) <= tolerance;
}
