import React, { useState } from 'react';
import {
  createSavedLayout,
  getPipelineArtifact,
  getPipelineResult,
  getPipelineStatus,
  runPipeline,
} from '../api';
import { useEditorStore } from '../store';
import { LayoutVariant, Placement, Point } from '../types';
import { preparePipelineResultLayouts } from '../utils/pipelineLayouts';

const DEFAULT_PIPELINE_POLL_TIMEOUT_SECONDS = 3600;
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
    currentUser,
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
    appendRunLog,
    clearRunLogs,
    setRunStatus,
    setLastDesignMeta,
    setAdditionRequests,
    setCaseId,
    setPipelineStage,
    layoutVariants,
    variantPickerOpen,
    savedLayouts,
    setSavedLayouts,
    setActiveRootLayoutId,
    selectSavedRender,
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
  const [progressPercent, setProgressPercent] = useState<number | null>(null);
  const [progressLabel, setProgressLabel] = useState<string | null>(null);
  const [readyNotice, setReadyNotice] = useState<string | null>(null);
  const [layoutName, setLayoutName] = useState('');
  const [savingLayout, setSavingLayout] = useState(false);
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const galleryOpen = variantPickerOpen && layoutVariants.length > 0;

  type LoadFinalResult = {
    variantCount: number;
  };

  const applyVariant = (variant: LayoutVariant) => {
    if (variant.design) {
      setFloorplan(variant.design.floorplan_geometry);
      setDesign(variant.design);
    }
    setStyledResult(variant.styled_result);
    selectLayoutVariant(variant.variant_id);
    setViewMode('2d');
    setVariantPickerOpen(false);
  };

  const handleGenerate = async () => {
    if (!floorplan) return;
    const manualPlacements = collectManualPlacements(design?.placements ?? []);
    setLoading(true);
    setError(null);
    setProgressPercent(0);
    setProgressLabel('Starting layout generation');
    setReadyNotice(null);
    clearRunLogs();
    setAdditionRequests([]);
    setRunStatus('Queued pipeline build', null);
    clearLayoutVariants();
    setStyledResult(null);
    setDesign({ floorplan_geometry: floorplan, placements: [] });
    setViolations([]);
    setLastDesignMeta({ iteration: null, attempt: null });
    setVariantPickerOpen(false);
    setViewMode('2d');
    try {
      const caseResp = await runPipeline({
        user_id: currentUser?.id || tenantId || 'demo_user',
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
      const caseId = caseResp.case_id;
      let done = false;
      let lastStage = '';
      let lastProgressLine = '';
      let lastLoadedVariantCount = 0;
      let lastActionIndex = 0;
      const loadFinal = async (caseId: string): Promise<LoadFinalResult> => {
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
        if (
          prepared.styledResult &&
          typeof prepared.styledResult === 'object' &&
          (Array.isArray((prepared.styledResult as any).objects) ||
            (prepared.styledResult as any).room)
        ) {
          setStyledResult(prepared.styledResult);
        }
        if (prepared.design) {
          setDesign(prepared.design);
        }
        if (prepared.variants.length > 0) {
          const primaryVariant =
            prepared.variants.find(
              (variant) => variant.variant_id === prepared.selectedVariantId
            ) ?? prepared.variants[0];
          setLayoutVariants(prepared.variants, primaryVariant.variant_id);
          applyVariant(primaryVariant);
          setProgressPercent(100);
          setProgressLabel(
            `Option 1/${prepared.variants.length} loaded`
          );
          setReadyNotice(formatReadyOptionNotice(prepared.variants.length));
        } else {
          clearLayoutVariants();
          setVariantPickerOpen(false);
          setProgressPercent(100);
          setProgressLabel('Layout ready');
          setReadyNotice(null);
        }
        setViolations([]);
        setLastDesignMeta({ iteration: null, attempt: null });
        return {
          variantCount: prepared.variants.length,
        };
      };
      const maxPollSeconds = PIPELINE_POLL_TIMEOUT_SECONDS;
      for (let i = 0; i < maxPollSeconds && !done; i += 1) {
        const status = await getPipelineStatus(caseId);
        const progressLabel = formatPipelineProgress(status);
        setProgressPercent(computePipelineProgressPercent(status));
        if (progressLabel) {
          setProgressLabel(progressLabel);
        }
        const readyOptionNotice = formatStatusReadyOptionNotice(status);
        if (readyOptionNotice) {
          setReadyNotice(readyOptionNotice);
        }
        setRunStatus(progressLabel, null);
        if (status.stage) {
          setPipelineStage(status.stage);
        }
        const statusActions = Array.isArray(status.actions) ? status.actions : [];
        if (statusActions.length > lastActionIndex) {
          for (const action of statusActions.slice(lastActionIndex)) {
            const line = formatPipelineStatusAction(action);
            if (line) {
              appendRunLog(line);
            }
          }
          lastActionIndex = statusActions.length;
        } else if (statusActions.length === 0) {
          if (status.stage && status.stage !== lastStage) {
            appendRunLog(`Stage: ${status.stage}`);
            lastStage = status.stage;
          }
          if (progressLabel && progressLabel !== lastProgressLine) {
            appendRunLog(progressLabel);
            lastProgressLine = progressLabel;
          }
        }
        if (status.stage === 'error') {
          setError(status.error ?? 'Pipeline error');
          setProgressLabel(status.error ?? 'Pipeline error');
          setProgressPercent(null);
          setReadyNotice(null);
          done = true;
          break;
        }
        if (status.stage === 'done') {
          const finalState = await loadFinal(caseId);
          if (finalState.variantCount > lastLoadedVariantCount) {
            appendRunLog(
              `Loaded ${finalState.variantCount} final design option${finalState.variantCount === 1 ? '' : 's'}.`
            );
            lastLoadedVariantCount = finalState.variantCount;
          }
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
          const progressLabel = formatPipelineProgress(status);
          setProgressPercent(100);
          if (progressLabel) {
            setProgressLabel(progressLabel);
          }
          setRunStatus(progressLabel, null);
          setPipelineStage('done');
          const statusActions = Array.isArray(status.actions) ? status.actions : [];
          if (statusActions.length > lastActionIndex) {
            for (const action of statusActions.slice(lastActionIndex)) {
              const line = formatPipelineStatusAction(action);
              if (line) {
                appendRunLog(line);
              }
            }
            lastActionIndex = statusActions.length;
          }
          const finalState = await loadFinal(caseId);
          appendRunLog(
            `Loaded ${finalState.variantCount} final design option${finalState.variantCount === 1 ? '' : 's'}.`
          );
          done = true;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generate failed');
      setRunStatus('error', null);
      setProgressLabel(err instanceof Error ? err.message : 'Generate failed');
      setProgressPercent(null);
      setReadyNotice(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveLayout = async () => {
    if (!floorplan) {
      setLayoutError('Create or load a floorplan before saving.');
      return;
    }
    setSavingLayout(true);
    setLayoutError(null);
    try {
      const saved = await createSavedLayout({
        name: layoutName.trim() || `Layout ${new Date().toLocaleString()}`,
        floorplan_json: floorplan as unknown as Record<string, unknown>,
        design_json: (design ?? null) as unknown as Record<string, unknown> | null,
        styled_result_json:
          (styledResult ?? null) as Record<string, unknown> | null,
        meta: {
          saved_by_user_id: currentUser?.id ?? 'user',
        },
      });
      setSavedLayouts([saved, ...savedLayouts.filter((item) => item.id !== saved.id)]);
      setActiveRootLayoutId(saved.id);
      selectSavedRender(null);
      setLayoutName('');
      appendRunLog(`Saved layout "${saved.name}" to the current account.`);
    } catch (saveError) {
      setLayoutError(saveError instanceof Error ? saveError.message : 'Failed to save layout.');
    } finally {
      setSavingLayout(false);
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
      <div className="toolbar-save-layout">
        <input
          type="text"
          value={layoutName}
          onChange={(event) => setLayoutName(event.target.value)}
          placeholder="Layout name"
          aria-label="Layout name"
        />
        <button
          className="primary-button"
          disabled={savingLayout || !floorplan}
          onClick={handleSaveLayout}
          type="button"
        >
          {savingLayout ? 'Saving...' : 'Save'}
        </button>
      </div>
      {(loading || progressLabel || readyNotice) && (
        <div className={`toolbar-loading${error ? ' error' : ''}`} aria-live="polite">
          {loading ? <span className="toolbar-spinner" aria-hidden="true" /> : null}
          <span className="toolbar-loading-copy">
            <strong>
              {readyNotice ?? progressLabel ?? 'Working'}
            </strong>
            {typeof progressPercent === 'number' ? (
              <span>{Math.round(progressPercent)}%</span>
            ) : null}
          </span>
        </div>
      )}
      {error && <span className="toolbar-status error">{error}</span>}
      {layoutError && <span className="toolbar-status error">{layoutError}</span>}
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

function formatPipelineProgress(status: {
  stage?: string | null;
  message?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
}): string | null {
  const message = typeof status.message === 'string' ? status.message.trim() : '';
  if (message) {
    return message;
  }
  const stage = typeof status.stage === 'string' ? status.stage.trim() : '';
  if (!stage) {
    return null;
  }
  const current =
    typeof status.progress_current === 'number' && Number.isFinite(status.progress_current)
      ? status.progress_current
      : null;
  const total =
    typeof status.progress_total === 'number' && Number.isFinite(status.progress_total)
      ? status.progress_total
      : null;
  if (current !== null && total !== null && total > 0) {
    return `${stage} ${current}/${total}`;
  }
  return stage;
}

function formatPipelineStatusAction(action: {
  stage?: string | null;
  message?: string | null;
  updated_at_utc?: string | null;
  error?: string | null;
}): string | null {
  const stage = typeof action.stage === 'string' ? action.stage.trim() : '';
  const error = typeof action.error === 'string' ? action.error.trim() : '';
  const message = typeof action.message === 'string' ? action.message.trim() : '';
  const body = error || message || stage;
  if (!body) {
    return null;
  }
  if (!stage || body === stage) {
    return body;
  }
  return `${stage}: ${body}`;
}

function computePipelineProgressPercent(status: {
  stage?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
}): number | null {
  const stage = String(status.stage ?? '').trim();
  if (stage === 'done') {
    return 100;
  }
  if (stage === 'error') {
    return null;
  }
  const stageRanges: Record<string, { start: number; end: number }> = {
    queued: { start: 2, end: 6 },
    running: { start: 6, end: 10 },
    room_interpreter: { start: 10, end: 18 },
    stylist_style_policy: { start: 18, end: 24 },
    cluster_forge: { start: 24, end: 32 },
    tier_count: { start: 32, end: 38 },
    cluster_output_merger: { start: 38, end: 42 },
    cluster_relation_planner: { start: 42, end: 54 },
    solver: { start: 54, end: 74 },
    stylist: { start: 74, end: 92 },
    layout_variants: { start: 92, end: 98 },
  };
  const range = stageRanges[stage] ?? { start: 6, end: 12 };
  const current =
    typeof status.progress_current === 'number' && Number.isFinite(status.progress_current)
      ? status.progress_current
      : null;
  const total =
    typeof status.progress_total === 'number' && Number.isFinite(status.progress_total)
      ? status.progress_total
      : null;
  if (current === null || total === null || total <= 0) {
    return range.start;
  }
  const ratio = Math.min(1, Math.max(0, current / total));
  return range.start + (range.end - range.start) * ratio;
}

function formatStatusReadyOptionNotice(status: {
  stage?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
}): string | null {
  if (status.stage !== 'stylist') {
    return null;
  }
  const current =
    typeof status.progress_current === 'number' && Number.isFinite(status.progress_current)
      ? status.progress_current
      : null;
  const total =
    typeof status.progress_total === 'number' && Number.isFinite(status.progress_total)
      ? status.progress_total
      : null;
  if (current === null || total === null || total <= 1 || current <= 1) {
    return null;
  }
  return `Option ${Math.min(current, total)}/${total} ready`;
}

function formatReadyOptionNotice(variantCount: number): string | null {
  return variantCount > 1 ? `Option 2/${variantCount} ready` : null;
}
