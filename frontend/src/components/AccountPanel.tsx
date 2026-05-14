import React, { useState } from 'react';
import { createSavedLayout } from '../api';
import { useEditorStore } from '../store';
import {
  DesignPlanV1,
  FloorplanGeometry,
  GeneratedRenderRecord,
  SavedLayoutRecord,
} from '../types';

function canReuseSavedRender(render: GeneratedRenderRecord): boolean {
  if (render.source !== 'snapshot_render') {
    return false;
  }
  const visibleObjects = render.meta.visible_objects;
  return Array.isArray(visibleObjects) && visibleObjects.length > 0;
}

function parseTimestampMs(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function resolveSavedLayoutIdForRender(
  render: GeneratedRenderRecord,
  savedLayouts: SavedLayoutRecord[]
): string | null {
  const explicitRootLayoutId = render.meta.root_layout_id;
  if (
    typeof explicitRootLayoutId === 'string' &&
    explicitRootLayoutId.trim()
  ) {
    return explicitRootLayoutId.trim();
  }
  if (savedLayouts.length === 0) {
    return null;
  }

  const renderTimeMs = parseTimestampMs(render.created_at);
  const layoutsWithTime = savedLayouts
    .map((layout) => ({
      layout,
      timeMs: parseTimestampMs(layout.updated_at ?? layout.created_at),
    }))
    .filter(
      (item): item is { layout: SavedLayoutRecord; timeMs: number } =>
        item.timeMs !== null
    )
    .sort((left, right) => right.timeMs - left.timeMs);

  if (renderTimeMs !== null) {
    const earlierOrEqual = layoutsWithTime.find((item) => item.timeMs <= renderTimeMs);
    if (earlierOrEqual) {
      return earlierOrEqual.layout.id;
    }
  }

  return layoutsWithTime[0]?.layout.id ?? savedLayouts[0]?.id ?? null;
}

export const AccountPanel: React.FC = () => {
  const {
    currentUser,
    floorplan,
    design,
    styledResult,
    savedLayouts,
    savedRenders,
    activeRootLayoutId,
    selectedSavedRenderId,
    setSavedLayouts,
    setFloorplan,
    setDesign,
    setStyledResult,
    clearLayoutVariants,
    setViewMode,
    setActiveRootLayoutId,
    selectSavedRender,
    appendRunLog,
  } = useEditorStore();
  const [layoutName, setLayoutName] = useState('');
  const [savingLayout, setSavingLayout] = useState(false);
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const visibleSavedRenders = activeRootLayoutId
    ? savedRenders.filter(
        (render) =>
          resolveSavedLayoutIdForRender(render, savedLayouts) === activeRootLayoutId
      )
    : savedRenders;

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

  const loadLayout = (layout: SavedLayoutRecord) => {
    const nextFloorplan = layout.floorplan_json as unknown as FloorplanGeometry;
    const nextDesign = layout.design_json as unknown as DesignPlanV1 | null;
    setFloorplan(nextFloorplan);
    setDesign(
      nextDesign ?? {
        floorplan_geometry: nextFloorplan,
        placements: [],
      }
    );
    setStyledResult(layout.styled_result_json ?? null);
    clearLayoutVariants();
    setActiveRootLayoutId(layout.id);
    selectSavedRender(null);
    setViewMode('2d');
    appendRunLog(`Loaded saved layout "${layout.name}".`);
  };

  const handleReuseSavedRender = (render: GeneratedRenderRecord) => {
    if (!canReuseSavedRender(render)) {
      appendRunLog(
        `Saved render "${render.id}" is missing edit metadata and cannot be reopened.`
      );
      return;
    }
    setViewMode('3d');
    selectSavedRender(render.id);
    appendRunLog(`Loaded saved render "${render.id}" into the 3D editor for more edits.`);
  };

  return (
    <section className="panel account-panel">
      <h3>My Workspace</h3>
      <div className="account-summary">
        <strong>{currentUser?.display_name || currentUser?.email}</strong>
        <span>{currentUser?.email}</span>
      </div>

      <div className="account-section">
        <div className="account-section-header">
          <strong>Save Layout</strong>
        </div>
        <input
          type="text"
          value={layoutName}
          onChange={(event) => setLayoutName(event.target.value)}
          placeholder="Conference room v2"
        />
        {layoutError ? <div className="inline-error">{layoutError}</div> : null}
        <button
          className="primary-button"
          disabled={savingLayout}
          onClick={handleSaveLayout}
          type="button"
        >
          {savingLayout ? 'Saving...' : 'Save current layout'}
        </button>
      </div>

      <div className="account-section">
        <div className="account-section-header">
          <strong>Saved Layouts</strong>
          <span>{savedLayouts.length}</span>
        </div>
        <div className="account-list">
          {savedLayouts.length === 0 ? (
            <p className="muted">No layouts saved yet.</p>
          ) : (
            savedLayouts.map((layout) => (
              <div key={layout.id} className="account-list-item">
                <div>
                  <div className="account-list-title">{layout.name}</div>
                  <div className="account-list-meta">
                    {layout.updated_at || layout.created_at || 'Saved recently'}
                  </div>
                </div>
                <button onClick={() => loadLayout(layout)} type="button">
                  Load
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="account-section">
        <div className="account-section-header">
          <strong>Saved Renders</strong>
          <span>{visibleSavedRenders.length}</span>
        </div>
        {activeRootLayoutId ? (
          <p className="muted">
            Showing renders linked to the loaded root layout.
          </p>
        ) : null}
        <div className="account-render-grid">
          {visibleSavedRenders.length === 0 ? (
            <p className="muted">
              {activeRootLayoutId
                ? 'No renders are linked to this root layout yet.'
                : 'Your Gemini renders will appear here.'}
            </p>
          ) : (
            visibleSavedRenders.map((render) => (
              <div
                key={render.id}
                className={`saved-render-card ${
                  selectedSavedRenderId === render.id ? 'selected' : ''
                }`}
              >
                <img src={render.file_url} alt={render.prompt} />
                <div className="saved-render-meta">
                  <strong>{render.model_name}</strong>
                  <span>{render.source}</span>
                  <p>{render.prompt}</p>
                  <div className="saved-render-actions">
                    <button
                      className="primary-button"
                      disabled={!canReuseSavedRender(render)}
                      onClick={() => handleReuseSavedRender(render)}
                      type="button"
                    >
                      {selectedSavedRenderId === render.id ? 'Loaded for edit' : 'Use for edit'}
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
};
