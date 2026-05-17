import React, { useMemo, useState } from 'react';
import { useEditorStore } from '../store';
import {
  DesignPlanV1,
  FloorplanGeometry,
  GeneratedRenderRecord,
  SavedLayoutRecord,
} from '../types';

function canReuseSavedRender(render: GeneratedRenderRecord): boolean {
  return render.source === 'snapshot_render';
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
    savedLayouts,
    savedRenders,
    activeRootLayoutId,
    selectedSavedRenderId,
    setFloorplan,
    setDesign,
    setStyledResult,
    clearLayoutVariants,
    setViewMode,
    setActiveRootLayoutId,
    selectSavedRender,
    appendRunLog,
  } = useEditorStore();
  const [layoutSearchQuery, setLayoutSearchQuery] = useState('');
  const [savedLayoutsOpen, setSavedLayoutsOpen] = useState(true);
  const [savedRendersOpen, setSavedRendersOpen] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const visibleSavedRenders = activeRootLayoutId
    ? savedRenders.filter(
        (render) =>
          resolveSavedLayoutIdForRender(render, savedLayouts) === activeRootLayoutId
      )
    : savedRenders;
  const filteredSavedLayouts = useMemo(() => {
    const query = layoutSearchQuery.trim().toLowerCase();
    if (!query) {
      return savedLayouts;
    }
    return savedLayouts.filter((layout) => {
      const haystack = [
        layout.name,
        layout.created_at ?? '',
        layout.updated_at ?? '',
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [layoutSearchQuery, savedLayouts]);

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
    setSavedLayoutsOpen(false);
    appendRunLog(`Loaded saved layout "${layout.name}".`);
  };

  const handleReuseSavedRender = (render: GeneratedRenderRecord) => {
    if (!canReuseSavedRender(render)) {
      appendRunLog(
        `Saved render "${render.id}" is not a 3D snapshot render and cannot be reopened.`
      );
      return;
    }
    setViewMode('3d');
    if (selectedSavedRenderId === render.id) {
      selectSavedRender(null);
      window.setTimeout(() => selectSavedRender(render.id), 0);
    } else {
      selectSavedRender(render.id);
    }
    appendRunLog(`Loaded saved render "${render.id}" into the 3D editor for more edits.`);
  };

  return (
    <section className={`panel account-panel collapsible-panel${isOpen ? ' open' : ''}`}>
      <button
        className="panel-collapse-trigger"
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
      >
        <span>
          <span className="panel-collapse-title">My Workspace</span>
          <span className="panel-collapse-subtitle">
            {savedLayouts.length} layouts · {visibleSavedRenders.length} renders
          </span>
        </span>
        <span className="panel-collapse-caret">{isOpen ? 'Hide' : 'Open'}</span>
      </button>

      {isOpen ? (
        <div className="panel-collapse-body account-panel-body">
          <div className="account-summary">
            <strong>{currentUser?.display_name || currentUser?.email}</strong>
            <span>{currentUser?.email}</span>
          </div>

          <div className="account-section account-section-compact">
            <div className="account-search-row">
              <input
                type="search"
                value={layoutSearchQuery}
                onChange={(event) => {
                  setLayoutSearchQuery(event.target.value);
                  setSavedLayoutsOpen(true);
                }}
                placeholder="Search saved layouts"
              />
            </div>
            <button
              className="account-section-trigger"
              type="button"
              onClick={() => setSavedLayoutsOpen((current) => !current)}
              aria-expanded={savedLayoutsOpen}
            >
              <span>Saved Layouts</span>
              <strong>{filteredSavedLayouts.length}</strong>
            </button>
            {savedLayoutsOpen ? (
              <div className="account-list account-scroll-list">
                {savedLayouts.length === 0 ? (
                  <p className="muted">No layouts saved yet.</p>
                ) : filteredSavedLayouts.length === 0 ? (
                  <p className="muted">No saved layouts match this search.</p>
                ) : (
                  filteredSavedLayouts.map((layout) => (
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
            ) : null}
          </div>

          <div className="account-section account-section-compact">
            <button
              className="account-section-trigger"
              type="button"
              onClick={() => setSavedRendersOpen((current) => !current)}
              aria-expanded={savedRendersOpen}
            >
              <span>Saved Renders</span>
              <strong>{visibleSavedRenders.length}</strong>
            </button>
            {savedRendersOpen ? (
              <>
                {activeRootLayoutId ? (
                  <p className="muted">
                    Showing renders linked to the loaded root layout.
                  </p>
                ) : null}
                <div className="account-render-grid account-render-scroll">
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
                        onDoubleClick={() => handleReuseSavedRender(render)}
                      >
                        <img src={render.file_url} alt={render.prompt} />
                        <div className="saved-render-meta">
                          <strong>{render.model_name}</strong>
                          <span>{render.source}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
};
