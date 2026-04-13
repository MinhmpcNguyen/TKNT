import React, { useState } from 'react';
import { createSavedLayout } from '../api';
import { useEditorStore } from '../store';
import { DesignPlanV1, FloorplanGeometry, SavedLayoutRecord } from '../types';

export const AccountPanel: React.FC = () => {
  const {
    currentUser,
    floorplan,
    design,
    styledResult,
    savedLayouts,
    savedRenders,
    setSavedLayouts,
    setFloorplan,
    setDesign,
    setStyledResult,
    clearLayoutVariants,
    setViewMode,
    appendRunLog,
  } = useEditorStore();
  const [layoutName, setLayoutName] = useState('');
  const [savingLayout, setSavingLayout] = useState(false);
  const [layoutError, setLayoutError] = useState<string | null>(null);

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
    setViewMode('2d');
    appendRunLog(`Loaded saved layout "${layout.name}".`);
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
          <span>{savedRenders.length}</span>
        </div>
        <div className="account-render-grid">
          {savedRenders.length === 0 ? (
            <p className="muted">Your Gemini renders will appear here.</p>
          ) : (
            savedRenders.map((render) => (
              <div key={render.id} className="saved-render-card">
                <img src={render.file_url} alt={render.prompt} />
                <div className="saved-render-meta">
                  <strong>{render.model_name}</strong>
                  <span>{render.source}</span>
                  <p>{render.prompt}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
};
