import React, { useEffect } from 'react';
import { Canvas2D } from './components/Canvas2D';
import { InspectorPanel } from './components/InspectorPanel';
import { LibraryPanel } from './components/LibraryPanel';
import { RunLogPanel } from './components/RunLogPanel';
import { UserInputPanel } from './components/UserInputPanel';
import { ValidationPanel } from './components/ValidationPanel';
import { Toolbar } from './components/Toolbar';
import { ClusterPreviewPanel } from './components/ClusterPreviewPanel';
import { LayoutVariantsGallery } from './components/LayoutVariantsGallery';
import { ThreeDPreviewPanel } from './components/ThreeDPreviewPanel';
import { useEditorStore } from './store';
import { fetchInventory } from './api';
import sampleFloorplan from './sample_floorplan.json';

export const App: React.FC = () => {
  const {
    setFloorplan,
    setInventory,
    tenantId,
    setBaseFloorplan,
    appendRunLog,
    viewMode,
    layoutVariants,
    variantPickerOpen,
  } = useEditorStore();

  useEffect(() => {
    setFloorplan(sampleFloorplan as any);
    setBaseFloorplan(sampleFloorplan as any);
    fetchInventory(tenantId)
      .then((items) => {
        setInventory(items);
        appendRunLog(`Loaded inventory: ${items.length} items`);
      })
      .catch((err) => {
        setInventory([]);
        appendRunLog(
          `Inventory load failed: ${err instanceof Error ? err.message : 'unknown error'}`
        );
      });
  }, [setFloorplan, setBaseFloorplan, setInventory, appendRunLog, tenantId]);

  return (
    <div className="app-root">
      <header>
        <h1>TKNT Floorplan Editor</h1>
      </header>
      <Toolbar />
      <div className="workspace">
        <LibraryPanel />
        <div className="canvas-wrapper">
          {variantPickerOpen && layoutVariants.length > 0 ? (
            <LayoutVariantsGallery />
          ) : viewMode === '3d' ? (
            <ThreeDPreviewPanel />
          ) : (
            <Canvas2D />
          )}
        </div>
        <div className="side-stack">
          <UserInputPanel />
          <InspectorPanel />
        </div>
        <div className="side-stack">
          <ClusterPreviewPanel />
          <ValidationPanel />
          <RunLogPanel />
        </div>
      </div>
    </div>
  );
};
