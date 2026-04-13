import React, { useEffect, useState } from 'react';
import { clearStoredAuthToken, fetchInventory, getCurrentUser, listGeneratedRenders, listSavedLayouts, logoutUser } from './api';
import { AuthScreen } from './components/AuthScreen';
import { AccountPanel } from './components/AccountPanel';
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
import sampleFloorplan from './sample_floorplan.json';

export const App: React.FC = () => {
  const {
    authResolved,
    currentUser,
    tenantId,
    setAuthResolved,
    setAuthSession,
    clearAuthSession,
    setFloorplan,
    setBaseFloorplan,
    setInventory,
    setSavedLayouts,
    setSavedRenders,
    appendRunLog,
    viewMode,
    layoutVariants,
    variantPickerOpen,
  } = useEditorStore();
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    setFloorplan(sampleFloorplan as any);
    setBaseFloorplan(sampleFloorplan as any);
  }, [setBaseFloorplan, setFloorplan]);

  useEffect(() => {
    let cancelled = false;
    const bootstrap = async () => {
      try {
        const user = await getCurrentUser();
        if (cancelled) {
          return;
        }
        const token = window.localStorage.getItem('tknt.authToken');
        if (!token) {
          clearStoredAuthToken();
          clearAuthSession();
          return;
        }
        setAuthSession(token, user);
      } catch {
        clearStoredAuthToken();
        clearAuthSession();
      } finally {
        if (!cancelled) {
          setAuthResolved(true);
        }
      }
    };
    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [clearAuthSession, setAuthResolved, setAuthSession]);

  useEffect(() => {
    if (!currentUser) {
      return;
    }
    let cancelled = false;
    const loadWorkspace = async () => {
      try {
        const [items, layouts, renders] = await Promise.all([
          fetchInventory(tenantId),
          listSavedLayouts(),
          listGeneratedRenders(),
        ]);
        if (cancelled) {
          return;
        }
        setInventory(items);
        setSavedLayouts(layouts);
        setSavedRenders(renders);
        setBootError(null);
        appendRunLog(`Loaded inventory: ${items.length} items`);
      } catch (loadError) {
        if (cancelled) {
          return;
        }
        setInventory([]);
        setSavedLayouts([]);
        setSavedRenders([]);
        const message =
          loadError instanceof Error ? loadError.message : 'unknown error';
        setBootError(message);
        appendRunLog(`Workspace load failed: ${message}`);
      }
    };
    void loadWorkspace();
    return () => {
      cancelled = true;
    };
  }, [
    appendRunLog,
    currentUser,
    setInventory,
    setSavedLayouts,
    setSavedRenders,
    tenantId,
  ]);

  const handleLogout = async () => {
    try {
      await logoutUser();
    } catch {
      // The client-side token is the only thing that matters for ending the session locally.
    } finally {
      clearStoredAuthToken();
      clearAuthSession();
      setBootError(null);
    }
  };

  if (!authResolved) {
    return <div className="app-loading">Loading workspace...</div>;
  }

  if (!currentUser) {
    return <AuthScreen />;
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>TKNT Floorplan Editor</h1>
          <p className="app-subtitle">
            Shared inventory for everyone, plus your own GLB objects, saved layouts,
            and render history.
          </p>
        </div>
        <div className="app-account-actions">
          <div className="app-account-chip">
            <strong>{currentUser.display_name || currentUser.email}</strong>
            <span>{currentUser.email}</span>
          </div>
          <button onClick={handleLogout} type="button">
            Log out
          </button>
        </div>
      </header>
      {bootError ? <div className="app-banner-error">{bootError}</div> : null}
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
          <AccountPanel />
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
